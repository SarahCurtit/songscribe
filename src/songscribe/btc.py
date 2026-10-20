"""BTC chord recognition (Park et al., ISMIR 2019) in PyTorch, large-vocabulary model.

A compact re-implementation of the network in https://github.com/jayg996/BTC-ISMIR19
(MIT, (c) 2019 Jonggwon Park), written so the published checkpoint loads unchanged. The
checkpoint is fetched on first use from a pinned upstream commit and checked against a
SHA-256, then cached; it is not kept in this repository. ``chords._recognise_btc`` is
the only caller, so ``torch`` is imported only when that backend runs.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch import nn

log = logging.getLogger(__name__)

# The checkpoint is a file in the upstream repository, so it is fetched from that commit.
UPSTREAM_COMMIT = "2682317be668032e6e4b269ded36adaa2ad57df0"
CHECKPOINT_URL = (
    "https://raw.githubusercontent.com/jayg996/BTC-ISMIR19/"
    f"{UPSTREAM_COMMIT}/test/btc_model_large_voca.pt"
)
CHECKPOINT_SHA256 = "1673d23f8f9a55ae7f9e8b80a51da616debb22675b8d8b67ea6ce0ef37b0ab51"

# Fixed by how the checkpoint was trained.
SAMPLE_RATE = 22_050
HOP_LENGTH = 2048
N_BINS = 144
BINS_PER_OCTAVE = 24
# The network sees 108 frames (10 s) at a time, and upstream computes the CQT per 10 s
# chunk; so does this, since the chunk edges are part of what the weights were trained on.
CHUNK_SECONDS = 10.0
TIMESTEPS = 108
HIDDEN = 128
LAYERS = 8
HEADS = 4
NUM_CHORDS = 170

# Seconds per frame. A 10 s chunk gives 108 CQT frames (not 10 s / hop = 107.7), so this is
# not HOP_LENGTH / SAMPLE_RATE: using that would drift 0.3% per chunk, ~0.7 s over a song.
FRAME_SECONDS = CHUNK_SECONDS / TIMESTEPS


def _timing_signal(length: int, channels: int) -> torch.Tensor:
    """Sinusoidal position signal, shape ``(1, length, channels)``."""
    half = channels // 2
    increment = math.log(1.0e4) / (half - 1)
    inverse = np.exp(np.arange(half) * -increment)
    scaled = np.arange(length)[:, None] * inverse[None, :]
    signal = np.concatenate([np.sin(scaled), np.cos(scaled)], axis=1)
    return torch.from_numpy(signal).float().unsqueeze(0)


class _LayerNorm(nn.Module):
    # Upstream normalises by the unbiased std and adds eps to it, unlike nn.LayerNorm.
    def __init__(self, features: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class _Attention(nn.Module):
    def __init__(self, hidden: int, heads: int, mask: torch.Tensor) -> None:
        super().__init__()
        self.heads = heads
        self.scale = (hidden // heads) ** -0.5
        self.query_linear = nn.Linear(hidden, hidden, bias=False)
        self.key_linear = nn.Linear(hidden, hidden, bias=False)
        self.value_linear = nn.Linear(hidden, hidden, bias=False)
        self.output_linear = nn.Linear(hidden, hidden, bias=False)
        self.register_buffer("mask", mask, persistent=False)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, depth = x.shape
        return x.view(batch, steps, self.heads, depth // self.heads).permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        queries = self._split(self.query_linear(x)) * self.scale
        keys = self._split(self.key_linear(x))
        values = self._split(self.value_linear(x))
        logits = queries @ keys.transpose(-1, -2)
        steps = logits.shape[-1]
        logits = logits + self.mask[:steps, :steps]
        context = torch.softmax(logits, dim=-1) @ values
        batch, _, steps, width = context.shape
        merged = context.permute(0, 2, 1, 3).reshape(batch, steps, self.heads * width)
        return self.output_linear(merged)


class _Conv(nn.Module):
    """Width-3 convolution over ``(batch, time, channels)``, zero-padded to keep the length."""

    def __init__(self, inputs: int, outputs: int) -> None:
        super().__init__()
        self.pad = nn.ConstantPad1d((1, 1), 0.0)
        self.conv = nn.Conv1d(inputs, outputs, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pad(x.transpose(1, 2))).transpose(1, 2)


class _FeedForward(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Conv(hidden, hidden), _Conv(hidden, hidden)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ReLU after both convolutions, the last included, as the checkpoint was trained.
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x


class _Block(nn.Module):
    def __init__(self, hidden: int, heads: int, mask: torch.Tensor) -> None:
        super().__init__()
        self.multi_head_attention = _Attention(hidden, heads, mask)
        self.positionwise_convolution = _FeedForward(hidden)
        self.layer_norm_mha = _LayerNorm(hidden)
        self.layer_norm_ffn = _LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.multi_head_attention(self.layer_norm_mha(x))
        return x + self.positionwise_convolution(self.layer_norm_ffn(x))


class _BiDirectionalLayer(nn.Module):
    """One attention block that only looks back and one that only looks ahead, merged."""

    def __init__(self, hidden: int, heads: int, steps: int) -> None:
        super().__init__()
        future = torch.triu(torch.full((steps, steps), -torch.inf), diagonal=1)
        self.attn_block = _Block(hidden, heads, future)
        self.backward_attn_block = _Block(hidden, heads, future.T.contiguous())
        self.linear = nn.Linear(hidden * 2, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.cat([self.attn_block(x), self.backward_attn_block(x)], dim=2))


class _Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("timing_signal", _timing_signal(TIMESTEPS, HIDDEN), persistent=False)
        self.embedding_proj = nn.Linear(N_BINS, HIDDEN, bias=False)
        self.self_attn_layers = nn.Sequential(
            *[_BiDirectionalLayer(HIDDEN, HEADS, TIMESTEPS) for _ in range(LAYERS)]
        )
        self.layer_norm = _LayerNorm(HIDDEN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding_proj(x) + self.timing_signal[:, : x.shape[1]]
        return self.layer_norm(self.self_attn_layers(x))


class _Output(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output_projection = nn.Linear(HIDDEN, NUM_CHORDS)
        # Present in the checkpoint but unused at inference upstream; kept so it loads strictly.
        self.lstm = nn.LSTM(HIDDEN, HIDDEN // 2, batch_first=True, bidirectional=True)


class BTC(nn.Module):
    """Maps ``(batch, 108, 144)`` normalised log-CQT frames to chord logits per frame."""

    def __init__(self) -> None:
        super().__init__()
        self.self_attn_layers = _Encoder()
        self.output_layer = _Output()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer.output_projection(self.self_attn_layers(x))


def checkpoint_path(cache_dir: Path | None = None) -> Path:
    """Return the cached checkpoint, downloading and verifying it on first use."""
    root = cache_dir or Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    path = root / "songscribe" / f"btc_large_voca_{UPSTREAM_COMMIT[:8]}.pt"
    if path.exists():
        return path
    log.info("downloading the BTC checkpoint to %s", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(CHECKPOINT_URL, timeout=60) as response:
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != CHECKPOINT_SHA256:
        raise RuntimeError(f"BTC checkpoint has SHA-256 {digest}, expected {CHECKPOINT_SHA256}")
    # Written whole under another name first, so a partial download is not taken for a cached one.
    partial = path.with_suffix(".part")
    partial.write_bytes(data)
    partial.replace(path)
    return path


def load(checkpoint: str | os.PathLike[str] | None, device: torch.device):
    """Return ``(model, mean, std)`` from ``checkpoint``, or from the cached default."""
    path = Path(checkpoint) if checkpoint else checkpoint_path()
    if not path.is_file():
        raise FileNotFoundError(f"BTC checkpoint not found: {path}")
    # The mean and std are stored as numpy scalars, which weights_only loading refuses. The
    # default file is checked against CHECKPOINT_SHA256; a path passed in is the user's own.
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = BTC()
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model, float(state["mean"]), float(state["std"])


def features(audio: np.ndarray) -> np.ndarray:
    """Log-CQT of 22.05 kHz mono ``audio`` as ``(frames, 144)``, computed per 10 s chunk."""
    import librosa

    chunk = int(SAMPLE_RATE * CHUNK_SECONDS)
    pieces = []
    start = 0
    while len(audio) > start + chunk:
        pieces.append(_cqt(librosa, audio[start : start + chunk]))
        start += chunk
    pieces.append(_cqt(librosa, audio[start:]))
    return np.log(np.abs(np.concatenate(pieces, axis=1)) + 1e-6).T


def _cqt(librosa, segment: np.ndarray) -> np.ndarray:
    return librosa.cqt(
        segment,
        sr=SAMPLE_RATE,
        n_bins=N_BINS,
        bins_per_octave=BINS_PER_OCTAVE,
        hop_length=HOP_LENGTH,
    )


def predict(model: BTC, mean: float, std: float, feature: np.ndarray, device) -> np.ndarray:
    """Chord class index per frame, from ``(frames, 144)`` log-CQT features."""
    normalised = (feature - mean) / std
    pad = (-len(normalised)) % TIMESTEPS
    padded = np.pad(normalised, ((0, pad), (0, 0)))
    windows = torch.tensor(padded, dtype=torch.float32, device=device).view(-1, TIMESTEPS, N_BINS)
    with torch.no_grad():
        classes = model(windows).argmax(-1).reshape(-1).cpu().numpy()
    return classes[: len(normalised)]
