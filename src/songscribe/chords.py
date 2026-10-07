"""Chord recognition from audio, via pluggable deep-learning backends.

Every production backend here is a pretrained **open-source** model with openly
published weights -- no hosted APIs, no proprietary checkpoints. See
``MODELS.md`` for the model/licence inventory.

Backends
--------
``madmom`` (default)
    CNN chord features + CRF decoding (Korzeniowski & Widmer). Pip-installable
    with bundled weights, so it works out of the box. Major/minor vocabulary.

``btc``
    Bi-directional Transformer for Chord recognition (Park et al., ISMIR'19).
    Large vocabulary including sevenths and inversions, and the best quality of
    the three -- but needs a checkpoint fetched separately.

``template``
    Chroma/CQT matched against triad templates. **Not** deep learning: it is
    the baseline the models have to beat, and the only backend that runs in CI
    without downloading weights. Never the default.

Adding a backend means writing one function that returns a list of
:class:`~songscribe.types.ChordEvent` and registering it in :data:`BACKENDS`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from .types import ChordEvent

log = logging.getLogger(__name__)

PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

# Semitone offsets from the root for each triad quality.
QUALITIES = {"": (0, 4, 7), "m": (0, 3, 7)}

# Label the models emit for "no chord sounding" -- silence, drums, spoken word.
NO_CHORD = "N"

DEFAULT_BACKEND = "madmom"

# Both DL backends expect 44.1 kHz; they were trained on it.
MODEL_SAMPLE_RATE = 44_100


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


def _recognise_madmom(audio: np.ndarray, sample_rate: int, **kwargs) -> list[ChordEvent]:
    """CNN chord features + CRF decoding, via madmom's pretrained weights."""
    from madmom.features.chords import CNNChordFeatureProcessor, CRFChordRecognitionProcessor

    if sample_rate != MODEL_SAMPLE_RATE:
        raise ValueError(f"madmom backend expects {MODEL_SAMPLE_RATE} Hz, got {sample_rate}")

    features = CNNChordFeatureProcessor()(audio)
    segments = CRFChordRecognitionProcessor()(features)
    return [
        ChordEvent(start=float(start), end=float(end), label=str(label))
        for start, end, label in segments
    ]


def _recognise_btc(
    audio: np.ndarray,
    sample_rate: int,
    checkpoint: str | None = None,
    device: str = "auto",
    **kwargs,
) -> list[ChordEvent]:
    """Bi-directional Transformer chord recognition (BTC, ISMIR'19).

    Not yet wired up: BTC has no pip package, so this needs the upstream
    repo vendored or added as a submodule plus its published checkpoint. The
    signature is fixed so the rest of the pipeline is already backend-agnostic.
    """
    raise NotImplementedError(
        "the btc backend is not implemented yet -- see MODELS.md for the "
        "checkpoint source, and use backend='madmom' in the meantime"
    )


def _recognise_template(
    audio: np.ndarray,
    sample_rate: int,
    hop_length: int = 2048,
    smoothing: int = 9,
    **kwargs,
) -> list[ChordEvent]:
    """Non-DL baseline: chroma frames matched to triad templates.

    Kept so there is always something to compare a model against, and so the
    pipeline is runnable with no weights on disk. ``smoothing`` is a frame
    count; at the default hop and 22.05 kHz that is roughly one bar.
    """
    import librosa

    chroma = librosa.feature.chroma_cqt(y=audio, sr=sample_rate, hop_length=hop_length)
    norms = np.linalg.norm(chroma, axis=0, keepdims=True)
    chroma = chroma / np.maximum(norms, 1e-9)

    templates, labels = chord_templates()
    best = _median_smooth(np.argmax(templates @ chroma, axis=0), smoothing)
    times = librosa.frames_to_time(np.arange(len(best) + 1), sr=sample_rate, hop_length=hop_length)
    return _runs_to_events(best, times, labels)


BACKENDS: dict[str, Callable[..., list[ChordEvent]]] = {
    "madmom": _recognise_madmom,
    "btc": _recognise_btc,
    "template": _recognise_template,
}

#: Sample rate each backend wants its audio in.
BACKEND_SAMPLE_RATES = {
    "madmom": MODEL_SAMPLE_RATE,
    "btc": MODEL_SAMPLE_RATE,
    "template": 22_050,
}


def recognise(
    audio: np.ndarray,
    sample_rate: int,
    backend: str = DEFAULT_BACKEND,
    **kwargs,
) -> list[ChordEvent]:
    """Estimate a chord timeline for ``audio`` (mono float32).

    Raises :class:`ValueError` for an unknown backend rather than silently
    falling back, so a typo in a config never quietly downgrades quality.
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown chord backend {backend!r}, expected one of {sorted(BACKENDS)}")
    log.debug("chord backend=%s sample_rate=%d", backend, sample_rate)
    return BACKENDS[backend](audio, sample_rate, **kwargs)


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def chord_templates() -> tuple[np.ndarray, list[str]]:
    """The 24 binary major/minor triad templates and their labels.

    Returns a ``(24, 12)`` matrix of L2-normalised templates, so matching a
    normalised chroma frame is a plain dot product.
    """
    templates, labels = [], []
    for quality, intervals in QUALITIES.items():
        for root, name in enumerate(PITCH_CLASSES):
            template = np.zeros(12)
            for interval in intervals:
                template[(root + interval) % 12] = 1.0
            templates.append(template / np.linalg.norm(template))
            labels.append(f"{name}{quality}")
    return np.vstack(templates), labels


def _median_smooth(indices: np.ndarray, width: int) -> np.ndarray:
    """Suppress single-frame flickers without blurring real changes."""
    if width <= 1:
        return indices
    padded = np.pad(indices, width // 2, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, width)
    return np.median(windows, axis=1).astype(int)


def _runs_to_events(indices: np.ndarray, times: np.ndarray, labels: list[str]) -> list[ChordEvent]:
    """Convert a per-frame label index sequence into timed chord spans."""
    events: list[ChordEvent] = []
    start = 0
    for frame in range(1, len(indices) + 1):
        if frame == len(indices) or indices[frame] != indices[start]:
            events.append(
                ChordEvent(
                    start=float(times[start]),
                    end=float(times[frame]),
                    label=labels[indices[start]],
                )
            )
            start = frame
    return events


def strip_no_chord(chords: list[ChordEvent]) -> list[ChordEvent]:
    """Drop ``N`` (no-chord) spans, which models emit over intros and drums."""
    return [c for c in chords if c.label != NO_CHORD]


def estimate_key(chords: list[ChordEvent]) -> str | None:
    """Guess the key as the longest-held chord label.

    Crude but serviceable for popular music, where the tonic usually gets the
    most time. A Krumhansl-style profile -- or just reading the key off a
    dedicated model -- would do better if this starts mattering.
    """
    sounding = strip_no_chord(chords)
    if not sounding:
        return None
    totals: dict[str, float] = {}
    for chord in sounding:
        totals[chord.label] = totals.get(chord.label, 0.0) + chord.duration
    return max(totals, key=totals.get)
