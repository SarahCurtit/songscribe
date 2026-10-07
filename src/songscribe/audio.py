"""Audio loading.

Each model in the pipeline was trained at its own sample rate -- Whisper on
16 kHz, the chord models on 44.1 kHz -- and resampling away from that costs
accuracy. So loading is parameterised rather than fixed, and the caller asks
for the rate its model wants. Always returns mono float32.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ASR_SAMPLE_RATE = 16_000


def load(path: str | Path, sample_rate: int) -> tuple[np.ndarray, int]:
    """Load ``path`` as mono float32 at ``sample_rate``."""
    import librosa

    audio, sr = librosa.load(str(path), sr=sample_rate, mono=True)
    return audio.astype(np.float32), int(sr)


def duration(path: str | Path) -> float:
    """Length of ``path`` in seconds, without decoding the whole file."""
    import librosa

    return float(librosa.get_duration(path=str(path)))
