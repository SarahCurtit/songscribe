"""Chord recognition from audio, via pluggable deep-learning backends.

Every production backend is a pretrained open-source model with openly
published weights -- see ``MODELS.md`` for the inventory and the bar a new one
has to clear. A backend is one function returning
:class:`~songscribe.types.ChordEvent` list, registered in :data:`BACKENDS`
with its required sample rate in :data:`BACKEND_SAMPLE_RATES`.
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

# Models label chords in Harte notation ("C:maj", "A:min"); charts want compact
# guitarist shorthand ("C", "Am"). Backends normalise on the way out so the rest
# of the pipeline only ever sees one vocabulary.
HARTE_QUALITIES = {
    "maj": "",
    "min": "m",
    "dim": "dim",
    "aug": "aug",
    "maj7": "maj7",
    "min7": "m7",
    "7": "7",
    "dim7": "dim7",
    "hdim7": "m7b5",
    "minmaj7": "mmaj7",
    "maj6": "6",
    "min6": "m6",
    "9": "9",
    "maj9": "maj9",
    "min9": "m9",
    "sus2": "sus2",
    "sus4": "sus4",
}

DEFAULT_BACKEND = "madmom"

# Both DL backends expect 44.1 kHz; they were trained on it.
MODEL_SAMPLE_RATE = 44_100


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


def _recognise_madmom(audio: np.ndarray, sample_rate: int, **kwargs) -> list[ChordEvent]:
    """CNN chord features + CRF decoding, via madmom's pretrained weights."""
    from madmom.audio.signal import Signal
    from madmom.features.chords import CNNChordFeatureProcessor, CRFChordRecognitionProcessor

    if sample_rate != MODEL_SAMPLE_RATE:
        raise ValueError(f"madmom backend expects {MODEL_SAMPLE_RATE} Hz, got {sample_rate}")

    # Wrap in a madmom Signal rather than handing over a bare array: the
    # processor chain reads the rate off the signal, and a plain ndarray would
    # silently inherit whatever default it was configured with.
    signal = Signal(audio, sample_rate=sample_rate, num_channels=1)
    features = CNNChordFeatureProcessor()(signal)
    # A structured array with ('start', 'end', 'label') fields, labels in Harte
    # notation -- verified against madmom 0.17.dev0.
    segments = CRFChordRecognitionProcessor()(features)
    return [
        ChordEvent(start=float(start), end=float(end), label=normalise_label(str(label)))
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
    best = _smooth_labels(np.argmax(templates @ chroma, axis=0), smoothing)
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


def normalise_label(label: str) -> str:
    """Convert a Harte-notation chord label to compact chart shorthand.

    ``C:maj`` -> ``C``, ``A:min`` -> ``Am``, ``X`` (unknown) -> ``N``.
    An unrecognised quality is passed through untouched rather than guessed
    at, so a model with a wider vocabulary than :data:`HARTE_QUALITIES` shows
    up as odd output instead of being silently mislabelled.
    """
    label = label.strip()
    if not label or label == "X":
        return NO_CHORD
    if ":" not in label:
        return label
    root, _, quality = label.partition(":")
    # Drop the inversion: "C:maj/3" names a bass note a chord chart won't print.
    quality = quality.partition("/")[0]
    if quality in HARTE_QUALITIES:
        return f"{root}{HARTE_QUALITIES[quality]}"
    return label


def _check_backend(backend: str) -> None:
    """Reject an unknown backend by name.

    Never falls back to another model: a typo in a config should fail loudly,
    not quietly downgrade the output to something worse.
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown chord backend {backend!r}, expected one of {sorted(BACKENDS)}")


def sample_rate_for(backend: str) -> int:
    """The sample rate ``backend`` needs its audio loaded at."""
    _check_backend(backend)
    return BACKEND_SAMPLE_RATES[backend]


def recognise(
    audio: np.ndarray,
    sample_rate: int,
    backend: str = DEFAULT_BACKEND,
    **kwargs,
) -> list[ChordEvent]:
    """Estimate a chord timeline for ``audio`` (mono float32)."""
    _check_backend(backend)
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


def _smooth_labels(indices: np.ndarray, width: int) -> np.ndarray:
    """Majority-vote each frame against its neighbours to suppress flicker.

    Deliberately not a median filter: these are *nominal* label indices, so the
    median of a window can name a chord that never occurred in it (the median
    of C and B is somewhere around F#). Mode is the only meaningful order
    statistic on unordered categories.
    """
    if width <= 1 or len(indices) == 0:
        return indices
    width |= 1  # even widths would leave sliding_window_view one frame short
    padded = np.pad(indices, width // 2, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, width)
    n_labels = int(indices.max()) + 1
    return np.array([np.bincount(window, minlength=n_labels).argmax() for window in windows])


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
