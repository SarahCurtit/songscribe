"""Chord recognition from audio, via pluggable deep-learning backends.

Every production backend is a pretrained open-source model with openly
published weights -- see ``MODELS.md`` for the bar a new one has to clear. A
backend is one function returning :class:`~songscribe.types.ChordEvent` list,
registered in :data:`BACKENDS` with its rate in :data:`BACKEND_SAMPLE_RATES`,
taking each setting it accepts as a defaulted parameter, never ``**kwargs``.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Iterable

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


def _recognise_madmom(audio: np.ndarray, sample_rate: int) -> list[ChordEvent]:
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


#: Parameter kinds a setting can be passed as. A backend's own positional-only
#: or ``*args`` parameters are unreachable through :func:`recognise`, which
#: forwards settings by keyword.
_KEYWORD_KINDS = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)

#: Variadic parameter kinds. They carry no default and need none -- an omitted
#: ``*args``/``**kwargs`` is simply empty -- so the rule below that every
#: declared setting be optional does not apply to them.
_VARIADIC_KINDS = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)


def backend_options(backend: str) -> set[str]:
    """The settings ``backend`` accepts, by name.

    Read off the backend function's own signature, so a backend declares its
    settings once by taking them as parameters. The first two are the audio and
    its sample rate, which :func:`recognise` passes positionally. An empty set
    means the backend takes no settings; a backend that accepts *anything* is a
    separate question, asked of :func:`accepts_any_option`. Raises
    :exc:`RuntimeError` for a backend that cannot be called with the audio and
    its rate alone.
    """
    _check_backend(backend)
    parameters = list(inspect.signature(BACKENDS[backend]).parameters.values())
    options = set()
    for parameter in parameters[2:]:
        # A setting without a default is not optional, so it cannot be reported
        # as one: `recognise` calls a backend with the audio and its rate alone,
        # and a caller who names nothing would get a TypeError from the call --
        # the one error path this module promises, routed around. Checked before
        # the keyword filter below, so a required positional-only parameter is
        # named here too rather than slipping through as "no settings".
        #
        # A misdeclared backend is this package's own bug, not a caller's bad
        # input, so it is not the ValueError the caller-facing checks raise: a
        # CLI turning ValueError into a usage error must not blame the user for
        # it. Nothing runs when a backend is added to BACKENDS -- a plain dict
        # -- so the fault surfaces on the first dispatch to it, which is why
        # this check is repeated on every call rather than done once.
        if parameter.default is inspect.Parameter.empty and parameter.kind not in _VARIADIC_KINDS:
            raise RuntimeError(
                f"chord backend {backend!r} declares setting {parameter.name!r} with no "
                "default; a backend must be callable with the audio and its rate alone"
            )
        if parameter.kind not in _KEYWORD_KINDS:
            continue
        options.add(parameter.name)
    return options


def accepts_any_option(backend: str) -> bool:
    """Whether ``backend`` keeps a catch-all, and so accepts every setting.

    Such a backend cannot be checked -- it really does take anything -- so this
    reports the exemption instead of folding it into :func:`backend_options`,
    where "anything" and "nothing" would be two indistinguishable empty
    answers. Nothing registered here is exempt, and a test holds that: a
    ``**kwargs`` on a backend is how the silently-ignored setting comes back.
    """
    _check_backend(backend)
    return any(
        p.kind is inspect.Parameter.VAR_KEYWORD
        for p in inspect.signature(BACKENDS[backend]).parameters.values()
    )


def _check_options(backend: str, options: Iterable[str]) -> None:
    """Reject settings ``backend`` has no parameter for.

    The same rule as :func:`_check_backend`, one level down: a misspelled
    setting that is quietly dropped is indistinguishable from a model that
    ignored it, which is how a metre setting went unnoticed on the default beat
    backend. Raised here rather than in each backend so there is one error path.
    """
    # Asked first, and of every backend: a catch-all is exempt from the list of
    # accepted names, not from `backend_options`' own rule that each declared
    # setting be optional. Returning early on the exemption would let a backend
    # with both a catch-all and a required setting past this gate and straight
    # into the bare TypeError from the call that this check replaces.
    accepted = backend_options(backend)
    if accepts_any_option(backend):
        return
    unknown = sorted(set(options) - accepted)
    if not unknown:
        return
    named = ", ".join(repr(name) for name in unknown)
    noun = "setting" if len(unknown) == 1 else "settings"
    detail = f"expected one of {sorted(accepted)}" if accepted else "which takes no settings"
    raise ValueError(f"unknown {noun} {named} for chord backend {backend!r}, {detail}")


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
    """Estimate a chord timeline for ``audio`` (mono float32).

    Extra keyword arguments are the chosen backend's settings; one it does not
    accept raises rather than being dropped.
    """
    _check_backend(backend)
    _check_options(backend, kwargs)
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
