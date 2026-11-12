"""Melody transcription: a monophonic lead line, via pluggable backends.

Same registry contract as :mod:`songscribe.chords` and :mod:`songscribe.beats`:
one function per backend in :data:`BACKENDS`, its rate in
:data:`BACKEND_SAMPLE_RATES`, every setting a defaulted parameter, never
``**kwargs``. A backend returns :class:`~songscribe.types.NoteEvent` list on
the recording's own seconds-from-start timeline.

Pitch tracking and note segmentation are two steps. A tracker gives a pitch per
frame; how long a note has to be and how far it may wander is a separate
decision, and :func:`notes_from_f0` keeps it out of the backend, where it is
testable without audio and reusable by the next tracker.

``pyin`` is the default of this stage because it is the only backend in it. It
is a signal-processing tracker, the floor a model has to beat, like the
``template`` chord backend. See ``MODELS.md``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from math import isfinite

import numpy as np

from .registry import Registry, positive_number, setting_rules, whole_number
from .types import NoteEvent

log = logging.getLogger(__name__)

DEFAULT_BACKEND = "pyin"

#: librosa's own default rate, and ample for a lead line: the highest
#: fundamental here is 2 kHz, far below the Nyquist limit of 11 kHz.
PYIN_SAMPLE_RATE = 22_050

#: The f0 range a lead is searched in, in hertz: C2 to C7, a bass voice up to
#: the top of a flute. The ceiling is an octave above a sung lead because pYIN
#: reports the octave below for notes near its own ceiling, so one set at the
#: written range buys octave errors rather than safety.
LEAD_FMIN_HZ = 65.41
LEAD_FMAX_HZ = 2093.0

#: pYIN's analysis window and step. 2048 samples at 22.05 kHz is 93 ms, which
#: holds six periods of the lowest searched pitch; the 512-sample hop gives a
#: pitch every 23 ms, finer than any note this is meant to find.
PYIN_FRAME_LENGTH = 2048
PYIN_HOP_LENGTH = 512

#: Shorter runs of one pitch are tracker flicker, not notes: 80 ms is a
#: semiquaver at 190 bpm, below anything a lead line sustains.
MIN_NOTE_SECONDS = 0.08

#: How far the pitch may wander from a note's running mean before it is a new
#: note. Above a semitone a real step would be swallowed; at half a semitone
#: ordinary vibrato would split one note into several.
NOTE_TOLERANCE_CENTS = 60.0

#: MIDI note 69 is A4, at 440 Hz: the anchor the conversion is defined from.
A4_HZ = 440.0
A4_MIDI = 69

CENTS_PER_SEMITONE = 100.0


def hz_to_midi(hz: np.ndarray | float) -> np.ndarray | float:
    """Frequencies in hertz as (fractional) MIDI note numbers.

    Local rather than librosa's, which this module imports only inside the backend.
    """
    return 12.0 * np.log2(np.asarray(hz, dtype=float) / A4_HZ) + A4_MIDI


def midi_to_hz(pitch: np.ndarray | float) -> np.ndarray | float:
    """MIDI note numbers as frequencies in hertz."""
    return A4_HZ * 2.0 ** ((np.asarray(pitch, dtype=float) - A4_MIDI) / 12.0)


# --------------------------------------------------------------------------- #
# Frames to notes
# --------------------------------------------------------------------------- #


def notes_from_f0(
    times: Sequence[float] | np.ndarray,
    f0_hz: Sequence[float] | np.ndarray,
    voiced: Sequence[bool] | np.ndarray,
    *,
    min_duration: float = MIN_NOTE_SECONDS,
    tolerance_cents: float = NOTE_TOLERANCE_CENTS,
    confidence: Sequence[float] | np.ndarray | None = None,
) -> list[NoteEvent]:
    """Segment a frame-wise f0 contour into notes.

    ``times`` are frame start times in seconds, evenly spaced as a tracker returns them;
    ``f0_hz`` is the pitch found in each frame, ``voiced`` whether one was found at all. A
    note runs while the frames stay voiced, contiguous, and within ``tolerance_cents`` of
    its running mean pitch, and is kept if it lasts ``min_duration``. Its pitch is the
    median of its frames, rounded, so a few frames at the window's edge cannot shift it.
    """
    times = np.asarray(times, dtype=float)
    f0_hz = np.asarray(f0_hz, dtype=float)
    voiced = np.asarray(voiced, dtype=bool)
    if not times.shape == f0_hz.shape == voiced.shape:
        raise ValueError(
            f"times, f0_hz and voiced must be the same length, got "
            f"{times.shape}, {f0_hz.shape} and {voiced.shape}"
        )
    if times.ndim != 1:
        raise ValueError(f"expected one frame per entry, got an array of shape {times.shape}")
    if confidence is not None:
        confidence = np.asarray(confidence, dtype=float)
        if confidence.shape != times.shape:
            raise ValueError(
                f"confidence must be one value per frame, got {confidence.shape} "
                f"for {times.shape[0]} frames"
            )
    if min_duration < 0:
        raise ValueError(f"min_duration must not be negative, got {min_duration}")
    if tolerance_cents <= 0:
        raise ValueError(f"tolerance_cents must be positive, got {tolerance_cents}")

    # One hop for every frame, from the median gap, so a frame that happens to
    # sit before a hole still ends one hop after it starts rather than
    # stretching over sound nothing was tracked in.
    hop = float(np.median(np.diff(times))) if times.size > 1 else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        pitches = hz_to_midi(np.where(f0_hz > 0, f0_hz, np.nan))
    usable = voiced & np.isfinite(pitches)
    tolerance = tolerance_cents / CENTS_PER_SEMITONE

    notes: list[NoteEvent] = []
    run: list[int] = []

    def close() -> None:
        if not run:
            return
        start, end = float(times[run[0]]), float(times[run[-1]] + hop)
        held = [pitches[i] for i in run]
        if end - start >= min_duration and end > start:
            notes.append(
                NoteEvent(
                    start=start,
                    end=end,
                    pitch=round(float(np.median(held))),
                    confidence=(
                        1.0 if confidence is None else float(np.mean(confidence[list(run)]))
                    ),
                )
            )
        run.clear()

    for frame in range(times.size):
        if not usable[frame]:
            close()
            continue
        if run:
            # A gap in the frames ends the note: welding across it would claim
            # time nothing was heard in. Half a hop of slack, which is float
            # noise rather than a gap.
            hole = times[frame] - times[run[-1]] > 1.5 * hop
            # Against the mean of the frames so far, not the first frame:
            # vibrato swings either side of the pitch, and a note starting on
            # the swing would otherwise be measured from its extreme.
            stepped = abs(pitches[frame] - np.mean([pitches[i] for i in run])) > tolerance
            if hole or stepped:
                close()
        run.append(frame)
    close()
    return notes


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


def pyin_f0(
    audio: np.ndarray,
    sample_rate: int,
    fmin: float = LEAD_FMIN_HZ,
    fmax: float = LEAD_FMAX_HZ,
    frame_length: int = PYIN_FRAME_LENGTH,
    hop_length: int = PYIN_HOP_LENGTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run pYIN over ``audio``; return ``(times, f0_hz, voiced, confidence)``.

    The contour before any segmentation, which is what
    ``eval.melody.raw_pitch_accuracy`` measures a tracker on.
    """
    import librosa

    if not fmin < fmax:
        raise ValueError(f"fmin must be below fmax, got {fmin} and {fmax}")

    f0, voiced, confidence = librosa.pyin(
        audio,
        sr=sample_rate,
        fmin=fmin,
        fmax=fmax,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    times = librosa.times_like(f0, sr=sample_rate, hop_length=hop_length)
    return times, f0, voiced, confidence


@setting_rules(
    fmin=positive_number("hertz"),
    fmax=positive_number("hertz"),
    frame_length=whole_number("samples"),
    hop_length=whole_number("samples"),
    min_duration=positive_number("seconds"),
    tolerance_cents=positive_number("cents"),
)
def _transcribe_pyin(
    audio: np.ndarray,
    sample_rate: int,
    fmin: float = LEAD_FMIN_HZ,
    fmax: float = LEAD_FMAX_HZ,
    frame_length: int = PYIN_FRAME_LENGTH,
    hop_length: int = PYIN_HOP_LENGTH,
    min_duration: float = MIN_NOTE_SECONDS,
    tolerance_cents: float = NOTE_TOLERANCE_CENTS,
) -> list[NoteEvent]:
    """Probabilistic YIN pitch tracking, then :func:`notes_from_f0`.

    Signal processing rather than a model, and installed already since librosa is
    a core dependency. It is monophonic, so on a full mix it tracks whatever is
    loudest and periodic, usually the bass. Run it on a separated stem, or narrow
    ``fmin``/``fmax`` to the lead's own range.
    """
    # pYIN is not a trained model and works at whatever rate it is handed;
    # BACKEND_SAMPLE_RATES only says which rate the caller is asked to load.
    times, f0, voiced, confidence = pyin_f0(
        audio,
        sample_rate,
        fmin=fmin,
        fmax=fmax,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    return notes_from_f0(
        times,
        f0,
        voiced,
        min_duration=min_duration,
        tolerance_cents=tolerance_cents,
        confidence=confidence,
    )


BACKENDS: dict[str, Callable[..., list[NoteEvent]]] = {
    "pyin": _transcribe_pyin,
}

#: Sample rate each backend wants its audio in.
BACKEND_SAMPLE_RATES = {
    "pyin": PYIN_SAMPLE_RATE,
}


_REGISTRY = Registry("melody", BACKENDS, BACKEND_SAMPLE_RATES)

# The checks live in :mod:`songscribe.registry`, shared with the other
# stages; these are this module's public names for them.
_check_backend = _REGISTRY.check_backend
_check_options = _REGISTRY.check_options
backend_options = _REGISTRY.options
accepts_any_option = _REGISTRY.accepts_any_option
sample_rate_for = _REGISTRY.sample_rate_for


def transcribe(
    audio: np.ndarray,
    sample_rate: int,
    backend: str = DEFAULT_BACKEND,
    **kwargs,
) -> list[NoteEvent]:
    """Estimate a monophonic melody for ``audio`` (mono float32).

    Extra keyword arguments are the chosen backend's settings; one it does not
    accept raises rather than being dropped.
    """
    _check_backend(backend)
    _check_options(backend, kwargs)
    log.debug("melody backend=%s sample_rate=%d", backend, sample_rate)
    return BACKENDS[backend](audio, sample_rate, **kwargs)


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def transpose(notes: Sequence[NoteEvent], semitones: int) -> list[NoteEvent]:
    """``notes`` with every pitch moved ``semitones`` up."""
    return [NoteEvent(n.start, n.end, n.pitch + semitones, n.confidence) for n in notes]


def pitch_profile(notes: Sequence[NoteEvent]) -> np.ndarray:
    """How long each of the twelve pitch classes sounds, as a 12-vector.

    Duration-weighted, so one long tonic counts for more than a passing note.
    """
    profile = np.zeros(12)
    for note in notes:
        if isfinite(note.duration) and note.duration > 0:
            profile[note.pitch_class] += note.duration
    return profile
