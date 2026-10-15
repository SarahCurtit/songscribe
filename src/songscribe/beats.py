"""Beat and downbeat tracking, via pluggable backends.

Chord models emit boundaries wherever their frames fall; a musician reads
bars. This module supplies the bar lines.

Same registry contract as :mod:`songscribe.chords`: one function per backend,
registered in :data:`BACKENDS` with its rate in :data:`BACKEND_SAMPLE_RATES`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import isnan, nan
from statistics import median

import numpy as np

from .types import BeatEvent

log = logging.getLogger(__name__)

DEFAULT_BACKEND = "madmom"

#: madmom's downbeat RNN was trained at 44.1 kHz, like its chord model.
BEAT_SAMPLE_RATE = 44_100

#: Frame rate of the RNN's activation function. The DBN has to be told, and
#: getting it wrong scales every beat time rather than failing.
MADMOM_FPS = 100

#: Metres the tracker is allowed to consider. The DBN takes a candidate list;
#: jazz standards are in 4/4 or 3/4, and narrowing from madmom's wider default
#: both halves the work and removes the 5/4 and 7/8 fits that would wreck a
#: bar grid built on top of it.
METRE_CANDIDATES = (3, 4)

#: Assumed when a backend tracks beats without identifying the downbeat, and
#: when a beat timeline carries no position information at all.
DEFAULT_METRE = 4

#: How far one bar may differ from the median before it is called unreliable,
#: as a fraction. This catches a *gross* grid fault -- a dropped or added bar,
#: or a stretch tracked at half tempo -- and deliberately nothing finer.
#:
#: Measured on one real 236 s take: across the steady body of the performance
#: individual tracked bars still scatter up to 6.4% around the median, because
#: the figure is dominated by the tracker placing each beat rather than by the
#: band's tempo moving (the median itself drifts under 1% over the same span).
#: Anything tighter than that therefore fires on a perfectly good grid. A
#: dropped bar, by contrast, is about 100% long, so there is a wide margin.
#:
#: Deciding *which* bars belong to the form is not this statistic's job -- an
#: out-of-tempo intro sits only 8-13% out, inside the same band as the jitter.
#: That needs the form's known bar count and an anchor, not a per-bar test.
BAR_DRIFT_TOLERANCE = 0.10

#: Bar lengths outside this range are not a tempo, they are a tracking error:
#: at four beats to the bar they bound 30-300 bpm. The band being half or
#: double the detected tempo is the failure this catches.
MIN_BAR_SECONDS = 0.8
MAX_BAR_SECONDS = 8.0


# --------------------------------------------------------------------------- #
# The grid
# --------------------------------------------------------------------------- #


# Lives here rather than in types.py, which holds event streams: a grid is
# derived from one.
@dataclass(frozen=True, slots=True)
class BeatGrid:
    """Bar lines in seconds from the start of the recording.

    Holds every downbeat rather than one bar length, because a real band
    drifts and a constant bar length accumulates error across a chorus: over
    32 bars, a 0.5% disagreement is most of a beat by the end.
    """

    downbeats: tuple[float, ...]
    beats_per_bar: int = DEFAULT_METRE

    def __post_init__(self) -> None:
        if self.beats_per_bar < 1:
            raise ValueError(f"beats_per_bar must be at least 1, got {self.beats_per_bar}")
        if any(b <= a for a, b in zip(self.downbeats, self.downbeats[1:], strict=False)):
            raise ValueError("downbeats must be strictly ascending")

    @property
    def bars(self) -> int:
        """Complete bars. ``n`` downbeats delimit ``n - 1`` of them: the last
        opens a bar whose end the tracker never saw."""
        return max(len(self.downbeats) - 1, 0)

    @property
    def bar_length(self) -> float:
        """Median bar length, or ``nan`` with fewer than two downbeats.

        The median, not the mean: one dropped bar doubles a single interval,
        which would drag a mean but leaves a median alone.
        """
        lengths = self.bar_lengths
        return median(lengths) if lengths else nan

    @property
    def bar_lengths(self) -> tuple[float, ...]:
        """Length of each complete bar, in order."""
        return tuple(b - a for a, b in zip(self.downbeats, self.downbeats[1:], strict=False))

    @property
    def is_plausible(self) -> bool:
        """Whether the median bar length is a tempo a band could be playing."""
        length = self.bar_length
        return not isnan(length) and MIN_BAR_SECONDS <= length <= MAX_BAR_SECONDS

    def bar_start(self, index: int) -> float:
        """Start of bar ``index``, counting from 0."""
        return self.downbeats[index]

    def bar_span(self, index: int) -> tuple[float, float]:
        """``(start, end)`` of complete bar ``index``."""
        if not 0 <= index < self.bars:
            raise IndexError(f"bar {index} outside the grid's {self.bars} complete bars")
        return self.downbeats[index], self.downbeats[index + 1]

    def bar_index_at(self, time: float) -> int | None:
        """Which complete bar ``time`` falls in, or ``None`` if outside.

        Time before the first downbeat is outside on purpose: whatever plays
        there -- an intro, stage noise, a count-in -- is not in bar 1.
        """
        if not self.downbeats or time < self.downbeats[0]:
            return None
        index = int(np.searchsorted(np.asarray(self.downbeats), time, side="right")) - 1
        return index if index < self.bars else None

    def drift(self) -> tuple[float, ...]:
        """Each bar's deviation from the median, as a fraction of it."""
        length = self.bar_length
        if isnan(length) or length <= 0:
            return ()
        return tuple(abs(bar - length) / length for bar in self.bar_lengths)

    def unreliable_bars(self, tolerance: float = BAR_DRIFT_TOLERANCE) -> tuple[int, ...]:
        """Bars whose length is further than ``tolerance`` from the median.

        A gross-fault detector, not a "is this bar in the form" test -- see
        :data:`BAR_DRIFT_TOLERANCE` for why the distinction matters.
        """
        return tuple(i for i, d in enumerate(self.drift()) if d > tolerance)


def grid(beats: Sequence[BeatEvent], beats_per_bar: int | None = None) -> BeatGrid:
    """Collapse a tracked beat timeline to its bar lines.

    ``beats_per_bar`` is inferred from the highest position seen when not
    given -- the tracker already decided the metre, and re-deriving it from
    beat spacing would only disagree with it.
    """
    if beats_per_bar is None:
        positions = [b.position for b in beats]
        beats_per_bar = max(positions) if positions else DEFAULT_METRE
    downbeats = tuple(sorted(b.time for b in beats if b.is_downbeat))
    return BeatGrid(downbeats=downbeats, beats_per_bar=beats_per_bar)


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


def _track_madmom(
    audio: np.ndarray,
    sample_rate: int,
    candidates: Sequence[int] = METRE_CANDIDATES,
    **kwargs,
) -> list[BeatEvent]:
    """RNN downbeat activations + DBN bar tracking, via madmom's weights."""
    # Validated before the import, so a wrong rate costs nothing and is
    # catchable in CI, where the weights are not installed.
    if sample_rate != BEAT_SAMPLE_RATE:
        raise ValueError(f"madmom backend expects {BEAT_SAMPLE_RATE} Hz, got {sample_rate}")

    from madmom.audio.signal import Signal
    from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor

    # Wrap in a madmom Signal for the same reason the chord backend does: the
    # processor chain reads the rate off the signal, and a bare ndarray would
    # inherit whatever default it was configured with.
    signal = Signal(audio, sample_rate=sample_rate, num_channels=1)
    activations = RNNDownBeatProcessor()(signal)
    tracker = DBNDownBeatTrackingProcessor(beats_per_bar=list(candidates), fps=MADMOM_FPS)
    # Rows are (time, position in bar), position 1-based -- verified against
    # madmom 0.17.dev0.
    return [
        BeatEvent(time=float(time), position=int(position))
        for time, position in tracker(activations)
    ]


def _track_librosa(
    audio: np.ndarray,
    sample_rate: int,
    beats_per_bar: int = DEFAULT_METRE,
    hop_length: int = 512,
    **kwargs,
) -> list[BeatEvent]:
    """Non-DL baseline: onset-envelope beat tracking, metre assumed.

    Kept so there is always a grid available with no weights on disk, the way
    the ``template`` chord backend keeps the pipeline runnable. It finds beats
    but **not** downbeats -- the bar phase here is assumed, not detected, so
    the first tracked beat is called bar 1. Never the default for that reason.
    """
    import librosa

    _, frames = librosa.beat.beat_track(y=audio, sr=sample_rate, hop_length=hop_length)
    times = librosa.frames_to_time(frames, sr=sample_rate, hop_length=hop_length)
    return [
        BeatEvent(time=float(time), position=index % beats_per_bar + 1)
        for index, time in enumerate(times)
    ]


BACKENDS: dict[str, Callable[..., list[BeatEvent]]] = {
    "madmom": _track_madmom,
    "librosa": _track_librosa,
}

#: Sample rate each backend wants its audio in.
BACKEND_SAMPLE_RATES = {
    "madmom": BEAT_SAMPLE_RATE,
    "librosa": 22_050,
}


def _check_backend(backend: str) -> None:
    """Reject an unknown backend by name.

    Never falls back to another tracker: a wrong grid is harder to notice than
    a missing one, because every number downstream still comes out looking
    like a measurement.
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown beat backend {backend!r}, expected one of {sorted(BACKENDS)}")


def sample_rate_for(backend: str) -> int:
    """The sample rate ``backend`` needs its audio loaded at."""
    _check_backend(backend)
    return BACKEND_SAMPLE_RATES[backend]


def track(
    audio: np.ndarray,
    sample_rate: int,
    backend: str = DEFAULT_BACKEND,
    **kwargs,
) -> list[BeatEvent]:
    """Track beats and downbeats in ``audio`` (mono float32)."""
    _check_backend(backend)
    log.debug("beat backend=%s sample_rate=%d", backend, sample_rate)
    return BACKENDS[backend](audio, sample_rate, **kwargs)
