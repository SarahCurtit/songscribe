"""Beat and downbeat tracking, via pluggable backends.

Chord models emit boundaries wherever their frames fall; a musician reads
bars. This module supplies the bar lines.

Same registry contract as :mod:`songscribe.chords`: one function per backend
in :data:`BACKENDS`, its rate in :data:`BACKEND_SAMPLE_RATES`, no ``**kwargs``.
"""

from __future__ import annotations

import logging
from bisect import bisect_right
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from math import isfinite, isnan, nan
from statistics import median

import numpy as np

from .registry import Registry, setting_rules, whole_at_least, whole_number
from .tempo import TempoCheck, TempoReference, check_tempo
from .types import DEFAULT_METRE, BeatEvent

log = logging.getLogger(__name__)

DEFAULT_BACKEND = "madmom"

#: madmom's downbeat RNN was trained at 44.1 kHz, like its chord model.
BEAT_SAMPLE_RATE = 44_100

#: Frame rate of the RNN's activation function. The DBN has to be told, and
#: getting it wrong scales every beat time rather than failing.
MADMOM_FPS = 100

#: Metres the tracker is allowed to consider. The DBN takes a candidate list;
#: jazz standards are in 4/4 or 3/4, the same pair madmom's reference
#: DBNDownBeatTracker program uses.
METRE_CANDIDATES = (3, 4)

#: How far one bar may differ from the median before it is called unreliable,
#: as a fraction. Catches gross faults only (a dropped bar is ~100% long);
#: good bars scatter up to 6.4%.
BAR_DRIFT_TOLERANCE = 0.10

#: A grid implying a tempo outside this range is not a tempo, it is a tracking
#: error: librosa output gone astray, or a corrupt or hand-edited ``.beats``
#: file. Not a half- or double-tempo lock from madmom -- its DBN is held to its
#: own 55-215 bpm, so an octave error there lands inside this range; that is
#: :func:`check_tempo`'s job. In bpm rather than bar length to respect the
#: metre: at three beats to the bar, 0.85 s is 212 bpm, not a leisurely tempo.
MIN_TEMPO_BPM = 30.0
MAX_TEMPO_BPM = 300.0

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
    #: Verdict from :func:`check_tempo`; ``None`` means no reference, not that
    #: it agreed. A verdict that no longer matches the downbeats raises, so
    #: ``replace(grid, downbeats=...)`` needs a re-check or ``tempo_check=None``.
    #: Excluded from equality: same bar lines, same grid.
    tempo_check: TempoCheck | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.beats_per_bar < 1:
            raise ValueError(f"beats_per_bar must be at least 1, got {self.beats_per_bar}")
        # Checked before the ordering test, which nan would pass silently: every
        # comparison against nan is False, so a nan downbeat would survive into
        # a grid whose bar lengths are all nan and whose health report is empty.
        if not all(isfinite(b) for b in self.downbeats):
            raise ValueError(f"downbeats must all be finite, got {self.downbeats}")
        if any(b <= a for a, b in zip(self.downbeats, self.downbeats[1:], strict=False)):
            raise ValueError("downbeats must be strictly ascending")
        check = self.tempo_check
        if check is not None:
            # Compared exactly: check_tempo computed ``tracked`` with this same
            # method, so the same bar lines give the same float, bit for bit.
            own = self.tempo_in(check.beats_per_bar)
            if not (own == check.tracked or (isnan(own) and isnan(check.tracked))):
                raise ValueError(
                    f"tempo_check is stale: it was made at {check.tracked:.1f} bpm, and these "
                    f"downbeats give {own:.1f}. Re-run check_tempo, or pass tempo_check=None"
                )

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
    def tempo(self) -> float:
        """Implied beats per minute, or ``nan`` without a measurable bar."""
        return self.tempo_in(self.beats_per_bar)

    def tempo_in(self, beats_per_bar: int) -> float:
        """Beats per minute if each bar held ``beats_per_bar`` beats.

        The bar rate in another metre's units: how :func:`check_tempo` lays
        these bar lines against a reference counted differently.
        """
        length = self.bar_length
        return nan if isnan(length) or length <= 0 else beats_per_bar * 60.0 / length

    @property
    def is_plausible(self) -> bool:
        """Whether the implied tempo is one a band could be playing.

        Blind to a tempo octave error by construction, since double a slow
        tempo is still a tempo a band could play: :func:`check_tempo` is what
        catches that, and it needs a reference the audio cannot supply.
        """
        tempo = self.tempo
        return not isnan(tempo) and MIN_TEMPO_BPM <= tempo <= MAX_TEMPO_BPM

    def bar_start(self, index: int) -> float:
        """Start of bar ``index``, counting from 0.

        Bounded rather than relying on tuple indexing, which would answer a
        negative index from the far end of the grid. The last downbeat is in
        range though it opens no complete bar: it is where the form stops.
        """
        if not 0 <= index < len(self.downbeats):
            raise IndexError(f"bar {index} outside the grid's {len(self.downbeats)} bar lines")
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
        # bisect on the tuple rather than np.searchsorted on a fresh array:
        # placing a whole chord timeline is one call per event, and this way
        # that costs no allocation.
        index = bisect_right(self.downbeats, time) - 1
        return index if index < self.bars else None

    def drift(self) -> tuple[float, ...]:
        """Each bar's deviation from the median, as a fraction of it."""
        length = self.bar_length
        if isnan(length) or length <= 0:
            return ()
        return tuple(abs(bar - length) / length for bar in self.bar_lengths)

    def unreliable_bars(self, tolerance: float = BAR_DRIFT_TOLERANCE) -> tuple[int, ...]:
        """Bars whose length is further than ``tolerance`` from the median.

        A gross-fault detector, not an "is this bar in the form" test: a tracked
        bar scatters by several percent even on a steady grid.
        """
        return tuple(i for i, d in enumerate(self.drift()) if d > tolerance)


def grid(
    beats: Sequence[BeatEvent],
    beats_per_bar: int | None = None,
    reference: TempoReference | None = None,
) -> BeatGrid:
    """Collapse a tracked beat timeline to its bar lines.

    ``beats_per_bar`` is inferred from the highest position seen when not
    given: the tracker already decided the metre.

    A ``reference`` is checked by :func:`check_tempo` into ``tempo_check``.
    """
    positions = [b.position for b in beats]
    if beats_per_bar is None:
        beats_per_bar = max(positions) if positions else DEFAULT_METRE
        # An excerpt shorter than a bar infers its own length as the metre.
        if positions and beats_per_bar not in METRE_CANDIDATES:
            log.warning("inferred metre %d from %d beats", beats_per_bar, len(positions))
    elif positions and max(positions) != beats_per_bar:
        # The caller still wins, but every tempo figure scales with the metre:
        # 4/4 beats read as 3/4 report 90 bpm for 120, and nothing else says so.
        log.warning(
            "metre %d given, but the tracked positions reach %d", beats_per_bar, max(positions)
        )
    downbeats = tuple(sorted(b.time for b in beats if b.is_downbeat))
    # A timeline trimmed mid-bar carries no downbeat, and the resulting grid
    # places nothing at all -- worth a word, since it is indistinguishable
    # downstream from a recording the tracker simply found no bars in.
    if beats and not downbeats:
        log.warning("%d beats but no downbeat among them: the grid is empty", len(beats))
    built = BeatGrid(downbeats=downbeats, beats_per_bar=beats_per_bar)
    if reference is None:
        return built
    # Checked here so the verdict travels with the grid.
    return replace(built, tempo_check=check_tempo(built, reference))


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


#: Text is not a list of metres: "34" and b"34" would be read as two bad metres.
_TEXT_TYPES = (str, bytes, bytearray, memoryview)


def _is_candidate_list(value: object) -> bool:
    """Whether ``value`` is shaped like a list of metres rather than one metre.

    An ndarray is not a :class:`~collections.abc.Sequence`, but madmom's DBN
    reads its metre setting with ``np.array(ndmin=1)``, so an array of metres is
    as much its native form as a list. A 0-d array is one value, not a list.
    """
    if isinstance(value, np.ndarray):
        return value.ndim >= 1
    return isinstance(value, Sequence) and not isinstance(value, _TEXT_TYPES)


def _one_metre(value: object) -> str | None:
    if _is_candidate_list(value):
        # A candidate list is madmom's form; this backend needs a single metre.
        return (
            f"must be one metre, got {value!r}: this backend assumes the metre rather "
            "than detecting it, so it cannot choose between several"
        )
    return whole_at_least(value, "beats to the bar")


def _metre_or_candidates(value: object) -> str | None:
    if not _is_candidate_list(value):
        return whole_at_least(value, "beats to the bar")
    if isinstance(value, np.ndarray) and value.ndim > 1:
        return f"must be a flat list of metres, got a {value.ndim}-D array"
    if len(value) == 0:
        kind = "array" if isinstance(value, np.ndarray) else type(value).__name__
        return f"must name at least one metre to choose between, got an empty {kind}"
    for position, metre in enumerate(value):
        # Name the bad entry alone; reprinting a long list hides it.
        if whole_at_least(metre, "beats to the bar") is not None:
            return (
                "must list metres that are each a whole number of at least 1, "
                f"but candidate {metre!r} at position {position} is not"
            )
    return None


@setting_rules(beats_per_bar=_metre_or_candidates)
def _track_madmom(
    audio: np.ndarray,
    sample_rate: int,
    beats_per_bar: int | Sequence[int] = METRE_CANDIDATES,
) -> list[BeatEvent]:
    """RNN downbeat activations + DBN bar tracking, via madmom's weights.

    ``beats_per_bar`` is one metre or a list to choose between -- madmom's own
    name, and the librosa backend's, so ``track(..., beats_per_bar=3)`` means
    the same thing whichever backend runs.
    """
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
    # Passed as given: the DBN reads it with np.array(ndmin=1), so an int and a
    # list both work without converting here.
    tracker = DBNDownBeatTrackingProcessor(beats_per_bar=beats_per_bar, fps=MADMOM_FPS)
    # Rows are (time, position in bar), position 1-based -- verified against
    # madmom 0.17.dev0.
    return [
        BeatEvent(time=float(time), position=int(position))
        for time, position in tracker(activations)
    ]


@setting_rules(
    beats_per_bar=_one_metre,
    hop_length=whole_number("samples"),
)
def _track_librosa(
    audio: np.ndarray,
    sample_rate: int,
    beats_per_bar: int = DEFAULT_METRE,
    hop_length: int = 512,
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


_REGISTRY = Registry("beat", BACKENDS, BACKEND_SAMPLE_RATES)

# The checks live in :mod:`songscribe.registry`, shared with the chord stage; these
# are this module's public names for them.
_check_backend = _REGISTRY.check_backend
_check_options = _REGISTRY.check_options
backend_options = _REGISTRY.options
accepts_any_option = _REGISTRY.accepts_any_option
sample_rate_for = _REGISTRY.sample_rate_for


def track(
    audio: np.ndarray,
    sample_rate: int,
    backend: str = DEFAULT_BACKEND,
    **kwargs,
) -> list[BeatEvent]:
    """Track beats and downbeats in ``audio`` (mono float32).

    Extra keyword arguments are the chosen backend's settings; one it does not
    accept raises rather than being dropped.
    """
    _check_backend(backend)
    _check_options(backend, kwargs)
    log.debug("beat backend=%s sample_rate=%d", backend, sample_rate)
    return BACKENDS[backend](audio, sample_rate, **kwargs)
