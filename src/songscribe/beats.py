"""Beat and downbeat tracking, via pluggable backends.

Chord models emit boundaries wherever their frames fall; a musician reads
bars. This module supplies the bar lines.

Same registry contract as :mod:`songscribe.chords`: one function per backend
in :data:`BACKENDS`, its rate in :data:`BACKEND_SAMPLE_RATES`, no ``**kwargs``.
"""

from __future__ import annotations

import json
import logging
from bisect import bisect_right
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from math import isfinite, isnan, nan
from pathlib import Path
from statistics import median

import numpy as np

from .registry import Registry
from .types import BeatEvent

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

#: Assumed when a backend tracks beats without identifying the downbeat, and
#: when a beat timeline carries no position information at all.
DEFAULT_METRE = 4

#: How far one bar may differ from the median before it is called unreliable,
#: as a fraction. Catches gross faults only (a dropped bar is ~100% long);
#: good bars scatter up to 6.4%. EVALUATION.md has the measurement.
BAR_DRIFT_TOLERANCE = 0.10

#: A grid implying a tempo outside this range is not a tempo, it is a tracking
#: error: librosa output gone astray, or a corrupt or hand-edited ``.beats``
#: file. Not a half- or double-tempo lock from madmom -- its DBN is held to its
#: own 55-215 bpm, so an octave error there lands inside this range and needs
#: an external tempo reference to catch -- :func:`check_tempo`, which is where
#: that case is now reported. Bounded in beats per minute rather
#: than in bar length so the metre is respected: at three beats to the bar,
#: 0.85 s is 212 bpm, not the leisurely tempo a bar-length bound would have
#: called it.
MIN_TEMPO_BPM = 30.0
MAX_TEMPO_BPM = 300.0

#: Ratios of tracked to reference tempo that are a tracking fault rather than a
#: tempo. Only the octave is listed: a tracker that misses the pulse locks onto
#: twice or half it, because every other beat still falls on a real beat, and
#: that is the error a reference is brought in to find. A ratio that is neither
#: 1 nor an octave is a disagreement about which tune is playing, not a fault
#: this check can name.
TEMPO_OCTAVE_FACTORS = (2.0, 0.5)

#: How far the ratio of tracked to reference tempo may sit from 1 -- or from an
#: octave of it -- and still count as that ratio, as a fraction of the ratio.
#:
#: Wide enough that an honest grid still matches: a reference tempo is rounded
#: (a chart says 72 where the band settles at 74), a bar-count reference
#: inherits the error in both ends of its span, and the band drifts under the
#: whole thing. The same take that sets :data:`BAR_DRIFT_TOLERANCE` moves under
#: 1% in its median bar, but the reference itself is the loose side here.
#:
#: Narrow enough to stay unambiguous: 1 and 2 are 100% apart, so 10% windows
#: around each cannot both match, and nothing lands in two verdicts at once.
TEMPO_RATIO_TOLERANCE = 0.10


# --------------------------------------------------------------------------- #
# The tempo reference
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class TempoReference:
    """The tempo a grid is expected to agree with, known from outside the audio.

    Two ways to say it, because that is how charts come: an explicit ``tempo``
    in bpm, or a ``bars`` count spanning a known ``span`` of seconds -- 32 bars
    of chorus across 96 s is 80 bpm in 4/4. Both are accepted so that a chart
    with no tempo marking still gives the check something to work with.

    An explicit tempo wins when both are given: it is stated by whoever knows
    the tune, where a bar count is divided by a span whose ends were read off a
    waveform.
    """

    tempo: float | None = None
    bars: int | None = None
    span: float | None = None
    beats_per_bar: int = DEFAULT_METRE

    def __post_init__(self) -> None:
        if self.beats_per_bar < 1:
            raise ValueError(f"beats_per_bar must be at least 1, got {self.beats_per_bar}")
        # Validated here rather than left to surface as a nan ratio: a
        # reference that cannot be turned into a tempo would make every grid
        # it is compared against look like it agreed.
        if self.tempo is not None and not (isfinite(self.tempo) and self.tempo > 0):
            raise ValueError(f"tempo must be a positive number of bpm, got {self.tempo}")
        if (self.bars is None) != (self.span is None):
            raise ValueError("a bar-count reference needs both bars and span")
        if self.bars is not None and self.bars < 1:
            raise ValueError(f"bars must be at least 1, got {self.bars}")
        if self.span is not None and not (isfinite(self.span) and self.span > 0):
            raise ValueError(f"span must be a positive number of seconds, got {self.span}")
        if self.tempo is None and self.bars is None:
            raise ValueError("a tempo reference needs either a tempo, or bars over a span")

    @property
    def expected_tempo(self) -> float:
        """The reference tempo in bpm, however it was supplied."""
        if self.tempo is not None:
            return self.tempo
        # A mean over the span, which is all a form-level bar count can say: it
        # is deliberately not asked to follow the band within the span.
        return self.bars * self.beats_per_bar * 60.0 / self.span

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> TempoReference:
        """Build from a parsed sidecar, rejecting keys it has no field for.

        An unknown key raises rather than being dropped, for the same reason a
        backend setting does: a misspelled ``tempo`` that is silently ignored
        leaves the check comparing against the bar count it was meant to
        override, or against nothing at all.
        """
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            named = ", ".join(repr(name) for name in unknown)
            noun = "key" if len(unknown) == 1 else "keys"
            raise ValueError(f"unknown tempo reference {noun} {named}, expected {sorted(known)}")
        tempo = values.get("tempo")
        bars = values.get("bars")
        span = values.get("span")
        return cls(
            tempo=None if tempo is None else float(tempo),
            bars=None if bars is None else int(bars),
            span=None if span is None else float(span),
            beats_per_bar=int(values.get("beats_per_bar", DEFAULT_METRE)),
        )


def load_tempo_reference(path: str | Path) -> TempoReference:
    """Read a tempo reference from a JSON sidecar, e.g. ``{"tempo": 48}``.

    A file beside the recording rather than anything read out of it: the whole
    point of the comparison is that the expectation comes from outside the
    audio, so a tracker's own output can never become its own reference.
    """
    with Path(path).open(encoding="utf-8") as handle:
        return TempoReference.from_mapping(json.load(handle))


@dataclass(frozen=True, slots=True)
class TempoCheck:
    """What comparing a tracked tempo against a reference found.

    Carried on the grid so a caller or the eval harness reads the verdict
    instead of re-deriving it, and so the tracked tempo stays visible next to
    the tempo it disagreed with.
    """

    tracked: float
    reference: float
    tolerance: float = TEMPO_RATIO_TOLERANCE

    @property
    def ratio(self) -> float:
        """Tracked tempo over reference tempo, or ``nan`` if either is absent."""
        if isnan(self.tracked) or isnan(self.reference) or self.reference <= 0:
            return nan
        return self.tracked / self.reference

    @property
    def agrees(self) -> bool:
        """Whether the grid is on the reference tempo, within tolerance."""
        return self._matches(1.0)

    @property
    def octave_factor(self) -> float | None:
        """Which octave of the reference the grid sits at, or ``None``."""
        return next((f for f in TEMPO_OCTAVE_FACTORS if self._matches(f)), None)

    @property
    def is_octave_error(self) -> bool:
        """Whether the grid is at twice or half the reference tempo."""
        return self.octave_factor is not None

    def _matches(self, factor: float) -> bool:
        ratio = self.ratio
        # Relative to the factor, not an absolute window: a tempo ratio is
        # multiplicative, so the same fractional slack has to be as wide at 2x
        # as at 1/2x, or half-tempo locks would be held to a stricter test.
        return not isnan(ratio) and abs(ratio / factor - 1.0) <= self.tolerance


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
    #: Verdict from :func:`check_tempo` when the grid was built against a
    #: reference. ``None`` means nothing outside the audio was available to
    #: compare with, which is not the same as having agreed with it.
    tempo_check: TempoCheck | None = None

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
        length = self.bar_length
        return nan if isnan(length) or length <= 0 else self.beats_per_bar * 60.0 / length

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

        A gross-fault detector, not a "is this bar in the form" test -- see
        "Beat-grid drift" in EVALUATION.md for why the distinction matters.
        """
        return tuple(i for i, d in enumerate(self.drift()) if d > tolerance)


def check_tempo(
    grid: BeatGrid,
    reference: TempoReference,
    tolerance: float = TEMPO_RATIO_TOLERANCE,
) -> TempoCheck:
    """Compare a grid against a reference tempo, warning on an octave error.

    This is the half- and double-tempo lock that :attr:`BeatGrid.is_plausible`
    cannot see: madmom's DBN will not report outside 55-215 bpm, so a ballad
    under 55 comes back at double tempo with twice the bars, and double a slow
    tempo is still a plausible tempo. Only a reference from outside the audio
    separates the two.

    Warns rather than raising, and never rewrites the grid. A 2x ratio is
    strong evidence but not proof -- the reference can be noting the half-time
    feel the chart is written in while the band counts the fast pulse -- and
    raising would reject a legitimately fast take on one number. Correcting the
    grid would be worse than either: a grid bent to match the reference it is
    supposed to be measured against has stopped being a measurement.
    """
    check = TempoCheck(tracked=grid.tempo, reference=reference.expected_tempo, tolerance=tolerance)
    factor = check.octave_factor
    if factor is not None:
        log.warning(
            "tracked tempo %.1f bpm is %.3gx the reference %.1f bpm: a tempo octave error, "
            "so the grid's %d bars are %s the form's. Reported, not corrected",
            check.tracked,
            factor,
            check.reference,
            grid.bars,
            "twice" if factor > 1.0 else "half",
        )
    elif isnan(check.ratio):
        log.debug("no measurable tempo to compare against the %.1f bpm reference", check.reference)
    elif not check.agrees:
        log.warning(
            "tracked tempo %.1f bpm disagrees with the reference %.1f bpm (%.2fx), "
            "which is not a half- or double-tempo lock",
            check.tracked,
            check.reference,
            check.ratio,
        )
    return check


def grid(
    beats: Sequence[BeatEvent],
    beats_per_bar: int | None = None,
    reference: TempoReference | None = None,
) -> BeatGrid:
    """Collapse a tracked beat timeline to its bar lines.

    ``beats_per_bar`` is inferred from the highest position seen when not
    given -- the tracker already decided the metre, and re-deriving it from
    beat spacing would only disagree with it.

    A ``reference`` is compared against the finished grid by
    :func:`check_tempo`, and its verdict travels on the grid's
    :attr:`~BeatGrid.tempo_check`.
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
    # Checked here rather than left to the caller so that a grid built with a
    # reference carries the verdict wherever it travels; the metre warning
    # above is a separate fault and both have to be able to fire alone.
    return replace(built, tempo_check=check_tempo(built, reference))


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


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
