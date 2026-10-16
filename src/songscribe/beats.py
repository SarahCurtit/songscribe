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
from dataclasses import dataclass, field, fields, replace
from itertools import pairwise
from math import isfinite, isnan, nan
from pathlib import Path
from statistics import median
from typing import Literal

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
#: tempo. Only octaves are listed: a tracker that misses the pulse locks onto a
#: multiple or fraction of it by twos, because every other (or every fourth)
#: beat still falls on a real beat, and that is the error a reference is
#: brought in to find. A ratio that is neither 1 nor one of these is a
#: disagreement about which tune is playing, not a fault this check can name.
#:
#: Two octaves each way and no further, because that is what can happen: madmom
#: reports 55-215 bpm and a plausible reference is 30-300, so the ratio is
#: bounded to 55/300 = 0.18 .. 215/30 = 7.2. A ballad under 54 bpm can come
#: back at 4x as readily as at 2x, but 8x and 1/8x are out of reach.
TEMPO_OCTAVE_FACTORS = (2.0, 0.5, 4.0, 0.25)

#: How each octave reads in a warning about the grid's bar count.
_OCTAVE_WORDS = {4.0: "four times", 2.0: "twice", 0.5: "half", 0.25: "a quarter of"}

#: How far the ratio of tracked to reference tempo may sit from 1 -- or from an
#: octave of it -- and still count as that ratio, as a fraction of the ratio.
#:
#: Wide enough that an honest grid still matches: a reference tempo is rounded
#: (a chart says 72 where the band settles at 74), a bar-count reference
#: inherits the error in both ends of its span, and the band drifts under the
#: whole thing. The same take that sets :data:`BAR_DRIFT_TOLERANCE` moves under
#: 1% in its median bar, but the reference itself is the loose side here.
#:
#: Narrow enough to stay unambiguous: see :data:`MAX_TEMPO_RATIO_TOLERANCE`.
TEMPO_RATIO_TOLERANCE = 0.10

# 1 and every octave, in order: the ratios whose windows must stay apart.
_VERDICT_RATIOS = sorted({1.0, *TEMPO_OCTAVE_FACTORS})

#: The widest tolerance at which no ratio can match two verdicts at once. The
#: window around ratio ``a`` reaches up to ``a(1 + t)`` and the one around the
#: next ratio ``b`` down to ``b(1 - t)``; they meet at ``t = (b - a) / (b + a)``,
#: which is 1/3 for any two ratios an octave apart. At or past it a grid could
#: both agree with the reference and be an octave off it, so the bound is
#: exclusive -- at equality the two windows share their edge.
MAX_TEMPO_RATIO_TOLERANCE = min((b - a) / (b + a) for a, b in pairwise(_VERDICT_RATIOS))


def _coerce_number(key: str, value: object) -> float | None:
    """A sidecar value as a float, ``None`` for absent or ``null``."""
    if value is None:
        return None
    # bool is an int subclass, and JSON ``true`` is not 1 bpm.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"tempo reference {key!r} must be a number, got {value!r}")
    return float(value)


def _coerce_count(key: str, value: object) -> int | None:
    """A sidecar value as a whole count: ``32.0`` is 32 bars, ``31.5`` is not a count."""
    number = _coerce_number(key, value)
    if number is None:
        return None
    if isinstance(value, int):
        return value
    if not number.is_integer():
        raise ValueError(f"tempo reference {key!r} must be a whole number, got {value!r}")
    return int(number)


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


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

    An explicit tempo wins when both are given and agree to within
    :data:`TEMPO_RATIO_TOLERANCE`: it is stated by whoever knows the tune,
    where a bar count is divided by a span whose ends were read off a waveform.
    When they disagree by more than that, construction raises -- see
    ``__post_init__``.

    ``beats_per_bar`` is ``None`` when the reference does not say. A bar count
    then needs no metre to be compared, since it is already a rate of bars; an
    explicit tempo is read in whatever metre the grid was tracked in, because a
    bare bpm figure carries no grouping of its own to compare bar lines with.
    """

    tempo: float | None = None
    bars: int | None = None
    span: float | None = None
    beats_per_bar: int | None = None

    def __post_init__(self) -> None:
        if self.beats_per_bar is not None and not _is_count(self.beats_per_bar):
            raise ValueError(f"beats_per_bar must be a whole number, got {self.beats_per_bar!r}")
        if self.beats_per_bar is not None and self.beats_per_bar < 1:
            raise ValueError(f"beats_per_bar must be at least 1, got {self.beats_per_bar}")
        # Validated here rather than left to surface as a nan ratio: a
        # reference that cannot be turned into a tempo would make every grid
        # it is compared against look like it agreed.
        if self.tempo is not None and not (isfinite(self.tempo) and self.tempo > 0):
            raise ValueError(f"tempo must be a positive number of bpm, got {self.tempo}")
        if (self.bars is None) != (self.span is None):
            raise ValueError("a bar-count reference needs both bars and span")
        if self.bars is not None and not _is_count(self.bars):
            raise ValueError(f"bars must be a whole number of bars, got {self.bars!r}")
        if self.bars is not None and self.bars < 1:
            raise ValueError(f"bars must be at least 1, got {self.bars}")
        if self.span is not None and not (isfinite(self.span) and self.span > 0):
            raise ValueError(f"span must be a positive number of seconds, got {self.span}")
        if self.tempo is None and self.bars is None:
            raise ValueError("a tempo reference needs either a tempo, or bars over a span")
        if self.tempo is not None and self.bars is not None:
            self._check_consistent()

    def _check_consistent(self) -> None:
        # Raises rather than warns. A reference whose two halves disagree has
        # not said what tempo to expect, and the one fault this whole check
        # exists for is a factor of two -- exactly what a half-time marking next
        # to a full-time bar count looks like. Letting the tempo win with a
        # warning would have the verdict decided by whichever half happened to
        # be stated first, in the one class that already raises on an unknown
        # key so that nothing in a sidecar is dropped without a word. The fix
        # is one edit to a file the user wrote; a quiet wrong verdict is not.
        # Within the tolerance the tempo still wins, since a span read off a
        # waveform is the looser of the two.
        metre = self.metre
        counted = self.bars * metre * 60.0 / self.span
        if abs(counted / self.tempo - 1.0) <= TEMPO_RATIO_TOLERANCE:
            return
        hint = (
            ""
            if self.beats_per_bar is not None
            else f"; give beats_per_bar if the tune is not in {DEFAULT_METRE}"
        )
        raise ValueError(
            f"tempo {self.tempo:g} bpm contradicts {self.bars} bars over {self.span:g} s, "
            f"which is {counted:.1f} bpm at {metre} beats to the bar{hint}"
        )

    @property
    def metre(self) -> int:
        """The stated metre, or :data:`DEFAULT_METRE` when none was given."""
        return DEFAULT_METRE if self.beats_per_bar is None else self.beats_per_bar

    @property
    def expected_tempo(self) -> float:
        """The reference tempo in bpm, in its own metre, however it was supplied."""
        return self.tempo_in(self.metre)

    def tempo_in(self, beats_per_bar: int) -> float:
        """The reference tempo counted at ``beats_per_bar`` beats to its bars.

        What a grid in that metre would have to read to share the reference's
        bar lines. An explicit tempo with no stated metre is taken to be in the
        metre asked for, having no bar lines of its own to keep.
        """
        if self.tempo is not None:
            if self.beats_per_bar is None:
                return self.tempo
            return self.tempo * beats_per_bar / self.beats_per_bar
        # A mean over the span, which is all a form-level bar count can say: it
        # is deliberately not asked to follow the band within the span.
        return self.bars * beats_per_bar * 60.0 / self.span

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
        # Each value coerced and checked here, so a wrong type is a ValueError
        # naming its key rather than a TypeError from deep inside int(), and a
        # fractional count is refused rather than truncated. ``null`` means
        # absent for every key alike.
        return cls(
            tempo=_coerce_number("tempo", values.get("tempo")),
            bars=_coerce_count("bars", values.get("bars")),
            span=_coerce_number("span", values.get("span")),
            beats_per_bar=_coerce_count("beats_per_bar", values.get("beats_per_bar")),
        )


def load_tempo_reference(path: str | Path) -> TempoReference:
    """Read a tempo reference from a JSON sidecar, e.g. ``{"tempo": 48}``.

    A file beside the recording rather than anything read out of it: the whole
    point of the comparison is that the expectation comes from outside the
    audio, so a tracker's own output can never become its own reference.

    Every way the file can be wrong raises :class:`ValueError` naming it, since
    a sidecar is found by path and the path is what the user has to go and fix.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        try:
            values = json.load(handle)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}: not valid JSON: {error}") from error
    if not isinstance(values, Mapping):
        raise ValueError(
            f"{path}: a tempo reference is a JSON object, got {type(values).__name__} {values!r}"
        )
    try:
        return TempoReference.from_mapping(values)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error


#: What a :class:`TempoCheck` found, as one value a caller can switch on.
TempoVerdict = Literal["agrees", "octave error", "disagrees", "unmeasurable"]


@dataclass(frozen=True, slots=True)
class TempoCheck:
    """What comparing a tracked tempo against a reference found.

    Carried on the grid so a caller or the eval harness reads the verdict
    instead of re-deriving it, and so the tracked tempo stays visible next to
    the tempo it disagreed with.

    Both tempos are counted at ``beats_per_bar`` beats to the bar, so the
    comparison is of bar lines: ``tracked`` is the grid's bar rate in that
    metre, which is not :attr:`BeatGrid.tempo` when the grid was tracked in a
    different one.
    """

    tracked: float
    reference: float
    tolerance: float = TEMPO_RATIO_TOLERANCE
    beats_per_bar: int = DEFAULT_METRE

    def __post_init__(self) -> None:
        # nan fails every comparison, so it would never match anything and
        # report an exact agreement as a disagreement; past the maximum, the
        # 1x and octave windows overlap and one grid gets two verdicts.
        if not 0.0 <= self.tolerance < MAX_TEMPO_RATIO_TOLERANCE:
            raise ValueError(
                f"tolerance must be at least 0 and below {MAX_TEMPO_RATIO_TOLERANCE:.4g}, "
                f"where the windows around 1x and an octave would overlap; got {self.tolerance}"
            )

    @property
    def ratio(self) -> float:
        """Tracked tempo over reference tempo, or ``nan`` if either is absent."""
        if isnan(self.tracked) or isnan(self.reference) or self.reference <= 0:
            return nan
        return self.tracked / self.reference

    @property
    def measurable(self) -> bool:
        """Whether there was a tempo to compare at all.

        ``False`` for a grid with under two downbeats. Then :attr:`agrees` and
        :attr:`is_octave_error` are both ``False`` too, which says nothing
        against the grid: read this, or :attr:`verdict`, before reading those.
        """
        return not isnan(self.ratio)

    @property
    def agrees(self) -> bool:
        """Whether the grid is on the reference tempo, within tolerance.

        ``False`` both when it disagrees and when it could not be measured;
        :attr:`verdict` tells the two apart.
        """
        return self._matches(1.0)

    @property
    def octave_factor(self) -> float | None:
        """Which octave of the reference the grid sits at, or ``None``."""
        return next((f for f in TEMPO_OCTAVE_FACTORS if self._matches(f)), None)

    @property
    def is_octave_error(self) -> bool:
        """Whether the grid is one of :data:`TEMPO_OCTAVE_FACTORS` off the reference."""
        return self.octave_factor is not None

    @property
    def verdict(self) -> TempoVerdict:
        """The finding as one value, keeping "no tempo" apart from "wrong tempo"."""
        if not self.measurable:
            return "unmeasurable"
        if self.agrees:
            return "agrees"
        if self.is_octave_error:
            return "octave error"
        return "disagrees"

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
    #:
    #: It describes the downbeats as they were when checked, and only their
    #: tempo: construction refuses a verdict whose tracked tempo this grid's
    #: own bar lines no longer give, so ``replace(grid, downbeats=...)`` either
    #: keeps a verdict that is still true of it or raises until the verdict is
    #: re-run or dropped with ``tempo_check=None``. Left out of equality and
    #: hashing: two grids over the same bar lines are the same grid whether or
    #: not anyone had a reference to hold one of them against.
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

    Compares bar lines, not beats: both tempos are counted in the reference's
    metre (the grid's, if the reference states none). madmom only chooses
    between 3 and 4, so a 6/8 chart comes back in 3 on the very same bar lines
    at half the beat rate -- a beat comparison would call that an octave error
    on a grid that is exactly right. What the reference is brought in to catch
    is an octave of the *bars*, and that is what this measures.
    """
    metre = grid.beats_per_bar if reference.beats_per_bar is None else reference.beats_per_bar
    check = TempoCheck(
        tracked=grid.tempo_in(metre),
        reference=reference.tempo_in(metre),
        tolerance=tolerance,
        beats_per_bar=metre,
    )
    verdict = check.verdict
    if verdict != "unmeasurable" and metre != grid.beats_per_bar:
        log.info(
            "grid tracked in %d, reference in %d: comparing bar lines, which reads the "
            "grid's %.1f bpm as %.1f bpm in %d",
            grid.beats_per_bar,
            metre,
            grid.tempo,
            check.tracked,
            metre,
        )
    if verdict == "octave error":
        factor = check.octave_factor
        log.warning(
            "tracked tempo %.1f bpm is %.3gx the reference %.1f bpm: a tempo octave error, "
            "so the grid's %d bars are %s the form's. Reported, not corrected",
            check.tracked,
            factor,
            check.reference,
            grid.bars,
            _OCTAVE_WORDS[factor],
        )
    elif verdict == "unmeasurable":
        log.debug("no measurable tempo to compare against the %.1f bpm reference", check.reference)
    elif verdict == "disagrees":
        # Names what was ruled out rather than "not an octave error": only the
        # listed octaves are tested, and a ratio past them is not cleared.
        log.warning(
            "tracked tempo %.1f bpm disagrees with the reference %.1f bpm (%.2fx), "
            "which is none of the octave locks checked for (%s)",
            check.tracked,
            check.reference,
            check.ratio,
            ", ".join(f"{f:g}x" for f in sorted(TEMPO_OCTAVE_FACTORS)),
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
