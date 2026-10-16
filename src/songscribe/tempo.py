"""Check a tracked beat grid against a tempo known from outside the audio.

Catches the half- or double-tempo lock that a plausible-range test cannot see.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, fields
from itertools import pairwise
from math import isfinite, isnan, nan
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from .types import DEFAULT_METRE

if TYPE_CHECKING:
    from .beats import BeatGrid

log = logging.getLogger(__name__)

#: Tracked/reference tempo ratios that mean a tracking fault, each with how it
#: reads in a warning: octaves, since a tracker that misses the pulse locks on
#: every other (or fourth) beat. Two octaves each way is all that can occur:
#: madmom's 55-215 bpm against a plausible 30-300 bounds the ratio to 0.18..7.2.
_OCTAVE_WORDS = {2.0: "twice", 0.5: "half", 4.0: "four times", 0.25: "a quarter of"}
TEMPO_OCTAVE_FACTORS = tuple(_OCTAVE_WORDS)

#: How far the tempo ratio may sit from 1 or an octave and still match, as a
#: fraction. Wide because references are loose (a chart says 72 where the band
#: settles at 74); capped by :data:`MAX_TEMPO_RATIO_TOLERANCE`.
TEMPO_RATIO_TOLERANCE = 0.10

# 1 and every octave, in order: the ratios whose windows must stay apart.
_VERDICT_RATIOS = sorted({1.0, *TEMPO_OCTAVE_FACTORS})

#: Exclusive bound on the tolerance: the windows around two ratios an octave
#: apart touch at 1/3, past which a grid could match two verdicts at once.
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


def is_count(value: object) -> bool:
    # Integral, not int, so np.int64(3) counts; bool is Integral too and is not a count.
    return isinstance(value, Integral) and not isinstance(value, bool)


# --------------------------------------------------------------------------- #
# The tempo reference
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class TempoReference:
    """The tempo a grid should agree with, from outside the audio.

    Either ``tempo`` in bpm, or ``bars`` over a ``span`` of seconds (32 bars in
    96 s is 80 bpm in 4/4). Both given: they must agree within
    :data:`TEMPO_RATIO_TOLERANCE`, else it raises, and the tempo wins.
    ``beats_per_bar`` unset means the grid's metre.
    """

    tempo: float | None = None
    bars: int | None = None
    span: float | None = None
    beats_per_bar: int | None = None

    def __post_init__(self) -> None:
        if self.beats_per_bar is not None and not is_count(self.beats_per_bar):
            raise ValueError(f"beats_per_bar must be a whole number, got {self.beats_per_bar!r}")
        if self.beats_per_bar is not None and self.beats_per_bar < 1:
            raise ValueError(f"beats_per_bar must be at least 1, got {self.beats_per_bar}")
        # Validated here: an unusable reference would otherwise yield a nan
        # ratio and every grid would look like it agreed.
        if self.tempo is not None and not (isfinite(self.tempo) and self.tempo > 0):
            raise ValueError(f"tempo must be a positive number of bpm, got {self.tempo}")
        if (self.bars is None) != (self.span is None):
            raise ValueError("a bar-count reference needs both bars and span")
        if self.bars is not None and not is_count(self.bars):
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
        # Raises, not warns: a half-time marking beside a full-time bar count is
        # exactly the factor-of-two fault this check hunts, so no half may win
        # silently. Within tolerance the tempo wins; a span is the looser half.
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

    Kept beside the recording, not read from it, so a tracker's output can
    never be its own reference. Any fault raises :class:`ValueError` naming the path.
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

    Both tempos are counted at ``beats_per_bar`` beats to the bar, so bar lines
    are compared; ``tracked`` differs from :attr:`BeatGrid.tempo` when the grid
    was tracked in another metre.
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
        # Relative to the factor: ratios are multiplicative, so 1/2x must get
        # the same slack as 2x.
        return not isnan(ratio) and abs(ratio / factor - 1.0) <= self.tolerance


def check_tempo(
    grid: BeatGrid,
    reference: TempoReference,
    tolerance: float = TEMPO_RATIO_TOLERANCE,
) -> TempoCheck:
    """Compare a grid against a reference tempo, warning on an octave error.

    Catches the half- or double-tempo lock :attr:`BeatGrid.is_plausible` cannot
    see (a ballad under 55 bpm comes back at 2x). Compares bar lines, not beats,
    so 6/8 tracked in 3 is not flagged. Warns, never raises or corrects: a
    grid bent to its reference has stopped being a measurement.
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
