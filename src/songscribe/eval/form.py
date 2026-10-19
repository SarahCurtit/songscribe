"""Split a performance into choruses, each mapped back onto the chart's clock.

A recording is an intro, the chart played several times, perhaps a short last
pass, and an ending. Once the first chorus is anchored, the chart's bar count
tiles the bar grid; each block then maps onto the reference's own timeline, so
every existing metric scores it unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite, nan
from statistics import median

from ..beats import BAR_DRIFT_TOLERANCE, BeatGrid
from ..types import ChordEvent
from .harte import transpose_label
from .metrics import MIN_SEGMENT, extent

#: Which full choruses are the head. In jazz the head is stated first and
#: played out last, with improvisation between; negative counts from the end.
HEAD_CHORUSES = (0, -1)

#: A shorter last block is an ending, not a partial chorus: scoring two bars
#: against a thirty-two-bar chart gives a confident number about nothing.
CHORUS_MIN_BARS = 4

#: How a take's key is decided: ``auto`` per chorus, ``none`` the chart's own.
TRANSPOSE_MODES = ("auto", "none")

#: Kinds whose time is scored.
SCORED_KINDS = ("head", "chorus", "partial")


@dataclass(frozen=True, slots=True)
class Chorus:
    """One block of the performance, on the recording's clock.

    ``kind`` is ``intro``, ``head``, ``chorus``, ``partial``, ``tail`` or
    ``unaligned`` (a block whose bars do not tile evenly). ``scale`` is the
    block's mean bar length over the chart's, and ``nan`` where nothing is
    mapped. ``bars`` counts the chart bars covered. ``shift`` is the key it is
    played in, in semitones above the chart's.
    """

    index: int
    start: float
    end: float
    first_bar: int
    bars: int
    scale: float
    kind: str
    shift: int = 0

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def is_scored(self) -> bool:
        return self.kind in SCORED_KINDS


def choruses(
    grid: BeatGrid,
    bars_per_chorus: int,
    anchor_bar: int,
    reference_duration: float,
    *,
    head_choruses: Sequence[int] = HEAD_CHORUSES,
    drift_tolerance: float = BAR_DRIFT_TOLERANCE,
) -> tuple[Chorus, ...]:
    """Tile the grid from ``anchor_bar`` in blocks of the chart's bar count.

    Bars before the anchor are the intro. A block holding a bar far from its
    own median length (and every block after it, the count being lost), or
    whose mean bar length is far from the other choruses', is ``unaligned``:
    named and left out, not scored as wrong.
    """
    if bars_per_chorus < 1:
        raise ValueError(f"bars_per_chorus must be at least 1, got {bars_per_chorus}")
    if not reference_duration > 0:
        raise ValueError(f"reference_duration must be positive, got {reference_duration}")
    downbeats = grid.downbeats
    if not 0 <= anchor_bar < max(grid.bars, 1):
        raise ValueError(f"anchor bar {anchor_bar} is outside the grid's {grid.bars} bars")
    if not downbeats:
        return ()

    reference_bar = reference_duration / bars_per_chorus

    blocks: list[tuple[int, int]] = []
    first = anchor_bar
    while first < grid.bars:
        count = min(bars_per_chorus, grid.bars - first)
        blocks.append((first, count))
        first += count

    def scale_of(block: tuple[int, int]) -> float:
        start, count = block
        return (downbeats[start + count] - downbeats[start]) / count / reference_bar

    full = [b for b in blocks if b[1] == bars_per_chorus]
    typical = median(scale_of(b) for b in full) if full else nan
    heads = {h if h >= 0 else len(full) + h for h in head_choruses}

    found: list[Chorus] = []
    if anchor_bar > 0 or downbeats[0] > MIN_SEGMENT:
        found.append(Chorus(0, 0.0, downbeats[anchor_bar], 0, anchor_bar, nan, "intro"))

    lost = False
    for block in blocks:
        start, count = block
        scale = scale_of(block)
        if _has_faulty_bar(grid.bar_lengths[start : start + count], drift_tolerance):
            # A bar added or dropped loses the count, so every later block would
            # be a bar out: refuse them too rather than score a shifted chart.
            lost = True
        if lost or (isfinite(typical) and abs(scale / typical - 1) > drift_tolerance):
            kind = "unaligned"
        elif count < bars_per_chorus:
            kind = "partial" if count >= CHORUS_MIN_BARS else "tail"
        else:
            kind = "head" if full.index(block) in heads else "chorus"
        mapped = kind in SCORED_KINDS
        found.append(
            Chorus(
                index=len(found),
                start=downbeats[start],
                end=downbeats[start + count],
                first_bar=start,
                bars=count,
                scale=scale if mapped else nan,
                kind=kind,
            )
        )
    return tuple(found)


def _has_faulty_bar(lengths: Sequence[float], tolerance: float) -> bool:
    """Whether a bar is far from its own block's median length.

    Against the block, not the take: a whole chorus at another tempo is no fault.
    """
    typical = median(lengths)
    return any(abs(length / typical - 1) > tolerance for length in lengths)


def unscored_seconds(found: Sequence[Chorus]) -> float:
    """Time in blocks no score covers: the intro, an ending, unaligned choruses."""
    return sum(c.duration for c in found if not c.is_scored)


def _clip(events: Sequence[ChordEvent], low: float, high: float) -> list[ChordEvent]:
    return [
        ChordEvent(max(e.start, low), min(e.end, high), e.label)
        for e in events
        if e.start < high and e.end > low and min(e.end, high) - max(e.start, low) > MIN_SEGMENT
    ]


def project_estimate(
    events: Sequence[ChordEvent], chorus: Chorus, origin: float = 0.0
) -> list[ChordEvent]:
    """The part of ``events`` inside ``chorus``, moved onto the reference's clock.

    ``origin`` is where the reference starts. Afterwards the block looks like a
    single-chorus track: estimate time outside it is gone, and reference time
    the estimate never reached will score as no chord.
    """
    if not chorus.is_scored:
        raise ValueError(f"a {chorus.kind} block has no mapping onto the chart")
    return [
        ChordEvent(
            origin + (e.start - chorus.start) / chorus.scale,
            origin + (e.end - chorus.start) / chorus.scale,
            e.label,
        )
        for e in _clip(events, chorus.start, chorus.end)
    ]


def truncate_reference(
    reference: Sequence[ChordEvent], bars_per_chorus: int, bars: int
) -> list[ChordEvent]:
    """The first ``bars`` bars of the chart, for scoring a partial last chorus.

    Exact because the chart is bar-indexed: its bars are equal slices of its span.
    """
    start, end = extent(list(reference))
    if not 0 < bars <= bars_per_chorus:
        raise ValueError(f"bars must be within 1..{bars_per_chorus}, got {bars}")
    return _clip(reference, start, start + (end - start) * bars / bars_per_chorus)


def transpose_events(events: Sequence[ChordEvent], semitones: int) -> list[ChordEvent]:
    """The chart moved to the key a chorus is played in; the estimate is never moved."""
    return [ChordEvent(e.start, e.end, transpose_label(e.label, semitones)) for e in events]
