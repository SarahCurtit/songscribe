"""Split a recording into choruses and convert their times to the chart's timeline.

A recording is usually an intro, the chart played several times, sometimes a
shorter last pass, and an ending. Starting from the anchor bar (the bar where
the first chorus begins, see ``form_start.locate``), the recording's bars are
cut into blocks of the chart's length. Each block's times are then converted
to the chart's timeline, so the existing metrics can score it unchanged.
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

#: Which full choruses count as the head, by position (negative counts from
#: the end). In jazz the head (the tune itself) is played first and last, with
#: improvised choruses in between.
HEAD_CHORUSES = (0, -1)

#: A last block shorter than a chorus is scored as a partial chorus only if it
#: has at least this many bars. Otherwise it is treated as an ending and not
#: scored, since a score over two bars of a 32-bar chart is not meaningful.
CHORUS_MIN_BARS = 4

#: Block kinds that are scored.
SCORED_KINDS = ("head", "chorus", "partial")


@dataclass(frozen=True, slots=True)
class Chorus:
    """One block of the recording; ``start``/``end`` are recording seconds.

    ``kind``: ``intro``, ``head``/``chorus`` (full), ``partial`` (short last block, scored),
    ``tail`` (short ending, not scored), ``unaligned`` (inconsistent bars, not scored).
    ``bars`` is the block's bar count, ``first_bar`` its grid index. ``scale`` is mean bar
    duration over the chart's (1.0 = chart tempo), ``nan`` when not scored.
    """

    index: int
    start: float
    end: float
    first_bar: int
    bars: int
    scale: float
    kind: str

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
    """Cut the bar grid into blocks of ``bars_per_chorus`` bars from ``anchor_bar``.

    A block is ``unaligned`` (kept, not scored) if a bar's length is off the block's median by
    more than ``drift_tolerance`` (a bar was added or dropped, so all later blocks are
    ``unaligned`` too), or if its mean bar length is off the median of the full choruses.
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
    # The intro is the bars before the anchor, plus any time before the first bar line.
    if anchor_bar > 0 or downbeats[0] > MIN_SEGMENT:
        found.append(Chorus(0, 0.0, downbeats[anchor_bar], 0, anchor_bar, nan, "intro"))

    lost = False
    for block in blocks:
        start, count = block
        scale = scale_of(block)
        if _has_faulty_bar(grid.bar_lengths[start : start + count], drift_tolerance):
            # After an added or dropped bar, every later block is shifted by a
            # bar, so mark them all unaligned instead of scoring them shifted.
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
    """Return whether any bar differs from the median bar length of ``lengths``.

    The median is taken over the block only, so a chorus played at a different
    tempo from the others is not flagged here.
    """
    typical = median(lengths)
    return any(abs(length / typical - 1) > tolerance for length in lengths)


def unscored_seconds(found: Sequence[Chorus]) -> float:
    """Total seconds in blocks that are not scored (intro, tail, unaligned)."""
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
    """Keep the events inside ``chorus`` and convert their times to the chart's timeline.

    Times are shifted to the chorus start and divided by ``chorus.scale``.
    ``origin`` is the time the chart starts at. Events outside the chorus are
    dropped, and events crossing its edges are cut. Parts of the chart with
    no event left are scored as no chord.
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
    """Return the first ``bars`` bars of the chart, to score against a partial chorus.

    The chart's bars are all the same length, so the cut is
    ``bars / bars_per_chorus`` of its total duration.
    """
    start, end = extent(list(reference))
    if not 0 < bars <= bars_per_chorus:
        raise ValueError(f"bars must be within 1..{bars_per_chorus}, got {bars}")
    return _clip(reference, start, start + (end - start) * bars / bars_per_chorus)


def transpose_events(events: Sequence[ChordEvent], semitones: int) -> list[ChordEvent]:
    """The chart moved to the key a chorus is played in; the estimate is never moved."""
    return [ChordEvent(e.start, e.end, transpose_label(e.label, semitones)) for e in events]
