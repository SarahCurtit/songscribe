"""Timeline algebra: make two chord timelines gapless and slice them into shared segments.

Split out of :mod:`songscribe.eval.metrics` so :mod:`songscribe.eval.substitution` can use
it without importing the module that imports it.
"""

from __future__ import annotations

from itertools import pairwise

from ..types import ChordEvent
from .harte import NO_CHORD_LABEL

#: Durations below this are treated as zero when merging timelines: boundaries
#: that should coincide often differ in the last float digit, and a sliver
#: segment would otherwise score a label nobody intended to annotate there.
MIN_SEGMENT = 1e-9


def extent(events: list[ChordEvent]) -> tuple[float, float]:
    """Span covered by ``events``, as ``(start, end)``."""
    if not events:
        return 0.0, 0.0
    return min(e.start for e in events), max(e.end for e in events)


def fill_gaps(
    events: list[ChordEvent],
    start: float,
    end: float,
    label: str = NO_CHORD_LABEL,
) -> list[ChordEvent]:
    """Clip ``events`` to ``[start, end]`` and fill any uncovered time.

    The fill label is ``N``, which is what a gap actually asserts, and what
    makes a holey estimate comparable to a gapless reference.
    """
    # The pipeline calls chords.strip_no_chord before rendering, so a chart's
    # intro and outro arrive here as missing time rather than as silence the
    # model failed to label.
    filled: list[ChordEvent] = []
    cursor = start

    # `cursor` never moves backwards, so overlapping input resolves as
    # earliest-wins: the later chord keeps only the time the earlier one did
    # not claim. Published annotations are gapless and non-overlapping, so
    # this only arises on a malformed file, where a deterministic reading
    # beats either guessing or refusing to score the track.
    for event in sorted(events, key=lambda e: (e.start, e.end)):
        clipped_start = max(event.start, cursor)
        clipped_end = min(event.end, end)
        if clipped_end - clipped_start <= MIN_SEGMENT:
            continue
        if clipped_start - cursor > MIN_SEGMENT:
            filled.append(ChordEvent(cursor, clipped_start, label))
        filled.append(ChordEvent(clipped_start, clipped_end, event.label))
        cursor = clipped_end

    if end - cursor > MIN_SEGMENT:
        filled.append(ChordEvent(cursor, end, label))
    return filled


def merge_timelines(
    reference: list[ChordEvent],
    estimate: list[ChordEvent],
) -> list[tuple[float, float, str, str]]:
    """Slice two timelines into sub-segments where both hold one label.

    The reference defines the evaluated span: estimate time outside it is
    discarded, and reference time the estimate never reached counts as ``N``.
    """
    start, end = extent(reference)
    if end - start <= MIN_SEGMENT:
        return []

    reference = fill_gaps(reference, start, end)
    estimate = fill_gaps(estimate, start, end)

    boundaries = sorted({start, end} | {b for e in reference + estimate for b in (e.start, e.end)})

    merged: list[tuple[float, float, str, str]] = []
    r = s = 0
    for low, high in pairwise(boundaries):
        if high - low <= MIN_SEGMENT:
            continue
        # Both timelines are gapless and sorted, so each pointer only ever
        # moves forward: advance it past segments that end at or before `low`.
        while r + 1 < len(reference) and reference[r].end <= low + MIN_SEGMENT:
            r += 1
        while s + 1 < len(estimate) and estimate[s].end <= low + MIN_SEGMENT:
            s += 1
        merged.append((low, high, reference[r].label, estimate[s].label))

    return merged
