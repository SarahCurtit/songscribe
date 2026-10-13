"""Data preparation for looking at chord annotations.

Plot-library-free on purpose. Every function here returns plain tuples, so the
shaping is unit-tested in CI with no plotting backend installed, and the
notebook in ``notebooks/`` stays a thin layer of matplotlib calls over it.
That split matters because the interesting bugs are in the shaping -- which
chord got which colour, which bin a duration fell in -- not in the drawing.

Colours are returned as **root pitch classes** (0-11), not RGB: same root,
same colour, whatever palette the caller picks. ``NO_COLOUR`` marks a span
with no root, meaning ``N`` or a label that would not parse.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ..types import ChordEvent
from .harte import PITCH_CLASSES, parse
from .metrics import ALIGNMENT_SEARCH_SECONDS, ALIGNMENT_SEARCH_STEP, dedupe_chart, score

#: Colour index for a span with no root: ``N``, ``X``, or an unparseable label.
NO_COLOUR = -1

#: Default width of a duration histogram bin, in seconds. Half a second is
#: about the finest division that still reads as a chord rather than flicker.
DURATION_BIN = 0.5


@dataclass(frozen=True, slots=True)
class Span:
    """One chord span, ready to draw as a rectangle on a timeline."""

    start: float
    end: float
    label: str
    colour: int

    @property
    def duration(self) -> float:
        return self.end - self.start


def chord_spans(
    events: list[ChordEvent],
    vocabulary: str | None = None,
    chart: bool = False,
) -> tuple[Span, ...]:
    """Chord events as drawable spans, coloured by root.

    ``vocabulary`` projects the labels first; ``chart`` additionally collapses
    them the way a printed chart would, which is how to see the difference
    between the raw timeline and what ``render.py`` prints.
    """
    if chart:
        if vocabulary is None:
            raise ValueError("chart=True needs a vocabulary to collapse into")
        events = dedupe_chart(events, vocabulary)
    elif vocabulary is not None:
        from .metrics import project

        projected = []
        for event in events:
            chord = parse(event.label)
            label = None if chord is None else project(chord, vocabulary)
            projected.append(ChordEvent(event.start, event.end, label or event.label))
        events = projected

    spans = []
    for event in events:
        chord = parse(event.label)
        colour = NO_COLOUR if chord is None or chord.root is None else chord.root
        spans.append(Span(event.start, event.end, event.label, colour))
    return tuple(spans)


def duration_bins(
    events: list[ChordEvent],
    width: float = DURATION_BIN,
    limit: float | None = None,
) -> tuple[tuple[float, int], ...]:
    """Histogram of chord durations as ``(bin start, count)`` pairs.

    Empty bins are included, so the shape of the harmonic rhythm is visible
    rather than being closed up by the plot. ``limit`` clips the long tail.
    """
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")

    durations = [e.duration for e in events if e.duration > 0]
    if not durations:
        return ()

    longest = min(max(durations), limit) if limit else max(durations)
    counts: Counter[int] = Counter()
    for duration in durations:
        if duration <= longest:
            counts[int(duration / width)] += 1
    return tuple((index * width, counts.get(index, 0)) for index in range(int(longest / width) + 1))


def label_time(events: list[ChordEvent], top: int | None = None) -> tuple[tuple[str, float], ...]:
    """Total seconds per chord label, most time first."""
    totals: Counter[str] = Counter()
    for event in events:
        totals[event.label] += event.duration
    ranked = tuple(totals.most_common())
    return ranked[:top] if top else ranked


def root_time(events: list[ChordEvent]) -> tuple[tuple[str, float], ...]:
    """Total seconds per root, in chromatic order with every root present.

    Includes roots with zero time, so the shape reads as a key profile rather
    than an arbitrary bar chart -- the gaps are the informative part.
    """
    totals = dict.fromkeys(PITCH_CLASSES, 0.0)
    for event in events:
        chord = parse(event.label)
        if chord is not None and chord.root is not None:
            totals[PITCH_CLASSES[chord.root]] += event.duration
    return tuple(totals.items())


def offset_curve(
    reference: list[ChordEvent],
    estimate: list[ChordEvent],
    vocabulary: str = "majmin",
    search: float = ALIGNMENT_SEARCH_SECONDS,
    step: float = ALIGNMENT_SEARCH_STEP,
) -> tuple[tuple[float, float], ...]:
    """WCSR as a function of a global time shift, as ``(offset, score)`` pairs.

    Plotted, a clean single peak at zero means the recording is on the
    reference's timeline; a peak elsewhere is the lead-in or wrong-master
    problem, and a flat curve means the estimate is not tracking at all.
    """
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")

    points = []
    steps = int(search / step)
    for index in range(-steps, steps + 1):
        offset = index * step
        shifted = [ChordEvent(e.start + offset, e.end + offset, e.label) for e in estimate]
        points.append((offset, score(reference, shifted, vocabulary).wcsr))
    return tuple(points)


def agreement_spans(
    reference: list[ChordEvent],
    estimate: list[ChordEvent],
    vocabulary: str = "majmin",
) -> tuple[tuple[float, float, bool | None], ...]:
    """Where the two sides agree, as ``(start, end, verdict)`` sub-segments.

    ``True`` correct, ``False`` wrong, ``None`` not scoreable -- the same three
    outcomes WCSR sums over, laid out in time so a run of errors is visible as
    a block rather than averaged into a single number.
    """
    from .metrics import compare, merge_timelines

    out = []
    for low, high, reference_label, estimate_label in merge_timelines(reference, estimate):
        verdict = compare(parse(reference_label), parse(estimate_label), vocabulary)
        out.append((low, high, verdict))
    return tuple(out)
