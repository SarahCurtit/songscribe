"""Chord recognition metrics: WCSR over a comparison vocabulary.

Weighted Chord Symbol Recall is the MIREX measure, and the only one worth
leading with: it is the fraction of *annotated time* a model labels correctly.
Counting segments instead would let a model win by getting a hundred passing
chords right and the tonic wrong.

The shape of the computation:

1. Reference and estimate are each made gapless over the reference's extent,
   filling anything uncovered with ``N`` -- the pipeline drops ``N`` spans, so
   an estimate arrives full of holes that mean "no chord here".
2. The two timelines are merged on the union of their boundaries, giving
   sub-segments over which both sides hold one label.
3. Each sub-segment is scored by projecting both labels into the comparison
   vocabulary. Reference segments the vocabulary cannot express are excluded
   from the score entirely, numerator and denominator both.
4. WCSR is correct duration over compared duration.

Step 3 is where the vocabularies differ, and it is the whole reason several
are reported: ``majmin`` asks whether the triad is right, ``sevenths`` whether
the seventh is too. The default backend only knows triads (see ``CLAUDE.md``),
so its ``sevenths`` score is bounded well below its ``majmin`` score by the
vocabulary rather than by the model -- which is exactly what the pair of
numbers is there to show.

Pure Python: no numpy, no mir_eval, so this runs in CI with no weights and no
extra dependencies. Where the projection rules differ from ``mir_eval``'s,
:data:`VOCABULARIES` says how.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from math import isnan, nan

from ..types import ChordEvent
from .harte import NO_CHORD_LABEL, Chord, parse

#: Durations below this are treated as zero when merging timelines: boundaries
#: that should coincide often differ in the last float digit, and a sliver
#: segment would otherwise score a label nobody intended to annotate there.
MIN_SEGMENT = 1e-9

#: Pitch classes two chords must share for the ``mirex`` comparison to call
#: them a match. Three is the MIREX definition -- a triad's worth of agreement.
MIREX_MIN_SHARED = 3

VOCABULARIES: dict[str, str] = {
    "root": "root only; ignores quality entirely",
    "majmin": "major and minor triads; sevenths reduce to their triad",
    "triads": "majmin plus dim, aug, sus2, sus4",
    "sevenths": "maj, min, 7, maj7, min7",
    "mirex": f"at least {MIREX_MIN_SHARED} pitch classes in common",
}

DEFAULT_VOCABULARIES = ("root", "majmin", "sevenths", "mirex")

#: Which (third, fifth) pair names which triad.
_TRIADS = {
    ("maj3", "P5"): "maj",
    ("min3", "P5"): "min",
    ("min3", "d5"): "dim",
    ("maj3", "A5"): "aug",
    ("sus4", "P5"): "sus4",
    ("sus2", "P5"): "sus2",
}

_MAJMIN = ("maj", "min")
_TRIAD_VOCABULARY = ("maj", "min", "dim", "aug", "sus2", "sus4")


# --------------------------------------------------------------------------- #
# Projection into a comparison vocabulary
# --------------------------------------------------------------------------- #


def triad_of(chord: Chord) -> str | None:
    """Name the triad a chord is built on, or ``None`` if it has none.

    Reads the third and fifth off the interval set and ignores everything
    above, so ``C:maj7``, ``C:7`` and ``C:maj6`` all come back as ``maj``.
    A major third outranks a suspension when both are present, since an added
    4th or 9th colours a triad rather than replacing it.
    """
    intervals = chord.intervals
    if 4 in intervals:
        third = "maj3"
    elif 3 in intervals:
        third = "min3"
    elif 5 in intervals:
        third = "sus4"
    elif 2 in intervals:
        third = "sus2"
    else:
        return None

    if 7 in intervals:
        fifth = "P5"
    elif 6 in intervals:
        fifth = "d5"
    elif 8 in intervals:
        fifth = "A5"
    else:
        return None

    return _TRIADS.get((third, fifth))


def seventh_of(chord: Chord) -> str | None:
    """Name a chord in the MIREX ``sevenths`` vocabulary, or ``None``.

    ``minmaj7`` is deliberately outside it, as in MIREX: it is rare enough
    that admitting it would change scores by noise and nothing else.
    """
    triad = triad_of(chord)
    if triad not in _MAJMIN:
        return None
    if 11 in chord.intervals:
        seventh = "maj7"
    elif 10 in chord.intervals:
        seventh = "b7"
    else:
        seventh = None

    if triad == "maj":
        return {None: "maj", "maj7": "maj7", "b7": "7"}[seventh]
    return {None: "min", "b7": "min7"}.get(seventh)


def project(chord: Chord | None, vocabulary: str) -> str | None:
    """Reduce a chord to a comparable class string in ``vocabulary``.

    Returns ``None`` when the chord cannot be expressed there -- an
    unparseable label, an ``X``, or a quality outside the vocabulary. The two
    sides read that differently: see :func:`compare`.
    """
    if vocabulary not in VOCABULARIES:
        raise ValueError(
            f"unknown vocabulary {vocabulary!r}, expected one of {sorted(VOCABULARIES)}"
        )
    if chord is None or chord.unknown:
        return None
    if chord.is_no_chord:
        return NO_CHORD_LABEL

    if vocabulary == "root":
        return chord.root_name

    if vocabulary == "majmin":
        triad = triad_of(chord)
        quality = triad if triad in _MAJMIN else None
    elif vocabulary == "triads":
        triad = triad_of(chord)
        quality = triad if triad in _TRIAD_VOCABULARY else None
    elif vocabulary == "sevenths":
        quality = seventh_of(chord)
    else:
        # "mirex" compares pitch-class overlap, which is pairwise and so has no
        # per-chord class. compare() handles it without calling project().
        raise ValueError(f"vocabulary {vocabulary!r} has no per-chord projection")

    return None if quality is None else f"{chord.root_name}:{quality}"


def compare(reference: Chord | None, estimate: Chord | None, vocabulary: str) -> bool | None:
    """Score one sub-segment: ``True`` correct, ``False`` wrong, ``None`` skip.

    ``None`` means the reference is not scoreable here -- it is unparseable,
    an ``X``, or a chord the vocabulary cannot express -- and the segment
    leaves the denominator. An unparseable or unprojectable *estimate* is
    simply wrong; a model is not excused by emitting something unreadable.
    """
    if reference is None or reference.unknown:
        return None

    if vocabulary == "mirex":
        if reference.is_no_chord:
            return estimate is not None and estimate.is_no_chord
        if estimate is None or estimate.is_no_chord:
            return False
        shared = reference.pitch_classes & estimate.pitch_classes
        return len(shared) >= MIREX_MIN_SHARED

    reference_class = project(reference, vocabulary)
    if reference_class is None:
        return None
    return reference_class == project(estimate, vocabulary)


# --------------------------------------------------------------------------- #
# Timeline algebra
# --------------------------------------------------------------------------- #


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

    The fill label is ``N``, which is what a gap actually asserts: the
    pipeline calls ``chords.strip_no_chord`` before rendering, so a chart's
    intro and outro arrive here as missing time, not as silence it failed to
    label. Scoring them as ``N`` is what makes an estimate comparable to a
    gapless reference.
    """
    filled: list[ChordEvent] = []
    cursor = start

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


def directional_hamming_distance(
    reference: list[ChordEvent],
    estimate: list[ChordEvent],
) -> float:
    """How much of ``reference`` is not covered by its best-matching estimate.

    For each reference segment, the single longest-overlapping estimate
    segment is kept and the rest of the segment counted as error, normalised
    by reference duration. Labels are ignored -- this measures boundaries
    only, which is why it catches a model that is harmonically right but
    smears every change across a bar.
    """
    total = sum(e.duration for e in reference)
    if total <= MIN_SEGMENT:
        return nan

    error = 0.0
    for segment in reference:
        best = max(
            (
                min(segment.end, other.end) - max(segment.start, other.start)
                for other in estimate
                if other.overlaps(segment.start, segment.end)
            ),
            default=0.0,
        )
        error += segment.duration - best
    return error / total


# --------------------------------------------------------------------------- #
# Scores
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Score:
    """WCSR for one vocabulary, with the durations it was computed from.

    Durations are kept because corpus totals must be summed from them rather
    than averaged from per-track scores -- those are two different numbers,
    and both get reported.
    """

    vocabulary: str
    correct: float
    compared: float
    ignored: float

    @property
    def wcsr(self) -> float:
        """Correct fraction of compared time; ``nan`` if nothing was compared."""
        return self.correct / self.compared if self.compared > MIN_SEGMENT else nan


@dataclass(frozen=True, slots=True)
class Segmentation:
    """Boundary-only agreement, independent of the chord labels."""

    over: float
    under: float

    @property
    def score(self) -> float:
        """The worse of the two directions, as in Harte's thesis and MIREX."""
        if isnan(self.over) or isnan(self.under):
            return nan
        return min(self.over, self.under)


def score(
    reference: list[ChordEvent],
    estimate: list[ChordEvent],
    vocabulary: str,
) -> Score:
    """WCSR of ``estimate`` against ``reference`` in one vocabulary."""
    if vocabulary not in VOCABULARIES:
        raise ValueError(
            f"unknown vocabulary {vocabulary!r}, expected one of {sorted(VOCABULARIES)}"
        )

    correct = compared = ignored = 0.0
    # Labels repeat heavily within a track, so parsing is memoised: a 200-chord
    # annotation holds maybe fifteen distinct labels.
    cache: dict[str, Chord | None] = {}

    for low, high, reference_label, estimate_label in merge_timelines(reference, estimate):
        duration = high - low
        for label in (reference_label, estimate_label):
            if label not in cache:
                cache[label] = parse(label)

        verdict = compare(cache[reference_label], cache[estimate_label], vocabulary)
        if verdict is None:
            ignored += duration
            continue
        compared += duration
        if verdict:
            correct += duration

    return Score(vocabulary=vocabulary, correct=correct, compared=compared, ignored=ignored)


def score_all(
    reference: list[ChordEvent],
    estimate: list[ChordEvent],
    vocabularies: tuple[str, ...] = DEFAULT_VOCABULARIES,
) -> dict[str, Score]:
    """WCSR in several vocabularies at once."""
    return {name: score(reference, estimate, name) for name in vocabularies}


def segmentation(
    reference: list[ChordEvent],
    estimate: list[ChordEvent],
) -> Segmentation:
    """Over- and under-segmentation of ``estimate`` against ``reference``."""
    start, end = extent(reference)
    if end - start <= MIN_SEGMENT:
        return Segmentation(over=nan, under=nan)

    filled_reference = fill_gaps(reference, start, end)
    filled_estimate = fill_gaps(estimate, start, end)
    return Segmentation(
        over=1.0 - directional_hamming_distance(filled_reference, filled_estimate),
        under=1.0 - directional_hamming_distance(filled_estimate, filled_reference),
    )
