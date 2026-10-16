"""Chord recognition metrics: WCSR, the MIREX measure, over a comparison vocabulary.

1. Make reference and estimate gapless over the reference's extent, filling with ``N``.
2. Merge the two on the union of their boundaries into single-label sub-segments.
3. Project both labels into the vocabulary; skip reference time it cannot express.
4. WCSR is correct duration over compared duration.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import isnan, nan
from statistics import median

from ..types import ChordEvent, Word
from .harte import NO_CHORD_LABEL, Chord, parse

#: Durations below this are treated as zero when merging timelines: boundaries
#: that should coincide often differ in the last float digit, and a sliver
#: segment would otherwise score a label nobody intended to annotate there.
MIN_SEGMENT = 1e-9

#: Pitch classes two chords must share for the ``mirex`` comparison to call
#: them a match. Three is the MIREX definition -- a triad's worth of agreement.
MIREX_MIN_SHARED = 3

#: How far either way :func:`best_offset` looks for a better alignment, and at
#: what resolution. Three seconds covers the realistic causes -- encoder
#: lead-in, a rip that kept some pre-gap, a differently trimmed master -- while
#: staying far short of the range where a sweep would start finding spurious
#: fits in a repetitive progression.
ALIGNMENT_SEARCH_SECONDS = 3.0
ALIGNMENT_SEARCH_STEP = 0.05

#: A best-fit time shift is only worth reporting when it beats the measured
#: score by this much *and* is this far from zero. Below either threshold it is
#: the ordinary slack of a model that places boundaries approximately, not a
#: recording on the wrong timeline. Pass both to :meth:`Alignment.is_suspicious`.
ALIGNMENT_WARN_GAIN = 0.05
ALIGNMENT_WARN_OFFSET = 0.15

VOCABULARIES: dict[str, str] = {
    "root": "root only; ignores quality entirely",
    "majmin": "major and minor triads; sevenths reduce to their triad",
    "triads": "majmin plus dim, aug, sus2, sus4",
    "sevenths": "maj, min, 7, maj7, min7",
    "mirex": f"at least {MIREX_MIN_SHARED} pitch classes in common",
}

#: Several are reported because they ask different questions: ``majmin`` whether the
#: triad is right, ``sevenths`` whether the seventh is too. The default backend only knows
#: triads (see ``CLAUDE.md``), so its ``sevenths`` score is bounded well below its
#: ``majmin`` score by the vocabulary rather than the model -- which the pair is there to show.
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
        # A power chord or bare root can never share three pitch classes with
        # anything, itself included, so it is skipped as mir_eval does rather
        # than counted as time no estimate could get right.
        if len(reference.pitch_classes) < MIREX_MIN_SHARED:
            return None
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


def directional_hamming_distance(
    reference: list[ChordEvent],
    estimate: list[ChordEvent],
) -> float:
    """How much of ``reference`` is not covered by its best-matching estimate.

    For each reference segment the single longest-overlapping estimate segment
    is kept and the remainder counted as error, normalised by reference
    duration. Labels are ignored: this measures boundaries only, so it catches
    a model that is harmonically right but smears every change across a bar.
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

    # Weighted by time, not counted by segment: counting segments would let a model win by
    # getting a hundred passing chords right and the tonic wrong.
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


# --------------------------------------------------------------------------- #
# Chart sequence
# --------------------------------------------------------------------------- #


#: Harte degree spelling of each semitone above the root, used to name a chord
#: by its content when no vocabulary class fits it.
_SEMITONE_DEGREES = ("1", "b2", "2", "b3", "3", "4", "b5", "5", "#5", "6", "b7", "7")


def _content_label(chord: Chord) -> str:
    """A canonical Harte degree-list label, e.g. ``B:(1,b3,b5)``.

    Spelled as Harte so it parses back to the same root and intervals -- viz
    colours chart spans by re-parsing their labels. The parenthesised form can
    never equal a projected ``root:quality`` class, nor ``N``.
    """
    degrees = ",".join(_SEMITONE_DEGREES[i] for i in sorted(chord.intervals))
    return f"{chord.root_name}:({degrees})"


def _class_label(label: str, vocabulary: str) -> str:
    """A chord label as its class in ``vocabulary``, or something stable if it has none."""
    chord = parse(label)
    projected = None if chord is None else project(chord, vocabulary)
    if projected is not None:
        return projected
    # A chord the vocabulary cannot express is kept rather than dropped:
    # dropping it would quietly shorten the sequence and let a wrong chord
    # disappear. It is keyed by its pitch content, not its spelling, so
    # `B:dim` and `Bdim` still match; only an unparseable label (or `X`)
    # falls back to its raw text. A reference `X` thus stays in the sequence
    # as a token no backend emits, unlike WCSR, which skips it: dropping it
    # would merge its neighbours. One that prints costs a backend something,
    # depending on what the backend put there and on the chords around it,
    # and can tip a close comparison; see EVALUATION.md, "What gets excluded".
    return label if chord is None or chord.unknown else _content_label(chord)


def dedupe_chart(events: list[ChordEvent], vocabulary: str = "majmin") -> list[ChordEvent]:
    """The chart's chord spans: projected into ``vocabulary``, collapsed, ``N`` gone.

    Collapsing is ``align.dedupe``'s -- merge repeats, drop flicker shorter
    than ``align.MIN_CHORD_DURATION`` -- so everything built on this describes
    the chart ``render.py`` prints rather than the raw frame timeline.
    """
    from ..align import dedupe

    projected = [
        ChordEvent(event.start, event.end, _class_label(event.label, vocabulary))
        for event in events
    ]

    # `N` goes before collapsing, as the pipeline's strip_no_chord runs before
    # align.dedupe: otherwise `C N C` keeps two Cs where the chart prints one.
    return dedupe([e for e in projected if e.label != NO_CHORD_LABEL])


def chart_sequence(
    events: list[ChordEvent],
    vocabulary: str = "majmin",
) -> tuple[str, ...]:
    """The chord sequence a chart would print, with no durations.

    ``N`` spans drop out: a chart has no symbol for them.
    """
    return tuple(e.label for e in dedupe_chart(events, vocabulary))


def sequence_similarity(reference: Sequence[Hashable], estimate: Sequence[Hashable]) -> float:
    """How close two chord sequences are, as 1 - normalised edit distance.

    Timing is absent by construction: only order and identity matter. A
    substitution is a wrong chord, an insertion a chord the model invented, a
    deletion one it missed -- the three things a musician would notice reading
    the chart. ``nan`` when both sequences are empty.
    """
    if not reference and not estimate:
        return nan
    longest = max(len(reference), len(estimate))
    if longest == 0:
        return nan
    return 1.0 - _edit_distance(reference, estimate) / longest


def _edit_distance(a: Sequence[Hashable], b: Sequence[Hashable]) -> int:
    """Levenshtein distance between two label sequences.

    Rolling two rows rather than a full matrix: a chart is a few hundred
    chords at most, but there is no reason to hold n*m cells for it.
    """
    if not a:
        return len(b)
    previous = list(range(len(b) + 1))
    for i, item in enumerate(a, start=1):
        current = [i]
        for j, other in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (item != other),
                )
            )
        previous = current
    return previous[-1]


def chart_score(
    reference: list[ChordEvent],
    estimate: list[ChordEvent],
    vocabulary: str = "majmin",
) -> float:
    """Sequence similarity of the charts two timelines would print."""
    return sequence_similarity(
        chart_sequence(reference, vocabulary), chart_sequence(estimate, vocabulary)
    )


# --------------------------------------------------------------------------- #
# Harmonic rhythm and the tempo-independent grid
# --------------------------------------------------------------------------- #

#: The unit is searched in this window around the median chord duration. The
#: anchor matters more than the search: "harmonic rhythm" means most chords
#: last one unit, so the typical chord length *is* the unit. Searching freely
#: instead degenerates -- an ever-finer grid explains any set of durations
#: ever better, and on real music it picks a unit near the beat rather than
#: the bar.
GRID_WINDOW = (0.5, 1.5)

#: Resolution of that search, as a fraction of the window.
GRID_STEPS = 60

#: Most units one chord may be held for; beyond this the grid has stopped
#: describing harmonic rhythm and is counting bars of a drone.
GRID_MAX_UNITS = 32

#: Largest fit residual at which a grid is still worth reporting. Measured:
#: material with a regular harmonic rhythm fits at ~0%, while real irregular
#: pop sits above 10% -- the Isophonics Eleanor Rigby reference is 10.2%. Past
#: this the chord durations are not near-multiples of any unit and a grid score
#: would be a confident number about nothing.
GRID_MAX_RESIDUAL = 0.08


@dataclass(frozen=True, slots=True)
class HarmonicRhythm:
    """An estimated harmonic grid, and how well it actually fits.

    ``residual`` is the mean rounding error as a fraction of total duration.
    Near zero means the chords really do fall on a grid; large means they do
    not, and :attr:`is_reliable` is how callers avoid pretending otherwise.
    """

    unit: float
    residual: float
    chords: int

    @property
    def is_reliable(self) -> bool:
        return (
            self.chords > 0
            and not isnan(self.unit)
            and not isnan(self.residual)
            and self.residual <= GRID_MAX_RESIDUAL
        )


def harmonic_rhythm(events: list[ChordEvent]) -> HarmonicRhythm:
    """Estimate the harmonic unit -- a bar, typically -- and its fit.

    No beat tracking: anchored on the median chord duration, since harmonic
    rhythm means most chords last one unit, then refined to whichever nearby
    value the durations are closest to whole multiples of. The unit scales
    with tempo, which is what makes a grid comparable across performances.
    """
    durations = [e.duration for e in events if e.duration > MIN_SEGMENT]
    if not durations:
        return HarmonicRhythm(unit=nan, residual=nan, chords=0)

    total = sum(durations)
    anchor = median(durations)
    low, high = anchor * GRID_WINDOW[0], anchor * GRID_WINDOW[1]

    best_unit, best_error = anchor, None
    # Walked coarsest-first with a strict improvement test, so an exact tie
    # keeps the coarser grid. That matters: when every chord lasts 2s, both a
    # 2s unit (one chord per bar) and a 1s unit (two per bar) fit perfectly,
    # and the musician's reading is the former.
    for step in range(GRID_STEPS, -1, -1):
        unit = low + (high - low) * step / GRID_STEPS
        if unit <= MIN_SEGMENT:
            continue
        counts = [min(GRID_MAX_UNITS, max(1, round(d / unit))) for d in durations]
        error = sum(abs(d - n * unit) for d, n in zip(durations, counts, strict=True)) / total
        if best_error is None or error < best_error:
            best_unit, best_error = unit, error

    return HarmonicRhythm(
        unit=best_unit,
        residual=nan if best_error is None else best_error,
        chords=len(durations),
    )


def harmonic_unit(events: list[ChordEvent]) -> float:
    """The estimated harmonic unit in seconds, discarding its fit."""
    return harmonic_rhythm(events).unit


def grid_sequence(
    events: list[ChordEvent],
    vocabulary: str = "majmin",
    unit: float | None = None,
) -> tuple[tuple[str, int], ...]:
    """The chord grid as ``(label, units held)`` pairs, free of tempo.

    ``("C:maj", 1), ("A:min", 1), ("F:maj", 2)`` reads as one bar of C, one of
    A minor, two of F -- the same grid at 90 or 180 bpm, because the unit is
    measured from the music rather than assumed.
    """
    # Built on dedupe_chart, so it describes the printed chart. `unit`
    # overrides the estimate when the true one is known, or when a comparison
    # needs both sides on one grid.
    collapsed = dedupe_chart(events, vocabulary)
    if not collapsed:
        return ()

    if unit is None:
        unit = harmonic_unit(collapsed)
    if isnan(unit) or unit <= MIN_SEGMENT:
        return tuple((e.label, 1) for e in collapsed)

    return tuple((e.label, max(1, round(e.duration / unit))) for e in collapsed)


def grid_score(
    reference: list[ChordEvent],
    estimate: list[ChordEvent],
    vocabulary: str = "majmin",
) -> float:
    """Similarity of two chord grids: order *and* harmonic rhythm.

    Stricter than :func:`chart_score`, which ignores how long each chord is
    held, so the gap between the two isolates harmonic-rhythm error. ``nan``
    when the reference's own chords do not fall on a grid: a number there
    would describe the estimator's guesswork rather than the model.
    """
    # Both sides are measured against the *reference's* unit, as WCSR takes
    # its time base from the reference. Letting each side pick its own would
    # normalise away the very error this metric exists to catch: a model that
    # systematically merges chords has a longer typical chord, so a per-side
    # unit would rescale it back to "one chord per bar" and call it correct.
    rhythm = harmonic_rhythm(dedupe_chart(reference, vocabulary))
    if not rhythm.is_reliable:
        return nan
    return sequence_similarity(
        grid_sequence(reference, vocabulary, unit=rhythm.unit),
        grid_sequence(estimate, vocabulary, unit=rhythm.unit),
    )


# --------------------------------------------------------------------------- #
# Chord per bar
# --------------------------------------------------------------------------- #


def bar_sequence(
    events: list[ChordEvent],
    downbeats: Sequence[float],
    vocabulary: str = "majmin",
) -> tuple[tuple[int, str], ...]:
    """One ``(bar, label)`` per complete bar: the label held longest inside it.

    Time no event covers counts as ``N``, so a bar nothing was played in reads
    ``N`` rather than vanishing from the sequence.
    """
    sequence: list[tuple[int, str]] = []
    for bar, (low, high) in enumerate(pairwise(downbeats)):
        held: dict[str, float] = {}
        covered = 0.0
        for event in events:
            overlap = min(event.end, high) - max(event.start, low)
            if overlap > MIN_SEGMENT:
                label = _class_label(event.label, vocabulary)
                held[label] = held.get(label, 0.0) + overlap
                covered += overlap
        if high - low - covered > MIN_SEGMENT:
            held[NO_CHORD_LABEL] = held.get(NO_CHORD_LABEL, 0.0) + (high - low - covered)
        # max() keeps the first of equal durations, so a tie goes to the earlier label.
        sequence.append((bar, max(held, key=lambda label: held[label])))
    return tuple(sequence)


def bar_score(
    reference: list[ChordEvent],
    estimate: list[ChordEvent],
    downbeats: Sequence[float],
    vocabulary: str = "majmin",
    estimate_downbeats: Sequence[float] | None = None,
) -> float:
    """Fraction of bars holding the right chord: right chord, right bar.

    Offset, tempo and total length cannot move it, because bars are counted
    rather than timed. The reference's bars are ``downbeats``; the estimate's
    are ``estimate_downbeats`` when its clock differs, so each side is read
    against its own bar lines. A reference bar marked ``X`` is skipped, as WCSR
    skips it; ``nan`` when no bar can be scored.
    """
    mine = bar_sequence(reference, downbeats, vocabulary)
    theirs = dict(bar_sequence(estimate, estimate_downbeats or downbeats, vocabulary))
    correct = compared = 0
    for bar, label in mine:
        chord = parse(label)
        if chord is not None and chord.unknown:
            continue
        compared += 1
        correct += theirs.get(bar) == label
    return correct / compared if compared else nan


# --------------------------------------------------------------------------- #
# Lyric-chord placement
# --------------------------------------------------------------------------- #


def placement_sequence(
    chords: list[ChordEvent],
    words: list[Word],
    vocabulary: str = "majmin",
    compare_labels: bool = True,
    **group_kwargs: object,
) -> tuple[tuple, ...]:
    """Where each chord change lands, as ``(line, word index[, label])``.

    This is the lead sheet's real question -- is the chord over the right
    syllable -- and it is answered in word indices, not seconds, so it needs no
    lyric text and is immune to offset and tempo alike.
    """
    from ..align import group_words, place_in_line

    spans = dedupe_chart(chords, vocabulary)
    placed: list[tuple] = []
    for line_index, line in enumerate(group_words(words, **group_kwargs)):  # type: ignore[arg-type]
        offsets = line.word_offsets()
        for placement in place_in_line(line, spans):
            word_index = offsets.index(placement.offset)
            # Dropping the label asks the weaker question -- is the *change*
            # in the right place -- which separates a mistimed chord from a
            # misnamed one.
            placed.append(
                (line_index, word_index, placement.label)
                if compare_labels
                else (line_index, word_index)
            )
    return tuple(placed)


def placement_score(
    reference_chords: list[ChordEvent],
    estimate_chords: list[ChordEvent],
    words: list[Word],
    vocabulary: str = "majmin",
    compare_labels: bool = True,
    **group_kwargs: object,
) -> float:
    """How well an estimate's chords land on the same syllables as the reference's.

    Both sides are placed against **one** word timeline, the reference's, so
    this measures the chord branch alone. Scoring each against its own ASR
    output would confound two independent failures, and the branches are kept
    independent everywhere else for exactly that reason.
    """
    return sequence_similarity(
        placement_sequence(reference_chords, words, vocabulary, compare_labels, **group_kwargs),
        placement_sequence(estimate_chords, words, vocabulary, compare_labels, **group_kwargs),
    )


# --------------------------------------------------------------------------- #
# Alignment check
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Alignment:
    """Whether a recording shares the reference's timeline.

    ``offset`` is the shift that scored best, ``shifted`` the score there and
    ``measured`` the score as actually reported.
    """

    offset: float
    shifted: float
    measured: float

    @property
    def gain(self) -> float:
        """How much the best shift beat the score being reported."""
        if isnan(self.shifted) or isnan(self.measured):
            return nan
        return self.shifted - self.measured

    def is_suspicious(self, min_gain: float, min_offset: float) -> bool:
        """Whether this looks like a misaligned recording rather than noise."""
        if isnan(self.gain):
            return False
        return self.gain >= min_gain and abs(self.offset) >= min_offset


def best_offset(
    reference: list[ChordEvent],
    estimate: list[ChordEvent],
    vocabulary: str = "majmin",
    search: float = ALIGNMENT_SEARCH_SECONDS,
    step: float = ALIGNMENT_SEARCH_STEP,
) -> Alignment:
    """Find the time shift of ``estimate`` that scores best against ``reference``.

    A large gain means the recording does not share the reference's timeline --
    an encoder lead-in, a kept pre-gap, a differently trimmed master -- and the
    reported score is measuring that mismatch rather than the model. Pure
    timeline arithmetic, so it costs no model time.
    """
    # Diagnostic only: the offset is a number to go and fix the audio with,
    # never a correction to fold into a score. Shifting an estimate until it
    # fits the reference best is fitting to the test set, and would turn a
    # broken corpus into a flattering one.
    measured = score(reference, estimate, vocabulary).wcsr
    best = Alignment(offset=0.0, shifted=measured, measured=measured)
    if not reference or not estimate or step <= 0:
        return best

    steps = int(search / step)
    for index in range(-steps, steps + 1):
        offset = index * step
        if index == 0:
            continue
        shifted = [ChordEvent(e.start + offset, e.end + offset, e.label) for e in estimate]
        candidate = score(reference, shifted, vocabulary).wcsr
        if not isnan(candidate) and (isnan(best.shifted) or candidate > best.shifted):
            best = Alignment(offset=offset, shifted=candidate, measured=measured)
    return best
