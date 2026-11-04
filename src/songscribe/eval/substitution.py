"""How close two chords are musically, rather than whether their symbols match.

A tritone substitution (``G:7`` for ``Db:7``) and a ii written for its V (``D:min7`` for
``G:7``) miss every vocabulary in :mod:`songscribe.eval.metrics`, and a rootless dominant
(``B:hdim7`` for ``G:7``) is outside ``majmin`` and ``sevenths``, so that time is dropped
rather than scored. This module grades the relation between the two chords and gives it
partial credit: :func:`metrics.compare` reads it as pass/fail for the ``substitutions``
vocabulary, :func:`graded_score` as a duration-weighted score. Neither changes WCSR.

The credits are musical judgement, not measured frequencies, so read
:attr:`GradedScore.by_relation` to see which relations a score came from.
:mod:`songscribe.eval.labels` has its own partial credit for locating the form; the two
are separate and not kept in step.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from math import nan

from ..types import ChordEvent
from .harte import Chord, parse
from .metrics import MIN_SEGMENT, MIREX_MIN_SHARED, merge_timelines, triad_of

#: Semitone above the root of a major and a minor third, and of the three
#: sevenths a chord symbol can name. ``bb7`` only ever comes from a diminished
#: seventh; a major sixth sits on the same semitone and is not a seventh, which
#: is why :func:`_seventh` asks the triad before accepting it.
_MAJOR_THIRD = 4
_MINOR_THIRD = 3
_MAJOR_SEVENTH = 11
_MINOR_SEVENTH = 10
_DIMINISHED_SEVENTH = 9

#: Intervals that stand in for a third in a suspension.
_SUSPENSIONS = (5, 2)

#: Root distances, in semitones, between a chord and its relative major or minor.
_RELATIVE_DISTANCES = (3, 9)

#: Semitones from a minor chord's root up to the dominant it is the ii of.
_II_TO_V = 5

#: A tritone, the root distance of a tritone substitution.
_TRITONE = 6

#: Where a diminished chord's root can sit relative to the dominant it stands
#: for: on its b9, 3rd, 5th or b7, which are the four notes left when the
#: dominant's root is dropped. Without this a dim chord and a dominant sharing
#: three notes by coincidence -- ``C:13`` against ``D:hdim7`` -- would read as a
#: rootless voicing of a chord neither contains.
_ROOTLESS_DISTANCES = (1, 4, 7, 10)

#: Shared pitch classes a relation needs before it counts as a substitution at
#: all. Two notes is a coincidence: ``C:maj`` and ``A:min`` share C and E and
#: are a plain error, which is also what ``mirex`` says about them.
_MIN_SHARED = MIREX_MIN_SHARED


@dataclass(frozen=True, slots=True)
class Relation:
    """One named way two chords can stand in for each other.

    ``holds`` is called as ``holds(reference, estimate)``. Most relations are
    symmetric and are written that way; the ones that are not -- a suspension,
    a rootless dominant, a ii written for its V -- test both orders themselves,
    so no caller has to care which side it holds for.
    """

    name: str
    credit: float
    holds: Callable[[Chord, Chord], bool]


@dataclass(frozen=True, slots=True)
class Grade:
    """The relation that best explains one pair of chords, and what it is worth."""

    relation: str
    credit: float


@dataclass(frozen=True, slots=True)
class GradedScore:
    """Partial credit over a timeline, duration-weighted exactly as WCSR is.

    ``by_relation`` holds compared duration per relation name, so it sums to
    :attr:`compared` and says where the credit came from. That breakdown is the
    only check on the credit table there is until cross-source frequencies exist.
    """

    credit: float = 0.0
    compared: float = 0.0
    ignored: float = 0.0
    by_relation: dict[str, float] = field(default_factory=dict)

    @property
    def score(self) -> float:
        """Credit as a fraction of compared time; ``nan`` if nothing was compared."""
        return self.credit / self.compared if self.compared > MIN_SEGMENT else nan


def _third(chord: Chord) -> int | None:
    """Absolute pitch class of the chord's third, or ``None`` if it has none."""
    if chord.root is None:
        return None
    if _MAJOR_THIRD in chord.intervals:
        return (chord.root + _MAJOR_THIRD) % 12
    if _MINOR_THIRD in chord.intervals:
        return (chord.root + _MINOR_THIRD) % 12
    return None


def _seventh(chord: Chord) -> int | None:
    """Absolute pitch class of the chord's seventh, or ``None`` if it has none.

    A pitch class rather than :func:`metrics.seventh_of`'s MIREX class, which
    is undefined off a major or minor triad. A ``bb7`` counts only on a
    diminished chord: the same semitone is a major sixth elsewhere, and reading
    ``C:maj6``'s A as a seventh would pair it with unrelated chords.
    """
    if chord.root is None:
        return None
    if _MAJOR_SEVENTH in chord.intervals:
        return (chord.root + _MAJOR_SEVENTH) % 12
    if _MINOR_SEVENTH in chord.intervals:
        return (chord.root + _MINOR_SEVENTH) % 12
    if _DIMINISHED_SEVENTH in chord.intervals and triad_of(chord) == "dim":
        return (chord.root + _DIMINISHED_SEVENTH) % 12
    return None


def _is_dominant(chord: Chord) -> bool:
    """A major triad carrying a minor seventh: ``C:7``, ``C:9``, ``C:7(#11)``."""
    return triad_of(chord) == "maj" and _MINOR_SEVENTH in chord.intervals


def _is_diminished(chord: Chord) -> bool:
    """``dim``, ``dim7`` or ``hdim7``: anything on a diminished triad.

    ``triad_of`` folds all three onto ``"dim"`` by reading the third and fifth,
    so ``Bdim`` and ``B:hdim7`` arrive together.
    """
    return triad_of(chord) == "dim"


def _is_suspended(chord: Chord) -> bool:
    """No third at all, with a fourth or second standing in for it."""
    if _MAJOR_THIRD in chord.intervals or _MINOR_THIRD in chord.intervals:
        return False
    return any(interval in chord.intervals for interval in _SUSPENSIONS)


def _shared(first: Chord, second: Chord) -> int:
    return len(first.pitch_classes & second.pitch_classes)


def _same(reference: Chord, estimate: Chord) -> bool:
    """The same notes, and no disagreement about which of them is the root.

    A symmetric chord has no determinate root -- a dim7 can be named from any
    of its four notes, an augmented triad from any of its three -- so two
    spellings of one are the same chord. ``C:maj6`` and ``A:min7`` are also the
    same notes but are not symmetric at a sixth, and the root choice there is a
    real difference in function: those are graded ``relative``.
    """
    if reference.pitch_classes != estimate.pitch_classes:
        return False
    if reference.root == estimate.root:
        return True
    shift = (estimate.root - reference.root) % 12
    rotated = frozenset((pitch + shift) % 12 for pitch in reference.pitch_classes)
    return rotated == reference.pitch_classes


def _extension(reference: Chord, estimate: Chord) -> bool:
    """Same root and triad, with a seventh that agrees as far as both name one.

    A chart's ``C:7`` voiced as ``C:13`` is one chord; so is a chart's ``C`` played
    as ``C:maj7``, which is the triad-model flattening this metric exists to stop
    punishing twice. A seventh present on both sides must match, so ``C:maj7``
    against ``C:7`` is not an extension -- that swap changes the chord's function,
    and it is graded as the error it is.
    """
    if reference.root != estimate.root:
        return False
    if triad_of(reference) is None or triad_of(reference) != triad_of(estimate):
        return False
    sevenths = (_seventh(reference), _seventh(estimate))
    return None in sevenths or sevenths[0] == sevenths[1]


def _sus_for_dominant(reference: Chord, estimate: Chord) -> bool:
    """Same root, one side suspended and the other dominant: the third deferred."""
    if reference.root != estimate.root:
        return False
    return (_is_suspended(reference) and _is_dominant(estimate)) or (
        _is_dominant(reference) and _is_suspended(estimate)
    )


def _guide_tones(reference: Chord, estimate: Chord) -> bool:
    """The third and the seventh land on the same pitch classes, in the same roles.

    Those two notes carry a chord's function; the root and fifth are the ones a
    player drops. Roles, not a set: ``G:7``'s third and seventh are ``(B, F)``
    and ``Db:7``'s are ``(F, B)`` -- swapped, which is ``tritone_sub``. A
    diminished chord against a dominant is ``dim_for_dominant``, so that one
    chord grades the same whichever of its notes was named as the root.
    """
    thirds = (_third(reference), _third(estimate))
    sevenths = (_seventh(reference), _seventh(estimate))
    if None in thirds or None in sevenths:
        return False
    if thirds[0] != thirds[1] or sevenths[0] != sevenths[1]:
        return False
    if _dim_for_dominant(reference, estimate):
        return False
    return _shared(reference, estimate) >= _MIN_SHARED


def _tritone_sub(reference: Chord, estimate: Chord) -> bool:
    """Two dominants a tritone apart, which share both guide tones inverted."""
    if not (_is_dominant(reference) and _is_dominant(estimate)):
        return False
    return (reference.root - estimate.root) % 12 == _TRITONE


def _dim_for_dominant(reference: Chord, estimate: Chord) -> bool:
    """One side is the other's dominant without its root.

    ``B:hdim7`` is ``G:9`` minus the G and ``B:dim7`` is ``G:7(b9)`` minus the
    G, which is how a player voices them. Shared notes rather than a subset,
    since a backend emits a bare ``G:7`` with no ninth to contain the A, plus
    ``_ROOTLESS_DISTANCES`` so the dim chord sits where that root would leave it.
    """
    for dim, dominant in ((reference, estimate), (estimate, reference)):
        if not (_is_diminished(dim) and _is_dominant(dominant)):
            continue
        if (dim.root - dominant.root) % 12 not in _ROOTLESS_DISTANCES:
            continue
        if _shared(dim, dominant) >= _MIN_SHARED:
            return True
    return False


def _ii_v_merge(reference: Chord, estimate: Chord) -> bool:
    """One side wrote the ii of a ii-V, the other wrote its V."""
    for minor, dominant in ((reference, estimate), (estimate, reference)):
        if (
            triad_of(minor) == "min"
            and _is_dominant(dominant)
            and minor.root is not None
            and (minor.root + _II_TO_V) % 12 == dominant.root
        ):
            return True
    return False


def _relative(reference: Chord, estimate: Chord) -> bool:
    """A relative major and minor sharing most of their notes: ``C:maj6``/``A:min7``."""
    if reference.root is None or estimate.root is None:
        return False
    if (reference.root - estimate.root) % 12 not in _RELATIVE_DISTANCES:
        return False
    return _shared(reference, estimate) >= _MIN_SHARED


def _parallel(reference: Chord, estimate: Chord) -> bool:
    """Same root, major against minor: the near-miss a triad model makes constantly."""
    if reference.root != estimate.root:
        return False
    triads = (triad_of(reference), triad_of(estimate))
    return all(triad in ("maj", "min") for triad in triads) and triads[0] != triads[1]


#: Every relation, with what a swap of that kind is worth. Ordered by credit for
#: reading; :func:`grade` picks the highest-credit relation that holds, so the
#: order carries no meaning beyond breaking a tie between equal credits.
#:
#: ``none`` is last and always holds, so a pair is never left ungraded.
RELATIONS: tuple[Relation, ...] = (
    Relation("same", 1.00, _same),
    Relation("extension", 0.95, _extension),
    Relation("sus_for_dominant", 0.90, _sus_for_dominant),
    Relation("guide_tones", 0.90, _guide_tones),
    Relation("tritone_sub", 0.80, _tritone_sub),
    Relation("dim_for_dominant", 0.80, _dim_for_dominant),
    Relation("ii_V_merge", 0.70, _ii_v_merge),
    Relation("relative", 0.70, _relative),
    Relation("parallel", 0.40, _parallel),
    Relation("none", 0.00, lambda reference, estimate: True),
)

#: The relation a pair falls to when nothing better holds.
NO_RELATION = RELATIONS[-1].name


def grade(
    reference: Chord | None,
    estimate: Chord | None,
    relations: tuple[Relation, ...] = RELATIONS,
) -> Grade | None:
    """Grade one pair of chords, or ``None`` if the reference is not scoreable.

    The tri-state and its asymmetry are :func:`metrics.compare`'s, for the same
    reason: an unparseable or ``X`` *reference* is a gap in the annotation and
    leaves the denominator, while an unparseable *estimate* is simply wrong --
    a model is not excused by emitting something unreadable.
    """
    if reference is None or reference.unknown:
        return None

    # "No chord" is a label the reference really does assert, so it is scored,
    # but no relation applies to it: either the estimate agrees or it does not.
    if reference.is_no_chord:
        if estimate is not None and estimate.is_no_chord:
            return Grade(relation=RELATIONS[0].name, credit=RELATIONS[0].credit)
        return Grade(relation=NO_RELATION, credit=0.0)
    if estimate is None or estimate.unknown or estimate.is_no_chord:
        return Grade(relation=NO_RELATION, credit=0.0)

    best = max(
        (relation for relation in relations if relation.holds(reference, estimate)),
        key=lambda relation: relation.credit,
        default=None,
    )
    if best is None:
        return Grade(relation=NO_RELATION, credit=0.0)
    return Grade(relation=best.name, credit=best.credit)


def graded_score(
    reference: list[ChordEvent],
    estimate: list[ChordEvent],
    relations: tuple[Relation, ...] = RELATIONS,
) -> GradedScore:
    """Duration-weighted partial credit of ``estimate`` against ``reference``.

    The same merge and the same label-keyed parse memo as :func:`metrics.score`,
    so corpus aggregation stays ``sum(credit) / sum(compared)`` and a long song
    does not count for more per second than a short one.
    """
    credit = compared = ignored = 0.0
    by_relation: dict[str, float] = {}
    cache: dict[str, Chord | None] = {}

    for low, high, reference_label, estimate_label in merge_timelines(reference, estimate):
        duration = high - low
        for label in (reference_label, estimate_label):
            if label not in cache:
                cache[label] = parse(label)

        graded = grade(cache[reference_label], cache[estimate_label], relations)
        if graded is None:
            ignored += duration
            continue
        compared += duration
        credit += graded.credit * duration
        by_relation[graded.relation] = by_relation.get(graded.relation, 0.0) + duration

    return GradedScore(credit=credit, compared=compared, ignored=ignored, by_relation=by_relation)
