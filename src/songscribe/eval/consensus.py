"""Scoring against every chord a bar's sources print, not just the first one.

A tune has no single correct chart. A lead sheet offers a bar two readings -- the
big chord and the small one printed above it -- and two sheets of one tune
disagree the same way: where one writes ``D:min`` another writes ``D:7``. A model
that prints the second is not wrong, but strict WCSR counts it as wrong, so part
of every published chord score is really chart choice.

The set of readings rides inside the annotation label, separated by
``harte.ALTERNATIVE_SEPARATOR`` (``D:min|D:7``), which is why none of this needed
a change to ``.lab`` or its reader. :func:`harte.parse` returns the primary chord
alone, so every existing metric stays strict and its numbers do not move. This
module is the opt-in other reading:

* :func:`score` and :func:`graded_score` count a segment correct when *any*
  source's chord matches, so strict and consensus can be printed side by side.
  The gap between them is how much apparent error was chart choice.
* :func:`disagreements` needs no estimate, no model and no audio: it reads the
  references alone and names, through :mod:`songscribe.eval.substitution`, which
  relation explains each way the sources differ. The pairs it cannot explain are
  either a relation the table is missing or a misread chart, and both want
  reading by eye.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..types import ChordEvent
from . import metrics, substitution
from .harte import ALTERNATIVE_SEPARATOR, Chord, alternatives, parse


def parse_alternatives(label: str) -> tuple[Chord | None, ...]:
    """Parse every chord a label prints, primary first.

    An alternative that does not parse comes back as ``None`` rather than
    failing the label: by the rule in :func:`verdict` an unreadable alternative
    can neither rescue a wrong estimate nor spoil a right one, so a typo in the
    small print costs only itself.
    """
    return tuple(parse(part) for part in alternatives(label))


def has_alternatives(events: Sequence[ChordEvent]) -> bool:
    """Whether any label in ``events`` prints more than one chord.

    Callers gate the consensus work on this so that a corpus of ordinary
    single-chord references does none of it and reports nothing extra.
    """
    return any(ALTERNATIVE_SEPARATOR in event.label for event in events)


def verdict(verdicts: Sequence[bool | None]) -> bool | None:
    """Combine one segment's per-alternative verdicts into one.

    Any source matching makes the segment correct; failing all the sources that
    could be scored makes it wrong; a segment no source could score is skipped.
    So ``True`` beats ``False`` beats ``None``, which is also what makes a label
    with a single chord score exactly as :func:`metrics.compare` scored it.
    """
    if any(item is True for item in verdicts):
        return True
    if any(item is False for item in verdicts):
        return False
    return None


def _compare_any(
    references: Sequence[Chord | None], estimate: Chord | None, vocabulary: str
) -> bool | None:
    return verdict([metrics.compare(chord, estimate, vocabulary) for chord in references])


def _grade_any(
    references: Sequence[Chord | None],
    estimate: Chord | None,
    relations: tuple[substitution.Relation, ...],
) -> substitution.Grade | None:
    """The best grade over the reference's chords, or ``None`` if none is scoreable.

    The best, not the first: a source the estimate matches exactly should not be
    passed over for one it merely resembles.
    """
    graded = [substitution.grade(chord, estimate, relations) for chord in references]
    present = [item for item in graded if item is not None]
    return max(present, key=lambda item: item.credit) if present else None


def compare(reference_label: str, estimate: Chord | None, vocabulary: str) -> bool | None:
    """Compare an estimate against every chord ``reference_label`` prints."""
    return _compare_any(parse_alternatives(reference_label), estimate, vocabulary)


def grade(
    reference_label: str,
    estimate: Chord | None,
    relations: tuple[substitution.Relation, ...] = substitution.RELATIONS,
) -> substitution.Grade | None:
    """Grade an estimate against the best chord ``reference_label`` prints."""
    return _grade_any(parse_alternatives(reference_label), estimate, relations)


def score(
    reference: list[ChordEvent],
    estimate: list[ChordEvent],
    vocabulary: str,
) -> metrics.Score:
    """WCSR of ``estimate`` against every chord the reference prints.

    The same merge, the same duration weighting and the same label-keyed memo as
    :func:`metrics.score`, so a consensus number aggregates over a corpus exactly
    as a strict one does and the two are comparable.
    """
    if vocabulary not in metrics.VOCABULARIES:
        raise ValueError(
            f"unknown vocabulary {vocabulary!r}, expected one of {sorted(metrics.VOCABULARIES)}"
        )

    correct = compared = ignored = 0.0
    references: dict[str, tuple[Chord | None, ...]] = {}
    estimates: dict[str, Chord | None] = {}

    for low, high, reference_label, estimate_label in metrics.merge_timelines(reference, estimate):
        duration = high - low
        if reference_label not in references:
            references[reference_label] = parse_alternatives(reference_label)
        if estimate_label not in estimates:
            estimates[estimate_label] = parse(estimate_label)

        result = _compare_any(references[reference_label], estimates[estimate_label], vocabulary)
        if result is None:
            ignored += duration
            continue
        compared += duration
        if result:
            correct += duration

    return metrics.Score(vocabulary=vocabulary, correct=correct, compared=compared, ignored=ignored)


def score_all(
    reference: list[ChordEvent],
    estimate: list[ChordEvent],
    vocabularies: tuple[str, ...] = metrics.DEFAULT_VOCABULARIES,
) -> dict[str, metrics.Score]:
    """Consensus WCSR in several vocabularies at once."""
    return {name: score(reference, estimate, name) for name in vocabularies}


def graded_score(
    reference: list[ChordEvent],
    estimate: list[ChordEvent],
    relations: tuple[substitution.Relation, ...] = substitution.RELATIONS,
) -> substitution.GradedScore:
    """Partial credit against the best chord the reference prints for each segment."""
    credit = compared = ignored = 0.0
    by_relation: dict[str, float] = {}
    references: dict[str, tuple[Chord | None, ...]] = {}
    estimates: dict[str, Chord | None] = {}

    for low, high, reference_label, estimate_label in metrics.merge_timelines(reference, estimate):
        duration = high - low
        if reference_label not in references:
            references[reference_label] = parse_alternatives(reference_label)
        if estimate_label not in estimates:
            estimates[estimate_label] = parse(estimate_label)

        best = _grade_any(references[reference_label], estimates[estimate_label], relations)
        if best is None:
            ignored += duration
            continue
        compared += duration
        credit += best.credit * duration
        by_relation[best.relation] = by_relation.get(best.relation, 0.0) + duration

    return substitution.GradedScore(
        credit=credit, compared=compared, ignored=ignored, by_relation=by_relation
    )


# --------------------------------------------------------------------------- #
# What the sources disagree about
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Disagreement:
    """One way the sources of a corpus disagree, and what explains it."""

    primary: str
    alternative: str
    #: The relation from :data:`substitution.RELATIONS` that holds between them.
    #: ``substitution.NO_RELATION`` means no theory in the table connects the
    #: two, which is the interesting case.
    relation: str
    credit: float
    #: Annotated seconds whose label prints this pair.
    duration: float
    segments: int

    @property
    def label(self) -> str:
        """The pair as a label would print it."""
        return f"{self.primary}{ALTERNATIVE_SEPARATOR}{self.alternative}"


@dataclass(frozen=True, slots=True)
class Disagreements:
    """Every disagreement among a corpus's sources, largest first."""

    #: Annotated seconds whose label prints more than one chord.
    duration: float = 0.0
    #: All annotated seconds, so :attr:`share` can say how much of the corpus
    #: the sources actually argue about.
    annotated: float = 0.0
    pairs: tuple[Disagreement, ...] = ()
    by_relation: dict[str, float] = field(default_factory=dict)

    @property
    def share(self) -> float:
        """Fraction of annotated time the sources print more than one chord for."""
        return self.duration / self.annotated if self.annotated > metrics.MIN_SEGMENT else 0.0

    @property
    def unexplained(self) -> tuple[Disagreement, ...]:
        """The pairs no relation in the table connects.

        Each is either a relation :mod:`songscribe.eval.substitution` is missing
        or a chart read wrong on the way in. Both want a person's eye, which is
        why these are listed rather than counted.
        """
        return tuple(pair for pair in self.pairs if pair.relation == substitution.NO_RELATION)


def disagreements(
    events: Sequence[ChordEvent],
    relations: tuple[substitution.Relation, ...] = substitution.RELATIONS,
    annotated: float | None = None,
) -> Disagreements:
    """Grade every disagreement among the chords a reference's labels print.

    Each non-primary chord is graded against the primary one, so a bar printing
    three chords contributes two pairs. Needs no estimate and no model: this is a
    property of the sources, and it is what says whether the relation table
    covers the ways real charts of one tune differ.

    ``annotated`` is the total time :attr:`Disagreements.share` is a share *of*,
    for a caller holding only the segments that carry alternatives -- a corpus
    survey, which has the corpus total already. It defaults to these events' own
    duration, which is what one track's reference wants.
    """
    if annotated is None:
        annotated = sum(event.duration for event in events)
    carried = 0.0
    durations: dict[tuple[str, str], float] = {}
    counts: dict[tuple[str, str], int] = {}

    for event in events:
        printed = alternatives(event.label)
        if len(printed) < 2:
            continue
        carried += event.duration
        for alternative in printed[1:]:
            pair = (printed[0], alternative)
            durations[pair] = durations.get(pair, 0.0) + event.duration
            counts[pair] = counts.get(pair, 0) + 1

    pairs = []
    by_relation: dict[str, float] = {}
    for (primary, alternative), duration in durations.items():
        graded = substitution.grade(parse(primary), parse(alternative), relations)
        # An unreadable or X primary grades to None. It is still a disagreement
        # worth printing, so it is reported as explained by nothing.
        relation = substitution.NO_RELATION if graded is None else graded.relation
        credit = 0.0 if graded is None else graded.credit
        pairs.append(
            Disagreement(
                primary=primary,
                alternative=alternative,
                relation=relation,
                credit=credit,
                duration=duration,
                segments=counts[(primary, alternative)],
            )
        )
        by_relation[relation] = by_relation.get(relation, 0.0) + duration

    pairs.sort(key=lambda pair: (-pair.duration, pair.label))
    return Disagreements(
        duration=carried,
        annotated=annotated,
        pairs=tuple(pairs),
        by_relation=dict(sorted(by_relation.items(), key=lambda item: -item[1])),
    )
