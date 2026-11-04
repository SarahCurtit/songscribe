from __future__ import annotations

from math import isnan

import pytest

from songscribe.eval import metrics, substitution
from songscribe.eval.harte import parse
from songscribe.types import ChordEvent


def timeline(*spec: tuple[float, float, str]) -> list[ChordEvent]:
    """A chord timeline from ``(start, end, label)`` triples."""
    return [ChordEvent(start, end, label) for start, end, label in spec]


def graded(reference: str, estimate: str) -> substitution.Grade | None:
    return substitution.grade(parse(reference), parse(estimate))


def relation(reference: str, estimate: str) -> str | None:
    result = graded(reference, estimate)
    return None if result is None else result.relation


def credit(reference: str, estimate: str) -> float | None:
    result = graded(reference, estimate)
    return None if result is None else result.credit


#: Labels swept pairwise by TestEveryRelationEarnsItsPlace: enough roots to
#: cover every distance between them, one quality per family, and the wide
#: chords whose three-note overlap is what makes an overlap-gated relation
#: too loose.
GRID = [
    f"{root}:{quality}"
    for root in ("C", "D", "Eb", "E", "F", "G", "A", "Bb", "B")
    for quality in (
        "maj",
        "min",
        "dim",
        "aug",
        "maj7",
        "min7",
        "7",
        "dim7",
        "hdim7",
        "minmaj7",
        "maj6",
        "min6",
        "sus4",
        "sus2",
        "sus4(b7)",
        "9",
        "11",
        "13",
        "7(b9)",
        # Both spellings of a dominant with a flattened fifth, which are not the
        # same chord: "7(b5)" adds the flat five and keeps the natural one,
        # "7(*5,b5)" replaces it. Only the second is a note from a hdim7.
        "7(b5)",
        "7(*5,b5)",
        # hdim7(9) is in the project's own chart, and min7(b5) is a fifth-note
        # spelling of a half-diminished chord: both reach dim_near_miss.
        "hdim7(9)",
        "min7(b5)",
    )
]

#: Pairs whose relation must not depend on which side is the reference. The
#: directional relations are tested separately, in both orders, by their own
#: classes -- they are symmetric in outcome but not in definition.
SYMMETRIC = [
    ("C:maj", "C:maj"),
    ("C:7", "C:13"),
    ("C:maj", "C:maj7"),
    ("B:hdim7", "Bb:maj7"),
    ("C:dim7", "Eb:dim7"),
    ("G:7", "Db:7"),
    ("C:maj6", "A:min7"),
    ("C:maj", "C:min"),
    ("C:maj", "F#:maj"),
]


class TestSame:
    def test_an_identical_label_is_the_same_chord(self):
        assert graded("C:maj", "C:maj") == substitution.Grade("same", 1.00)

    def test_an_enharmonic_respelling_is_the_same_chord(self):
        assert relation("Db:maj", "C#:maj") == "same"

    def test_the_same_notes_on_another_root_is_relative_not_same(self):
        # C:maj6 and A:min7 are the same four pitch classes, but the chord is not
        # symmetric at a sixth, so which note is the root is a real difference.
        assert parse("C:maj6").pitch_classes == parse("A:min7").pitch_classes
        assert relation("C:maj6", "A:min7") == "relative"

    @pytest.mark.parametrize(
        ("reference", "estimate"),
        [
            ("C:dim7", "Eb:dim7"),
            ("C:dim7", "Gb:dim7"),
            ("C:dim7", "A:dim7"),
            ("C:aug", "E:aug"),
            ("C:aug", "Ab:aug"),
        ],
    )
    def test_a_symmetric_chord_named_from_another_of_its_notes_is_the_same_chord(
        self, reference, estimate
    ):
        # A dim7 can be named from any of its four notes and an augmented triad
        # from any of its three; annotators and backends pick freely.
        assert parse(reference).pitch_classes == parse(estimate).pitch_classes
        assert graded(reference, estimate) == substitution.Grade("same", 1.00)
        assert metrics.compare(parse(reference), parse(estimate), "substitutions") is True


class TestExtension:
    @pytest.mark.parametrize(
        ("reference", "estimate"),
        [
            ("C:7", "C:13"),
            ("C:7", "C:9"),
            ("C:maj", "C:maj(9)"),
            # The chart writes the altered dominant; the player plays the plain
            # one. Harte's 7(#5) keeps a natural fifth, so both read as a major
            # triad with a minor seventh.
            ("A:7(#5)", "A:7"),
            ("C:7(b5)", "C:7"),
            # A triad model flattening a seventh, which is the whole reason this
            # relation exists: `sevenths` already punishes it once.
            ("C:7", "C:maj"),
            ("C:maj", "C:maj7"),
            ("C:min", "C:min7"),
            ("C:dim", "C:dim7"),
        ],
    )
    def test_the_same_chord_voiced_wider(self, reference, estimate):
        assert relation(reference, estimate) == "extension"
        assert credit(reference, estimate) == 0.95

    def test_a_seventh_both_sides_name_must_agree(self):
        # maj7 against a dominant 7 on one root changes the function, so it is
        # the weaker `seventh_swap` rather than an extension.
        assert relation("C:maj7", "C:7") == "seventh_swap"

    def test_a_different_root_is_not_an_extension(self):
        assert relation("C:7", "D:13") == "none"


class TestSusForDominant:
    @pytest.mark.parametrize(
        ("reference", "estimate"),
        [("C:sus4(b7)", "C:7"), ("C:7", "C:sus4"), ("C:sus4", "C:9"), ("C:9", "C:sus2")],
    )
    def test_a_suspension_defers_the_third_of_its_dominant(self, reference, estimate):
        assert relation(reference, estimate) == "sus_for_dominant"
        assert credit(reference, estimate) == 0.90

    def test_a_suspension_against_a_plain_triad_is_the_weaker_relation(self):
        # No seventh on the other side to corroborate a dominant reading, so it
        # is graded below `sus_for_dominant` rather than as one.
        assert relation("C:sus4", "C:maj") == "sus_for_triad"


class TestNearMissesBelowTheThreshold:
    """Relations that fail `substitutions` but are not graded as plain errors."""

    @pytest.mark.parametrize(
        ("reference", "estimate", "name"),
        [
            ("C:maj7", "C:7", "seventh_swap"),
            ("C:7", "C:maj7", "seventh_swap"),
            ("C:13", "C:maj7", "seventh_swap"),
            ("C:min7", "C:minmaj7", "seventh_swap"),
            ("C:sus4", "C:maj", "sus_for_triad"),
            ("C:maj", "C:sus4", "sus_for_triad"),
            ("C:sus4(b7)", "C:maj", "sus_for_triad"),
            ("C:sus2", "C:min", "sus_for_triad"),
            # One suspension against the other: same root, same deferral.
            ("C:sus4", "C:sus2", "sus_for_triad"),
            ("C:sus4(b7)", "C:sus2", "sus_for_triad"),
        ],
    )
    def test_a_near_miss_keeps_partial_credit(self, reference, estimate, name):
        assert relation(reference, estimate) == name
        assert 0.0 < credit(reference, estimate) < metrics.SUBSTITUTION_MIN_CREDIT

    @pytest.mark.parametrize(
        ("reference", "estimate"),
        [("C:maj7", "C:7"), ("C:sus4", "C:maj"), ("C:maj", "C:min")],
    )
    def test_a_near_miss_still_fails_the_vocabulary(self, reference, estimate):
        assert metrics.compare(parse(reference), parse(estimate), "substitutions") is False

    def test_a_seventh_swap_outscores_an_unrelated_chord(self):
        # C:maj7 against C:7 agrees on three notes of four and is correct under
        # both majmin and mirex; C:maj against F#:maj shares none.
        assert credit("C:maj7", "C:7") > credit("C:maj", "F#:maj")
        assert metrics.compare(parse("C:maj7"), parse("C:7"), "majmin") is True
        assert metrics.compare(parse("C:maj7"), parse("C:7"), "mirex") is True


class TestGuideTones:
    @pytest.mark.parametrize(
        ("reference", "estimate"), [("B:hdim7", "Bb:maj7"), ("Bb:maj7", "B:hdim7")]
    )
    def test_the_third_and_seventh_in_the_same_roles(self, reference, estimate):
        # Both carry D as the third and A as the seventh, and share D F A.
        assert substitution._third(parse(reference)) == substitution._third(parse(estimate))
        assert substitution._seventh(parse(reference)) == substitution._seventh(parse(estimate))
        assert relation(reference, estimate) == "guide_tones"
        assert credit(reference, estimate) == 0.90

    def test_two_shared_notes_are_not_enough(self):
        # C:maj7 and Db:min7 agree on the third and seventh and on nothing else:
        # two notes out of four, which is what `mirex` calls a miss.
        assert len(parse("C:maj7").pitch_classes & parse("Db:min7").pitch_classes) == 2
        assert relation("C:maj7", "Db:min7") == "none"
        assert metrics.compare(parse("C:maj7"), parse("Db:min7"), "mirex") is False

    def test_a_chord_with_no_triad_has_no_guide_tones(self):
        # C:maj(*5) omits its fifth, so triad_of cannot classify it and
        # `extension` does not apply; with no seventh there are no guide tones.
        assert metrics.triad_of(parse("C:maj(*5)")) is None
        assert relation("C:maj(*5)", "C:maj7") == "none"

    def test_a_chord_without_a_seventh_has_no_guide_tones(self):
        assert relation("C:maj6", "A:maj6") == "none"

    def test_a_major_sixth_is_not_read_as_a_seventh(self):
        # Both sit on the same semitone above the root. Counting C:maj6's A as a
        # seventh would pair it with chords it shares no function with.
        assert substitution._seventh(parse("C:maj6")) is None
        assert substitution._seventh(parse("C:dim7")) is not None


class TestTritoneSub:
    @pytest.mark.parametrize(("reference", "estimate"), [("G:7", "Db:7"), ("Db:7", "G:7")])
    def test_two_dominants_a_tritone_apart(self, reference, estimate):
        assert relation(reference, estimate) == "tritone_sub"
        assert credit(reference, estimate) == 0.80

    def test_it_is_not_graded_as_guide_tones(self):
        # G:7 is (B, F) and Db:7 is (F, B): equal as sets, swapped as a pair.
        # Comparing sets would collapse the two relations and pay this 0.90.
        assert substitution._third(parse("G:7")) == substitution._seventh(parse("Db:7"))
        assert substitution._seventh(parse("G:7")) == substitution._third(parse("Db:7"))
        assert relation("G:7", "Db:7") != "guide_tones"

    def test_both_sides_must_be_dominant(self):
        assert relation("G:7", "Db:maj7") == "none"


class TestDimForDominant:
    @pytest.mark.parametrize(
        ("reference", "estimate"),
        [
            ("B:hdim7", "G:7"),
            ("G:7", "B:hdim7"),
            ("B:dim7", "G:7(b9)"),
            ("D:dim7", "G:7"),
            ("E:hdim7", "C:7"),
        ],
    )
    def test_a_diminished_chord_is_its_dominant_without_the_root(self, reference, estimate):
        assert relation(reference, estimate) == "dim_for_dominant"
        assert credit(reference, estimate) == 0.80

    @pytest.mark.parametrize("root", ["Db", "E", "G", "Bb"])
    def test_every_spelling_of_one_dim7_grades_alike(self, root):
        # A dim7 is symmetric, so these are four names for one chord. Grading
        # them differently would make the score depend on the annotator's choice.
        assert parse(f"{root}:dim7").pitch_classes == parse("E:dim7").pitch_classes
        assert graded(f"{root}:dim7", "C:7") == substitution.Grade("dim_for_dominant", 0.80)

    def test_a_dim_chord_on_the_dominants_own_root_is_not_rootless(self):
        # "Without its root" means nothing when both name the same root.
        assert relation("C:hdim7", "C:7(b5)") == "none"

    def test_three_shared_notes_is_not_enough_to_be_rootless(self):
        # Each shares three notes with the dominant but carries neither its third
        # nor its seventh, so it is not that chord with the root removed.
        assert len(parse("C:13").pitch_classes & parse("D:hdim7").pitch_classes) == 3
        assert relation("C:13", "D:hdim7") == "none"
        assert len(parse("C:11").pitch_classes & parse("G:hdim7").pitch_classes) == 3
        assert relation("C:11", "G:hdim7") == "none"

    def test_the_guide_tones_of_the_dominant_must_survive(self):
        # Dropping G from G:7 leaves B, D and F: the third and seventh are what
        # a rootless voicing keeps.
        assert {substitution._third(parse("G:7")), substitution._seventh(parse("G:7"))} <= parse(
            "B:hdim7"
        ).pitch_classes
        assert relation("B:hdim7", "G:7") == "dim_for_dominant"

    def test_an_unrelated_diminished_chord_is_not(self):
        # E:hdim7 is the ii of D minor, not a rootless A7: it shares only two
        # pitch classes with it.
        assert relation("E:hdim7", "A:7") == "none"

    def test_a_diminished_chord_on_the_dominant_root_is_not(self):
        assert relation("C:dim7", "C:7") == "none"


class TestIiVMerge:
    @pytest.mark.parametrize(
        ("reference", "estimate"),
        [("D:min7", "G:7"), ("G:7", "D:min7"), ("D:min", "G:7"), ("D:min7", "G:9")],
    )
    def test_one_side_wrote_the_ii_and_the_other_the_v(self, reference, estimate):
        assert relation(reference, estimate) == "ii_V_merge"
        assert credit(reference, estimate) == 0.70

    def test_the_wrong_distance_is_not_a_ii_v(self):
        assert relation("D:min7", "A:7") == "none"


class TestRelative:
    @pytest.mark.parametrize(
        ("reference", "estimate"),
        [("C:maj6", "A:min7"), ("A:min7", "C:maj6"), ("C:min7", "Eb:maj7")],
    )
    def test_a_relative_pair_sharing_most_of_its_notes(self, reference, estimate):
        assert relation(reference, estimate) == "relative"
        assert credit(reference, estimate) == 0.70

    def test_bare_relative_triads_share_too_little_to_count(self):
        # C:maj and A:min share two pitch classes. Different third, different
        # fifth, and only the root pair suggesting a relation -- which is an
        # error, and is also what `mirex` has always said about this pair.
        assert len(parse("C:maj").pitch_classes & parse("A:min").pitch_classes) == 2
        assert relation("C:maj", "A:min") == "none"
        assert metrics.compare(parse("C:maj"), parse("A:min"), "mirex") is False

    def test_a_mediant_is_not_a_relative(self):
        assert relation("C:maj", "E:min") == "none"

    @pytest.mark.parametrize(
        ("reference", "estimate"),
        [("C:11", "Eb:11"), ("C:13", "A:sus4"), ("C:maj13", "A:sus2"), ("C:aug", "A:minmaj7")],
    )
    def test_wide_chords_a_third_apart_are_not_relatives(self, reference, estimate):
        # Three shared notes is easy for six-note chords, so the relation needs
        # one major and one minor triad rather than overlap alone.
        assert len(parse(reference).pitch_classes & parse(estimate).pitch_classes) >= 3
        assert relation(reference, estimate) == "none"


class TestParallel:
    @pytest.mark.parametrize(
        ("reference", "estimate"),
        [("C:maj", "C:min"), ("C:min", "C:maj"), ("C:maj7", "C:min7")],
    )
    def test_major_against_minor_on_one_root(self, reference, estimate):
        assert relation(reference, estimate) == "parallel"
        assert credit(reference, estimate) == 0.40

    def test_a_diminished_triad_is_not_a_parallel(self):
        assert relation("C:maj", "C:dim") == "none"


class TestUnrelated:
    @pytest.mark.parametrize(
        ("reference", "estimate"),
        [("C:maj", "F#:maj"), ("C:maj", "E:min"), ("C:maj", "A:min"), ("C:maj", "Db:min")],
    )
    def test_a_wrong_chord_earns_nothing(self, reference, estimate):
        assert graded(reference, estimate) == substitution.Grade("none", 0.0)


class TestSymmetry:
    @pytest.mark.parametrize(("reference", "estimate"), SYMMETRIC)
    def test_the_grade_does_not_depend_on_the_side(self, reference, estimate):
        assert graded(reference, estimate) == graded(estimate, reference)


class TestHighestCreditWins:
    def test_a_pair_several_relations_hold_for_takes_the_best(self):
        holding = [
            r.name for r in substitution.RELATIONS if r.holds(*map(parse, ("C:maj7", "C:maj9")))
        ]
        assert "extension" in holding and "guide_tones" in holding
        assert graded("C:maj7", "C:maj9") == substitution.Grade("extension", 0.95)

    def test_the_table_credits_are_non_increasing(self):
        # What makes reading the table top to bottom and picking the highest
        # credit the same answer.
        credits = [r.credit for r in substitution.RELATIONS]
        assert credits == sorted(credits, reverse=True)

    @pytest.mark.parametrize(
        ("reference", "estimate", "name"),
        [
            ("C:maj", "C:maj", "same"),
            ("C:dim7", "Eb:dim7", "same"),
            ("C:7", "C:13", "extension"),
            ("C:sus4(b7)", "C:7", "sus_for_dominant"),
            ("B:hdim7", "Bb:maj7", "guide_tones"),
            ("G:7", "Db:7", "tritone_sub"),
            ("B:hdim7", "G:7", "dim_for_dominant"),
            ("D:min7", "G:7", "ii_V_merge"),
            ("C:maj6", "A:min7", "relative"),
            ("C:sus4", "C:maj", "sus_for_triad"),
            ("C:maj7", "C:7", "seventh_swap"),
            ("C:dim7", "C:min6", "dim_near_miss"),
            ("C:maj", "C:min", "parallel"),
            ("C:maj", "F#:maj", "none"),
        ],
    )
    def test_each_documented_example_grades_as_printed(self, reference, estimate, name):
        # The pairs EVALUATION.md's table prints against each relation.
        assert relation(reference, estimate) == name

    def test_the_documented_examples_cover_every_relation(self):
        assert {
            name
            for _, _, name in [
                ("C:maj", "C:maj", "same"),
                ("C:7", "C:13", "extension"),
                ("C:sus4(b7)", "C:7", "sus_for_dominant"),
                ("B:hdim7", "Bb:maj7", "guide_tones"),
                ("G:7", "Db:7", "tritone_sub"),
                ("B:hdim7", "G:7", "dim_for_dominant"),
                ("D:min7", "G:7", "ii_V_merge"),
                ("C:maj6", "A:min7", "relative"),
                ("C:sus4", "C:maj", "sus_for_triad"),
                ("C:maj7", "C:7", "seventh_swap"),
                ("C:dim7", "C:min6", "dim_near_miss"),
                ("C:maj", "C:min", "parallel"),
                ("C:maj", "F#:maj", "none"),
            ]
        } == {r.name for r in substitution.RELATIONS}


#: The quality pairs ``dim_near_miss`` wins over ``GRID``. Not the whole set it
#: can reach -- a wider spelling of the diminished family reaches more, and the
#: rule in ``_dim_near_miss`` is the definition. This is the regression pin.
DIM_NEAR_MISS_WINS = [
    ("dim", "min"),
    ("dim", "min7(b5)"),
    ("dim7", "min6"),
    ("hdim7", "7(*5,b5)"),
]


class TestDiminishedNearMiss:
    """A diminished chord a note from another chord on the same root."""

    @pytest.mark.parametrize(("first", "second"), DIM_NEAR_MISS_WINS)
    def test_it_holds_both_ways_round_at_its_stated_credit(self, first, second):
        for reference, estimate in ((first, second), (second, first)):
            assert graded(f"C:{reference}", f"C:{estimate}") == substitution.Grade(
                "dim_near_miss", 0.60
            )

    def test_it_wins_for_no_other_quality_pair_in_the_grid(self):
        # Pinned so that widening the rule shows up here rather than silently
        # paying credit somewhere nobody looked.
        chords = {label: parse(label) for label in GRID}
        won = {
            tuple(sorted((a.split(":", 1)[1], b.split(":", 1)[1])))
            for a in GRID
            for b in GRID
            if substitution.grade(chords[a], chords[b]).relation == "dim_near_miss"
        }
        assert won == {tuple(sorted(pair)) for pair in DIM_NEAR_MISS_WINS}

    def test_a_flat_five_added_is_not_a_flat_five_substituted(self):
        # Harte's "7(b5)" keeps the natural fifth, so it is a five-note chord
        # three notes from a hdim7. Omitting the fifth it replaces is what makes
        # the four-note chord this relation is about.
        assert sorted(parse("C:7(b5)").pitch_classes) == [0, 4, 6, 7, 10]
        assert sorted(parse("C:7(*5,b5)").pitch_classes) == [0, 4, 6, 10]
        assert relation("C:hdim7", "C:7(b5)") == "none"
        assert relation("C:hdim7", "C:7(*5,b5)") == "dim_near_miss"

    @pytest.mark.parametrize(
        ("estimate", "name"),
        # A note apart as well, but these two relations describe them better and
        # outrank or tie it. Pinned because the docstring says so.
        [("C:min7", "guide_tones"), ("C:dim7", "seventh_swap")],
    )
    def test_the_relations_that_claim_a_near_miss_first_keep_it(self, estimate, name):
        assert relation("C:hdim7", estimate) == name

    def test_major_for_minor_is_still_only_parallel(self):
        # Neither side is diminished, so the signature triad-model error keeps
        # the low credit it is meant to have.
        assert graded("C:maj", "C:min") == substitution.Grade("parallel", 0.40)


class TestAddingARelationOnlyEverRaisesCredit:
    """Swept over ``GRID``: a new row must not move an existing verdict.

    ``grade`` returns the highest-credit relation that holds, so a row below
    ``metrics.SUBSTITUTION_MIN_CREDIT`` cannot change the ``substitutions``
    vocabulary and cannot lower any pair's credit. That is the argument for
    adding one to a table whose numbers are already published, so it is checked
    rather than asserted in prose.
    """

    WITHOUT_NEW_ROW = tuple(r for r in substitution.RELATIONS if r.name != "dim_near_miss")

    def test_no_pair_loses_credit(self):
        chords = [parse(label) for label in GRID]
        lowered = [
            (a, b)
            for a in chords
            for b in chords
            if substitution.grade(a, b).credit
            < substitution.grade(a, b, self.WITHOUT_NEW_ROW).credit
        ]
        assert lowered == []

    def test_no_substitutions_verdict_moves(self):
        chords = [parse(label) for label in GRID]
        threshold = metrics.SUBSTITUTION_MIN_CREDIT
        moved = [
            (a, b)
            for a in chords
            for b in chords
            if (substitution.grade(a, b).credit >= threshold)
            != (substitution.grade(a, b, self.WITHOUT_NEW_ROW).credit >= threshold)
        ]
        assert moved == []

    def test_the_pairs_it_does_raise_were_all_graded_zero(self):
        chords = {label: parse(label) for label in GRID}
        raised = [
            substitution.grade(chords[a], chords[b], self.WITHOUT_NEW_ROW).credit
            for a in GRID
            for b in GRID
            if substitution.grade(chords[a], chords[b]).relation == "dim_near_miss"
        ]
        assert raised and set(raised) == {0.0}


class TestEveryRelationEarnsItsPlace:
    """Swept over ``GRID`` pairwise rather than over hand-picked pairs."""

    def test_every_relation_wins_for_some_pair(self):
        chords = [parse(label) for label in GRID]
        winners = {
            substitution.grade(first, second).relation for first in chords for second in chords
        }
        assert winners == {r.name for r in substitution.RELATIONS}

    def test_no_grade_depends_on_which_side_is_the_reference(self):
        chords = [parse(label) for label in GRID]
        asymmetric = [
            (a, b)
            for a in chords
            for b in chords
            if substitution.grade(a, b) != substitution.grade(b, a)
        ]
        assert asymmetric == []


class TestNoChordAndUnscoreable:
    def test_no_chord_against_no_chord_is_right(self):
        assert graded("N", "N") == substitution.Grade("same", 1.00)

    @pytest.mark.parametrize(("reference", "estimate"), [("N", "C:maj"), ("C:maj", "N")])
    def test_no_chord_against_a_chord_is_wrong(self, reference, estimate):
        assert credit(reference, estimate) == 0.0

    @pytest.mark.parametrize("reference", ["X", "C:nonsense", ""])
    def test_an_unscoreable_reference_is_skipped(self, reference):
        # The same asymmetry as metrics.compare: a gap in the annotation leaves
        # the denominator rather than counting against the model.
        assert graded(reference, "C:maj") is None

    def test_a_none_reference_is_skipped(self):
        assert substitution.grade(None, parse("C:maj")) is None

    @pytest.mark.parametrize("estimate", ["X", "C:nonsense", ""])
    def test_an_unreadable_estimate_is_simply_wrong(self, estimate):
        assert credit("C:maj", estimate) == 0.0

    def test_a_none_estimate_is_simply_wrong(self):
        result = substitution.grade(parse("C:maj"), None)
        assert result == substitution.Grade("none", 0.0)


class TestSubstitutionsVocabulary:
    def test_it_is_registered_with_a_description(self):
        assert metrics.VOCABULARIES["substitutions"]

    def test_it_is_not_a_default_vocabulary(self):
        # The default WCSR table stays exactly as published; -v opts in.
        assert "substitutions" not in metrics.DEFAULT_VOCABULARIES

    def test_it_is_pairwise_so_it_has_no_per_chord_projection(self):
        assert "substitutions" in metrics.PAIRWISE_VOCABULARIES
        with pytest.raises(ValueError, match="no per-chord projection"):
            metrics.project(parse("C:maj"), "substitutions")

    @pytest.mark.parametrize(
        ("reference", "estimate"),
        [
            ("C:maj", "C:maj"),
            ("C:7", "C:13"),
            ("C:7", "C:maj"),
            ("C:sus4(b7)", "C:7"),
            ("G:7", "Db:7"),
            ("B:hdim7", "G:7"),
        ],
    )
    def test_a_substitution_worth_the_threshold_passes(self, reference, estimate):
        assert credit(reference, estimate) >= metrics.SUBSTITUTION_MIN_CREDIT
        assert metrics.compare(parse(reference), parse(estimate), "substitutions") is True

    @pytest.mark.parametrize(
        ("reference", "estimate"),
        [("D:min7", "G:7"), ("C:maj6", "A:min7"), ("C:maj", "C:min"), ("C:maj", "F#:maj")],
    )
    def test_a_substitution_below_the_threshold_fails(self, reference, estimate):
        assert credit(reference, estimate) < metrics.SUBSTITUTION_MIN_CREDIT
        assert metrics.compare(parse(reference), parse(estimate), "substitutions") is False

    def test_the_threshold_sits_between_relative_and_the_rootless_dominants(self):
        credits = {r.name: r.credit for r in substitution.RELATIONS}
        assert credits["relative"] < metrics.SUBSTITUTION_MIN_CREDIT
        assert credits["ii_V_merge"] < metrics.SUBSTITUTION_MIN_CREDIT
        assert credits["tritone_sub"] >= metrics.SUBSTITUTION_MIN_CREDIT
        assert credits["dim_for_dominant"] >= metrics.SUBSTITUTION_MIN_CREDIT

    def test_an_unscoreable_reference_leaves_the_denominator(self):
        assert metrics.compare(parse("X"), parse("C:maj"), "substitutions") is None

    def test_it_scores_more_than_sevenths_on_a_flattened_seventh(self):
        # The headline case: a triad model against a jazz chart. `sevenths`
        # calls it a miss, `substitutions` calls it an extension.
        reference = timeline((0.0, 4.0, "C:7"))
        estimate = timeline((0.0, 4.0, "C:maj"))
        assert metrics.score(reference, estimate, "sevenths").wcsr == 0.0
        assert metrics.score(reference, estimate, "substitutions").wcsr == 1.0


class TestGradedScore:
    def test_a_perfect_estimate_scores_one(self):
        events = timeline((0.0, 2.0, "C:maj"), (2.0, 4.0, "A:min"))
        assert substitution.graded_score(events, events).score == 1.0

    def test_credit_is_weighted_by_duration_not_counted_by_segment(self):
        reference = timeline((0.0, 1.0, "C:maj"), (1.0, 11.0, "C:7"))
        # One short exact second, then ten seconds flattened to the triad.
        estimate = timeline((0.0, 1.0, "C:maj"), (1.0, 11.0, "C:maj"))
        result = substitution.graded_score(reference, estimate)
        assert result.compared == pytest.approx(11.0)
        assert result.credit == pytest.approx(1.0 + 10 * 0.95)
        assert result.score == pytest.approx((1.0 + 9.5) / 11.0)

    def test_by_relation_accounts_for_every_compared_second(self):
        reference = timeline((0.0, 2.0, "C:maj"), (2.0, 4.0, "C:7"), (4.0, 6.0, "C:maj"))
        estimate = timeline((0.0, 2.0, "C:maj"), (2.0, 4.0, "C:maj"), (4.0, 6.0, "F#:maj"))
        result = substitution.graded_score(reference, estimate)
        assert result.by_relation == {
            "same": pytest.approx(2.0),
            "extension": pytest.approx(2.0),
            "none": pytest.approx(2.0),
        }
        assert sum(result.by_relation.values()) == pytest.approx(result.compared)

    def test_an_unscoreable_reference_second_is_ignored_not_counted(self):
        reference = timeline((0.0, 2.0, "C:maj"), (2.0, 4.0, "X"))
        estimate = timeline((0.0, 4.0, "C:maj"))
        result = substitution.graded_score(reference, estimate)
        assert result.ignored == pytest.approx(2.0)
        assert result.compared == pytest.approx(2.0)
        assert result.score == 1.0

    def test_an_empty_reference_scores_nothing_rather_than_zero(self):
        result = substitution.graded_score([], timeline((0.0, 2.0, "C:maj")))
        assert result.compared == 0.0
        assert isnan(result.score)

    def test_an_all_unscoreable_reference_scores_nothing(self):
        result = substitution.graded_score(timeline((0.0, 2.0, "X")), timeline((0.0, 2.0, "C:maj")))
        assert isnan(result.score)

    def test_it_differs_from_the_substitutions_wcsr_in_both_directions(self):
        # The two answer different questions and neither bounds the other. A
        # passing substitution scores 1.0 as WCSR and its own credit here; a
        # failing one scores 0.0 as WCSR and keeps partial credit here.
        passing = timeline((0.0, 4.0, "C:7")), timeline((0.0, 4.0, "C:maj"))
        assert substitution.graded_score(*passing).score == pytest.approx(0.95)
        assert metrics.score(*passing, "substitutions").wcsr == 1.0

        failing = timeline((0.0, 4.0, "C:maj")), timeline((0.0, 4.0, "C:min"))
        assert substitution.graded_score(*failing).score == pytest.approx(0.40)
        assert metrics.score(*failing, "substitutions").wcsr == 0.0

    def test_estimate_time_past_the_reference_is_discarded(self):
        # The reference defines the evaluated span, as it does for WCSR.
        reference = timeline((0.0, 2.0, "C:maj"))
        estimate = timeline((0.0, 2.0, "C:maj"), (2.0, 20.0, "F#:maj"))
        result = substitution.graded_score(reference, estimate)
        assert result.compared == pytest.approx(2.0)
        assert result.score == 1.0

    def test_substitutions_can_score_below_majmin(self):
        # majmin drops a reference it cannot express; substitutions scores it,
        # so a wider denominator can put it below on the same timeline.
        reference = timeline((0.0, 2.0, "C:maj"), (2.0, 4.0, "C:sus4"))
        estimate = timeline((0.0, 4.0, "C:maj"))
        assert metrics.score(reference, estimate, "majmin").wcsr == 1.0
        assert metrics.score(reference, estimate, "substitutions").wcsr == pytest.approx(0.5)

    def test_a_custom_relation_table_is_respected(self):
        # grade() must read the table it is given, not the module default.
        only_same = (substitution.RELATIONS[0],)
        assert substitution.grade(parse("C:7"), parse("C:13"), only_same) == substitution.Grade(
            "none", 0.0
        )
        assert substitution.grade(parse("C:maj"), parse("C:maj"), only_same) == substitution.Grade(
            "same", 1.00
        )
