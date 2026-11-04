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


#: Pairs whose relation must not depend on which side is the reference. The
#: directional relations are tested separately, in both orders, by their own
#: classes -- they are symmetric in outcome but not in definition.
SYMMETRIC = [
    ("C:maj", "C:maj"),
    ("C:7", "C:13"),
    ("C:maj", "C:maj7"),
    ("C:maj7", "Db:min7"),
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
        # C:maj6 and A:min7 are the same four pitch classes. Grading that "same"
        # would make `relative` unreachable and call a sixth chord named as its
        # relative minor a perfect match.
        assert parse("C:maj6").pitch_classes == parse("A:min7").pitch_classes
        assert relation("C:maj6", "A:min7") == "relative"


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
        # maj7 against a dominant 7 on one root is a change of function -- tonic
        # against dominant -- so it is graded an error, not an extension.
        assert relation("C:maj7", "C:7") == "none"

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

    def test_a_suspension_against_a_plain_triad_is_not_it(self):
        # No seventh on the other side, so there is no dominant for the third to
        # be deferred from.
        assert relation("C:sus4", "C:maj") == "none"


class TestGuideTones:
    def test_the_third_and_seventh_in_the_same_roles(self):
        # C:maj7's third and seventh are E and B; Db:min7's are E and B too,
        # in the same roles, although the roots are a semitone apart.
        assert relation("C:maj7", "Db:min7") == "guide_tones"
        assert credit("C:maj7", "Db:min7") == 0.90

    def test_an_altered_chord_no_triad_names_still_matches_on_guide_tones(self):
        # C:maj(*5) has its fifth omitted, so triad_of cannot classify it and
        # `extension` does not apply -- but with no seventh there are no guide
        # tones either, and the pair is an error.
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
            # A dim7 is symmetric, so every inversion of it is the same rootless
            # dominant.
            ("D:dim7", "G:7"),
            ("E:hdim7", "C:7"),
        ],
    )
    def test_a_diminished_chord_is_its_dominant_without_the_root(self, reference, estimate):
        assert relation(reference, estimate) == "dim_for_dominant"
        assert credit(reference, estimate) == 0.80

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
        [("C:maj", "F#:maj"), ("C:maj", "E:min"), ("C:maj7", "C:7"), ("C:maj", "A:min")],
    )
    def test_a_wrong_chord_earns_nothing(self, reference, estimate):
        assert graded(reference, estimate) == substitution.Grade("none", 0.0)


class TestSymmetry:
    @pytest.mark.parametrize(("reference", "estimate"), SYMMETRIC)
    def test_the_grade_does_not_depend_on_the_side(self, reference, estimate):
        assert graded(reference, estimate) == graded(estimate, reference)


class TestHighestCreditWins:
    def test_a_pair_several_relations_hold_for_takes_the_best(self):
        # C:7 against C:maj is both an extension (0.95) and, were the table
        # walked in order, nothing else -- but D:min7 against G:9 is a ii-V and
        # shares three pitch classes, so the pick must not depend on table order.
        assert credit("D:min7", "G:9") == 0.70

    def test_every_relation_in_the_table_is_reachable(self):
        reached = {
            relation(reference, estimate)
            for reference, estimate in [
                ("C:maj", "C:maj"),
                ("C:7", "C:13"),
                ("C:sus4(b7)", "C:7"),
                ("C:maj7", "Db:min7"),
                ("G:7", "Db:7"),
                ("B:hdim7", "G:7"),
                ("D:min7", "G:7"),
                ("C:maj6", "A:min7"),
                ("C:maj", "C:min"),
                ("C:maj", "F#:maj"),
            ]
        }
        assert reached == {r.name for r in substitution.RELATIONS}


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

    def test_it_never_scores_below_the_wcsr_it_grades(self):
        # Partial credit is added on top of exact matches, never taken away, so
        # the graded score is a ceiling on the substitutions WCSR.
        reference = timeline((0.0, 2.0, "C:7"), (2.0, 4.0, "D:min7"), (4.0, 6.0, "C:maj"))
        estimate = timeline((0.0, 2.0, "C:maj"), (2.0, 4.0, "G:7"), (4.0, 6.0, "C:maj"))
        assert (
            substitution.graded_score(reference, estimate).score
            >= metrics.score(reference, estimate, "substitutions").wcsr
        )
