from __future__ import annotations

import pytest

from songscribe.eval import consensus, metrics, substitution
from songscribe.eval.harte import ALTERNATIVE_SEPARATOR, alternatives, parse, transpose_label
from songscribe.types import ChordEvent


def timeline(*spec: tuple[float, float, str]) -> list[ChordEvent]:
    return [ChordEvent(start, end, label) for start, end, label in spec]


def one_bar(label: str) -> list[ChordEvent]:
    return timeline((0.0, 4.0, label))


#: The bars of Beautiful Love where TheFeelBook's chart prints a small alternate
#: chord above the big one, as ``chart2chords.py`` recorded them: bars 4 and 20,
#: 14 and 15. Real source disagreement rather than invented examples, which is
#: what makes them worth pinning -- see the Stage 4 ticket.
CHART_ALTERNATES = [
    ("D:min", "D:7"),
    ("B:hdim7(9)", "G:7(#11)"),
    ("E:hdim7", "E:7"),
]


class TestSplittingALabel:
    def test_one_chord_is_a_one_item_tuple(self):
        # So no caller has to ask whether a label carries alternatives first.
        assert alternatives("D:min") == ("D:min",)

    def test_several_chords_come_back_in_order(self):
        assert alternatives("D:min|D:7|F:maj6") == ("D:min", "D:7", "F:maj6")

    def test_space_around_the_separator_is_ignored(self):
        # lab.parse joins a label's whitespace-separated fields with a space, so
        # a hand-edited "D:min | D:7" arrives spaced.
        assert alternatives("D:min | D:7") == ("D:min", "D:7")

    def test_the_separator_is_not_harte_syntax(self):
        # The whole reason the set can ride inside the label.
        assert parse(f"C:maj{ALTERNATIVE_SEPARATOR}") is not None


class TestParsingKeepsEveryMetricStrict:
    def test_parse_returns_the_primary_chord_alone(self):
        assert parse("D:min|D:7") == parse("D:min")

    def test_parse_alternatives_returns_all_of_them(self):
        assert consensus.parse_alternatives("D:min|D:7") == (parse("D:min"), parse("D:7"))

    def test_an_unreadable_alternative_is_none_rather_than_fatal(self):
        # A typo in the small print costs only itself: by the verdict rule it can
        # neither rescue a wrong estimate nor spoil a right one.
        assert consensus.parse_alternatives("D:min|nonsense") == (parse("D:min"), None)

    def test_an_empty_primary_does_not_parse(self):
        assert parse("|D:7") is None

    def test_every_strict_metric_reads_the_primary_chord(self):
        # The point of parsing to the primary: a reference carrying alternatives
        # scores in the existing harness exactly as its big chords alone would.
        carried = one_bar("D:min|D:7")
        plain = one_bar("D:min")
        estimate = one_bar("D:7")
        for vocabulary in metrics.DEFAULT_VOCABULARIES:
            assert metrics.score(carried, estimate, vocabulary).wcsr == pytest.approx(
                metrics.score(plain, estimate, vocabulary).wcsr
            )


class TestTransposing:
    def test_every_alternative_moves(self):
        # Moving only the primary would leave the rest naming chords in the old key.
        assert transpose_label("D:min|D:7", 2) == "E:min|E:7"

    def test_three_alternatives_all_move(self):
        assert transpose_label("C:maj|A:min7|Eb:7", 1) == "C#:maj|A#:min7|E:7"

    def test_an_unreadable_alternative_is_left_as_written(self):
        assert transpose_label("C:maj|nonsense", 2) == "D:maj|nonsense"


class TestTheVerdictLattice:
    @pytest.mark.parametrize(
        ("verdicts", "expected"),
        [
            ([True], True),
            ([False], False),
            ([None], None),
            ([False, True], True),
            ([None, False], False),
            ([None, True], True),
            ([None, None], None),
            ([], None),
        ],
    )
    def test_any_source_matching_makes_the_segment_correct(self, verdicts, expected):
        assert consensus.verdict(verdicts) is expected

    def test_a_single_chord_scores_exactly_as_metrics_compare_did(self):
        chord = parse("C:maj")
        for vocabulary in metrics.VOCABULARIES:
            assert consensus.compare("C:maj", chord, vocabulary) is metrics.compare(
                chord, chord, vocabulary
            )


class TestConsensusScore:
    def test_a_reference_printing_one_chord_scores_identically(self):
        # The invariant that keeps every existing corpus's numbers where they are.
        reference = one_bar("D:min")
        estimate = one_bar("D:7")
        for vocabulary in metrics.VOCABULARIES:
            strict = metrics.score(reference, estimate, vocabulary)
            loose = consensus.score(reference, estimate, vocabulary)
            assert (loose.correct, loose.compared, loose.ignored) == (
                strict.correct,
                strict.compared,
                strict.ignored,
            )

    def test_matching_the_alternative_is_correct_where_strict_is_wrong(self):
        reference = one_bar("D:min|D:7")
        estimate = one_bar("D:7")
        assert metrics.score(reference, estimate, "sevenths").wcsr == 0.0
        assert consensus.score(reference, estimate, "sevenths").wcsr == 1.0

    def test_matching_neither_is_still_wrong(self):
        reference = one_bar("D:min|D:7")
        estimate = one_bar("D:maj")
        assert consensus.score(reference, estimate, "sevenths").wcsr == 0.0

    def test_it_is_duration_weighted_like_wcsr(self):
        reference = timeline((0.0, 3.0, "D:min|D:7"), (3.0, 4.0, "G:maj"))
        estimate = timeline((0.0, 3.0, "D:7"), (3.0, 4.0, "A:maj"))
        score = consensus.score(reference, estimate, "majmin")
        assert score.correct == pytest.approx(3.0)
        assert score.compared == pytest.approx(4.0)
        assert score.wcsr == pytest.approx(0.75)

    def test_a_segment_no_source_can_score_is_skipped(self):
        # X on every alternative is a gap in the annotation, so it leaves the
        # denominator rather than counting against the model.
        reference = one_bar("X|X")
        score = consensus.score(reference, one_bar("C:maj"), "majmin")
        assert (score.compared, score.ignored) == (0.0, pytest.approx(4.0))

    def test_a_readable_alternative_rescues_an_unscoreable_primary(self):
        reference = one_bar("X|C:maj")
        score = consensus.score(reference, one_bar("C:maj"), "majmin")
        assert (score.correct, score.compared) == (pytest.approx(4.0), pytest.approx(4.0))

    def test_an_alternative_can_lower_the_score_by_widening_the_denominator(self):
        # Measured on the real backing track and documented in EVALUATION.md:
        # consensus moves a score both ways. `sevenths` cannot express a
        # half-diminished chord, so strictly that bar is excluded; its
        # alternative is a dominant, which `sevenths` can express, so the bar
        # joins the denominator and a wrong estimate now counts against it.
        reference = timeline((0.0, 2.0, "C:maj"), (2.0, 4.0, "E:hdim7|E:7"))
        estimate = timeline((0.0, 2.0, "C:maj"), (2.0, 4.0, "F:maj"))

        strict = metrics.score(reference, estimate, "sevenths")
        loose = consensus.score(reference, estimate, "sevenths")
        assert (strict.compared, strict.ignored) == (pytest.approx(2.0), pytest.approx(2.0))
        assert (loose.compared, loose.ignored) == (pytest.approx(4.0), 0.0)
        assert strict.wcsr == pytest.approx(1.0)
        assert loose.wcsr == pytest.approx(0.5)

    def test_a_vocabulary_that_excludes_nothing_can_only_rise(self):
        # root and mirex keep every segment, so their denominator cannot grow
        # and consensus is monotone for them.
        reference = timeline((0.0, 2.0, "C:maj"), (2.0, 4.0, "E:hdim7|E:7"))
        estimate = timeline((0.0, 2.0, "C:maj"), (2.0, 4.0, "E:7"))
        for vocabulary in ("root", "mirex"):
            strict = metrics.score(reference, estimate, vocabulary)
            loose = consensus.score(reference, estimate, vocabulary)
            assert loose.compared == pytest.approx(strict.compared)
            assert loose.correct >= strict.correct

    def test_an_unknown_vocabulary_raises(self):
        with pytest.raises(ValueError, match="unknown vocabulary"):
            consensus.score(one_bar("C:maj"), one_bar("C:maj"), "nope")

    def test_score_all_covers_several_vocabularies(self):
        scores = consensus.score_all(one_bar("D:min|D:7"), one_bar("D:7"))
        assert set(scores) == set(metrics.DEFAULT_VOCABULARIES)
        assert scores["sevenths"].wcsr == 1.0


class TestConsensusGradedScore:
    def test_it_takes_the_best_alternative_not_the_first(self):
        # D:maj is parallel to D:min (0.40) and an extension short of D:7 (0.95).
        reference = one_bar("D:min|D:7")
        assert substitution.graded_score(reference, one_bar("D:maj")).score == pytest.approx(0.40)
        assert consensus.graded_score(reference, one_bar("D:maj")).score == pytest.approx(0.95)

    def test_an_exact_match_with_any_source_is_full_credit(self):
        graded = consensus.graded_score(one_bar("D:min|D:7"), one_bar("D:7"))
        assert graded.score == pytest.approx(1.0)
        assert graded.by_relation == {"same": pytest.approx(4.0)}

    def test_by_relation_accounts_for_every_compared_second(self):
        reference = timeline((0.0, 2.0, "D:min|D:7"), (2.0, 4.0, "G:maj"))
        graded = consensus.graded_score(reference, timeline((0.0, 2.0, "D:7"), (2.0, 4.0, "G:min")))
        assert sum(graded.by_relation.values()) == pytest.approx(graded.compared)

    def test_a_plain_reference_grades_exactly_as_the_strict_table_does(self):
        reference = timeline((0.0, 2.0, "D:min"), (2.0, 4.0, "G:maj"))
        estimate = timeline((0.0, 2.0, "D:7"), (2.0, 4.0, "G:min"))
        strict = substitution.graded_score(reference, estimate)
        loose = consensus.graded_score(reference, estimate)
        assert loose == strict

    def test_it_honours_a_callers_own_relation_table(self):
        only_same = (substitution.RELATIONS[0], substitution.RELATIONS[-1])
        graded = consensus.graded_score(one_bar("D:min|D:7"), one_bar("D:maj"), only_same)
        assert graded.score == pytest.approx(0.0)


class TestHasAlternatives:
    def test_false_for_an_ordinary_reference(self):
        assert not consensus.has_alternatives(timeline((0.0, 1.0, "C:maj"), (1.0, 2.0, "G:7")))

    def test_true_when_any_one_segment_carries_them(self):
        assert consensus.has_alternatives(timeline((0.0, 1.0, "C:maj"), (1.0, 2.0, "D:min|D:7")))

    def test_false_for_an_empty_timeline(self):
        assert not consensus.has_alternatives([])


class TestDisagreements:
    def test_an_ordinary_reference_has_nothing_to_report(self):
        found = consensus.disagreements(timeline((0.0, 4.0, "C:maj")))
        assert found.pairs == ()
        assert found.duration == 0.0
        assert found.share == 0.0

    def test_each_pair_carries_its_duration_and_segment_count(self):
        found = consensus.disagreements(
            timeline((0.0, 2.0, "D:min|D:7"), (2.0, 3.0, "D:min|D:7"), (3.0, 4.0, "G:maj"))
        )
        assert len(found.pairs) == 1
        pair = found.pairs[0]
        assert (pair.primary, pair.alternative) == ("D:min", "D:7")
        assert pair.duration == pytest.approx(3.0)
        assert pair.segments == 2
        assert pair.label == "D:min|D:7"

    def test_the_share_is_of_all_annotated_time(self):
        found = consensus.disagreements(timeline((0.0, 1.0, "D:min|D:7"), (1.0, 4.0, "G:maj")))
        assert found.duration == pytest.approx(1.0)
        assert found.annotated == pytest.approx(4.0)
        assert found.share == pytest.approx(0.25)

    def test_a_caller_may_supply_the_total_it_is_a_share_of(self):
        # What survey does: it holds only the contested segments but knows the
        # corpus total already.
        found = consensus.disagreements(timeline((0.0, 1.0, "D:min|D:7")), annotated=10.0)
        assert found.share == pytest.approx(0.10)

    def test_a_bar_printing_three_chords_contributes_two_pairs(self):
        found = consensus.disagreements(timeline((0.0, 4.0, "C:maj|C:maj7|C:7")))
        assert {pair.label for pair in found.pairs} == {"C:maj|C:maj7", "C:maj|C:7"}
        # Counted once as contested time, not once per pair.
        assert found.duration == pytest.approx(4.0)

    def test_pairs_come_back_largest_first(self):
        found = consensus.disagreements(
            timeline((0.0, 1.0, "C:maj|C:min"), (1.0, 4.0, "D:min|D:7"))
        )
        assert [pair.label for pair in found.pairs] == ["D:min|D:7", "C:maj|C:min"]

    def test_each_pair_is_graded_by_the_relation_table(self):
        found = consensus.disagreements(timeline((0.0, 4.0, "G:7|Db:7")))
        assert (found.pairs[0].relation, found.pairs[0].credit) == ("tritone_sub", 0.80)

    def test_by_relation_sums_to_the_contested_time(self):
        found = consensus.disagreements(timeline((0.0, 2.0, "G:7|Db:7"), (2.0, 4.0, "C:maj|C:min")))
        assert sum(found.by_relation.values()) == pytest.approx(found.duration)

    def test_a_pair_no_relation_explains_is_listed_for_reading(self):
        found = consensus.disagreements(timeline((0.0, 4.0, "C:maj|F#:maj")))
        assert [pair.label for pair in found.unexplained] == ["C:maj|F#:maj"]

    def test_an_explained_pair_is_not_listed(self):
        found = consensus.disagreements(timeline((0.0, 4.0, "G:7|Db:7")))
        assert found.unexplained == ()

    def test_an_unreadable_primary_is_reported_rather_than_dropped(self):
        # It is still a disagreement worth printing; nothing in the table can
        # explain it, which is exactly what `unexplained` is for.
        found = consensus.disagreements(timeline((0.0, 4.0, "X|C:maj")))
        assert [pair.label for pair in found.unexplained] == ["X|C:maj"]
        assert found.pairs[0].credit == 0.0


class TestTheRealChartAlternates:
    """The bars where the only lead sheet in the project prints two chords.

    Pinned with their relations so that a change to the table shows what it does
    to real source disagreement, not only to invented pairs.
    """

    @pytest.mark.parametrize(("primary", "alternative"), CHART_ALTERNATES)
    def test_both_chords_of_each_bar_parse(self, primary, alternative):
        assert parse(primary) is not None
        assert parse(alternative) is not None

    @pytest.mark.parametrize(
        ("primary", "alternative", "name"),
        [
            ("D:min", "D:7", "parallel"),
            ("B:hdim7(9)", "G:7(#11)", "dim_for_dominant"),
            # The one the table does not explain: a half-diminished chord and the
            # dominant on the same root differ in both the third and the fifth.
            # Whether that deserves a relation wants more than one tune's charts.
            ("E:hdim7", "E:7", "none"),
        ],
    )
    def test_the_relation_each_pair_reaches(self, primary, alternative, name):
        graded = substitution.grade(parse(primary), parse(alternative))
        assert graded.relation == name

    def test_one_of_the_three_reaches_no_relation(self):
        found = consensus.disagreements(
            [
                ChordEvent(start=float(i), end=float(i) + 1.0, label=f"{a}|{b}")
                for i, (a, b) in enumerate(CHART_ALTERNATES)
            ]
        )
        assert len(found.pairs) == 3
        assert [pair.label for pair in found.unexplained] == ["E:hdim7|E:7"]
