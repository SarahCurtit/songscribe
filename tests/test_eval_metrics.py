from __future__ import annotations

from itertools import pairwise
from math import isnan

import pytest

from songscribe.eval import metrics
from songscribe.eval.harte import parse
from songscribe.types import ChordEvent


def timeline(*spec: tuple[float, float, str]) -> list[ChordEvent]:
    """A chord timeline from ``(start, end, label)`` triples."""
    return [ChordEvent(start, end, label) for start, end, label in spec]


def wcsr(reference, estimate, vocabulary="majmin") -> float:
    return metrics.score(reference, estimate, vocabulary).wcsr


class TestTriadProjection:
    @pytest.mark.parametrize(
        ("label", "expected"),
        [
            ("C:maj", "maj"),
            ("C:min", "min"),
            ("C:dim", "dim"),
            ("C:aug", "aug"),
            ("C:sus2", "sus2"),
            ("C:sus4", "sus4"),
        ],
    )
    def test_plain_triads(self, label, expected):
        assert metrics.triad_of(parse(label)) == expected

    @pytest.mark.parametrize("label", ["C:maj7", "C:7", "C:maj6", "C:9", "C:13", "C:maj(9)"])
    def test_extensions_reduce_to_the_underlying_major_triad(self, label):
        assert metrics.triad_of(parse(label)) == "maj"

    @pytest.mark.parametrize("label", ["C:min7", "C:min6", "C:minmaj7", "C:min9"])
    def test_extensions_reduce_to_the_underlying_minor_triad(self, label):
        assert metrics.triad_of(parse(label)) == "min"

    def test_half_diminished_reduces_to_diminished(self):
        assert metrics.triad_of(parse("C:hdim7")) == "dim"

    def test_a_chord_with_no_fifth_has_no_triad(self):
        assert metrics.triad_of(parse("C:maj(*5)")) is None

    def test_a_bare_power_chord_has_no_triad(self):
        assert metrics.triad_of(parse("C:(5)")) is None


class TestSeventhProjection:
    @pytest.mark.parametrize(
        ("label", "expected"),
        [
            ("C:maj", "maj"),
            ("C:min", "min"),
            ("C:maj7", "maj7"),
            ("C:min7", "min7"),
            ("C:7", "7"),
            ("C:maj6", "maj"),
            ("C:min6", "min"),
        ],
    )
    def test_the_mirex_sevenths_vocabulary(self, label, expected):
        assert metrics.seventh_of(parse(label)) == expected

    @pytest.mark.parametrize("label", ["C:minmaj7", "C:dim", "C:dim7", "C:hdim7", "C:sus4"])
    def test_qualities_outside_the_vocabulary(self, label):
        assert metrics.seventh_of(parse(label)) is None


class TestProject:
    def test_root_vocabulary_ignores_quality(self):
        assert metrics.project(parse("C:min7"), "root") == metrics.project(parse("C:maj"), "root")

    def test_root_vocabulary_canonicalises_enharmonics(self):
        assert metrics.project(parse("Db:maj"), "root") == metrics.project(parse("C#:min"), "root")

    def test_no_chord_projects_into_every_vocabulary(self):
        for vocabulary in ("root", "majmin", "triads", "sevenths"):
            assert metrics.project(parse("N"), vocabulary) == "N"

    def test_unknown_chord_projects_nowhere(self):
        assert metrics.project(parse("X"), "majmin") is None

    def test_unparseable_chord_projects_nowhere(self):
        assert metrics.project(None, "majmin") is None

    def test_majmin_rejects_a_suspension_that_triads_accepts(self):
        assert metrics.project(parse("C:sus4"), "majmin") is None
        assert metrics.project(parse("C:sus4"), "triads") == "C:sus4"

    def test_unknown_vocabulary_raises(self):
        with pytest.raises(ValueError, match="unknown vocabulary"):
            metrics.project(parse("C"), "nonsense")

    def test_mirex_has_no_per_chord_projection(self):
        # It is a pairwise overlap test, so there is nothing to project to.
        with pytest.raises(ValueError, match="no per-chord projection"):
            metrics.project(parse("C"), "mirex")


class TestCompare:
    def test_identical_chords_match(self):
        assert metrics.compare(parse("C:maj"), parse("C"), "majmin") is True

    def test_wrong_quality_does_not_match_in_majmin(self):
        assert metrics.compare(parse("C:maj"), parse("Cm"), "majmin") is False

    def test_wrong_quality_matches_in_root(self):
        assert metrics.compare(parse("C:maj"), parse("Cm"), "root") is True

    def test_unscoreable_reference_is_skipped(self):
        assert metrics.compare(parse("X"), parse("C"), "majmin") is None
        assert metrics.compare(None, parse("C"), "majmin") is None
        assert metrics.compare(parse("C:sus4"), parse("C"), "majmin") is None

    def test_unparseable_estimate_is_wrong_not_skipped(self):
        # A model is not excused by emitting something unreadable.
        assert metrics.compare(parse("C:maj"), None, "majmin") is False

    def test_estimate_outside_the_vocabulary_is_wrong(self):
        assert metrics.compare(parse("C:maj"), parse("C:sus4"), "majmin") is False

    def test_no_chord_matches_no_chord(self):
        assert metrics.compare(parse("N"), parse("N"), "majmin") is True
        assert metrics.compare(parse("N"), parse("C"), "majmin") is False


class TestMirexComparison:
    def test_three_shared_pitch_classes_is_a_match(self):
        # C:maj7 is C E G B; C:maj is C E G. Three in common.
        assert metrics.compare(parse("C:maj7"), parse("C:maj"), "mirex") is True

    def test_two_shared_pitch_classes_is_not(self):
        # C major and A minor share C and E only.
        assert metrics.compare(parse("C:maj"), parse("A:min"), "mirex") is False

    def test_a_relative_minor_third_apart_is_not_a_match(self):
        assert metrics.compare(parse("C:maj"), parse("E:min"), "mirex") is False

    def test_no_chord_against_a_chord_is_wrong_in_both_directions(self):
        assert metrics.compare(parse("N"), parse("C"), "mirex") is False
        assert metrics.compare(parse("C"), parse("N"), "mirex") is False

    def test_no_chord_against_no_chord_is_right(self):
        assert metrics.compare(parse("N"), parse("N"), "mirex") is True


class TestFillGaps:
    def test_an_uncovered_middle_becomes_no_chord(self):
        filled = metrics.fill_gaps(timeline((0.0, 1.0, "C"), (3.0, 4.0, "G")), 0.0, 4.0)
        assert [(e.start, e.end, e.label) for e in filled] == [
            (0.0, 1.0, "C"),
            (1.0, 3.0, "N"),
            (3.0, 4.0, "G"),
        ]

    def test_leading_and_trailing_time_becomes_no_chord(self):
        filled = metrics.fill_gaps(timeline((1.0, 2.0, "C")), 0.0, 3.0)
        assert [e.label for e in filled] == ["N", "C", "N"]

    def test_events_are_clipped_to_the_window(self):
        filled = metrics.fill_gaps(timeline((0.0, 10.0, "C")), 2.0, 4.0)
        assert [(e.start, e.end) for e in filled] == [(2.0, 4.0)]

    def test_an_empty_timeline_becomes_one_no_chord_span(self):
        filled = metrics.fill_gaps([], 0.0, 5.0)
        assert [(e.start, e.end, e.label) for e in filled] == [(0.0, 5.0, "N")]

    def test_result_is_gapless(self):
        filled = metrics.fill_gaps(timeline((0.5, 1.0, "C"), (2.0, 2.5, "G")), 0.0, 3.0)
        assert filled[0].start == 0.0
        assert filled[-1].end == 3.0
        for before, after in pairwise(filled):
            assert before.end == pytest.approx(after.start)


class TestMergeTimelines:
    def test_boundaries_from_both_sides_appear(self):
        merged = metrics.merge_timelines(
            timeline((0.0, 2.0, "C"), (2.0, 4.0, "G")),
            timeline((0.0, 1.0, "C"), (1.0, 4.0, "G")),
        )
        assert [(low, high) for low, high, _, _ in merged] == [
            (0.0, 1.0),
            (1.0, 2.0),
            (2.0, 4.0),
        ]

    def test_each_sub_segment_carries_both_labels(self):
        merged = metrics.merge_timelines(
            timeline((0.0, 2.0, "C")),
            timeline((0.0, 1.0, "C"), (1.0, 2.0, "F")),
        )
        assert [(r, e) for _, _, r, e in merged] == [("C", "C"), ("C", "F")]

    def test_the_reference_defines_the_evaluated_span(self):
        merged = metrics.merge_timelines(
            timeline((1.0, 2.0, "C")),
            timeline((0.0, 10.0, "C")),
        )
        assert [(low, high) for low, high, _, _ in merged] == [(1.0, 2.0)]

    def test_an_empty_reference_scores_nothing(self):
        assert metrics.merge_timelines([], timeline((0.0, 4.0, "C"))) == []

    def test_a_zero_length_reference_scores_nothing(self):
        assert metrics.merge_timelines(timeline((1.0, 1.0, "C")), timeline((0.0, 4.0, "C"))) == []


class TestWcsr:
    def test_a_perfect_estimate_scores_one(self):
        reference = timeline((0.0, 2.0, "C:maj"), (2.0, 4.0, "G:maj"))
        assert wcsr(reference, timeline((0.0, 2.0, "C"), (2.0, 4.0, "G"))) == 1.0

    def test_half_the_time_wrong_scores_a_half(self):
        reference = timeline((0.0, 2.0, "C:maj"), (2.0, 4.0, "G:maj"))
        assert wcsr(reference, timeline((0.0, 2.0, "C"), (2.0, 4.0, "F"))) == 0.5

    def test_it_weights_by_duration_not_by_segment_count(self):
        # Three short right chords and one long wrong one: counting segments
        # would say 75%, counting time says 25%.
        reference = timeline(
            (0.0, 1.0, "C:maj"), (1.0, 2.0, "G:maj"), (2.0, 3.0, "A:min"), (3.0, 12.0, "F:maj")
        )
        estimate = timeline((0.0, 1.0, "C"), (1.0, 2.0, "G"), (2.0, 3.0, "Am"), (3.0, 12.0, "Bb"))
        assert wcsr(reference, estimate) == pytest.approx(3.0 / 12.0)

    def test_a_misplaced_boundary_costs_only_the_time_it_misplaces(self):
        reference = timeline((0.0, 2.0, "C:maj"), (2.0, 4.0, "G:maj"))
        estimate = timeline((0.0, 1.0, "C"), (1.0, 4.0, "G"))
        assert wcsr(reference, estimate) == pytest.approx(0.75)

    def test_time_the_estimate_never_reached_counts_against_it(self):
        # The pipeline strips N spans, so an estimate arrives with holes. A
        # hole over an annotated chord is a miss, not an exemption.
        reference = timeline((0.0, 4.0, "C:maj"))
        assert wcsr(reference, timeline((0.0, 2.0, "C"))) == pytest.approx(0.5)

    def test_an_empty_estimate_scores_zero(self):
        assert wcsr(timeline((0.0, 4.0, "C:maj")), []) == 0.0

    def test_an_empty_reference_scores_nan(self):
        assert isnan(wcsr([], timeline((0.0, 4.0, "C"))))

    def test_silence_annotated_and_predicted_scores_one(self):
        reference = timeline((0.0, 2.0, "N"), (2.0, 4.0, "C:maj"))
        # The estimate has no N span at all: the gap means the same thing.
        assert wcsr(reference, timeline((2.0, 4.0, "C"))) == 1.0


class TestExcludedReferenceTime:
    def test_unknown_reference_segments_leave_the_denominator(self):
        reference = timeline((0.0, 2.0, "X"), (2.0, 4.0, "C:maj"))
        score = metrics.score(reference, timeline((0.0, 4.0, "C")), "majmin")
        assert score.ignored == pytest.approx(2.0)
        assert score.compared == pytest.approx(2.0)
        assert score.wcsr == 1.0

    def test_a_reference_quality_outside_the_vocabulary_is_excluded(self):
        reference = timeline((0.0, 2.0, "C:sus4"), (2.0, 4.0, "C:maj"))
        score = metrics.score(reference, timeline((0.0, 4.0, "C")), "majmin")
        assert score.ignored == pytest.approx(2.0)
        assert score.wcsr == 1.0

    def test_the_same_segment_is_scored_under_a_wider_vocabulary(self):
        reference = timeline((0.0, 2.0, "C:sus4"), (2.0, 4.0, "C:maj"))
        score = metrics.score(reference, timeline((0.0, 4.0, "C")), "triads")
        assert score.ignored == 0.0
        assert score.wcsr == pytest.approx(0.5)

    def test_an_unparseable_reference_label_is_excluded(self):
        reference = timeline((0.0, 2.0, "C:gibberish"), (2.0, 4.0, "C:maj"))
        score = metrics.score(reference, timeline((0.0, 4.0, "C")), "majmin")
        assert score.ignored == pytest.approx(2.0)

    def test_a_reference_of_nothing_but_excluded_chords_scores_nan(self):
        score = metrics.score(timeline((0.0, 4.0, "X")), timeline((0.0, 4.0, "C")), "majmin")
        assert isnan(score.wcsr)


class TestVocabularyCeiling:
    """The headline reason several vocabularies are reported at once.

    A triad-only backend can score well on ``majmin`` and badly on
    ``sevenths`` without being any worse at hearing chords -- the gap is the
    vocabulary, not the model (see ``CLAUDE.md``, "where accuracy is actually
    lost"). The pair of numbers is what makes that visible.
    """

    def test_a_triad_estimate_of_a_seventh_is_right_in_majmin(self):
        reference = timeline((0.0, 4.0, "C:maj7"))
        assert wcsr(reference, timeline((0.0, 4.0, "C")), "majmin") == 1.0

    def test_and_wrong_in_sevenths(self):
        reference = timeline((0.0, 4.0, "C:maj7"))
        assert wcsr(reference, timeline((0.0, 4.0, "C")), "sevenths") == 0.0

    def test_and_right_again_under_mirex_overlap(self):
        reference = timeline((0.0, 4.0, "C:maj7"))
        assert wcsr(reference, timeline((0.0, 4.0, "C")), "mirex") == 1.0

    def test_a_dominant_seventh_flattened_to_its_triad(self):
        reference = timeline((0.0, 4.0, "G:7"))
        assert wcsr(reference, timeline((0.0, 4.0, "G")), "majmin") == 1.0
        assert wcsr(reference, timeline((0.0, 4.0, "G")), "sevenths") == 0.0


class TestScoreAll:
    def test_it_returns_one_score_per_vocabulary(self):
        reference = timeline((0.0, 4.0, "C:min7"))
        scores = metrics.score_all(reference, timeline((0.0, 4.0, "C")))
        assert set(scores) == set(metrics.DEFAULT_VOCABULARIES)
        assert scores["root"].wcsr == 1.0
        assert scores["majmin"].wcsr == 0.0

    def test_unknown_vocabulary_raises(self):
        with pytest.raises(ValueError, match="unknown vocabulary"):
            metrics.score(timeline((0.0, 1.0, "C")), timeline((0.0, 1.0, "C")), "nope")


class TestSegmentation:
    def test_identical_boundaries_score_one(self):
        reference = timeline((0.0, 2.0, "C"), (2.0, 4.0, "G"))
        result = metrics.segmentation(reference, reference)
        assert result.over == pytest.approx(1.0)
        assert result.under == pytest.approx(1.0)
        assert result.score == pytest.approx(1.0)

    def test_labels_are_ignored(self):
        reference = timeline((0.0, 2.0, "C"), (2.0, 4.0, "G"))
        estimate = timeline((0.0, 2.0, "Eb"), (2.0, 4.0, "F#m"))
        assert metrics.segmentation(reference, estimate).score == pytest.approx(1.0)

    def test_an_estimate_that_splits_every_chord_is_over_segmented(self):
        reference = timeline((0.0, 4.0, "C"))
        estimate = timeline((0.0, 2.0, "C"), (2.0, 4.0, "G"))
        result = metrics.segmentation(reference, estimate)
        assert result.over == pytest.approx(0.5)
        assert result.under == pytest.approx(1.0)
        assert result.score == pytest.approx(0.5)

    def test_an_estimate_that_merges_every_chord_is_under_segmented(self):
        reference = timeline((0.0, 2.0, "C"), (2.0, 4.0, "G"))
        estimate = timeline((0.0, 4.0, "C"))
        result = metrics.segmentation(reference, estimate)
        assert result.over == pytest.approx(1.0)
        assert result.under == pytest.approx(0.5)
        assert result.score == pytest.approx(0.5)

    def test_an_empty_reference_scores_nan(self):
        assert isnan(metrics.segmentation([], timeline((0.0, 4.0, "C"))).score)


class TestVocabularyRegistry:
    def test_every_default_vocabulary_is_documented(self):
        for vocabulary in metrics.DEFAULT_VOCABULARIES:
            assert vocabulary in metrics.VOCABULARIES
            assert metrics.VOCABULARIES[vocabulary]

    def test_every_registered_vocabulary_can_actually_score(self):
        reference = timeline((0.0, 1.0, "C:maj"))
        estimate = timeline((0.0, 1.0, "C"))
        for vocabulary in metrics.VOCABULARIES:
            assert metrics.score(reference, estimate, vocabulary).wcsr == 1.0
