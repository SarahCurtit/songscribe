from __future__ import annotations

from itertools import pairwise
from math import isnan

import pytest

from songscribe.eval import viz
from songscribe.types import ChordEvent


def timeline(*spec: tuple[float, float, str]) -> list[ChordEvent]:
    return [ChordEvent(start, end, label) for start, end, label in spec]


@pytest.fixture
def turnaround() -> list[ChordEvent]:
    return timeline(
        (0.0, 2.0, "C:maj"),
        (2.0, 4.0, "G:maj"),
        (4.0, 6.0, "A:min"),
        (6.0, 8.0, "F:maj"),
    )


class TestChordSpans:
    def test_spans_carry_times_labels_and_a_root_colour(self, turnaround):
        spans = viz.chord_spans(turnaround)
        assert [s.label for s in spans] == ["C:maj", "G:maj", "A:min", "F:maj"]
        assert [s.colour for s in spans] == [0, 7, 9, 5]  # C, G, A, F
        assert spans[0].duration == 2.0

    def test_the_same_root_gets_the_same_colour_whatever_the_quality(self):
        spans = viz.chord_spans(timeline((0.0, 1.0, "C:maj"), (1.0, 2.0, "C:min7")))
        assert spans[0].colour == spans[1].colour

    def test_enharmonic_roots_share_a_colour(self):
        spans = viz.chord_spans(timeline((0.0, 1.0, "Db:maj"), (1.0, 2.0, "C#:maj")))
        assert spans[0].colour == spans[1].colour

    def test_rootless_labels_get_the_no_colour_marker(self):
        spans = viz.chord_spans(timeline((0.0, 1.0, "N"), (1.0, 2.0, "X"), (2.0, 3.0, "???")))
        assert [s.colour for s in spans] == [viz.NO_COLOUR] * 3

    def test_a_vocabulary_projects_the_labels(self, turnaround):
        spans = viz.chord_spans(timeline((0.0, 1.0, "C:maj7")), vocabulary="majmin")
        assert spans[0].label == "C:maj"

    def test_chart_mode_collapses_what_a_chart_would_print(self):
        events = timeline((0.0, 1.0, "C:maj7"), (1.0, 2.0, "C:maj"), (2.0, 4.0, "G:maj"))
        spans = viz.chord_spans(events, vocabulary="majmin", chart=True)
        assert [s.label for s in spans] == ["C:maj", "G:maj"]
        # And the collapsed span covers both originals.
        assert spans[0].start == 0.0
        assert spans[0].end == 2.0

    def test_chart_mode_needs_a_vocabulary(self, turnaround):
        with pytest.raises(ValueError, match="needs a vocabulary"):
            viz.chord_spans(turnaround, chart=True)

    def test_an_empty_timeline_gives_no_spans(self):
        assert viz.chord_spans([]) == ()


class TestDurationBins:
    def test_it_counts_durations_into_bins(self):
        events = timeline((0.0, 1.0, "C"), (1.0, 2.0, "G"), (2.0, 4.0, "A"))
        bins = dict(viz.duration_bins(events, width=0.5))
        assert bins[1.0] == 2  # two one-second chords
        assert bins[2.0] == 1  # one two-second chord

    def test_empty_bins_are_kept_so_the_shape_survives(self):
        # A gap in the histogram is informative; closing it up would lie.
        events = timeline((0.0, 0.5, "C"), (0.5, 3.5, "G"))
        bins = viz.duration_bins(events, width=0.5)
        assert len(bins) > 2
        assert any(count == 0 for _, count in bins)

    def test_the_tail_can_be_clipped(self):
        events = timeline((0.0, 1.0, "C"), (1.0, 101.0, "G"))
        bins = viz.duration_bins(events, width=0.5, limit=2.0)
        assert max(start for start, _ in bins) <= 2.0

    def test_zero_length_spans_are_ignored(self):
        assert viz.duration_bins(timeline((1.0, 1.0, "C"))) == ()

    def test_an_empty_timeline_gives_no_bins(self):
        assert viz.duration_bins([]) == ()

    def test_a_non_positive_width_raises(self, turnaround):
        with pytest.raises(ValueError, match="width must be positive"):
            viz.duration_bins(turnaround, width=0.0)


class TestLabelAndRootTime:
    def test_label_time_ranks_by_seconds(self):
        events = timeline((0.0, 1.0, "C"), (1.0, 6.0, "G"), (6.0, 7.0, "C"))
        assert viz.label_time(events) == (("G", 5.0), ("C", 2.0))

    def test_label_time_can_be_truncated(self, turnaround):
        assert len(viz.label_time(turnaround, top=2)) == 2

    def test_root_time_always_has_twelve_roots_in_chromatic_order(self, turnaround):
        roots = viz.root_time(turnaround)
        assert len(roots) == 12
        assert [name for name, _ in roots][:3] == ["C", "C#", "D"]

    def test_root_time_keeps_unused_roots_at_zero(self):
        # The gaps are the key profile; dropping them would hide it.
        roots = dict(viz.root_time(timeline((0.0, 4.0, "C:maj"))))
        assert roots["C"] == 4.0
        assert roots["F#"] == 0.0

    def test_root_time_pools_qualities_and_enharmonics(self):
        roots = dict(viz.root_time(timeline((0.0, 1.0, "Db:maj"), (1.0, 3.0, "C#:min7"))))
        assert roots["C#"] == 3.0

    def test_rootless_labels_contribute_to_no_root(self):
        assert sum(seconds for _, seconds in viz.root_time(timeline((0.0, 5.0, "N")))) == 0.0


class TestOffsetCurve:
    def test_it_samples_symmetrically_around_zero(self, turnaround):
        curve = viz.offset_curve(turnaround, turnaround, search=1.0, step=0.5)
        assert [offset for offset, _ in curve] == [-1.0, -0.5, 0.0, 0.5, 1.0]

    def test_an_aligned_estimate_peaks_at_zero(self, turnaround):
        curve = viz.offset_curve(turnaround, turnaround, search=2.0, step=0.5)
        best = max(curve, key=lambda point: point[1])
        assert best[0] == 0.0
        assert best[1] == 1.0

    def test_a_shifted_estimate_peaks_at_the_shift(self, turnaround):
        shifted = [ChordEvent(e.start + 1.0, e.end + 1.0, e.label) for e in turnaround]
        curve = viz.offset_curve(turnaround, shifted, search=2.0, step=0.5)
        assert max(curve, key=lambda point: point[1])[0] == pytest.approx(-1.0)

    def test_an_empty_reference_gives_a_curve_of_nans(self, turnaround):
        curve = viz.offset_curve([], turnaround, search=0.5, step=0.5)
        assert all(isnan(wcsr) for _, wcsr in curve)

    def test_a_non_positive_step_raises(self, turnaround):
        with pytest.raises(ValueError, match="step must be positive"):
            viz.offset_curve(turnaround, turnaround, step=0.0)


class TestAgreementSpans:
    def test_it_marks_correct_and_wrong_time(self):
        reference = timeline((0.0, 2.0, "C:maj"), (2.0, 4.0, "G:maj"))
        estimate = timeline((0.0, 2.0, "C:maj"), (2.0, 4.0, "F:maj"))
        spans = viz.agreement_spans(reference, estimate)
        assert spans == ((0.0, 2.0, True), (2.0, 4.0, False))

    def test_unscoreable_reference_time_is_marked_none(self):
        reference = timeline((0.0, 2.0, "X"), (2.0, 4.0, "C:maj"))
        estimate = timeline((0.0, 4.0, "C:maj"))
        verdicts = [verdict for _, _, verdict in viz.agreement_spans(reference, estimate)]
        assert verdicts == [None, True]

    def test_the_spans_tile_the_reference_without_gaps(self):
        reference = timeline((0.0, 2.0, "C:maj"), (2.0, 5.0, "G:maj"))
        estimate = timeline((0.0, 1.0, "C:maj"), (1.0, 5.0, "G:maj"))
        spans = viz.agreement_spans(reference, estimate)
        assert spans[0][0] == 0.0
        assert spans[-1][1] == 5.0
        for before, after in pairwise(spans):
            assert before[1] == pytest.approx(after[0])

    def test_an_empty_reference_gives_nothing_to_draw(self, turnaround):
        assert viz.agreement_spans([], turnaround) == ()
