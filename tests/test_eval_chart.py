from __future__ import annotations

from math import isnan

import pytest

from songscribe.align import MIN_CHORD_DURATION
from songscribe.eval import metrics
from songscribe.types import ChordEvent


def timeline(*spec: tuple[float, float, str]) -> list[ChordEvent]:
    return [ChordEvent(start, end, label) for start, end, label in spec]


def shift(events: list[ChordEvent], offset: float) -> list[ChordEvent]:
    return [ChordEvent(e.start + offset, e.end + offset, e.label) for e in events]


@pytest.fixture
def turnaround() -> list[ChordEvent]:
    return timeline(
        (0.0, 2.0, "C:maj"),
        (2.0, 4.0, "G:maj"),
        (4.0, 6.0, "A:min"),
        (6.0, 8.0, "F:maj"),
    )


class TestChartSequence:
    def test_it_is_the_printed_chord_order(self, turnaround):
        assert metrics.chart_sequence(turnaround) == ("C:maj", "G:maj", "A:min", "F:maj")

    def test_repeats_collapse(self):
        events = timeline((0.0, 2.0, "C:maj"), (2.0, 4.0, "C:maj"), (4.0, 6.0, "G:maj"))
        assert metrics.chart_sequence(events) == ("C:maj", "G:maj")

    def test_no_chord_spans_drop_out(self):
        # A chart has no symbol for silence.
        events = timeline((0.0, 2.0, "N"), (2.0, 4.0, "C:maj"), (4.0, 6.0, "N"))
        assert metrics.chart_sequence(events) == ("C:maj",)

    def test_a_chord_either_side_of_a_no_chord_gap_prints_once(self):
        # The pipeline strips N before align.dedupe, so the repeat merges
        # across the gap; collapsing first would leave a phantom insertion.
        events = timeline(
            (0.0, 4.0, "C:maj"),
            (4.0, 6.0, "N"),
            (6.0, 10.0, "C:maj"),
            (10.0, 14.0, "G:maj"),
        )
        assert metrics.chart_sequence(events) == ("C:maj", "G:maj")

    def test_flicker_shorter_than_the_chart_threshold_is_dropped(self):
        # align.dedupe's threshold, reused: a chord too short to print is not
        # part of the chart, so it must not be part of the chart's score.
        brief = MIN_CHORD_DURATION / 2
        events = timeline(
            (0.0, 2.0, "C:maj"),
            (2.0, 2.0 + brief, "F#:maj"),
            (2.0 + brief, 4.0, "G:maj"),
        )
        assert metrics.chart_sequence(events) == ("C:maj", "G:maj")

    def test_labels_are_projected_before_collapsing(self):
        # C:maj7 and C:maj are one chord on a majmin chart, so they merge.
        events = timeline((0.0, 2.0, "C:maj7"), (2.0, 4.0, "C:maj"), (4.0, 6.0, "G:maj"))
        assert metrics.chart_sequence(events, "majmin") == ("C:maj", "G:maj")

    def test_they_stay_distinct_under_a_vocabulary_that_separates_them(self):
        events = timeline((0.0, 2.0, "C:maj7"), (2.0, 4.0, "C:maj"))
        assert metrics.chart_sequence(events, "sevenths") == ("C:maj7", "C:maj")

    def test_a_label_outside_the_vocabulary_keeps_its_spelling(self):
        # Dropping it would shorten the sequence and let a wrong chord vanish.
        events = timeline((0.0, 2.0, "C:sus4"), (2.0, 4.0, "C:maj"))
        assert metrics.chart_sequence(events, "majmin") == ("C:sus4", "C:maj")

    def test_enharmonic_spellings_collapse(self):
        events = timeline((0.0, 2.0, "Db:maj"), (2.0, 4.0, "C#:maj"))
        assert metrics.chart_sequence(events) == ("C#:maj",)

    def test_an_empty_timeline_is_an_empty_chart(self):
        assert metrics.chart_sequence([]) == ()


class TestSequenceSimilarity:
    def test_identical_sequences_score_one(self, turnaround):
        assert metrics.chart_score(turnaround, turnaround) == 1.0

    def test_one_wrong_chord_in_four(self, turnaround):
        wrong = timeline(
            (0.0, 2.0, "C:maj"), (2.0, 4.0, "G:maj"), (4.0, 6.0, "A:min"), (6.0, 8.0, "D:maj")
        )
        assert metrics.chart_score(turnaround, wrong) == pytest.approx(0.75)

    def test_an_inserted_chord_costs_one_edit(self, turnaround):
        extra = timeline(
            (0.0, 2.0, "C:maj"),
            (2.0, 4.0, "G:maj"),
            (4.0, 6.0, "A:min"),
            (6.0, 8.0, "F:maj"),
            (8.0, 10.0, "D:maj"),
        )
        # Five chords, one of them spurious.
        assert metrics.chart_score(turnaround, extra) == pytest.approx(0.8)

    def test_a_missing_chord_costs_one_edit(self, turnaround):
        short = timeline((0.0, 2.0, "C:maj"), (2.0, 4.0, "G:maj"), (4.0, 6.0, "A:min"))
        assert metrics.chart_score(turnaround, short) == pytest.approx(0.75)

    def test_an_empty_estimate_scores_zero(self, turnaround):
        assert metrics.chart_score(turnaround, []) == 0.0

    def test_two_empty_charts_are_nan_not_perfect(self):
        # Nothing was measured; claiming 100% would be a lie.
        assert isnan(metrics.chart_score([], []))

    def test_order_matters(self):
        forward = timeline((0.0, 2.0, "C:maj"), (2.0, 4.0, "G:maj"))
        backward = timeline((0.0, 2.0, "G:maj"), (2.0, 4.0, "C:maj"))
        assert metrics.chart_score(forward, backward) < 1.0


class TestTimingInvariance:
    """The property that makes this the right metric for a playable chart.

    A transcription that is uniformly late is annoying to nobody: the musician
    reads the same progression. WCSR punishes it heavily, which is correct for
    "is each second labelled right" and wrong for "is the chart playable".
    """

    @pytest.mark.parametrize("offset", [0.1, 0.5, 1.0, 2.0, 5.0, -1.7])
    def test_a_global_shift_does_not_change_the_chart_score(self, turnaround, offset):
        assert metrics.chart_score(turnaround, shift(turnaround, offset)) == 1.0

    def test_while_wcsr_collapses_under_the_same_shift(self, turnaround):
        shifted = shift(turnaround, 2.0)
        assert metrics.score(turnaround, shifted, "majmin").wcsr < 0.6
        assert metrics.chart_score(turnaround, shifted) == 1.0

    @pytest.mark.parametrize("factor", [0.5, 0.9, 0.95, 0.98, 1.02, 1.05, 1.1, 1.5, 2.0])
    def test_a_tempo_difference_does_not_change_the_chart_score(self, turnaround, factor):
        # A speed difference moves every boundary by a different amount, which
        # is why it wrecks WCSR, but the printed order is unchanged -- so the
        # chart is unaffected, at any factor. Measured on the real corpus: 0.90x
        # to 1.50x all score 67.44% where WCSR spans 46.89%-79.51%.
        stretched = [ChordEvent(e.start * factor, e.end * factor, e.label) for e in turnaround]
        assert metrics.chart_score(turnaround, stretched) == 1.0

    @pytest.mark.parametrize("factor", [0.9, 1.1])
    def test_offset_and_tempo_together_are_both_absorbed(self, turnaround, factor):
        both = [
            ChordEvent(e.start * factor + 1.7, e.end * factor + 1.7, e.label) for e in turnaround
        ]
        assert metrics.chart_score(turnaround, both) == 1.0

    def test_the_flicker_threshold_is_the_one_tempo_dependent_edge(self):
        # The single residual: dedupe's threshold is absolute seconds, so a
        # chord within the stretch factor of MIN_CHORD_DURATION can cross it
        # and appear in one chart but not the other. The window is tiny -- here
        # 0.245s-0.25s for a 2% stretch -- and the threshold stays absolute on
        # purpose, because that is what render.py prints and because flicker is
        # an absolute-time phenomenon, not a proportional one.
        just_under = MIN_CHORD_DURATION - 0.004
        events = [
            ChordEvent(0.0, 2.0, "C:maj"),
            ChordEvent(2.0, 2.0 + just_under, "F:maj"),
            ChordEvent(2.0 + just_under, 4.0, "G:maj"),
        ]
        stretched = [ChordEvent(e.start * 1.02, e.end * 1.02, e.label) for e in events]
        assert metrics.chart_sequence(events) == ("C:maj", "G:maj")
        assert metrics.chart_sequence(stretched) == ("C:maj", "F:maj", "G:maj")

    def test_a_chord_clear_of_the_threshold_is_unaffected_by_tempo(self):
        clear = MIN_CHORD_DURATION * 2
        events = [
            ChordEvent(0.0, 2.0, "C:maj"),
            ChordEvent(2.0, 2.0 + clear, "F:maj"),
            ChordEvent(2.0 + clear, 6.0, "G:maj"),
        ]
        for factor in (0.9, 1.0, 1.1):
            stretched = [ChordEvent(e.start * factor, e.end * factor, e.label) for e in events]
            assert metrics.chart_sequence(stretched) == ("C:maj", "F:maj", "G:maj")

    def test_but_reordering_is_not_tolerated(self, turnaround):
        # Timing-invariant is not error-blind: getting the progression wrong
        # still costs, which is the whole point of keeping the metric.
        scrambled = timeline(
            (0.0, 2.0, "A:min"), (2.0, 4.0, "F:maj"), (4.0, 6.0, "C:maj"), (6.0, 8.0, "G:maj")
        )
        assert metrics.chart_score(turnaround, scrambled) < 0.6


class TestOverSegmentationIsVisibleHere:
    """Why the chart score is not just WCSR with the timing removed.

    A model that flickers between plausible chords can score respectably on
    WCSR -- most seconds carry a defensible label -- while printing a chart
    with three times too many changes, which is unplayable. WCSR barely
    notices; the chart score does.
    """

    def test_a_thrashing_estimate_keeps_a_fair_wcsr(self):
        reference = timeline((0.0, 8.0, "C:maj"))
        # Alternates between C and its relative minor every half second: every
        # frame is at worst a near miss, but the chart is nonsense.
        thrash = [
            ChordEvent(t / 2, (t + 1) / 2, "C:maj" if t % 2 == 0 else "A:min") for t in range(16)
        ]
        wcsr = metrics.score(reference, thrash, "majmin").wcsr
        chart = metrics.chart_score(reference, thrash)
        assert wcsr == pytest.approx(0.5)
        assert chart < wcsr
        assert chart < 0.2

    def test_the_chart_counts_reveal_the_cause(self):
        reference = timeline((0.0, 8.0, "C:maj"))
        thrash = [
            ChordEvent(t / 2, (t + 1) / 2, "C:maj" if t % 2 == 0 else "A:min") for t in range(16)
        ]
        assert len(metrics.chart_sequence(reference)) == 1
        assert len(metrics.chart_sequence(thrash)) == 16


class TestRunnerReportsTheChart:
    @pytest.fixture
    def corpus(self, tmp_path):
        from songscribe import lab
        from songscribe.eval import dataset

        reference = timeline(
            (0.0, 2.0, "C:maj"), (2.0, 4.0, "G:maj"), (4.0, 6.0, "A:min"), (6.0, 8.0, "F:maj")
        )
        lab.write(tmp_path / "ann" / "a.lab", reference)
        # The same chart, two seconds late.
        lab.write(tmp_path / "est" / "a.lab", shift(reference, 2.0))
        manifest = tmp_path / "m.json"
        dataset.write_manifest(manifest, dataset.build("tiny", [{"id": "a", "reference": "a.lab"}]))
        return dataset.load_manifest(manifest, annotation_root=tmp_path / "ann"), tmp_path / "est"

    def test_the_chart_score_survives_what_wcsr_does_not(self, corpus):
        from songscribe.eval import runner

        data, estimates = corpus
        report = runner.evaluate(data, estimates_dir=estimates)
        assert report.mean_chart() == 1.0
        assert report.weighted("majmin") < 0.6

    def test_the_chart_chord_counts_are_recorded(self, corpus):
        from songscribe.eval import runner

        data, estimates = corpus
        (result,) = runner.evaluate(data, estimates_dir=estimates).results
        assert result.reference_chart_chords == 4
        assert result.estimate_chart_chords == 4

    def test_the_report_and_json_both_carry_it(self, corpus):
        from songscribe.eval import runner
        from songscribe.eval.cli import format_report, report_to_dict

        data, estimates = corpus
        report = runner.evaluate(data, estimates_dir=estimates)
        assert "chord order as printed" in format_report(report)
        assert report_to_dict(report)["chart"]["mean"] == 1.0
