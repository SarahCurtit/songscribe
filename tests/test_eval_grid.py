from __future__ import annotations

from math import isnan

import pytest

from songscribe.eval import metrics
from songscribe.types import ChordEvent

LABELS = ("C:maj", "A:min", "F:maj", "G:maj")

#: Durations that fit no unit in the search window: spread wide, with ratios
#: that are not small whole numbers. Picked by measurement, not by eye -- most
#: irregular-looking sets still fit *something* to within a few percent, which
#: is the limitation :class:`TestTheReliabilityGateIsWeak` records.
IRREGULAR = [0.9, 3.1, 1.4, 2.05, 4.7, 1.15, 3.85, 2.45]


def grid_of(durations: list[float], labels: tuple[str, ...] = LABELS) -> list[ChordEvent]:
    """A chord timeline from a list of durations, cycling through labels."""
    events, cursor = [], 0.0
    for index, duration in enumerate(durations):
        events.append(ChordEvent(cursor, cursor + duration, labels[index % len(labels)]))
        cursor += duration
    return events


def stretch(events: list[ChordEvent], factor: float) -> list[ChordEvent]:
    return [ChordEvent(e.start * factor, e.end * factor, e.label) for e in events]


def units(events: list[ChordEvent]) -> list[int]:
    return [n for _, n in metrics.grid_sequence(events)]


class TestHarmonicRhythm:
    @pytest.mark.parametrize(
        ("durations", "expected_unit"),
        [
            ([2.0] * 8, 2.0),
            ([1.0] * 16, 1.0),
            ([4.0] * 6, 4.0),
            ([2.0, 2.0, 4.0, 2.0] * 2, 2.0),
            ([4.0, 2.0, 2.0] * 3, 2.0),
        ],
    )
    def test_it_finds_the_unit_on_regular_material(self, durations, expected_unit):
        rhythm = metrics.harmonic_rhythm(grid_of(durations))
        assert rhythm.unit == pytest.approx(expected_unit, rel=0.05)
        assert rhythm.residual == pytest.approx(0.0, abs=0.01)
        assert rhythm.is_reliable

    def test_a_perfect_fit_prefers_the_coarser_grid(self):
        # Every chord lasts 2s, so a 2s unit (one per bar) and a 1s unit (two
        # per bar) both fit exactly. A musician reads the former.
        assert metrics.harmonic_rhythm(grid_of([2.0] * 8)).unit == pytest.approx(2.0, rel=0.05)
        assert units(grid_of([2.0] * 8)) == [1] * 8

    def test_one_chord_per_bar_and_two_per_bar_are_distinguishable(self):
        # The thing the metric exists to tell apart. Same total length, same
        # chord order, different harmonic rhythm.
        one_per_bar = grid_of([2.0] * 4)
        two_per_bar = grid_of([1.0] * 8)
        assert units(one_per_bar) == [1, 1, 1, 1]
        assert units(two_per_bar) == [1] * 8
        assert metrics.grid_score(one_per_bar, two_per_bar) < 0.6

    def test_a_held_chord_counts_as_several_units(self):
        assert units(grid_of([16.0, 2.0, 2.0, 2.0])) == [8, 1, 1, 1]

    def test_irregular_material_is_flagged_unreliable(self):
        # Durations that are not near-multiples of anything: no estimate of the
        # bar length can rescue this, and saying so beats guessing.
        rhythm = metrics.harmonic_rhythm(grid_of(IRREGULAR))
        assert rhythm.residual > metrics.GRID_MAX_RESIDUAL
        assert not rhythm.is_reliable

    def test_an_empty_timeline_has_no_rhythm(self):
        rhythm = metrics.harmonic_rhythm([])
        assert isnan(rhythm.unit)
        assert rhythm.chords == 0
        assert not rhythm.is_reliable

    def test_harmonic_unit_is_the_bare_number(self):
        events = grid_of([2.0] * 8)
        assert metrics.harmonic_unit(events) == metrics.harmonic_rhythm(events).unit


class TestGridIsTempoFree:
    """The property the metric exists for: a grid is bars, not seconds."""

    @pytest.mark.parametrize("factor", [0.5, 0.8, 0.9, 1.1, 1.25, 1.5, 2.0])
    def test_the_grid_survives_any_tempo(self, factor):
        base = grid_of([2.0, 2.0, 4.0, 2.0] * 2)
        assert units(stretch(base, factor)) == units(base)

    @pytest.mark.parametrize("factor", [0.8, 1.25])
    def test_the_unit_scales_with_the_tempo(self, factor):
        base = grid_of([2.0] * 8)
        scaled = metrics.harmonic_rhythm(stretch(base, factor)).unit
        assert scaled == pytest.approx(2.0 * factor, rel=0.05)

    @pytest.mark.parametrize("factor", [0.95, 1.05])
    def test_a_slight_tempo_difference_still_scores_perfectly(self, factor):
        # The reference's unit is used for both sides, so a small tempo
        # difference rounds to the same bar counts.
        base = grid_of([2.0, 2.0, 4.0, 2.0] * 2)
        assert metrics.grid_score(base, stretch(base, factor)) == 1.0


class TestGridCatchesWhatChartCannot:
    def test_a_swapped_harmonic_rhythm_passes_chart_and_fails_grid(self):
        # Identical chord order; the reference holds the first chord for two
        # bars and the estimate holds the second for two instead.
        reference = grid_of([4.0, 2.0, 2.0] * 3, ("C:maj", "A:min", "F:maj"))
        swapped = grid_of([2.0, 4.0, 2.0] * 3, ("C:maj", "A:min", "F:maj"))
        assert metrics.chart_score(reference, swapped) == 1.0
        assert metrics.grid_score(reference, swapped) < 0.5

    def test_the_right_rhythm_scores_full_marks(self):
        reference = grid_of([4.0, 2.0, 2.0] * 3, ("C:maj", "A:min", "F:maj"))
        assert metrics.grid_score(reference, reference) == 1.0

    def test_a_per_side_unit_would_have_hidden_a_systematic_merge(self):
        # The reason both sides share the reference's unit. The estimate holds
        # every chord twice as long -- a model that merges. Were each side
        # normalised to its own unit, both would read "one chord per bar" and
        # the error would vanish.
        reference = grid_of([2.0] * 8)
        merged = grid_of([4.0] * 8)
        assert units(reference) == units(merged)  # identical in isolation
        assert metrics.grid_score(reference, merged) < 1.0  # but not when compared


class TestGridReliabilityGating:
    def test_an_unreliable_reference_yields_no_score(self):
        reference = grid_of(IRREGULAR)
        assert isnan(metrics.grid_score(reference, reference))

    def test_a_reliable_reference_does_yield_one(self):
        reference = grid_of([2.0] * 8)
        assert metrics.grid_score(reference, reference) == 1.0

    def test_an_empty_reference_yields_no_score(self):
        assert isnan(metrics.grid_score([], grid_of([2.0] * 4)))


class TestTheReliabilityGateIsWeak:
    """Known limitation: the residual gate is weak on short timelines.

    The unit is searched over a +/-50% window, so on 8-12 chord fixtures almost any
    durations fit some unit within a few percent; the gate only discriminates at
    real-track lengths (30-50 chords). ``GRID_MAX_RESIDUAL`` is calibrated on limited
    data, so a known ``unit=`` is the reliable path.
    """

    @pytest.mark.parametrize(
        "durations",
        [
            [1.0, 1.414, 2.0, 2.828, 4.0, 1.414, 2.0, 2.828],
            [1.1, 2.3, 3.7, 5.3, 1.7, 2.9, 4.1, 1.3],
            [2.0 * (1 + 0.4 * (-1) ** i) for i in range(12)],
        ],
    )
    def test_plainly_irregular_short_timelines_still_pass_the_gate(self, durations):
        # None of these is a grid a musician would recognise, yet all fit
        # under the threshold. Treat `grid` as indicative, not decisive.
        assert metrics.harmonic_rhythm(grid_of(durations)).is_reliable

    def test_supplying_the_unit_sidesteps_the_estimate_entirely(self):
        # The exact path: with the bar length known, no search happens and the
        # grid is simply read off.
        events = grid_of([2.0, 2.0, 4.0, 2.0])
        assert [n for _, n in metrics.grid_sequence(events, unit=2.0)] == [1, 1, 2, 1]
        assert [n for _, n in metrics.grid_sequence(events, unit=1.0)] == [2, 2, 4, 2]


class TestRunnerReportsTheGrid:
    @pytest.fixture
    def corpus(self, tmp_path):
        from songscribe import lab
        from songscribe.eval import dataset

        reference = grid_of([4.0, 2.0, 2.0] * 3, ("C:maj", "A:min", "F:maj"))
        swapped = grid_of([2.0, 4.0, 2.0] * 3, ("C:maj", "A:min", "F:maj"))
        lab.write(tmp_path / "ann" / "a.lab", reference)
        lab.write(tmp_path / "est" / "a.lab", swapped)
        manifest = tmp_path / "m.json"
        dataset.write_manifest(manifest, dataset.build("tiny", [{"id": "a", "reference": "a.lab"}]))
        return dataset.load_manifest(manifest, annotation_root=tmp_path / "ann"), tmp_path / "est"

    def test_the_grid_is_scored_and_the_chart_is_not_fooled(self, corpus):
        from songscribe.eval import runner

        data, estimates = corpus
        report = runner.evaluate(data, estimates_dir=estimates)
        assert report.mean_chart() == 1.0
        assert report.mean_grid() < 0.5
        assert report.graded_grids() == 1

    def test_the_harmonic_rhythm_is_recorded(self, corpus):
        from songscribe.eval import runner

        data, estimates = corpus
        (result,) = runner.evaluate(data, estimates_dir=estimates).results
        assert result.harmonic is not None
        assert result.harmonic.is_reliable
        assert result.harmonic.unit == pytest.approx(2.0, rel=0.05)

    def test_the_report_and_json_carry_it(self, corpus):
        from songscribe.eval import runner
        from songscribe.eval.cli import format_report, report_to_dict

        data, estimates = corpus
        report = runner.evaluate(data, estimates_dir=estimates)
        assert "harmonic rhythm in bars" in format_report(report)
        document = report_to_dict(report)
        assert document["grid"]["graded_tracks"] == 1
        assert document["tracks"][0]["grid"]["reliable"] is True

    def test_an_ungradeable_reference_is_absent_not_zero(self, tmp_path):
        from songscribe import lab
        from songscribe.eval import dataset, runner

        irregular = grid_of(IRREGULAR)
        lab.write(tmp_path / "ann" / "a.lab", irregular)
        lab.write(tmp_path / "est" / "a.lab", irregular)
        manifest = tmp_path / "m.json"
        dataset.write_manifest(manifest, dataset.build("tiny", [{"id": "a", "reference": "a.lab"}]))
        data = dataset.load_manifest(manifest, annotation_root=tmp_path / "ann")

        report = runner.evaluate(data, estimates_dir=tmp_path / "est")
        assert report.graded_grids() == 0
        assert isnan(report.mean_grid())
        # A perfect chart next to no grid at all: the chart is still measurable.
        assert report.mean_chart() == 1.0
