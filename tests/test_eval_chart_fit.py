from __future__ import annotations

import numpy as np
import pytest

from songscribe.eval import chart_fit, form, labels
from songscribe.types import ChordEvent

CHART = ("C", "Am", "Dm", "G7", "Em", "A7", "Dm", "G7")
PER_BAR = 4
BAR = 2.0


def timeline(names, start=0.0) -> list[ChordEvent]:
    return [
        ChordEvent(start + i * BAR, start + (i + 1) * BAR, name) for i, name in enumerate(names)
    ]


def symbols_of(events) -> np.ndarray:
    end = max(e.end for e in events)
    return labels.beat_symbols(events, np.linspace(0.0, end, round(end / BAR) * PER_BAR + 1))


def fitted(estimate_names, starts, chart_names=CHART, shifts=chart_fit.AS_WRITTEN, need=4):
    chart = chart_fit.chart_beats(timeline(chart_names), len(chart_names), PER_BAR, shifts)
    return chart_fit.fit_chart(
        symbols_of(timeline(estimate_names)), starts, chart, shifts, need=need
    )


class TestChartBeats:
    def test_one_row_of_beats_per_bar_and_one_slice_per_key(self):
        chart = chart_fit.chart_beats(timeline(CHART), len(CHART), PER_BAR, (0, 2))
        assert chart.shape == (2, len(CHART), PER_BAR)
        assert chart[0, 0].tolist() == [labels.symbol("C")] * PER_BAR
        assert chart[1, 0].tolist() == [labels.symbol("D")] * PER_BAR

    def test_a_bar_with_two_chords_splits_its_beats(self):
        chart = chart_fit.chart_beats(
            [ChordEvent(0, 1, "C"), ChordEvent(1, 2, "G")], 1, PER_BAR, (0,)
        )
        assert chart[0, 0].tolist() == [labels.symbol("C")] * 2 + [labels.symbol("G")] * 2

    def test_a_chart_that_starts_late_is_cut_from_where_it_starts(self):
        late = chart_fit.chart_beats(timeline(CHART, start=0.005), len(CHART), PER_BAR, (0,))
        assert late[0, 0].tolist() == [labels.symbol("C")] * PER_BAR

    @pytest.mark.parametrize("bars", [0, -1])
    def test_a_chart_needs_at_least_a_bar(self, bars):
        with pytest.raises(ValueError, match="bars"):
            chart_fit.chart_beats(timeline(CHART), bars, PER_BAR, (0,))

    def test_an_empty_chart_covers_no_time(self):
        with pytest.raises(ValueError, match="no time"):
            chart_fit.chart_beats([ChordEvent(0, 0, "C")], 4, PER_BAR, (0,))


class TestFit:
    def test_the_chart_agrees_fully_with_itself_at_its_start_and_less_elsewhere(self):
        result = fitted((*CHART, *CHART), [0, 4, 8])
        assert result.best(0) == pytest.approx(1.0)
        assert result.best(1) < 0.5
        assert result.best(2) < 0.5

    def test_the_start_is_found_after_an_intro(self):
        result = fitted(("F#", "B", "Ebm", *CHART, *CHART), [0, 4, 12, 16])
        assert int(np.argmax([result.best(i) for i in range(4)])) == 2

    def test_an_estimate_that_only_roughly_agrees_scores_in_between(self):
        rough = list(CHART)
        rough[1], rough[4] = "C", "F#"
        score = fitted(rough, [0], need=8).best(0)
        assert 0.5 < score < 1.0

    def test_the_same_root_with_the_other_third_counts_for_half(self):
        parallel = fitted(("Cm", *CHART[1:]), [0], need=8).best(0)
        assert parallel == pytest.approx((7 + labels.ROOT_CREDIT) / 8)

    def test_a_start_with_too_few_bars_left_is_not_fitted(self):
        result = fitted(CHART, [0, 20], need=4)
        assert np.isnan(result.fit[1]).all()
        assert result.best(1) == -np.inf

    def test_a_recording_with_too_few_bars_for_any_start_raises(self):
        with pytest.raises(ValueError, match="too short"):
            fitted(CHART[:3], [0], need=4)

    def test_a_partial_chorus_is_fitted_on_the_bars_there_are(self):
        assert fitted(CHART[:5], [0], need=4).best(0) == pytest.approx(1.0)


class TestChance:
    def test_the_right_chart_beats_every_shuffled_copy(self):
        result = fitted((*CHART, *CHART), [0, 8])
        assert result.best(0) > result.chance + 0.2

    def test_a_shuffled_chart_does_not_beat_its_own_shuffles(self):
        order = (3, 6, 1, 4, 0, 7, 2, 5)
        result = fitted((*CHART, *CHART), [0, 8], chart_names=[CHART[i] for i in order])
        assert result.best(0) < result.chance

    def test_a_chart_of_other_chords_does_not_beat_the_chance_level_either(self):
        other = ("F#", "Bb", "Eb", "B", "Ab", "D", "F#", "Bb")
        result = fitted((*CHART, *CHART), [0, 8], chart_names=other)
        assert result.best(0) < result.chance

    def test_one_chord_throughout_cannot_beat_its_shuffles(self):
        same = ("C",) * 8
        result = fitted(same * 2, [0, 8], chart_names=same)
        assert result.best(0) == pytest.approx(result.chance)

    def test_it_is_the_same_on_every_run(self):
        assert fitted(CHART * 2, [0, 8]).chance == fitted(CHART * 2, [0, 8]).chance

    def test_searching_more_starts_cannot_lower_it(self):
        few = fitted(CHART * 3, [0, 8]).chance
        many = fitted(CHART * 3, list(range(0, 17, 4))).chance
        assert many >= few


class TestKeys:
    def moved(self, semitones):
        return [e.label for e in form.transpose_events(timeline(CHART), semitones)]

    @pytest.mark.parametrize("semitones", [-5, -2, 0, 3, 6])
    def test_the_key_the_estimate_is_in_is_the_one_that_fits(self, semitones):
        estimate = [*self.moved(semitones), *self.moved(semitones)]
        result = fitted(estimate, [0, 8], shifts=chart_fit.ANY_KEY)
        assert result.key(0) == semitones
        assert result.best(0) == pytest.approx(1.0)
        assert result.key_margin(0) > 0.3

    def test_with_the_chart_as_written_only_the_chart_as_written_fits(self):
        result = fitted(self.moved(3) * 2, [0, 8])
        assert result.best(0) < result.chance

    def test_twelve_keys_give_the_chance_level_twelve_chances(self):
        one = fitted(CHART * 2, [0, 8]).chance
        twelve = fitted(CHART * 2, [0, 8], shifts=chart_fit.ANY_KEY).chance
        assert twelve >= one

    def test_a_single_key_has_no_margin_over_another(self):
        assert np.isnan(fitted(CHART * 2, [0, 8]).key_margin(0))
