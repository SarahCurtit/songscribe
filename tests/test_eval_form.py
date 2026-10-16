from __future__ import annotations

from itertools import pairwise
from math import isnan

import pytest

from songscribe.beats import BeatGrid
from songscribe.eval import form
from songscribe.types import ChordEvent

BARS = 32
REFERENCE_DURATION = 60.0  # 1.875 s a bar in the chart
BAR = 2.0


def grid_of(intro: int, choruses: int, tail: int, lead: float = 3.0, bar: float = BAR):
    """Bar lines for a take: ``lead`` seconds of nothing, an intro, choruses, a last pass."""
    count = intro + choruses * BARS + tail
    return BeatGrid(downbeats=tuple(lead + i * bar for i in range(count + 1)))


def split(grid, anchor, **kwargs):
    return form.choruses(grid, BARS, anchor, REFERENCE_DURATION, **kwargs)


class TestChoruses:
    def test_an_intro_four_choruses_and_a_partial_pass(self):
        found = split(grid_of(intro=7, choruses=4, tail=22), anchor=7)
        assert [c.kind for c in found] == [
            "intro",
            "head",
            "chorus",
            "chorus",
            "head",
            "partial",
        ]
        assert [c.bars for c in found] == [7, 32, 32, 32, 32, 22]
        assert [c.first_bar for c in found] == [0, 7, 39, 71, 103, 135]

    def test_blocks_tile_the_grid_without_gaps(self):
        found = split(grid_of(intro=7, choruses=4, tail=22), anchor=7)
        for earlier, later in pairwise(found[1:]):
            assert earlier.end == later.start

    def test_intro_time_is_unscored_and_includes_the_lead_in(self):
        grid = grid_of(intro=7, choruses=2, tail=0)
        found = split(grid, anchor=7)
        assert found[0].start == 0.0
        assert found[0].end == grid.downbeats[7]
        assert form.unscored_seconds(found) == pytest.approx(3.0 + 7 * BAR)

    def test_no_intro_block_when_the_form_starts_the_recording(self):
        found = split(grid_of(intro=0, choruses=2, tail=0, lead=0.0), anchor=0)
        assert [c.kind for c in found] == ["head", "head"]

    def test_a_lead_in_before_bar_one_is_still_an_intro(self):
        found = split(grid_of(intro=0, choruses=2, tail=0), anchor=0)
        assert (found[0].kind, found[0].bars) == ("intro", 0)

    def test_the_chorus_in_between_the_heads_is_a_solo_chorus(self):
        found = split(grid_of(intro=0, choruses=3, tail=0, lead=0.0), anchor=0)
        assert [c.kind for c in found] == ["head", "chorus", "head"]

    def test_a_single_chorus_is_the_head(self):
        found = split(grid_of(intro=0, choruses=1, tail=0, lead=0.0), anchor=0)
        assert [c.kind for c in found] == ["head"]

    def test_which_choruses_are_heads_is_a_setting(self):
        found = split(grid_of(intro=0, choruses=3, tail=0, lead=0.0), 0, head_choruses=(1,))
        assert [c.kind for c in found] == ["chorus", "head", "chorus"]

    def test_a_short_last_block_is_an_ending_not_a_partial_chorus(self):
        found = split(grid_of(intro=0, choruses=2, tail=form.CHORUS_MIN_BARS - 1), 0)
        assert found[-1].kind == "tail"
        assert isnan(found[-1].scale)

    def test_a_block_of_the_minimum_length_is_partial(self):
        found = split(grid_of(intro=0, choruses=2, tail=form.CHORUS_MIN_BARS), 0)
        assert found[-1].kind == "partial"

    def test_scale_is_the_blocks_bar_length_over_the_charts(self):
        found = split(grid_of(intro=0, choruses=2, tail=0, lead=0.0), anchor=0)
        assert found[0].scale == pytest.approx(BAR / (REFERENCE_DURATION / BARS))

    def test_a_dropped_bar_makes_that_chorus_unaligned_and_names_it(self):
        base = grid_of(intro=0, choruses=2, tail=0, lead=0.0)
        # The tracker missed one downbeat inside the second chorus.
        downbeats = tuple(d for i, d in enumerate(base.downbeats) if i != BARS + 10)
        found = split(BeatGrid(downbeats=downbeats), anchor=0)
        assert [c.kind for c in found] == ["head", "unaligned"]
        assert isnan(found[1].scale)
        assert form.unscored_seconds(found) == pytest.approx(found[1].duration)

    def test_the_count_is_lost_after_a_dropped_bar_so_later_choruses_are_refused(self):
        base = grid_of(intro=0, choruses=4, tail=0, lead=0.0)
        downbeats = tuple(d for i, d in enumerate(base.downbeats) if i != BARS + 10)
        found = split(BeatGrid(downbeats=downbeats), anchor=0)
        assert [c.kind for c in found][:2] == ["head", "unaligned"]
        assert all(c.kind == "unaligned" for c in found[1:])

    def test_a_chorus_played_at_another_tempo_is_unaligned(self):
        downbeats = [0.0]
        for i in range(3 * BARS):
            downbeats.append(downbeats[-1] + BAR * (0.8 if i // BARS == 1 else 1.0))
        found = split(BeatGrid(downbeats=tuple(downbeats)), 0)
        assert [c.kind for c in found] == ["head", "unaligned", "head"]

    def test_gentle_drift_across_the_take_stays_aligned(self):
        downbeats = [0.0]
        for i in range(3 * BARS):
            downbeats.append(downbeats[-1] + BAR * (1 - 0.03 * i / (3 * BARS)))
        found = split(BeatGrid(downbeats=tuple(downbeats)), anchor=0)
        assert [c.kind for c in found] == ["head", "chorus", "head"]

    @pytest.mark.parametrize("anchor", [-1, 1_000])
    def test_an_anchor_outside_the_grid_raises(self, anchor):
        with pytest.raises(ValueError):
            split(grid_of(0, 1, 0), anchor)

    def test_degenerate_arguments_raise(self):
        grid = grid_of(0, 1, 0)
        with pytest.raises(ValueError):
            form.choruses(grid, 0, 0, 60.0)
        with pytest.raises(ValueError):
            form.choruses(grid, BARS, 0, 0.0)

    def test_an_empty_grid_has_no_choruses(self):
        assert form.choruses(BeatGrid(downbeats=()), BARS, 0, 60.0) == ()


class TestProjectEstimate:
    def chorus(self, **kwargs):
        defaults = {
            "index": 1,
            "start": 10.0,
            "end": 74.0,
            "first_bar": 0,
            "bars": BARS,
            "scale": BAR / (REFERENCE_DURATION / BARS),
            "kind": "head",
        }
        return form.Chorus(**{**defaults, **kwargs})

    def test_time_moves_onto_the_charts_clock(self):
        (event,) = form.project_estimate([ChordEvent(12.0, 14.0, "C")], self.chorus())
        assert event.start == pytest.approx(2.0 / 2.0 * 1.875)
        assert event.end == pytest.approx(4.0 / 2.0 * 1.875)

    def test_origin_is_where_the_reference_starts(self):
        (event,) = form.project_estimate([ChordEvent(10.0, 12.0, "C")], self.chorus(), origin=5.0)
        assert event.start == pytest.approx(5.0)

    def test_time_outside_the_block_is_discarded_and_edges_are_clipped(self):
        events = [
            ChordEvent(0.0, 8.0, "junk"),
            ChordEvent(8.0, 12.0, "A"),
            ChordEvent(72.0, 80.0, "B"),
            ChordEvent(80.0, 90.0, "junk"),
        ]
        projected = form.project_estimate(events, self.chorus())
        assert [e.label for e in projected] == ["A", "B"]
        assert projected[0].start == 0.0
        assert projected[1].end == pytest.approx(REFERENCE_DURATION)

    @pytest.mark.parametrize("kind", ["intro", "tail", "unaligned"])
    def test_a_block_with_no_mapping_cannot_be_projected(self, kind):
        with pytest.raises(ValueError):
            form.project_estimate([], self.chorus(kind=kind, scale=float("nan")))


class TestTruncateReference:
    REFERENCE = (ChordEvent(0.0, 30.0, "C"), ChordEvent(30.0, 60.0, "G"))

    def test_keeps_the_first_bars_of_the_chart(self):
        kept = form.truncate_reference(self.REFERENCE, BARS, 16)
        assert kept == [ChordEvent(0.0, 30.0, "C")]

    def test_clips_a_chord_that_straddles_the_cut(self):
        kept = form.truncate_reference(self.REFERENCE, BARS, 24)
        assert kept[-1] == ChordEvent(30.0, 45.0, "G")

    def test_the_whole_chart_is_unchanged(self):
        assert form.truncate_reference(self.REFERENCE, BARS, BARS) == list(self.REFERENCE)

    @pytest.mark.parametrize("bars", [0, BARS + 1])
    def test_bars_outside_the_chart_raise(self, bars):
        with pytest.raises(ValueError):
            form.truncate_reference(self.REFERENCE, BARS, bars)
