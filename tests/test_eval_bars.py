from __future__ import annotations

from math import isnan

import pytest

from songscribe.eval import metrics
from songscribe.types import ChordEvent

DOWNBEATS = [0.0, 2.0, 4.0, 6.0, 8.0]


def chords(*spans):
    return [ChordEvent(a, b, label) for a, b, label in spans]


REFERENCE = chords((0, 2, "C"), (2, 4, "A:min"), (4, 6, "F"), (6, 8, "G"))


class TestBarSequence:
    def test_one_label_per_complete_bar(self):
        assert metrics.bar_sequence(REFERENCE, DOWNBEATS) == (
            (0, "C:maj"),
            (1, "A:min"),
            (2, "F:maj"),
            (3, "G:maj"),
        )

    def test_the_label_held_longest_in_the_bar_wins(self):
        events = chords((0, 0.5, "G"), (0.5, 2, "C"))
        assert metrics.bar_sequence(events, [0.0, 2.0]) == ((0, "C:maj"),)

    def test_a_tie_goes_to_the_earlier_label(self):
        events = chords((0, 1, "G"), (1, 2, "C"))
        assert metrics.bar_sequence(events, [0.0, 2.0]) == ((0, "G:maj"),)

    def test_quality_is_projected_into_the_vocabulary(self):
        events = chords((0, 2, "C:maj7"))
        assert metrics.bar_sequence(events, [0.0, 2.0], "majmin") == ((0, "C:maj"),)
        assert metrics.bar_sequence(events, [0.0, 2.0], "sevenths") == ((0, "C:maj7"),)

    def test_split_time_in_one_class_adds_up(self):
        # C and C7 are both C:maj under majmin, so together they outweigh the A.
        events = chords((0, 0.7, "C"), (0.7, 1.4, "C:7"), (1.4, 2.0, "A:min"))
        assert metrics.bar_sequence(events, [0.0, 2.0]) == ((0, "C:maj"),)

    def test_a_bar_nothing_covers_reads_as_no_chord(self):
        assert metrics.bar_sequence(chords((0, 2, "C")), [0.0, 2.0, 4.0]) == (
            (0, "C:maj"),
            (1, "N"),
        )

    def test_no_bar_lines_means_no_bars(self):
        assert metrics.bar_sequence(REFERENCE, []) == ()
        assert metrics.bar_sequence(REFERENCE, [1.0]) == ()


class TestBarScore:
    def test_a_perfect_estimate_scores_one(self):
        assert metrics.bar_score(REFERENCE, REFERENCE, DOWNBEATS) == 1.0

    def test_each_wrong_bar_costs_one_bar(self):
        wrong = chords((0, 2, "C"), (2, 4, "D:min"), (4, 6, "F"), (6, 8, "E"))
        assert metrics.bar_score(REFERENCE, wrong, DOWNBEATS) == 0.5

    def test_a_flicker_inside_a_bar_costs_nothing(self):
        flicker = chords((0, 2, "C"), (2, 3.5, "A:min"), (3.5, 4, "E"), (4, 8, "F"))
        assert metrics.bar_score(REFERENCE, flicker, DOWNBEATS) == 0.75

    def test_tempo_does_not_move_it(self):
        slow_downbeats = [d * 1.5 for d in DOWNBEATS]
        slow = [ChordEvent(e.start * 1.5, e.end * 1.5, e.label) for e in REFERENCE]
        assert metrics.bar_score(REFERENCE, slow, DOWNBEATS, estimate_downbeats=slow_downbeats) == 1

    def test_offset_does_not_move_it(self):
        late_downbeats = [d + 10.0 for d in DOWNBEATS]
        late = [ChordEvent(e.start + 10.0, e.end + 10.0, e.label) for e in REFERENCE]
        assert metrics.bar_score(REFERENCE, late, DOWNBEATS, estimate_downbeats=late_downbeats) == 1

    def test_an_estimate_that_stops_early_loses_the_bars_it_missed(self):
        assert metrics.bar_score(REFERENCE, REFERENCE[:2], DOWNBEATS) == 0.5

    def test_an_unknown_reference_bar_is_skipped_not_counted(self):
        reference = chords((0, 2, "C"), (2, 4, "X"))
        estimate = chords((0, 2, "C"), (2, 4, "G"))
        assert metrics.bar_score(reference, estimate, [0.0, 2.0, 4.0]) == 1.0

    def test_no_scoreable_bar_is_nan(self):
        assert isnan(metrics.bar_score(REFERENCE, REFERENCE, []))
        assert isnan(metrics.bar_score(chords((0, 2, "X")), REFERENCE, [0.0, 2.0]))

    def test_the_vocabulary_decides_what_counts_as_right(self):
        estimate = chords((0, 2, "C:7"), (2, 4, "A:min"), (4, 6, "F"), (6, 8, "G"))
        assert metrics.bar_score(REFERENCE, estimate, DOWNBEATS, "majmin") == 1.0
        assert metrics.bar_score(REFERENCE, estimate, DOWNBEATS, "sevenths") == pytest.approx(0.75)
