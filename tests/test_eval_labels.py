from __future__ import annotations

import numpy as np
import pytest

from songscribe.eval import harte, labels
from songscribe.types import ChordEvent

BEAT = 0.5


def edges(beats: int, start: float = 0.0) -> np.ndarray:
    return start + BEAT * np.arange(beats + 1)


def events(*spans: tuple[float, float, str]) -> list[ChordEvent]:
    return [ChordEvent(a, b, label) for a, b, label in spans]


class TestSymbol:
    @pytest.mark.parametrize("same", [("C", "C:maj", "C7", "Cmaj7", "C9", "Csus4", "B#")])
    def test_a_root_with_a_major_third_or_none_is_one_symbol(self, same):
        assert len({labels.symbol(label) for label in same}) == 1

    def test_any_minor_third_makes_the_minor_symbol(self):
        minors = {labels.symbol(label) for label in ("Cm", "C:min7", "Cm7b5", "Cdim", "C:hdim7")}
        assert len(minors) == 1
        assert minors != {labels.symbol("C")}

    def test_enharmonic_spellings_are_the_same_symbol(self):
        assert labels.symbol("Bb") == labels.symbol("A#") == labels.symbol("Bb:7")

    @pytest.mark.parametrize("label", ["N", "X", "", "not a chord"])
    def test_no_chord_and_unreadable_labels_have_no_symbol(self, label):
        assert labels.symbol(label) == labels.NO_SYMBOL


class TestBeatSymbols:
    def test_each_beat_takes_the_chord_sounding_for_most_of_it(self):
        out = labels.beat_symbols(events((0, 1, "C"), (1, 2, "G")), edges(4))
        assert list(out) == [labels.symbol("C")] * 2 + [labels.symbol("G")] * 2

    def test_a_chord_that_arrives_a_little_early_does_not_change_the_beat_before(self):
        out = labels.beat_symbols(events((0, 0.9, "C"), (0.9, 2, "G")), edges(4))
        assert list(out) == [labels.symbol("C")] * 2 + [labels.symbol("G")] * 2

    def test_a_flicker_inside_a_beat_is_outvoted(self):
        flicker = events((0, 0.1, "C"), (0.1, 0.2, "F#"), (0.2, 1.0, "C"))
        assert list(labels.beat_symbols(flicker, edges(2))) == [labels.symbol("C")] * 2

    def test_an_exact_tie_goes_to_the_chord_that_arrived_last(self):
        assert labels.beat_symbols(events((0, 0.25, "C"), (0.25, 0.5, "G")), edges(1))[0] == (
            labels.symbol("G")
        )

    def test_a_beat_that_is_mostly_no_chord_or_uncovered_has_no_symbol(self):
        out = labels.beat_symbols(events((0, 0.1, "C"), (0.1, 1.0, "N")), edges(2))
        assert list(out) == [labels.NO_SYMBOL] * 2
        assert labels.beat_symbols([], edges(3)).tolist() == [labels.NO_SYMBOL] * 3

    def test_events_out_of_order_and_past_the_grid_are_fine(self):
        out = labels.beat_symbols(events((5, 6, "C"), (0, 1, "G")), edges(2))
        assert list(out) == [labels.symbol("G")] * 2


class TestAgreement:
    def test_equal_symbols_agree_the_same_root_in_part_and_nothing_else_not_at_all(self):
        c, cm, g = (np.array([labels.symbol(x)]) for x in ("C", "Cm", "G"))
        assert labels.agreement(c, c)[0] == 1.0
        assert labels.agreement(c, cm)[0] == labels.ROOT_CREDIT
        assert labels.agreement(c, g)[0] == 0.0

    def test_beats_without_a_chord_agree_with_nothing_not_even_each_other(self):
        none = np.array([labels.NO_SYMBOL])
        assert labels.agreement(none, none)[0] == 0.0
        assert labels.agreement(none, np.array([labels.symbol("C")]))[0] == 0.0


class TestTranspose:
    @pytest.mark.parametrize(
        ("label", "semitones", "expected"),
        [
            ("C", 2, "D"),
            ("Bb:min7", 3, "C#:min7"),
            ("Am", -2, "Gm"),
            ("G7/b7", 5, "C7/b7"),
            ("C", 12, "C"),
            ("N", 4, "N"),
            ("X", 4, "X"),
        ],
    )
    def test_the_root_moves_and_the_rest_stays(self, label, semitones, expected):
        assert harte.transpose_label(label, semitones) == expected

    def test_an_unreadable_label_comes_back_untouched(self):
        assert harte.transpose_label("not a chord", 3) == "not a chord"
