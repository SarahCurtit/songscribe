from __future__ import annotations

from math import isnan

import pytest

from songscribe.eval import metrics
from songscribe.types import ChordEvent, Word

# Invented placeholder words, per the project's rule: real lyrics are
# copyrighted, and this metric never needs them -- it works in word *indices*,
# so the text is arbitrary scaffolding.
WORDS = [
    Word(0.0, 0.4, "lantern"),
    Word(0.5, 0.9, "over"),
    Word(1.0, 1.6, "water"),
    Word(3.1, 3.5, "counting"),
    Word(3.6, 4.0, "every"),
    Word(4.1, 4.8, "window"),
]


def timeline(*spec: tuple[float, float, str]) -> list[ChordEvent]:
    return [ChordEvent(start, end, label) for start, end, label in spec]


def shift(events: list[ChordEvent], offset: float) -> list[ChordEvent]:
    return [ChordEvent(e.start + offset, e.end + offset, e.label) for e in events]


@pytest.fixture
def reference() -> list[ChordEvent]:
    """One chord per line, changing on the first word of each."""
    return timeline((0.0, 3.0, "C:maj"), (3.1, 5.0, "G:maj"))


class TestPlacementSequence:
    def test_it_reports_line_and_word_indices(self, reference):
        placed = metrics.placement_sequence(reference, WORDS)
        assert placed == ((0, 0, "C:maj"), (1, 0, "G:maj"))

    def test_a_mid_line_change_lands_on_the_word_being_sung(self):
        # The chord changes while "water" (index 2) is sounding.
        chords = timeline((0.0, 1.0, "C:maj"), (1.0, 3.0, "A:min"))
        placed = metrics.placement_sequence(chords, WORDS)
        assert placed == ((0, 0, "C:maj"), (0, 2, "A:min"))

    def test_labels_can_be_dropped(self, reference):
        placed = metrics.placement_sequence(reference, WORDS, compare_labels=False)
        assert placed == ((0, 0), (1, 0))

    def test_no_words_means_no_placements(self, reference):
        assert metrics.placement_sequence(reference, []) == ()

    def test_no_chords_means_no_placements(self):
        assert metrics.placement_sequence([], WORDS) == ()


class TestPlacementScore:
    def test_identical_chords_score_one(self, reference):
        assert metrics.placement_score(reference, reference, WORDS) == 1.0

    def test_a_chord_on_the_wrong_syllable_costs(self, reference):
        # Same two chords, but the second change arrives a word late.
        late = timeline((0.0, 3.6, "C:maj"), (3.6, 5.0, "G:maj"))
        assert metrics.placement_score(reference, late, WORDS) < 1.0

    def test_the_right_chord_on_the_right_syllable_is_what_counts(self, reference):
        wrong_label = timeline((0.0, 3.0, "C:maj"), (3.1, 5.0, "F:maj"))
        # In the right place, so the label-free comparison is perfect...
        assert metrics.placement_score(reference, wrong_label, WORDS, compare_labels=False) == 1.0
        # ...but it is still the wrong chord.
        assert metrics.placement_score(reference, wrong_label, WORDS) < 1.0

    def test_an_empty_estimate_scores_zero(self, reference):
        assert metrics.placement_score(reference, [], WORDS) == 0.0

    def test_two_empty_sides_are_nan_not_perfect(self):
        assert isnan(metrics.placement_score([], [], WORDS))


class TestPlacementIgnoresWhatAMusicianIgnores:
    """Timing only matters here through which syllable a chord lands on."""

    @pytest.mark.parametrize("offset", [0.05, 0.1])
    def test_a_small_shift_that_keeps_every_syllable_is_free(self, reference, offset):
        # Chords still change during the same words, so the chart is identical.
        assert metrics.placement_score(reference, shift(reference, offset), WORDS) == 1.0

    def test_a_shift_large_enough_to_cross_a_syllable_does_cost(self, reference):
        # Not timing-blind: once the change moves onto the next word, the
        # printed chart genuinely differs and the score must say so.
        assert metrics.placement_score(reference, shift(reference, 1.0), WORDS) < 1.0

    def test_a_chord_spilling_over_a_line_break_displaces_the_next_one(self, reference):
        """Pins a real ``align.py`` behaviour this metric surfaces.

        A 0.2s shift spills the first chord's *end* past line 2's first word, and
        ``place_in_line``'s "two changes on one syllable, keep the first" rule then
        drops line 2's real chord -- printed wrong, with no sign of the right one.
        """
        spilled = metrics.placement_sequence(shift(reference, 0.2), WORDS)
        assert spilled == ((0, 0, "C:maj"), (1, 0, "C:maj"))
        assert "G:maj" not in [label for _, _, label in spilled]
        # Which the score reports, rather than smoothing over.
        assert metrics.placement_score(reference, shift(reference, 0.2), WORDS) == 0.5

    def test_both_sides_share_the_reference_word_timeline(self, reference):
        # The signature takes one `words`, not two, so the chord branch is
        # measured without the ASR's errors leaking in.
        import inspect

        parameters = inspect.signature(metrics.placement_score).parameters
        assert "words" in parameters
        assert "estimate_words" not in parameters
