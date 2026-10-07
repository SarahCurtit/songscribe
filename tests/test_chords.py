from __future__ import annotations

import numpy as np

from songscribe.chords import PITCH_CLASSES, chord_templates, estimate_key
from songscribe.types import ChordEvent


class TestChordTemplates:
    def test_covers_every_major_and_minor_triad(self):
        _, labels = chord_templates()
        assert len(labels) == 24
        assert len(set(labels)) == 24
        for name in PITCH_CLASSES:
            assert name in labels
            assert f"{name}m" in labels

    def test_templates_are_unit_normalised(self):
        templates, _ = chord_templates()
        assert templates.shape == (24, 12)
        np.testing.assert_allclose(np.linalg.norm(templates, axis=1), 1.0)

    def test_each_triad_activates_exactly_three_pitch_classes(self):
        templates, _ = chord_templates()
        assert np.all(np.count_nonzero(templates, axis=1) == 3)

    def test_a_clean_triad_matches_its_own_template_best(self):
        templates, labels = chord_templates()
        # A perfect A minor chroma frame: A, C, E.
        frame = np.zeros(12)
        for pitch_class in ("A", "C", "E"):
            frame[PITCH_CLASSES.index(pitch_class)] = 1.0
        frame /= np.linalg.norm(frame)
        assert labels[int(np.argmax(templates @ frame))] == "Am"


class TestEstimateKey:
    def test_picks_the_longest_held_chord(self):
        chords = [
            ChordEvent(0.0, 8.0, "G"),
            ChordEvent(8.0, 10.0, "C"),
            ChordEvent(10.0, 12.0, "D"),
        ]
        assert estimate_key(chords) == "G"

    def test_sums_time_across_repeats(self):
        chords = [
            ChordEvent(0.0, 3.0, "Em"),
            ChordEvent(3.0, 7.0, "A"),
            ChordEvent(7.0, 11.0, "Em"),
        ]
        assert estimate_key(chords) == "Em"

    def test_returns_none_without_chords(self):
        assert estimate_key([]) is None
