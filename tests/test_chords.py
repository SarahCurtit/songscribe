from __future__ import annotations

import numpy as np
import pytest

from songscribe.chords import (
    NO_CHORD,
    PITCH_CLASSES,
    _smooth_labels,
    chord_templates,
    estimate_key,
    normalise_label,
    sample_rate_for,
)
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


class TestSmoothLabels:
    def test_removes_a_single_frame_flicker(self):
        indices = np.array([3, 3, 3, 17, 3, 3, 3])
        assert list(_smooth_labels(indices, 3)) == [3, 3, 3, 3, 3, 3, 3]

    def test_never_invents_a_label_absent_from_the_window(self):
        # A median filter would answer 10 here -- a chord neither neighbour
        # played. Mode can only ever return a label that actually occurred.
        indices = np.array([0, 0, 0, 20, 20, 20])
        smoothed = _smooth_labels(indices, 3)
        assert set(smoothed) <= {0, 20}

    def test_preserves_a_genuine_change(self):
        indices = np.array([5] * 6 + [9] * 6)
        smoothed = _smooth_labels(indices, 3)
        assert smoothed[0] == 5
        assert smoothed[-1] == 9

    def test_length_is_unchanged(self):
        indices = np.array([1, 2, 1, 2, 1, 2, 1])
        for width in (1, 3, 4, 5, 9):
            assert len(_smooth_labels(indices, width)) == len(indices)

    def test_width_one_is_a_no_op(self):
        indices = np.array([4, 11, 4])
        assert list(_smooth_labels(indices, 1)) == [4, 11, 4]

    def test_empty_input(self):
        assert len(_smooth_labels(np.array([], dtype=int), 9)) == 0


class TestNormaliseLabel:
    @pytest.mark.parametrize(
        ("harte", "compact"),
        [
            ("C:maj", "C"),
            ("A:min", "Am"),
            ("F#:min", "F#m"),
            ("Bb:maj", "Bb"),
            ("G:7", "G7"),
            ("C:maj7", "Cmaj7"),
            ("D:min7", "Dm7"),
            ("B:hdim7", "Bm7b5"),
            ("E:sus4", "Esus4"),
        ],
    )
    def test_converts_harte_to_chart_shorthand(self, harte, compact):
        assert normalise_label(harte) == compact

    def test_drops_the_inversion(self):
        assert normalise_label("C:maj/3") == "C"
        assert normalise_label("A:min/5") == "Am"

    def test_no_chord_and_unknown_collapse_to_n(self):
        assert normalise_label("N") == NO_CHORD
        assert normalise_label("X") == NO_CHORD
        assert normalise_label("  ") == NO_CHORD

    def test_already_compact_labels_pass_through(self):
        # The template backend emits these directly; normalising twice is safe.
        for label in ("C", "Am", "F#m", "Bb"):
            assert normalise_label(label) == label

    def test_unrecognised_quality_is_not_guessed_at(self):
        assert normalise_label("C:min11") == "C:min11"

    def test_output_is_idempotent(self):
        for harte in ("C:maj", "A:min", "G:7", "C:maj/3", "X"):
            once = normalise_label(harte)
            assert normalise_label(once) == once


class TestSampleRateFor:
    def test_returns_the_rate_each_backend_wants(self):
        assert sample_rate_for("madmom") == 44_100
        assert sample_rate_for("template") == 22_050

    def test_rejects_an_unknown_backend(self):
        with pytest.raises(ValueError, match="unknown chord backend"):
            sample_rate_for("definitely-not-a-model")


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
