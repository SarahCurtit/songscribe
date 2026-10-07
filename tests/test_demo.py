from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from songscribe.demo import (
    DEFAULT_PROGRESSION,
    chord_frequencies,
    parse_chord,
    synthesise,
    write_annotation,
)


class TestParseChord:
    @pytest.mark.parametrize(
        ("label", "root", "intervals"),
        [
            ("C", 0, (0, 4, 7)),
            ("Am", 9, (0, 3, 7)),
            ("F#m", 6, (0, 3, 7)),
            ("G7", 7, (0, 4, 7, 10)),
            ("Dm7", 2, (0, 3, 7, 10)),
            ("Cmaj7", 0, (0, 4, 7, 11)),
            ("Bdim", 11, (0, 3, 6)),
        ],
    )
    def test_splits_root_from_quality(self, label, root, intervals):
        assert parse_chord(label) == (root, intervals)

    def test_accepts_flats(self):
        # Bb and A# are the same pitch class; the module writes sharps.
        assert parse_chord("Bb") == parse_chord("A#")
        assert parse_chord("Ebm") == parse_chord("D#m")

    def test_two_character_roots_win_over_one(self):
        assert parse_chord("C#")[0] == 1
        assert parse_chord("C")[0] == 0

    def test_tolerates_surrounding_whitespace(self):
        assert parse_chord("  Am  ") == parse_chord("Am")

    @pytest.mark.parametrize("label", ["", "   ", "H", "Cm11", "Xyz", "C:maj"])
    def test_rejects_what_it_cannot_render(self, label):
        with pytest.raises(ValueError):
            parse_chord(label)


class TestChordFrequencies:
    def test_voices_the_root_in_the_requested_octave(self):
        # A3 is 220 Hz by definition, being an octave below A4 = 440.
        assert chord_frequencies("Am", octave=3)[0] == pytest.approx(220.0)

    def test_c3_matches_the_standard_table(self):
        assert chord_frequencies("C", octave=3)[0] == pytest.approx(130.81, abs=0.01)

    def test_one_frequency_per_interval(self):
        assert len(chord_frequencies("C")) == 3
        assert len(chord_frequencies("C7")) == 4

    def test_a_major_triad_is_root_major_third_fifth(self):
        root, third, fifth = chord_frequencies("C", octave=3)
        assert third / root == pytest.approx(2 ** (4 / 12))
        assert fifth / root == pytest.approx(2 ** (7 / 12))

    def test_octave_doubles_the_frequency(self):
        low = chord_frequencies("C", octave=3)[0]
        high = chord_frequencies("C", octave=4)[0]
        assert high / low == pytest.approx(2.0)


class TestSynthesise:
    def test_duration_follows_tempo_and_bar_count(self):
        # 4 beats at 96 bpm = 2.5s per bar; 4 chords twice = 8 bars = 20s.
        audio, _events = synthesise(bpm=96.0, beats_per_bar=4, repeats=2, sample_rate=22_050)
        assert len(audio) / 22_050 == pytest.approx(20.0, abs=0.01)

    def test_one_event_per_bar_in_order(self):
        _audio, events = synthesise(progression=("C", "G"), repeats=3)
        assert [e.label for e in events] == ["C", "G", "C", "G", "C", "G"]

    def test_events_are_contiguous_and_cover_the_audio(self):
        audio, events = synthesise(sample_rate=22_050)
        for earlier, later in pairwise(events):
            assert earlier.end == pytest.approx(later.start)
        assert events[-1].end == pytest.approx(len(audio) / 22_050, abs=0.01)

    def test_returns_mono_float32(self):
        audio, _events = synthesise(sample_rate=22_050)
        assert audio.ndim == 1
        assert audio.dtype == np.float32

    def test_leaves_headroom_so_16_bit_output_cannot_clip(self):
        audio, _events = synthesise(sample_rate=22_050, noise=0.05)
        assert np.max(np.abs(audio)) < 1.0

    def test_is_deterministic_including_noise(self):
        first, _ = synthesise(sample_rate=22_050, noise=0.01)
        second, _ = synthesise(sample_rate=22_050, noise=0.01)
        np.testing.assert_array_equal(first, second)

    def test_default_progression_is_in_the_majmin_vocabulary(self):
        # The default backend only knows major and minor triads; the shipped
        # demo must not ask it for anything it cannot label.
        for label in DEFAULT_PROGRESSION:
            _root, intervals = parse_chord(label)
            assert intervals in {(0, 4, 7), (0, 3, 7)}

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"progression": ()},
            {"bpm": 0},
            {"bpm": -5},
            {"repeats": 0},
            {"beats_per_bar": 0},
            {"sample_rate": 0},
            {"noise": -1.0},
        ],
    )
    def test_rejects_degenerate_arguments(self, kwargs):
        # Every one of these otherwise reaches numpy and surfaces as an opaque
        # "zero-size array to reduction operation" instead of naming the arg.
        with pytest.raises(ValueError):
            synthesise(**kwargs)


class TestWriteAnnotation:
    def test_writes_tab_separated_start_end_label(self, tmp_path):
        _audio, events = synthesise(progression=("C", "G"), repeats=1)
        path = tmp_path / "demo.lab"
        write_annotation(path, events)

        rows = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(rows) == 2
        for row, event in zip(rows, events, strict=True):
            start, end, label = row.split("\t")
            assert float(start) == pytest.approx(event.start)
            assert float(end) == pytest.approx(event.end)
            assert label == event.label

    def test_ends_with_a_single_newline(self, tmp_path):
        _audio, events = synthesise(progression=("C",), repeats=1)
        path = tmp_path / "demo.lab"
        write_annotation(path, events)
        text = path.read_text(encoding="utf-8")
        assert text.endswith("\n")
        assert not text.endswith("\n\n")
