from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from songscribe.demo import (
    DEFAULT_PROGRESSION,
    chord_frequencies,
    main,
    parse_chord,
    synthesise,
    synthesise_performance,
    write_annotation,
)

SR = 8_000
PROGRESSION = ("C", "Am", "F", "G")


class TestSynthesisePerformance:
    def take(self, **kwargs):
        defaults = {"progression": PROGRESSION, "bpm": 120.0, "repeats": 3, "sample_rate": SR}
        return synthesise_performance(**{**defaults, **kwargs})

    def test_the_plain_take_matches_synthesise(self):
        # With no extra options, synthesise_performance must give the same audio as synthesise.
        audio, _ = synthesise(progression=PROGRESSION, bpm=120.0, repeats=3, sample_rate=SR)
        np.testing.assert_allclose(self.take().audio, audio, atol=1e-6)

    def test_a_pickup_is_the_last_chord_ending_on_the_first_downbeat(self):
        take = self.take(pickup_beats=3)
        assert take.downbeats[0] == pytest.approx(1.5)  # 3 beats at 120 bpm
        first = take.events[0]
        assert (first.start, first.label) == (0.0, PROGRESSION[-1])
        assert first.end == pytest.approx(take.downbeats[0])
        assert take.events[1].label == PROGRESSION[0]
        assert len(take.downbeats) == 3 * 4 + 1

    def test_a_pickup_is_not_in_the_chart_and_adds_no_bar(self):
        plain, with_pickup = self.take(), self.take(pickup_beats=3)
        assert with_pickup.reference == plain.reference
        assert with_pickup.downbeats == pytest.approx([d + 1.5 for d in plain.downbeats])

    def test_the_pickup_beats_are_listed_before_bar_zero(self):
        beats = self.take(pickup_beats=3).beat_events()
        assert [b.position for b in beats[:5]] == [2, 3, 4, 1, 2]
        assert [b.time for b in beats[:4]] == pytest.approx([0.0, 0.5, 1.0, 1.5])

    def test_a_pickup_keeps_the_first_bars_tempo_under_drift(self):
        take = self.take(pickup_beats=3, tempo_drift=0.2)
        assert take.downbeats[0] == pytest.approx(1.5)

    def test_drift_shortens_bars_progressively(self):
        take = self.take(tempo_drift=0.2)
        lengths = [b - a for a, b in pairwise(take.downbeats)]
        assert lengths == sorted(lengths, reverse=True)
        assert lengths[0] / lengths[-1] == pytest.approx(1.2, abs=0.01)

    def test_a_substitution_changes_one_bar_of_one_chorus(self):
        take = self.take(substitutions={1: {2: "Dm"}})
        labels = [e.label for e in take.events]
        assert labels[4 + 2] == "Dm"
        assert labels.count("Dm") == 1
        assert [e.label for e in take.reference] == list(PROGRESSION)

    def test_the_reference_is_one_chorus_from_zero(self):
        take = self.take(pickup_beats=3, tempo_drift=0.1)
        assert take.reference[0].start == 0.0
        assert take.reference[-1].end == pytest.approx(4 * 2.0)
        assert take.bars_per_chorus == 4

    def test_the_beats_close_the_last_bar(self):
        take = self.take()
        beats = take.beat_events()
        assert [b.position for b in beats[:5]] == [1, 2, 3, 4, 1]
        assert beats[-1].time == take.downbeats[-1]
        assert sum(b.position == 1 for b in beats) == len(take.downbeats)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"pickup_beats": -1},
            {"pickup_beats": 4},
            {"tempo_drift": -1.0},
            {"substitutions": {5: {0: "C"}}},
            {"substitutions": {0: {9: "C"}}},
            {"bpm": 0},
        ],
    )
    def test_rejects_degenerate_arguments(self, kwargs):
        with pytest.raises(ValueError):
            self.take(**kwargs)

    def test_the_command_line_writes_audio_chart_and_beats(self, tmp_path):
        out = tmp_path / "demo.wav"
        argv = ["-o", str(out), "--sample-rate", str(SR), "--pickup-beats", "3", "-r", "2"]
        assert main([*argv, "--vary", "1:0=Em"]) == 0
        assert out.exists()
        assert (tmp_path / "demo.lab").exists()
        assert (tmp_path / "demo.reference.lab").exists()
        first_beat = (tmp_path / "demo.beats").read_text(encoding="utf-8").splitlines()[0]
        assert float(first_beat.split()[0]) == 0.0
        assert first_beat.split()[1] == "2"
        assert "Em" in (tmp_path / "demo.lab").read_text(encoding="utf-8")

    def test_a_malformed_vary_is_reported(self, tmp_path, capsys):
        assert main(["-o", str(tmp_path / "d.wav"), "--vary", "nonsense"]) == 1
        assert "--vary" in capsys.readouterr().err


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
