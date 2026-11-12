"""The melody stage: the registry contract, the segmenter, and pYIN end to end.

The segmenter is handed invented f0 contours, so it is tested without audio at
all. The one test that does run a tracker runs it on the demo, which is the
only audio this repo may contain -- and on librosa alone, so it runs in CI.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest

from songscribe import melody
from songscribe.demo import synthesise_performance
from songscribe.melody import (
    MIN_NOTE_SECONDS,
    backend_options,
    hz_to_midi,
    midi_to_hz,
    notes_from_f0,
    transcribe,
)

HOP = 0.01


def contour(pitches, hop: float = HOP):
    """A frame-wise f0 contour from MIDI pitches; ``None`` is an unvoiced frame."""
    times = np.arange(len(pitches)) * hop
    f0 = np.array([np.nan if p is None else float(midi_to_hz(p)) for p in pitches])
    return times, f0, np.array([p is not None for p in pitches])


class TestPitchConversion:
    def test_a440_is_midi_69(self):
        assert hz_to_midi(440.0) == pytest.approx(69.0)
        assert midi_to_hz(69) == pytest.approx(440.0)

    def test_an_octave_is_twelve_semitones(self):
        assert hz_to_midi(880.0) == pytest.approx(81.0)

    def test_the_two_are_inverses(self):
        pitches = np.arange(21, 109)
        np.testing.assert_allclose(hz_to_midi(midi_to_hz(pitches)), pitches, atol=1e-9)


class TestNotesFromF0:
    def test_a_held_pitch_is_one_note(self):
        notes = notes_from_f0(*contour([72] * 50))
        assert len(notes) == 1
        assert notes[0].pitch == 72
        assert (notes[0].start, notes[0].end) == pytest.approx((0.0, 0.5))

    def test_a_step_starts_a_new_note(self):
        notes = notes_from_f0(*contour([72] * 20 + [74] * 20))
        assert [n.pitch for n in notes] == [72, 74]
        assert notes[0].end == pytest.approx(notes[1].start)

    def test_unvoiced_frames_end_the_note(self):
        notes = notes_from_f0(*contour([72] * 20 + [None] * 20 + [72] * 20))
        assert [n.pitch for n in notes] == [72, 72]
        assert notes[0].end == pytest.approx(0.2)
        assert notes[1].start == pytest.approx(0.4)

    def test_vibrato_inside_the_tolerance_stays_one_note(self):
        # +/- 25 cents, which a singer does without changing note.
        swing = [72.25 if frame % 2 else 71.75 for frame in range(40)]
        notes = notes_from_f0(*contour(swing))
        assert len(notes) == 1
        assert notes[0].pitch == 72

    def test_a_swing_wider_than_the_tolerance_splits(self):
        # Not a semitone -- 90 cents, well short of one -- but past the window,
        # so it is two notes. The rule is the tolerance, not the semitone.
        notes = notes_from_f0(*contour([72.0] * 20 + [72.9] * 20))
        assert [n.pitch for n in notes] == [72, 73]

    def test_a_note_shorter_than_the_minimum_is_dropped(self):
        frames = max(1, int(MIN_NOTE_SECONDS / HOP) - 2)
        assert notes_from_f0(*contour([72] * frames)) == []

    def test_min_duration_is_the_caller_s_to_set(self):
        notes = notes_from_f0(*contour([72, 72]), min_duration=0.01)
        assert [n.pitch for n in notes] == [72]

    def test_a_note_played_slightly_sharp_is_still_that_note(self):
        notes = notes_from_f0(*contour([72.3] * 20))
        assert [n.pitch for n in notes] == [72]

    def test_the_pitch_is_the_median_not_the_mean(self):
        # Fourteen frames at 72.59 and six at 72.0: the mean is 72.41 and
        # would round down, the median is 72.59 and rounds to where the note
        # actually sat. Every frame is inside the window, so this is one note.
        notes = notes_from_f0(*contour([72.0] * 6 + [72.59] * 14))
        assert [n.pitch for n in notes] == [73]

    def test_zero_hertz_counts_as_unvoiced(self):
        times, f0, voiced = contour([72] * 20 + [72] * 20)
        f0[20:] = 0.0
        notes = notes_from_f0(times, f0, voiced)
        assert [n.pitch for n in notes] == [72]
        assert notes[0].end == pytest.approx(0.2)

    def test_confidence_is_averaged_over_the_note(self):
        times, f0, voiced = contour([72] * 20)
        notes = notes_from_f0(times, f0, voiced, confidence=np.linspace(0.0, 1.0, 20))
        assert notes[0].confidence == pytest.approx(0.5)

    def test_without_confidence_a_note_is_certain(self):
        assert notes_from_f0(*contour([72] * 20))[0].confidence == 1.0

    def test_a_dropped_frame_does_not_stretch_the_note(self):
        # Times with a hole in them: the note ends one hop after its last
        # frame, not where the contour resumes.
        times, f0, voiced = contour([72] * 20)
        times = times + np.where(np.arange(20) >= 10, 1.0, 0.0)
        notes = notes_from_f0(times, f0, voiced)
        assert notes[0].end == pytest.approx(0.1)

    def test_mismatched_lengths_are_refused(self):
        times, f0, voiced = contour([72] * 20)
        with pytest.raises(ValueError, match="same length"):
            notes_from_f0(times[:5], f0, voiced)

    def test_a_confidence_of_the_wrong_length_is_refused(self):
        times, f0, voiced = contour([72] * 20)
        with pytest.raises(ValueError, match="one value per frame"):
            notes_from_f0(times, f0, voiced, confidence=np.zeros(3))

    def test_an_empty_contour_gives_no_notes(self):
        assert notes_from_f0(np.array([]), np.array([]), np.array([], dtype=bool)) == []

    def test_a_nonsense_tolerance_is_refused(self):
        with pytest.raises(ValueError, match="tolerance_cents must be positive"):
            notes_from_f0(*contour([72] * 20), tolerance_cents=0)


class TestBackendSettings:
    def test_reports_what_the_backend_accepts(self):
        assert backend_options("pyin") == {
            "fmin",
            "fmax",
            "frame_length",
            "hop_length",
            "min_duration",
            "tolerance_cents",
        }

    def test_no_registered_backend_keeps_a_catch_all(self):
        # A `**kwargs` would switch the setting check off for that backend.
        for backend in melody.BACKENDS:
            assert not melody.accepts_any_option(backend)

    def test_every_registered_setting_has_a_default(self):
        for backend in melody.BACKENDS:
            backend_options(backend)

    def test_every_backend_declares_a_sample_rate(self):
        assert set(melody.BACKENDS) == set(melody.BACKEND_SAMPLE_RATES)

    def test_an_unknown_backend_raises_rather_than_falling_back(self):
        with pytest.raises(ValueError, match="unknown melody backend 'crepe'"):
            transcribe(np.zeros(10, dtype=np.float32), 22_050, backend="crepe")

    def test_an_unknown_setting_raises(self):
        with pytest.raises(ValueError) as excinfo:
            transcribe(np.zeros(10, dtype=np.float32), 22_050, fmni=100.0)
        assert "'fmni'" in str(excinfo.value)
        assert "fmin" in str(excinfo.value)

    def test_a_setting_value_is_checked_before_any_audio_is_touched(self):
        with pytest.raises(ValueError, match="setting 'fmin' for melody backend 'pyin'"):
            transcribe(np.zeros(10, dtype=np.float32), 22_050, fmin=0)
        with pytest.raises(ValueError, match="setting 'hop_length'"):
            transcribe(np.zeros(10, dtype=np.float32), 22_050, hop_length=1.5)

    def test_a_range_the_wrong_way_round_is_refused(self):
        with pytest.raises(ValueError, match="fmin must be below fmax"):
            transcribe(np.zeros(1024, dtype=np.float32), 22_050, fmin=800.0, fmax=200.0)


#: 160 bpm and one chorus keeps pYIN near a second, and the hop is widened for
#: the same reason: 46 ms frames still give eight to a note at this tempo.
DEMO_BPM = 160.0
DEMO_HOP = 1024


@lru_cache(maxsize=1)
def tracked_demo():
    """The demo's lead line, and what pYIN makes of it. Cached: it is the slow test."""
    take = synthesise_performance(bpm=DEMO_BPM, repeats=1, sample_rate=22_050, melody=True)
    return take, transcribe(take.audio, take.sample_rate, hop_length=DEMO_HOP)


class TestPyinOnTheDemo:
    """The one end-to-end test: synthetic audio in, notes out, no weights."""

    def test_the_written_line_comes_back_note_for_note(self):
        take, found = tracked_demo()
        assert [n.pitch for n in found] == [n.pitch for n in take.notes]

    def test_the_onsets_land_within_a_fraction_of_a_beat(self):
        take, found = tracked_demo()
        beat = 60.0 / DEMO_BPM
        for note, played in zip(found, take.notes, strict=True):
            assert note.start == pytest.approx(played.start, abs=0.25 * beat)

    def test_the_tracker_reports_a_confidence_it_did_not_invent(self):
        _, found = tracked_demo()
        assert all(0.0 <= n.confidence <= 1.0 for n in found)
        assert any(n.confidence < 1.0 for n in found)

    def test_silence_yields_no_notes(self):
        assert transcribe(np.zeros(11_025, dtype=np.float32), 22_050, hop_length=DEMO_HOP) == []


class TestHelpers:
    def test_transposing_moves_every_pitch(self):
        take = synthesise_performance(repeats=1, sample_rate=8_000, melody=True)
        moved = melody.transpose(take.reference_notes, 3)
        assert [n.pitch for n in moved] == [n.pitch + 3 for n in take.reference_notes]
        assert [n.start for n in moved] == [n.start for n in take.reference_notes]

    def test_the_pitch_profile_is_duration_weighted(self):
        from songscribe.types import NoteEvent

        profile = melody.pitch_profile(
            [NoteEvent(0.0, 2.0, 60), NoteEvent(2.0, 2.5, 64), NoteEvent(3.0, 3.5, 72)]
        )
        assert profile[0] == pytest.approx(2.5)  # C4 and C5 are one pitch class
        assert profile[4] == pytest.approx(0.5)
        assert profile.sum() == pytest.approx(3.0)
