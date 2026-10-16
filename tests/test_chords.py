from __future__ import annotations

import numpy as np
import pytest

from songscribe import chords
from songscribe.chords import (
    NO_CHORD,
    PITCH_CLASSES,
    _smooth_labels,
    backend_options,
    chord_templates,
    estimate_key,
    normalise_label,
    recognise,
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


class TestBackendSettings:
    def test_reports_what_each_backend_accepts(self):
        assert backend_options("template") == {"hop_length", "smoothing"}
        # The madmom chord processors are used as pretrained; nothing to tune.
        assert backend_options("madmom") == set()

    def test_no_registered_backend_keeps_a_catch_all(self):
        # A `**kwargs` on a backend would silently switch this validation off
        # for it, which is the bug the check exists to prevent.
        for backend in chords.BACKENDS:
            assert not chords.accepts_any_option(backend)

    def test_a_catch_all_is_reported_apart_from_the_accepted_names(self, monkeypatch):
        # "accepts anything" and "accepts nothing" must not both read as a
        # falsy `backend_options`: the madmom backend genuinely accepts
        # nothing, and conflating the two is how a check gets skipped.
        monkeypatch.setitem(chords.BACKENDS, "fake", lambda audio, rate, smoothing=9, **kw: [])
        monkeypatch.setitem(chords.BACKEND_SAMPLE_RATES, "fake", 8_000)
        assert chords.accepts_any_option("fake")
        assert backend_options("fake") == {"smoothing"}
        assert not chords.accepts_any_option("madmom")
        assert backend_options("madmom") == set()

    def test_a_catch_all_backend_takes_any_setting(self, monkeypatch):
        # The exemption is real, not just reported: nothing can be rejected on
        # behalf of a backend that will accept it.
        monkeypatch.setitem(chords.BACKENDS, "fake", lambda audio, rate, **kw: [])
        monkeypatch.setitem(chords.BACKEND_SAMPLE_RATES, "fake", 8_000)
        assert recognise(np.zeros(10, dtype=np.float32), 8_000, backend="fake", nonsense=1) == []

    def test_a_setting_without_a_default_is_rejected(self, monkeypatch):
        # Not reportable as an accepted setting: the backend could not be
        # called without it, so `recognise` would raise TypeError from the call
        # instead of naming the fault here. Caught on every dispatch, so it
        # surfaces even when the caller names no settings at all. A misdeclared
        # backend is this package's bug, so it is not the caller-facing
        # ValueError -- a CLI mapping that to a usage error must not blame the
        # user for it.
        monkeypatch.setitem(chords.BACKENDS, "fake", lambda audio, rate, checkpoint: [])
        monkeypatch.setitem(chords.BACKEND_SAMPLE_RATES, "fake", 8_000)
        with pytest.raises(RuntimeError, match="'checkpoint' with no default") as excinfo:
            recognise(np.zeros(10, dtype=np.float32), 8_000, backend="fake")
        assert not isinstance(excinfo.value, ValueError)

    def test_a_catch_all_does_not_excuse_a_setting_without_a_default(self, monkeypatch):
        # The exemption covers the list of accepted names only. A backend with
        # both still cannot be called with the audio and its rate alone, so
        # checking the catch-all first would hand the caller the bare TypeError
        # this validation exists to replace.
        monkeypatch.setitem(chords.BACKENDS, "fake", lambda audio, rate, checkpoint, **kw: [])
        monkeypatch.setitem(chords.BACKEND_SAMPLE_RATES, "fake", 8_000)
        assert chords.accepts_any_option("fake")
        with pytest.raises(RuntimeError, match="'checkpoint' with no default"):
            recognise(np.zeros(10, dtype=np.float32), 8_000, backend="fake")

    def test_a_required_positional_only_setting_is_rejected(self, monkeypatch):
        # Unreachable by keyword *and* required: reporting it as "no settings"
        # and then failing at the call is the worst of both. Being callable with
        # the audio and its rate alone is checked before reachability.
        def fake(audio, sample_rate, hop_length, /):
            return []

        monkeypatch.setitem(chords.BACKENDS, "fake", fake)
        monkeypatch.setitem(chords.BACKEND_SAMPLE_RATES, "fake", 8_000)
        with pytest.raises(RuntimeError, match="'hop_length' with no default"):
            backend_options("fake")
        with pytest.raises(RuntimeError, match="'hop_length' with no default"):
            recognise(np.zeros(10, dtype=np.float32), 8_000, backend="fake")

    def test_every_registered_setting_has_a_default(self):
        # The same rule over the shipped backends, so `btc` growing a required
        # `checkpoint` fails here rather than at the call. The guard is that the
        # call itself does not raise -- there is nothing to assert about the
        # value, so don't add an assertion that hides which backend failed.
        for backend in chords.BACKENDS:
            backend_options(backend)

    def test_recognise_rejects_an_unknown_setting(self):
        with pytest.raises(ValueError) as excinfo:
            recognise(np.zeros(10, dtype=np.float32), 22_050, backend="template", smoothng=9)
        message = str(excinfo.value)
        assert "'smoothng'" in message
        assert "'template'" in message
        # The accepted settings are listed, so the typo is obvious from the error.
        assert "smoothing" in message

    def test_rejects_a_setting_before_the_model_is_imported(self):
        # Runs with no weights on disk: the setting is checked at the registry,
        # so madmom is never reached.
        with pytest.raises(ValueError, match="which takes no settings"):
            recognise(np.zeros(10, dtype=np.float32), 44_100, backend="madmom", metre=4)

    def test_names_every_unknown_setting(self):
        with pytest.raises(ValueError, match="unknown settings 'hop', 'smooth'"):
            recognise(np.zeros(10, dtype=np.float32), 22_050, backend="template", hop=1, smooth=2)

    def test_accepted_settings_reach_the_backend(self, monkeypatch):
        seen = {}

        def fake(audio, sample_rate, smoothing=9):
            seen["smoothing"] = smoothing
            return []

        monkeypatch.setitem(chords.BACKENDS, "fake", fake)
        monkeypatch.setitem(chords.BACKEND_SAMPLE_RATES, "fake", 8_000)
        assert recognise(np.zeros(10, dtype=np.float32), 8_000, backend="fake", smoothing=3) == []
        assert seen == {"smoothing": 3}


class TestBackendSettingValues:
    AUDIO = np.zeros(10, dtype=np.float32)

    @pytest.mark.parametrize(
        ("setting", "value", "fault"),
        [
            ("hop_length", 0, "must be at least 1, got 0"),
            ("hop_length", -4, "must be at least 1, got -4"),
            ("smoothing", -4, "must be at least 0, got -4"),
            ("smoothing", np.int64(-1), "must be at least 0, got np.int64(-1)"),
        ]
        + [
            (setting, value, f"must be a whole number of {unit}, got {value!r}")
            for setting, unit in [("hop_length", "samples"), ("smoothing", "frames")]
            for value in [True, False, 2.5, "9", np.float64(9.0), np.bool_(True)]
        ],
    )
    def test_a_template_setting_that_is_not_a_whole_number_in_range_is_refused(
        self, setting, value, fault
    ):
        # A caller fault, so ValueError, raised at the registry before librosa
        # is imported, naming the setting, the backend and what was wrong.
        with pytest.raises(ValueError) as excinfo:
            recognise(self.AUDIO, 22_050, backend="template", **{setting: value})
        assert str(excinfo.value) == f"setting {setting!r} for chord backend 'template' {fault}"

    @staticmethod
    def _capture(monkeypatch):
        seen = {}

        def fake(audio, sample_rate, hop_length=2048, smoothing=9):
            seen.update(hop_length=hop_length, smoothing=smoothing)
            return []

        # Rules are keyed by backend name, so they still apply to the fake.
        monkeypatch.setitem(chords.BACKENDS, "template", fake)
        return seen

    def test_valid_template_settings_reach_the_backend(self, monkeypatch):
        seen = self._capture(monkeypatch)
        recognise(self.AUDIO, 22_050, backend="template", hop_length=1024, smoothing=1)
        assert seen == {"hop_length": 1024, "smoothing": 1}

    def test_smoothing_zero_still_turns_smoothing_off(self, monkeypatch):
        # 0 means "off" to _smooth_labels; only the hop has to be at least 1.
        seen = self._capture(monkeypatch)
        recognise(self.AUDIO, 22_050, backend="template", smoothing=0)
        assert seen["smoothing"] == 0
        flicker = np.array([0, 1, 0, 1, 0])
        np.testing.assert_array_equal(_smooth_labels(flicker, 0), flicker)

    def test_numpy_integers_reach_the_backend(self, monkeypatch):
        # NumPy integers pass the rules and reach the backend as given, not converted.
        seen = self._capture(monkeypatch)
        hop, width = np.int64(2048), np.int32(9)
        recognise(self.AUDIO, 22_050, backend="template", hop_length=hop, smoothing=width)
        assert seen["hop_length"] is hop
        assert seen["smoothing"] is width

    def test_smooth_labels_takes_a_numpy_width(self):
        flicker = np.array([0, 0, 0, 1, 0, 0, 0])
        np.testing.assert_array_equal(_smooth_labels(flicker, np.int64(3)), np.zeros(7))

    def test_every_rule_guards_a_setting_its_backend_declares(self):
        # A renamed parameter must not leave its rule on a dead name.
        for backend, rules in chords.SETTING_RULES.items():
            assert set(rules) <= backend_options(backend), backend


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
