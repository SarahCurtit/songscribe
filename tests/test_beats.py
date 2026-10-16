from __future__ import annotations

import sys
import types
from dataclasses import replace
from math import isnan

import numpy as np
import pytest

from songscribe import beats
from songscribe.beats import BeatGrid
from songscribe.types import BeatEvent

from .conftest import BALLAD_BPM, click_track, metronome


class TestBeatGrid:
    def test_complete_bars_are_one_fewer_than_downbeats(self):
        # The last downbeat opens a bar whose end was never seen.
        grid = BeatGrid(downbeats=(0.0, 2.0, 4.0, 6.0))
        assert grid.bars == 3

    def test_no_downbeats_means_no_bars_and_no_length(self):
        grid = BeatGrid(downbeats=())
        assert grid.bars == 0
        assert isnan(grid.bar_length)

    def test_one_downbeat_has_no_measurable_length(self):
        assert isnan(BeatGrid(downbeats=(1.5,)).bar_length)

    def test_bar_length_is_the_median_not_the_mean(self):
        # One dropped bar doubles an interval. The mean would read 2.5s; the
        # median stays on the tempo the band is actually playing.
        grid = BeatGrid(downbeats=(0.0, 2.0, 4.0, 8.0, 10.0))
        assert grid.bar_length == 2.0

    def test_bar_spans_tile_the_grid(self):
        grid = BeatGrid(downbeats=(0.0, 2.0, 4.0, 6.0))
        assert [grid.bar_span(i) for i in range(grid.bars)] == [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0)]

    def test_bar_span_rejects_an_incomplete_bar(self):
        grid = BeatGrid(downbeats=(0.0, 2.0))
        with pytest.raises(IndexError, match="outside the grid"):
            grid.bar_span(1)

    def test_bar_index_at_finds_the_containing_bar(self):
        grid = BeatGrid(downbeats=(10.0, 12.0, 14.0, 16.0))
        assert grid.bar_index_at(10.0) == 0
        assert grid.bar_index_at(11.9) == 0
        assert grid.bar_index_at(12.0) == 1
        assert grid.bar_index_at(15.5) == 2

    def test_time_before_the_first_downbeat_is_outside_the_grid(self):
        # An intro, stage noise or a count-in is not in bar 1, and saying so is
        # the whole point of anchoring on a downbeat.
        grid = BeatGrid(downbeats=(10.0, 12.0, 14.0))
        assert grid.bar_index_at(0.0) is None
        assert grid.bar_index_at(9.999) is None

    def test_time_past_the_last_complete_bar_is_outside_the_grid(self):
        grid = BeatGrid(downbeats=(0.0, 2.0, 4.0))
        assert grid.bar_index_at(4.0) is None
        assert grid.bar_index_at(99.0) is None

    def test_empty_grid_contains_no_time(self):
        assert BeatGrid(downbeats=()).bar_index_at(1.0) is None

    def test_rejects_downbeats_out_of_order(self):
        with pytest.raises(ValueError, match="strictly ascending"):
            BeatGrid(downbeats=(0.0, 4.0, 2.0))

    def test_rejects_duplicate_downbeats(self):
        with pytest.raises(ValueError, match="strictly ascending"):
            BeatGrid(downbeats=(0.0, 2.0, 2.0))

    def test_rejects_a_metre_below_one(self):
        with pytest.raises(ValueError, match="beats_per_bar"):
            BeatGrid(downbeats=(0.0, 2.0), beats_per_bar=0)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_a_non_finite_downbeat(self, bad):
        # nan passes every ordering test, since comparing against it is always
        # False, so without this check it reaches a grid whose bar lengths are
        # all nan and whose unreliable_bars() reports nothing wrong.
        with pytest.raises(ValueError, match="finite"):
            BeatGrid(downbeats=(0.0, bad, 2.0))

    def test_bar_start_rejects_an_index_outside_the_grid(self):
        grid = BeatGrid(downbeats=(10.0, 12.0, 14.0))
        with pytest.raises(IndexError, match="outside the grid"):
            grid.bar_start(-1)
        with pytest.raises(IndexError, match="outside the grid"):
            grid.bar_start(3)

    def test_bar_start_reaches_the_final_bar_line(self):
        # It opens no complete bar, but it is where the form stops.
        assert BeatGrid(downbeats=(10.0, 12.0, 14.0)).bar_start(2) == 14.0


class TestDrift:
    def test_an_even_grid_does_not_drift(self):
        grid = BeatGrid(downbeats=(0.0, 2.0, 4.0, 6.0))
        assert grid.drift() == (0.0, 0.0, 0.0)
        assert grid.unreliable_bars() == ()

    def test_real_band_drift_stays_within_tolerance(self):
        # The measured case: the median bar moves under 1% across a take.
        lengths = [1.6906, 1.6950, 1.6990, 1.7047]
        downbeats = [0.0]
        for length in lengths:
            downbeats.append(downbeats[-1] + length)
        grid = BeatGrid(downbeats=tuple(downbeats))
        assert max(grid.drift()) < 0.01
        assert grid.unreliable_bars() == ()

    def test_tracker_jitter_on_real_audio_is_not_called_unreliable(self):
        # Measured on one real take: across the steady body of the performance
        # individual tracked bars scatter to 6.4% around the median, dominated
        # by beat placement rather than by the band. A threshold under that
        # would condemn a perfectly good grid, which is why it sits at 10%.
        lengths = [1.73, 1.62, 1.83, 1.73, 1.70, 1.76, 1.73]
        downbeats = [0.0]
        for length in lengths:
            downbeats.append(downbeats[-1] + length)
        grid = BeatGrid(downbeats=tuple(downbeats))
        assert max(grid.drift()) > 0.06
        assert grid.unreliable_bars() == ()

    def test_a_dropped_bar_is_named_not_averaged_away(self):
        grid = BeatGrid(downbeats=(0.0, 2.0, 4.0, 8.0, 10.0))
        assert grid.unreliable_bars() == (2,)

    def test_a_bar_a_long_way_out_of_tempo_is_flagged(self):
        # Bounds the tolerance from above: the longest intro bar on the measured
        # take was 12% over the body tempo, and must not read as healthy.
        grid = BeatGrid(downbeats=(0.0, 1.9376, 3.6676, 5.3976, 7.1276))
        assert grid.unreliable_bars() == (0,)

    def test_drift_is_empty_without_a_measurable_length(self):
        assert BeatGrid(downbeats=(1.0,)).drift() == ()


class TestPlausibility:
    @pytest.mark.parametrize("bar_length", [1.0, 1.6906, 2.5, 7.5])
    def test_accepts_a_tempo_a_band_could_play(self, bar_length):
        assert BeatGrid(downbeats=(0.0, bar_length, 2 * bar_length)).is_plausible

    @pytest.mark.parametrize("bar_length", [0.4, 0.79, 8.5, 20.0])
    def test_rejects_a_tempo_no_band_plays(self, bar_length):
        assert not BeatGrid(downbeats=(0.0, bar_length, 2 * bar_length)).is_plausible

    def test_an_unmeasurable_grid_is_not_plausible(self):
        assert not BeatGrid(downbeats=()).is_plausible

    def test_the_bound_is_on_tempo_not_bar_length(self):
        # A 7.5s bar is 32 bpm in 4/4 and 24 bpm in 3/4. A bound written in bar
        # length would accept both; only the slower one is outside 30-300 bpm.
        in_four = BeatGrid(downbeats=(0.0, 7.5, 15.0), beats_per_bar=4)
        in_three = BeatGrid(downbeats=(0.0, 7.5, 15.0), beats_per_bar=3)
        assert in_four.tempo == pytest.approx(32.0)
        assert in_three.tempo == pytest.approx(24.0)
        assert in_four.is_plausible
        assert not in_three.is_plausible

    def test_tempo_reads_off_the_metre(self):
        grid = BeatGrid(downbeats=(0.0, 1.7200, 3.4400), beats_per_bar=4)
        assert grid.tempo == pytest.approx(139.53, abs=0.01)

    def test_tempo_is_nan_without_a_measurable_bar(self):
        assert isnan(BeatGrid(downbeats=(1.0,)).tempo)


class TestGridAgainstAReference:
    def test_the_verdict_travels_on_the_grid(self, caplog):
        with caplog.at_level("WARNING", logger="songscribe.beats"):
            built = beats.grid(
                click_track(2 * BALLAD_BPM, bars=16),
                reference=beats.TempoReference(tempo=BALLAD_BPM),
            )
        assert built.tempo_check is not None
        assert built.tempo_check.is_octave_error
        assert built.tempo_check.tracked == pytest.approx(96.0)
        assert built.tempo_check.reference == pytest.approx(48.0)
        assert "96.0 bpm" in caplog.text
        assert "48.0 bpm" in caplog.text

    def test_a_grid_built_without_a_reference_has_no_verdict(self, caplog):
        # Not the same as agreement: nothing outside the audio was available.
        with caplog.at_level("WARNING", logger="songscribe.beats"):
            built = beats.grid(click_track(2 * BALLAD_BPM, bars=16))
        assert built.tempo_check is None
        assert caplog.text == ""

    def test_the_grid_keeps_its_own_bar_lines_whatever_the_reference_says(self):
        tracked = click_track(2 * BALLAD_BPM, bars=16)
        built = beats.grid(tracked, reference=beats.TempoReference(tempo=BALLAD_BPM))
        assert built.downbeats == tuple(b.time for b in tracked if b.is_downbeat)
        assert built.bars == 15

    def test_an_agreeing_reference_leaves_the_grid_quiet(self, caplog):
        with caplog.at_level("WARNING", logger="songscribe.beats"):
            built = beats.grid(
                click_track(BALLAD_BPM, bars=8), reference=beats.TempoReference(bars=32, span=160.0)
            )
        assert built.tempo_check.agrees
        assert caplog.text == ""

    def test_the_tempo_warning_fires_without_the_metre_warning(self, caplog):
        with caplog.at_level("WARNING", logger="songscribe.beats"):
            beats.grid(
                click_track(2 * BALLAD_BPM, bars=16),
                beats_per_bar=4,
                reference=beats.TempoReference(tempo=BALLAD_BPM),
            )
        assert "octave error" in caplog.text
        assert "metre" not in caplog.text

    def test_the_metre_warning_fires_without_the_tempo_warning(self, caplog):
        # A 3/4 click declared as 4/4: the metre is contradicted, while the
        # tempo that follows from the caller's metre matches the reference.
        # Two different faults, and neither may swallow the other.
        with caplog.at_level("WARNING", logger="songscribe.beats"):
            beats.grid(
                click_track(36.0, bars=8, metre=3),
                beats_per_bar=4,
                reference=beats.TempoReference(tempo=BALLAD_BPM),
            )
        assert "metre 4 given" in caplog.text
        assert "octave error" not in caplog.text

    def test_a_verdict_is_not_part_of_the_grid_identity(self):
        # Was unequal: the same bar lines, one held against a reference.
        plain = beats.grid(click_track(2 * BALLAD_BPM, bars=16))
        checked = beats.grid(
            click_track(2 * BALLAD_BPM, bars=16), reference=beats.TempoReference(tempo=BALLAD_BPM)
        )
        assert checked.tempo_check is not None
        assert plain == checked
        assert hash(plain) == hash(checked)

    def test_a_verdict_cannot_outlive_the_tempo_it_describes(self):
        # Was carried silently onto bar lines at half the tempo it reported.
        checked = beats.grid(
            click_track(2 * BALLAD_BPM, bars=16), reference=beats.TempoReference(tempo=BALLAD_BPM)
        )
        halved = tuple(5.0 * bar for bar in range(8))
        with pytest.raises(ValueError, match=r"tempo_check is stale: it was made at 96\.0 bpm"):
            replace(checked, downbeats=halved)
        assert replace(checked, downbeats=halved, tempo_check=None).tempo_check is None

    def test_trimming_that_keeps_the_tempo_keeps_a_verdict_still_true(self):
        # grid()'s own replace pattern: fewer bar lines at the same tempo, and
        # the verdict -- which is about tempo alone -- still holds of them.
        checked = beats.grid(
            click_track(2 * BALLAD_BPM, bars=16), reference=beats.TempoReference(tempo=BALLAD_BPM)
        )
        trimmed = replace(checked, downbeats=checked.downbeats[:4])
        assert trimmed.tempo_check.is_octave_error
        assert trimmed.tempo_check.tracked == pytest.approx(trimmed.tempo)


class TestGrid:
    def test_keeps_only_the_downbeats(self):
        grid = beats.grid(metronome(3, bar_length=2.0))
        assert grid.downbeats == (0.0, 2.0, 4.0)
        assert grid.bars == 2

    def test_infers_the_metre_from_the_positions(self):
        assert beats.grid(metronome(2, metre=3)).beats_per_bar == 3
        assert beats.grid(metronome(2, metre=4)).beats_per_bar == 4

    def test_an_explicit_metre_wins(self, caplog):
        with caplog.at_level("WARNING", logger="songscribe.beats"):
            built = beats.grid(metronome(2, metre=4), beats_per_bar=3)
        assert built.beats_per_bar == 3
        assert "metre 3 given" in caplog.text

    def test_an_explicit_metre_that_agrees_does_not_warn(self, caplog):
        with caplog.at_level("WARNING", logger="songscribe.beats"):
            beats.grid(metronome(2, metre=3), beats_per_bar=3)
        assert caplog.text == ""

    def test_empty_input_gives_an_empty_grid(self):
        grid = beats.grid([])
        assert grid.downbeats == ()
        assert grid.beats_per_bar == beats.DEFAULT_METRE

    def test_sorts_downbeats_it_is_handed_out_of_order(self):
        unsorted = [BeatEvent(4.0, 1), BeatEvent(0.0, 1), BeatEvent(2.0, 1)]
        assert beats.grid(unsorted).downbeats == (0.0, 2.0, 4.0)

    def test_warns_when_no_beat_is_a_downbeat(self, caplog):
        # A .beats cache trimmed mid-bar. The grid places nothing, which is
        # otherwise indistinguishable from a file the tracker found no bars in.
        tracked = [BeatEvent(0.5, 2), BeatEvent(1.0, 3), BeatEvent(1.5, 4)]
        with caplog.at_level("WARNING", logger="songscribe.beats"):
            built = beats.grid(tracked)
        assert built.downbeats == ()
        assert "no downbeat" in caplog.text

    def test_warns_when_the_inferred_metre_is_not_a_candidate(self, caplog):
        tracked = [BeatEvent(0.0, 1), BeatEvent(0.5, 2)]
        with caplog.at_level("WARNING", logger="songscribe.beats"):
            built = beats.grid(tracked)
        assert built.beats_per_bar == 2
        assert "inferred metre" in caplog.text

    def test_does_not_warn_on_a_healthy_grid(self, caplog):
        with caplog.at_level("WARNING", logger="songscribe.beats"):
            beats.grid(metronome(3))
        assert caplog.text == ""

    def test_a_grid_that_starts_late_keeps_its_offset(self):
        # Seven bars of intro before the form: the grid must not be rebased to
        # zero, or the anchor that uses it is meaningless.
        grid = beats.grid(metronome(3, bar_length=2.0, start=14.0))
        assert grid.downbeats == (14.0, 16.0, 18.0)
        assert grid.bar_index_at(1.0) is None


class TestRegistry:
    def test_every_backend_declares_a_sample_rate(self):
        assert set(beats.BACKENDS) == set(beats.BACKEND_SAMPLE_RATES)

    def test_the_default_backend_is_registered(self):
        assert beats.DEFAULT_BACKEND in beats.BACKENDS

    def test_the_default_is_not_the_weights_free_baseline(self):
        # librosa assumes the bar phase rather than detecting it; defaulting to
        # it would quietly produce a grid that is a beat or two off.
        assert beats.DEFAULT_BACKEND != "librosa"

    def test_returns_the_rate_each_backend_wants(self):
        assert beats.sample_rate_for("madmom") == 44_100
        assert beats.sample_rate_for("librosa") == 22_050

    def test_rejects_an_unknown_backend(self):
        with pytest.raises(ValueError, match="unknown beat backend"):
            beats.sample_rate_for("definitely-not-a-tracker")

    def test_track_rejects_an_unknown_backend_before_touching_the_audio(self):
        with pytest.raises(ValueError, match="unknown beat backend"):
            beats.track(np.zeros(10, dtype=np.float32), 44_100, backend="nope")

    def test_track_dispatches_to_the_named_backend(self, monkeypatch):
        tracked = [BeatEvent(0.0, 1), BeatEvent(0.5, 2)]
        monkeypatch.setitem(beats.BACKENDS, "fake", lambda audio, rate, **kw: tracked)
        monkeypatch.setitem(beats.BACKEND_SAMPLE_RATES, "fake", 8_000)
        assert beats.track(np.zeros(10, dtype=np.float32), 8_000, backend="fake") == tracked

    def test_track_passes_keyword_arguments_through(self, monkeypatch):
        seen = {}

        def fake(audio, rate, **kwargs):
            seen.update(kwargs)
            return []

        monkeypatch.setitem(beats.BACKENDS, "fake", fake)
        monkeypatch.setitem(beats.BACKEND_SAMPLE_RATES, "fake", 8_000)
        beats.track(np.zeros(10, dtype=np.float32), 8_000, backend="fake", beats_per_bar=3)
        assert seen == {"beats_per_bar": 3}

    def test_madmom_refuses_the_wrong_sample_rate(self):
        # Checked before the model is imported, so this runs with no weights.
        with pytest.raises(ValueError, match="expects 44100 Hz"):
            beats.track(np.zeros(10, dtype=np.float32), 22_050, backend="madmom")

    @pytest.mark.parametrize(("given", "expected"), [(3, 3), (None, beats.METRE_CANDIDATES)])
    def test_madmom_honours_beats_per_bar(self, monkeypatch, given, expected):
        # A metre named right has to reach the DBN, not just survive the
        # registry's setting check. Stubbed modules stand in for madmom, which
        # CI does not install, and for its weights.
        seen = {}

        class Tracker:
            def __init__(self, beats_per_bar, fps):
                seen["beats_per_bar"] = beats_per_bar

            def __call__(self, activations):
                return [(0.5, 1), (1.0, 2)]

        signal = types.ModuleType("madmom.audio.signal")
        signal.Signal = lambda audio, **kwargs: audio
        downbeats = types.ModuleType("madmom.features.downbeats")
        downbeats.RNNDownBeatProcessor = lambda: lambda signal: signal
        downbeats.DBNDownBeatTrackingProcessor = Tracker
        monkeypatch.setitem(sys.modules, "madmom.audio.signal", signal)
        monkeypatch.setitem(sys.modules, "madmom.features.downbeats", downbeats)

        kwargs = {} if given is None else {"beats_per_bar": given}
        tracked = beats.track(np.zeros(10, dtype=np.float32), 44_100, **kwargs)
        assert seen == {"beats_per_bar": expected}
        assert tracked == [BeatEvent(0.5, 1), BeatEvent(1.0, 2)]


class TestBackendSettings:
    def test_reports_what_each_backend_accepts(self):
        assert beats.backend_options("madmom") == {"beats_per_bar"}
        assert beats.backend_options("librosa") == {"beats_per_bar", "hop_length"}

    def test_the_metre_is_named_the_same_on_every_backend(self):
        # Whichever backend runs, `beats_per_bar` is the metre; the check must
        # not be what stops a caller setting it on the default one.
        for backend in beats.BACKENDS:
            assert "beats_per_bar" in beats.backend_options(backend)

    def test_no_registered_backend_keeps_a_catch_all(self):
        # A `**kwargs` on a backend would silently switch this validation off
        # for it, which is the bug the check exists to prevent.
        for backend in beats.BACKENDS:
            assert not beats.accepts_any_option(backend)

    def test_a_catch_all_is_reported_apart_from_the_accepted_names(self, monkeypatch):
        # "accepts anything" and "accepts nothing" must not both read as a
        # falsy `backend_options`, or a caller testing the set skips the check.
        monkeypatch.setitem(beats.BACKENDS, "fake", lambda audio, rate, hop_length=512, **kw: [])
        monkeypatch.setitem(beats.BACKEND_SAMPLE_RATES, "fake", 8_000)
        assert beats.accepts_any_option("fake")
        assert beats.backend_options("fake") == {"hop_length"}

    def test_a_catch_all_backend_takes_any_setting(self, monkeypatch):
        # The exemption is real, not just reported: nothing can be rejected on
        # behalf of a backend that will accept it.
        monkeypatch.setitem(beats.BACKENDS, "fake", lambda audio, rate, **kw: [])
        monkeypatch.setitem(beats.BACKEND_SAMPLE_RATES, "fake", 8_000)
        assert beats.track(np.zeros(10, dtype=np.float32), 8_000, backend="fake", nonsense=1) == []

    def test_a_setting_without_a_default_is_rejected(self, monkeypatch):
        # Not reportable as an accepted setting: the backend could not be
        # called without it, so `track` would raise TypeError from the call
        # instead of naming the fault here. Caught on every dispatch, so it
        # surfaces even when the caller names no settings at all. A misdeclared
        # backend is this package's bug, so it is not the caller-facing
        # ValueError -- a CLI mapping that to a usage error must not blame the
        # user for it.
        monkeypatch.setitem(beats.BACKENDS, "fake", lambda audio, rate, beats_per_bar: [])
        monkeypatch.setitem(beats.BACKEND_SAMPLE_RATES, "fake", 8_000)
        with pytest.raises(RuntimeError, match="'beats_per_bar' with no default") as excinfo:
            beats.track(np.zeros(10, dtype=np.float32), 8_000, backend="fake")
        assert not isinstance(excinfo.value, ValueError)

    def test_a_catch_all_does_not_excuse_a_setting_without_a_default(self, monkeypatch):
        # The exemption covers the list of accepted names only. A backend with
        # both still cannot be called with the audio and its rate alone, so
        # checking the catch-all first would hand the caller the bare TypeError
        # this validation exists to replace.
        monkeypatch.setitem(beats.BACKENDS, "fake", lambda audio, rate, checkpoint, **kw: [])
        monkeypatch.setitem(beats.BACKEND_SAMPLE_RATES, "fake", 8_000)
        assert beats.accepts_any_option("fake")
        with pytest.raises(RuntimeError, match="'checkpoint' with no default"):
            beats.track(np.zeros(10, dtype=np.float32), 8_000, backend="fake")

    def test_a_required_positional_only_setting_is_rejected(self, monkeypatch):
        # Unreachable by keyword *and* required: reporting it as "no settings"
        # and then failing at the call is the worst of both. Being callable with
        # the audio and its rate alone is checked before reachability.
        def fake(audio, sample_rate, hop_length, /):
            return []

        monkeypatch.setitem(beats.BACKENDS, "fake", fake)
        monkeypatch.setitem(beats.BACKEND_SAMPLE_RATES, "fake", 8_000)
        with pytest.raises(RuntimeError, match="'hop_length' with no default"):
            beats.backend_options("fake")
        with pytest.raises(RuntimeError, match="'hop_length' with no default"):
            beats.track(np.zeros(10, dtype=np.float32), 8_000, backend="fake")

    def test_every_registered_setting_has_a_default(self):
        # The guard is that the call itself does not raise -- there is nothing
        # to assert about the value, so don't add an assertion that hides which
        # backend failed.
        for backend in beats.BACKENDS:
            beats.backend_options(backend)

    def test_track_rejects_an_unknown_setting(self):
        # The error for a stale name (`candidates`) must point at `beats_per_bar`.
        with pytest.raises(ValueError) as excinfo:
            beats.track(np.zeros(10, dtype=np.float32), 44_100, backend="madmom", candidates=(3,))
        message = str(excinfo.value)
        assert "'candidates'" in message
        assert "'madmom'" in message
        assert "beats_per_bar" in message

    def test_rejects_a_setting_before_the_model_is_imported(self):
        # Runs with no weights: the setting is checked at the registry, so the
        # wrong-rate guard inside the backend is never reached either.
        with pytest.raises(ValueError, match="unknown setting 'metre'"):
            beats.track(np.zeros(10, dtype=np.float32), 22_050, backend="madmom", metre=4)

    def test_names_every_unknown_setting(self):
        with pytest.raises(ValueError, match="unknown settings 'hop', 'metre'"):
            beats.track(np.zeros(10, dtype=np.float32), 22_050, backend="librosa", metre=3, hop=1)

    def test_accepted_settings_reach_the_backend(self, monkeypatch):
        seen = {}

        def fake(audio, sample_rate, beats_per_bar=beats.DEFAULT_METRE):
            seen["beats_per_bar"] = beats_per_bar
            return []

        monkeypatch.setitem(beats.BACKENDS, "fake", fake)
        monkeypatch.setitem(beats.BACKEND_SAMPLE_RATES, "fake", 8_000)
        audio = np.zeros(10, dtype=np.float32)
        assert beats.track(audio, 8_000, backend="fake", beats_per_bar=3) == []
        assert seen == {"beats_per_bar": 3}


def _capture(monkeypatch, backend):
    """Swap ``backend`` for a fake that records its settings, keeping its rules."""
    seen = {}

    def fake(audio, sample_rate, beats_per_bar=beats.DEFAULT_METRE, hop_length=512):
        seen.update(beats_per_bar=beats_per_bar, hop_length=hop_length)
        return []

    monkeypatch.setitem(beats.BACKENDS, backend, fake)
    return seen


class TestBackendSettingValues:
    AUDIO = np.zeros(10, dtype=np.float32)

    def test_a_metre_list_on_librosa_is_refused_as_a_caller_fault(self):
        # A candidate list is refused as a ValueError, which a CLI catches, not a TypeError.
        with pytest.raises(ValueError) as excinfo:
            beats.track(self.AUDIO, 22_050, backend="librosa", beats_per_bar=(3, 4))
        message = str(excinfo.value)
        assert "'beats_per_bar'" in message
        assert "'librosa'" in message
        assert "must be one metre, got (3, 4)" in message
        assert not isinstance(excinfo.value, TypeError)

    @pytest.mark.parametrize("backend", ["madmom", "librosa"])
    @pytest.mark.parametrize(
        ("metre", "fault"),
        [
            (0, "must be at least 1, got 0"),
            (-4, "must be at least 1, got -4"),
            (True, "must be a whole number of beats to the bar, got True"),
            (3.5, "must be a whole number of beats to the bar, got 3.5"),
            ("4", "must be a whole number of beats to the bar, got '4'"),
        ],
    )
    def test_a_metre_that_is_not_a_positive_whole_number_is_refused(self, backend, metre, fault):
        # Checked at the registry, so the madmom case runs with no weights.
        with pytest.raises(ValueError) as excinfo:
            beats.track(self.AUDIO, 8_000, backend=backend, beats_per_bar=metre)
        assert str(excinfo.value) == (
            f"setting 'beats_per_bar' for beat backend {backend!r} {fault}"
        )

    @pytest.mark.parametrize(
        ("candidates", "named"),
        [
            ([3, 0], "candidate 0 at position 1"),
            ((4, -4), "candidate -4 at position 1"),
            ([True, 4], "candidate True at position 0"),
            ((3, 4.0), "candidate 4.0 at position 1"),
            (np.array([3.0, 4.0]), "candidate np.float64(3.0) at position 0"),
            (np.array([4, 0]), "candidate np.int64(0) at position 1"),
            (np.array([True, False]), "candidate np.True_ at position 0"),
        ],
    )
    def test_every_madmom_candidate_must_be_a_metre(self, candidates, named):
        # The bad entry is named on its own rather than the list reprinted, so
        # a long candidate list does not leave the caller hunting for it.
        with pytest.raises(ValueError) as excinfo:
            beats.track(self.AUDIO, 44_100, backend="madmom", beats_per_bar=candidates)
        assert str(excinfo.value) == (
            "setting 'beats_per_bar' for beat backend 'madmom' must list metres that are "
            f"each a whole number of at least 1, but {named} is not"
        )

    @pytest.mark.parametrize(
        ("empty", "kind"),
        [([], "list"), ((), "tuple"), (np.array([], dtype=int), "array")],
    )
    def test_madmom_needs_a_candidate_to_choose(self, empty, kind):
        with pytest.raises(ValueError) as excinfo:
            beats.track(self.AUDIO, 44_100, backend="madmom", beats_per_bar=empty)
        assert str(excinfo.value).endswith(
            f"must name at least one metre to choose between, got an empty {kind}"
        )

    def test_madmom_refuses_a_table_of_metres(self):
        with pytest.raises(ValueError, match="must be a flat list of metres, got a 2-D array"):
            beats.track(self.AUDIO, 44_100, backend="madmom", beats_per_bar=np.array([[3, 4]]))

    @pytest.mark.parametrize("backend", ["madmom", "librosa"])
    @pytest.mark.parametrize("metre", [b"34", bytearray(b"34"), memoryview(b"34")])
    def test_bytes_are_not_a_list_of_metres(self, backend, metre):
        # b"34" iterates to 51 and 52, which madmom would have tracked as two
        # metres; refused as one bad value, the same way "34" is.
        with pytest.raises(ValueError) as excinfo:
            beats.track(self.AUDIO, 44_100, backend=backend, beats_per_bar=metre)
        assert str(excinfo.value) == (
            f"setting 'beats_per_bar' for beat backend {backend!r} must be a whole number "
            f"of beats to the bar, got {metre!r}"
        )

    def test_a_metre_array_on_librosa_is_told_it_must_be_one_metre(self):
        with pytest.raises(ValueError) as excinfo:
            beats.track(self.AUDIO, 22_050, backend="librosa", beats_per_bar=np.array([3, 4]))
        assert "must be one metre, got array([3, 4]): this backend assumes" in str(excinfo.value)

    @pytest.mark.parametrize("hop", [0, -512, 512.0, True, np.float64(512.0)])
    def test_a_hop_that_is_not_a_positive_whole_number_is_refused(self, hop):
        with pytest.raises(ValueError) as excinfo:
            beats.track(self.AUDIO, 22_050, backend="librosa", hop_length=hop)
        message = str(excinfo.value)
        assert message.startswith("setting 'hop_length' for beat backend 'librosa' must be")
        assert repr(hop) in message

    @pytest.mark.parametrize("metre", [4, 3, (3, 4), [3, 4], [7]])
    def test_madmom_still_takes_one_metre_or_a_list(self, monkeypatch, metre):
        # The list form is madmom's own and must keep working; it reaches the
        # backend exactly as given, not converted.
        seen = _capture(monkeypatch, "madmom")
        assert beats.track(self.AUDIO, 44_100, backend="madmom", beats_per_bar=metre) == []
        assert seen["beats_per_bar"] is metre

    @pytest.mark.parametrize(
        "metre",
        [
            np.int64(3),
            np.array([3, 4]),
            np.array([7], dtype=np.int32),
            [np.int64(3), np.int64(4)],
        ],
        ids=repr,
    )
    def test_madmom_takes_numpy_metres(self, monkeypatch, metre):
        # The DBN reads its metre with np.array(ndmin=1), so an array is as
        # native to it as a list; NumPy integers are accepted too.
        seen = _capture(monkeypatch, "madmom")
        assert beats.track(self.AUDIO, 44_100, backend="madmom", beats_per_bar=metre) == []
        assert seen["beats_per_bar"] is metre

    def test_valid_librosa_settings_reach_the_backend(self, monkeypatch):
        seen = _capture(monkeypatch, "librosa")
        beats.track(self.AUDIO, 22_050, backend="librosa", beats_per_bar=3, hop_length=256)
        assert seen == {"beats_per_bar": 3, "hop_length": 256}

    def test_numpy_integers_reach_librosa(self, monkeypatch):
        seen = _capture(monkeypatch, "librosa")
        metre, hop = np.int64(3), np.int32(256)
        beats.track(self.AUDIO, 22_050, backend="librosa", beats_per_bar=metre, hop_length=hop)
        assert seen["beats_per_bar"] is metre
        assert seen["hop_length"] is hop

    @pytest.mark.parametrize("backend", ["madmom", "librosa"])
    @pytest.mark.parametrize("flag", [True, np.bool_(True)], ids=repr)
    def test_a_bool_is_still_not_a_metre(self, backend, flag):
        with pytest.raises(ValueError, match="must be a whole number of beats to the bar"):
            beats.track(self.AUDIO, 44_100, backend=backend, beats_per_bar=flag)

    def test_every_rule_guards_a_setting_its_backend_declares(self):
        # A renamed parameter must not leave its rule on a dead name.
        for backend, rules in beats.SETTING_RULES.items():
            assert set(rules) <= beats.backend_options(backend), backend

    def test_every_metre_setting_has_a_rule(self):
        # The metre is the one setting every backend shares, so it is the one
        # a caller moves between them; each backend has to vet it.
        for backend in beats.BACKENDS:
            assert "beats_per_bar" in beats.SETTING_RULES.get(backend, {}), backend
