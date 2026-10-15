from __future__ import annotations

import sys
import types
from math import isnan

import numpy as np
import pytest

from songscribe import beats
from songscribe.beats import BeatGrid
from songscribe.types import BeatEvent


def metronome(count: int, bar_length: float = 2.0, start: float = 0.0, metre: int = 4):
    """``count`` bars of perfectly even beats, downbeats included."""
    return [
        BeatEvent(time=start + bar * bar_length + beat * bar_length / metre, position=beat + 1)
        for bar in range(count)
        for beat in range(metre)
    ]


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
            assert beats.backend_options(backend) is not None

    def test_track_rejects_an_unknown_setting(self):
        # `candidates` was madmom's own name for the metre until it was renamed
        # to match librosa's, so it is exactly the stale setting a config can
        # still carry -- and the error has to point at the name that replaced it.
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
