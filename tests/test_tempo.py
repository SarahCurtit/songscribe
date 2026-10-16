from __future__ import annotations

from math import isnan

import numpy as np
import pytest

from songscribe import beats, tempo
from songscribe.beats import BeatGrid

from .conftest import BALLAD_BPM, click_track


class TestTempoReference:
    def test_reads_an_explicit_tempo(self):
        assert tempo.TempoReference(tempo=BALLAD_BPM).expected_tempo == pytest.approx(48.0)

    def test_derives_a_tempo_from_bars_over_a_span(self):
        # 32 bars of chorus across 160 s, four beats to the bar: the reference a
        # chart with no tempo marking can still supply.
        assert tempo.TempoReference(bars=32, span=160.0).expected_tempo == pytest.approx(48.0)

    def test_the_derived_tempo_reads_off_the_metre(self):
        reference = tempo.TempoReference(bars=32, span=160.0, beats_per_bar=3)
        assert reference.expected_tempo == pytest.approx(36.0)

    def test_an_explicit_tempo_wins_over_a_bar_count_that_agrees_with_it(self):
        # The marking is stated; a span's ends were read off a waveform. 32 bars
        # over 162 s is 47.4 bpm, inside the tolerance, so the marking stands.
        reference = tempo.TempoReference(tempo=BALLAD_BPM, bars=32, span=162.0)
        assert reference.expected_tempo == pytest.approx(48.0)

    def test_a_bar_count_contradicting_the_tempo_is_refused(self):
        # 32 bars over 80 s is 96 bpm, an octave off the marking: exactly the
        # fault the check exists to find, so it may not be silently dropped.
        with pytest.raises(ValueError, match=r"contradicts 32 bars over 80 s, which is 96\.0 bpm"):
            tempo.TempoReference(tempo=BALLAD_BPM, bars=32, span=80.0)

    def test_a_contradiction_is_refused_from_a_sidecar_too(self):
        with pytest.raises(ValueError, match="contradicts"):
            tempo.TempoReference.from_mapping({"tempo": 48, "bars": 32, "span": 80})

    def test_an_unstated_metre_is_named_as_the_assumption(self):
        # 32 bars of 3/4 at 48 bpm take 120 s; read in the default 4 that is
        # 64 bpm, so the message says which metre it assumed and how to fix it.
        with pytest.raises(ValueError, match="give beats_per_bar"):
            tempo.TempoReference(tempo=BALLAD_BPM, bars=32, span=120.0)
        stated = tempo.TempoReference(tempo=BALLAD_BPM, bars=32, span=120.0, beats_per_bar=3)
        assert stated.expected_tempo == pytest.approx(48.0)

    def test_a_fractional_bar_count_is_not_a_bar_count(self):
        with pytest.raises(ValueError, match="whole number of bars"):
            tempo.TempoReference(bars=31.5, span=160.0)

    def test_a_fractional_metre_is_refused(self):
        with pytest.raises(ValueError, match="beats_per_bar must be a whole number"):
            tempo.TempoReference(tempo=BALLAD_BPM, beats_per_bar=6.5)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_a_tempo_that_is_not_a_tempo(self, bad):
        # A reference that cannot be turned into a tempo would make every grid
        # compared against it look like it agreed.
        with pytest.raises(ValueError, match="tempo must be"):
            tempo.TempoReference(tempo=bad)

    def test_rejects_bars_without_a_span(self):
        with pytest.raises(ValueError, match="both bars and span"):
            tempo.TempoReference(bars=32)

    def test_rejects_a_span_without_bars(self):
        with pytest.raises(ValueError, match="both bars and span"):
            tempo.TempoReference(span=160.0)

    def test_rejects_an_empty_reference(self):
        with pytest.raises(ValueError, match="either a tempo"):
            tempo.TempoReference()

    @pytest.mark.parametrize("bad", [0.0, -4.0, float("nan"), float("inf")])
    def test_rejects_a_span_that_is_not_a_duration(self, bad):
        with pytest.raises(ValueError, match="span must be"):
            tempo.TempoReference(bars=32, span=bad)

    def test_rejects_a_bar_count_below_one(self):
        with pytest.raises(ValueError, match="bars must be"):
            tempo.TempoReference(bars=0, span=10.0)

    def test_rejects_a_metre_below_one(self):
        with pytest.raises(ValueError, match="beats_per_bar"):
            tempo.TempoReference(tempo=BALLAD_BPM, beats_per_bar=0)

    def test_numpy_counts_are_counts(self):
        # A bar count read off an array is as whole as a Python int.
        reference = tempo.TempoReference(bars=np.int64(32), span=160.0, beats_per_bar=np.int32(4))
        assert reference.expected_tempo == pytest.approx(48.0)

    @pytest.mark.parametrize("flag", [True, np.bool_(True)], ids=repr)
    def test_a_bool_is_not_a_count(self, flag):
        with pytest.raises(ValueError, match="bars must be a whole number"):
            tempo.TempoReference(bars=flag, span=160.0)
        with pytest.raises(ValueError, match="beats_per_bar must be a whole number"):
            tempo.TempoReference(tempo=BALLAD_BPM, beats_per_bar=flag)


class TestLoadingATempoReference:
    def test_builds_from_a_mapping_either_way(self):
        stated = tempo.TempoReference.from_mapping({"tempo": 48})
        counted = tempo.TempoReference.from_mapping({"bars": 32, "span": 160})
        assert stated.expected_tempo == pytest.approx(48.0)
        assert counted.expected_tempo == pytest.approx(48.0)

    def test_rejects_a_key_it_has_no_field_for(self):
        # A misspelled tempo that was dropped would leave the check comparing
        # against the bar count it was meant to override.
        with pytest.raises(ValueError, match="unknown tempo reference key 'bpm'"):
            tempo.TempoReference.from_mapping({"bpm": 48})

    def test_names_every_unknown_key(self):
        with pytest.raises(ValueError, match="unknown tempo reference keys 'bpm', 'metre'"):
            tempo.TempoReference.from_mapping({"bpm": 48, "metre": 4})

    def test_reads_a_json_sidecar(self, tmp_path):
        path = tmp_path / "tempo.json"
        path.write_text('{"bars": 32, "span": 160.0, "beats_per_bar": 4}', encoding="utf-8")
        assert tempo.load_tempo_reference(path).expected_tempo == pytest.approx(48.0)

    def test_a_sidecar_tempo_survives_the_round_trip(self, tmp_path):
        path = tmp_path / "tempo.json"
        path.write_text('{"tempo": 48}', encoding="utf-8")
        assert tempo.load_tempo_reference(path) == tempo.TempoReference(tempo=48.0)

    def test_a_null_metre_means_unstated_like_every_other_key(self):
        # Was int(None), a TypeError outside the module's ValueError contract.
        reference = tempo.TempoReference.from_mapping({"tempo": 48, "beats_per_bar": None})
        assert reference.beats_per_bar is None
        assert reference.expected_tempo == pytest.approx(48.0)

    def test_a_fractional_bar_count_is_refused_not_truncated(self):
        # Was silently 31 bars, 46.5 bpm where the file said 47.25.
        with pytest.raises(ValueError, match=r"'bars' must be a whole number, got 31\.5"):
            tempo.TempoReference.from_mapping({"bars": 31.5, "span": 160.0})

    def test_a_whole_float_is_a_count(self):
        reference = tempo.TempoReference.from_mapping({"bars": 32.0, "span": 160.0})
        assert reference.bars == 32
        assert isinstance(reference.bars, int)

    @pytest.mark.parametrize(
        ("key", "value"),
        [("tempo", "48"), ("tempo", True), ("span", [160]), ("beats_per_bar", "4")],
    )
    def test_a_value_of_the_wrong_type_is_a_value_error_naming_its_key(self, key, value):
        values = {"bars": 32, "span": 160.0} if key != "tempo" else {}
        values[key] = value
        with pytest.raises(ValueError, match=f"'{key}' must be"):
            tempo.TempoReference.from_mapping(values)

    @pytest.mark.parametrize("text", ["[]", "48", '"tempo"', "null"])
    def test_a_sidecar_that_is_not_an_object_names_the_file(self, tmp_path, text):
        # Was AttributeError for [] and TypeError for 48, neither naming the path.
        path = tmp_path / "tempo.json"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match=r"tempo\.json: a tempo reference is a JSON object"):
            tempo.load_tempo_reference(path)

    def test_a_bad_value_in_a_sidecar_names_the_file(self, tmp_path):
        path = tmp_path / "tempo.json"
        path.write_text('{"bpm": 48}', encoding="utf-8")
        with pytest.raises(ValueError, match=r"tempo\.json: unknown tempo reference key 'bpm'"):
            tempo.load_tempo_reference(path)

    def test_a_sidecar_that_is_not_json_names_the_file(self, tmp_path):
        path = tmp_path / "tempo.json"
        path.write_text("tempo: 48", encoding="utf-8")
        with pytest.raises(ValueError, match=r"tempo\.json: not valid JSON"):
            tempo.load_tempo_reference(path)


class TestCheckTempo:
    def test_a_grid_on_the_reference_tempo_passes_clean(self, caplog):
        tracked = beats.grid(click_track(BALLAD_BPM, bars=8))
        with caplog.at_level("WARNING", logger="songscribe.tempo"):
            check = tempo.check_tempo(tracked, tempo.TempoReference(tempo=BALLAD_BPM))
        assert check.agrees
        assert not check.is_octave_error
        assert check.ratio == pytest.approx(1.0)
        assert caplog.text == ""

    def test_double_tempo_warns_and_names_both_tempos(self, caplog):
        # The reported failure: madmom's DBN will not go below 55 bpm, so this
        # ballad comes back at 96 with twice the bars it should have.
        tracked = beats.grid(click_track(2 * BALLAD_BPM, bars=16))
        with caplog.at_level("WARNING", logger="songscribe.tempo"):
            check = tempo.check_tempo(tracked, tempo.TempoReference(tempo=BALLAD_BPM))
        assert check.is_octave_error
        assert check.octave_factor == 2.0
        assert "96.0 bpm" in caplog.text
        assert "48.0 bpm" in caplog.text
        assert "is 2x" in caplog.text
        assert "twice" in caplog.text

    def test_half_tempo_warns_and_names_both_tempos(self, caplog):
        tracked = beats.grid(click_track(BALLAD_BPM / 2, bars=4))
        with caplog.at_level("WARNING", logger="songscribe.tempo"):
            check = tempo.check_tempo(tracked, tempo.TempoReference(tempo=BALLAD_BPM))
        assert check.octave_factor == 0.5
        assert "24.0 bpm" in caplog.text
        assert "48.0 bpm" in caplog.text
        assert "is 0.5x" in caplog.text
        assert "half" in caplog.text

    def test_both_reference_forms_reach_the_same_verdict(self, caplog):
        # 32 bars across 160 s is the same 48 bpm said the other way, and a
        # chart with no tempo marking has to catch the same octave error.
        tracked = beats.grid(click_track(2 * BALLAD_BPM, bars=16))
        with caplog.at_level("WARNING", logger="songscribe.tempo"):
            stated = tempo.check_tempo(tracked, tempo.TempoReference(tempo=BALLAD_BPM))
            counted = tempo.check_tempo(tracked, tempo.TempoReference(bars=32, span=160.0))
        assert counted.octave_factor == stated.octave_factor == 2.0
        assert counted.reference == pytest.approx(stated.reference)
        assert counted.ratio == pytest.approx(stated.ratio)
        assert caplog.text.count("96.0 bpm is 2x the reference 48.0 bpm") == 2

    def test_the_grid_is_reported_on_not_corrected(self, caplog):
        # Flag and report: a grid halved to match the reference it is supposed
        # to be measured against has stopped being a measurement.
        tracked = beats.grid(click_track(2 * BALLAD_BPM, bars=16))
        with caplog.at_level("WARNING", logger="songscribe.tempo"):
            tempo.check_tempo(tracked, tempo.TempoReference(tempo=BALLAD_BPM))
        assert tracked.downbeats == tuple(2.5 * bar for bar in range(16))
        assert tracked.tempo == pytest.approx(2 * BALLAD_BPM)
        assert tracked.bars == 15

    def test_the_range_check_cannot_see_the_octave_error(self):
        # Why a reference is needed at all: 96 bpm is a tempo a band could be
        # playing, so the 30-300 bpm window calls the doubled grid healthy.
        tracked = beats.grid(click_track(2 * BALLAD_BPM, bars=16))
        assert tracked.is_plausible
        check = tempo.check_tempo(tracked, tempo.TempoReference(tempo=BALLAD_BPM))
        assert check.is_octave_error

    def test_a_band_drifting_off_the_marking_still_agrees(self, caplog):
        # 3% over the chart's tempo is a band, not a tracking fault.
        tracked = beats.grid(click_track(BALLAD_BPM * 1.03, bars=8))
        with caplog.at_level("WARNING", logger="songscribe.tempo"):
            check = tempo.check_tempo(tracked, tempo.TempoReference(tempo=BALLAD_BPM))
        assert check.agrees
        assert not check.is_octave_error
        assert caplog.text == ""

    def test_drift_into_the_doubled_tempo_is_still_the_doubled_tempo(self):
        # Why the comparison is a window around 2 rather than equality.
        tracked = beats.grid(click_track(2 * BALLAD_BPM * 1.05, bars=16))
        check = tempo.check_tempo(tracked, tempo.TempoReference(tempo=BALLAD_BPM))
        assert check.octave_factor == 2.0

    def test_a_disagreement_that_is_not_an_octave_says_so(self, caplog):
        tracked = beats.grid(click_track(BALLAD_BPM * 1.5, bars=8))
        with caplog.at_level("WARNING", logger="songscribe.tempo"):
            check = tempo.check_tempo(tracked, tempo.TempoReference(tempo=BALLAD_BPM))
        assert not check.agrees
        assert not check.is_octave_error
        assert "72.0 bpm" in caplog.text
        assert "48.0 bpm" in caplog.text
        assert "none of the octave locks checked for (0.25x, 0.5x, 2x, 4x)" in caplog.text
        assert check.verdict == "disagrees"

    def test_an_unmeasurable_grid_claims_nothing(self, caplog):
        # One downbeat has no tempo, and "not 2x the reference" would be a
        # finding the grid cannot support.
        with caplog.at_level("WARNING", logger="songscribe.tempo"):
            check = tempo.check_tempo(
                BeatGrid(downbeats=(1.0,)), tempo.TempoReference(tempo=BALLAD_BPM)
            )
        assert isnan(check.ratio)
        assert not check.agrees
        assert not check.is_octave_error
        assert caplog.text == ""

    def test_an_unmeasurable_grid_is_told_apart_from_a_contradicted_one(self):
        # agrees is False for both; a caller reading only that would treat a
        # grid with no bars as one the reference contradicts.
        reference = tempo.TempoReference(tempo=BALLAD_BPM)
        silent = tempo.check_tempo(BeatGrid(downbeats=(1.0,)), reference)
        contradicted = tempo.check_tempo(beats.grid(click_track(BALLAD_BPM * 1.5, 8)), reference)
        assert not silent.agrees and not contradicted.agrees
        assert not silent.measurable
        assert contradicted.measurable
        assert silent.verdict == "unmeasurable"
        assert contradicted.verdict == "disagrees"

    def test_every_verdict_is_reachable(self):
        reference = tempo.TempoReference(tempo=BALLAD_BPM)
        verdicts = {
            tempo.check_tempo(beats.grid(click_track(bpm, 8)), reference).verdict
            for bpm in (BALLAD_BPM, 2 * BALLAD_BPM, 1.5 * BALLAD_BPM)
        }
        assert verdicts == {"agrees", "octave error", "disagrees"}

    def test_a_tighter_tolerance_can_be_asked_for(self):
        tracked = beats.grid(click_track(2 * BALLAD_BPM * 1.05, bars=16))
        reference = tempo.TempoReference(tempo=BALLAD_BPM)
        assert tempo.check_tempo(tracked, reference, tolerance=0.01).octave_factor is None


class TestToleranceBounds:
    def test_the_bound_is_where_the_windows_meet(self):
        # 1(1 + t) = 2(1 - t) at t = 1/3, and every listed ratio is an octave
        # from its neighbour, so that is the bound for all of them.
        assert pytest.approx(1 / 3) == tempo.MAX_TEMPO_RATIO_TOLERANCE

    @pytest.mark.parametrize("bad", [float("nan"), -0.01, 1 / 3, 0.5, float("inf")])
    def test_a_tolerance_that_breaks_the_verdict_is_refused(self, bad):
        tracked = beats.grid(click_track(BALLAD_BPM, bars=8))
        with pytest.raises(ValueError, match="tolerance must be"):
            tempo.check_tempo(tracked, tempo.TempoReference(tempo=BALLAD_BPM), tolerance=bad)

    def test_nan_no_longer_turns_an_exact_match_into_a_disagreement(self):
        # Was: 48 against 48 "disagrees ... (1.00x)", since nan fails every test.
        with pytest.raises(ValueError, match="tolerance"):
            tempo.TempoCheck(tracked=48.0, reference=48.0, tolerance=float("nan"))

    def test_one_grid_cannot_get_two_verdicts(self):
        # Was: at 0.5, 67 bpm against 48 both agreed and was a double-tempo
        # lock. Just under the bound, nothing between 1x and 2x matches both.
        tolerance = tempo.MAX_TEMPO_RATIO_TOLERANCE - 1e-9
        for ratio in np.linspace(0.2, 5.0, 2000):
            check = tempo.TempoCheck(tracked=48.0 * ratio, reference=48.0, tolerance=tolerance)
            matched = [f for f in (1.0, *tempo.TEMPO_OCTAVE_FACTORS) if check._matches(f)]
            assert len(matched) <= 1, (ratio, matched)

    def test_zero_tolerance_asks_for_an_exact_ratio(self):
        check = tempo.TempoCheck(tracked=48.0, reference=48.0, tolerance=0.0)
        assert check.agrees


class TestFurtherOctaves:
    def test_a_quadruple_tempo_lock_is_an_octave_error(self, caplog):
        # Was: "(4.00x), which is not a half- or double-tempo lock" -- the
        # same fault one octave further out, ruled out by the message.
        tracked = beats.grid(click_track(128.0, bars=16))
        with caplog.at_level("WARNING", logger="songscribe.tempo"):
            check = tempo.check_tempo(tracked, tempo.TempoReference(tempo=32.0))
        assert check.octave_factor == 4.0
        assert check.verdict == "octave error"
        assert "128.0 bpm is 4x the reference 32.0 bpm" in caplog.text
        assert "four times the form's" in caplog.text

    def test_a_quarter_tempo_lock_is_an_octave_error(self, caplog):
        tracked = beats.grid(click_track(30.0, bars=4))
        with caplog.at_level("WARNING", logger="songscribe.tempo"):
            check = tempo.check_tempo(tracked, tempo.TempoReference(tempo=120.0))
        assert check.octave_factor == 0.25
        assert "a quarter of the form's" in caplog.text

    def test_past_the_listed_octaves_the_message_does_not_clear_it(self, caplog):
        # 8x is out of madmom's reach, but if it came it must not be told it is
        # not an octave error: the message names what was checked instead.
        tracked = beats.grid(click_track(240.0, bars=16))
        with caplog.at_level("WARNING", logger="songscribe.tempo"):
            check = tempo.check_tempo(tracked, tempo.TempoReference(tempo=30.0))
        assert check.verdict == "disagrees"
        assert "none of the octave locks checked for" in caplog.text
        assert "not a half- or double-tempo lock" not in caplog.text


class TestDifferingMetres:
    def test_a_compound_chart_against_its_simple_tracking_agrees(self, caplog):
        # The review's case: a 6/8 chart, 8 bars over 16 s, and madmom -- which
        # only picks 3 or 4 -- tracks it in 3 on the very same bar lines. Beat
        # rates are 180 against 90, a ratio of exactly 0.5, which was reported
        # as "half the form's" for a grid whose downbeats are the form's own.
        tracked = beats.grid(click_track(90.0, bars=8, metre=3))
        assert tracked.downbeats == (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0)
        assert tracked.tempo == pytest.approx(90.0)
        reference = tempo.TempoReference(bars=8, span=16.0, beats_per_bar=6)
        assert reference.expected_tempo == pytest.approx(180.0)
        with caplog.at_level("INFO", logger="songscribe.tempo"):
            check = tempo.check_tempo(tracked, reference)
        assert check.ratio == pytest.approx(1.0)
        assert check.agrees
        assert not check.is_octave_error
        assert check.tracked == pytest.approx(180.0)
        assert check.beats_per_bar == 6
        # Said, at INFO, so the 180 is not mistaken for what was tracked.
        assert "grid tracked in 3, reference in 6: comparing bar lines" in caplog.text
        assert not [r for r in caplog.records if r.levelname == "WARNING"]

    def test_eight_against_four_on_the_same_bar_lines_agrees(self):
        tracked = beats.grid(click_track(120.0, bars=8, metre=4))
        check = tempo.check_tempo(tracked, tempo.TempoReference(bars=8, span=16.0, beats_per_bar=8))
        assert check.verdict == "agrees"

    def test_an_explicit_compound_tempo_against_simple_tracking_agrees(self):
        # 180 eighths a minute in 6 is 30 bars a minute, as is 90 in 3.
        tracked = beats.grid(click_track(90.0, bars=8, metre=3))
        check = tempo.check_tempo(tracked, tempo.TempoReference(tempo=180.0, beats_per_bar=6))
        assert check.verdict == "agrees"

    def test_a_real_octave_error_is_still_caught_across_metres(self):
        # Twice the 6/8 chart's bars, tracked in 3: the fault the check is for.
        tracked = beats.grid(click_track(180.0, bars=16, metre=3))
        check = tempo.check_tempo(tracked, tempo.TempoReference(bars=8, span=16.0, beats_per_bar=6))
        assert check.octave_factor == 2.0

    def test_a_bar_count_with_no_metre_needs_none(self):
        # Already a rate of bars, so a correct waltz grid agrees with it; the
        # old default of 4 read it as 160 bpm against the waltz's 120.
        tracked = beats.grid(click_track(120.0, bars=8, metre=3))
        check = tempo.check_tempo(tracked, tempo.TempoReference(bars=8, span=12.0))
        assert check.verdict == "agrees"

    def test_a_bare_tempo_is_read_in_the_grid_metre(self):
        # A tempo with no metre has no bar lines of its own, so it is compared
        # as beats -- the only comparison it supports -- and a correctly
        # tracked waltz agrees with its marking.
        tracked = beats.grid(click_track(120.0, bars=8, metre=3))
        check = tempo.check_tempo(tracked, tempo.TempoReference(tempo=120.0))
        assert check.verdict == "agrees"
        assert check.beats_per_bar == 3

    def test_matching_metres_say_nothing_about_metre(self, caplog):
        with caplog.at_level("INFO", logger="songscribe.tempo"):
            tempo.check_tempo(
                beats.grid(click_track(BALLAD_BPM, 8)),
                tempo.TempoReference(tempo=BALLAD_BPM, beats_per_bar=4),
            )
        assert "comparing bar lines" not in caplog.text
