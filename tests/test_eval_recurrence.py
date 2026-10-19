"""Finding the form's start from the audio recurring one form later, checked against the chart."""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
import soundfile as sf

from songscribe import beats, lab
from songscribe.demo import synthesise_performance
from songscribe.eval import chroma, cli, form, form_start, recurrence

SAMPLE_RATE = 11_025
BPM = 240.0  # a beat every quarter second
CHART = ("Dm", "G7", "C", "Am", "Em", "A7", "Dm", "G7", "Cmaj7", "Fmaj7", "Bdim", "E7")
CHART = (*CHART, "Am", "Dm", "G7", "C")
BARS = len(CHART)
#: Far from the tune's keys, so nothing in it resembles the chart by accident.
INTRO = ("F#", "B", "F#", "B7", "Ebm")
SOLO_CHANGES = {1: {2: "F", 5: "Bb", 9: "Dm"}, 2: {3: "C", 6: "G", 11: "Am"}}
SOLO_VOICINGS = {1: [10, 14], 2: [14]}


_SPECTRA: dict[int, chroma.Chromagram] = {}


def spectrum_of(performance):
    """The take's chromagram, computed once (takes are cached, so their id is stable)."""
    if id(performance) not in _SPECTRA:
        _SPECTRA[id(performance)] = chroma.chromagram(performance.audio, SAMPLE_RATE)
    return _SPECTRA[id(performance)]


def locate(performance, tracker_phase=0, shifts=chroma.AS_WRITTEN, reference=None):
    grid = beats.grid(performance.beat_events(tracker_phase))
    templates = chroma.bar_templates(reference or performance.reference, BARS)
    return form_start.locate(
        spectrum_of(performance), grid, templates, min_bars=form.CHORUS_MIN_BARS, shifts=shifts
    )


def the_take(intro=(), pickup=0, varied=False):
    """A cached take of the chart four times, optionally with an intro, pickup and solo changes."""
    options = {"intro": tuple(intro), "pickup_beats": pickup}
    if varied:
        options |= {"noise": 0.02}
    return _build(tuple(options.items()), varied)


@cache
def _build(options, varied):
    kwargs = dict(options)
    if varied:
        kwargs["substitutions"] = SOLO_CHANGES
        kwargs["colour"] = SOLO_VOICINGS
    return synthesise_performance(
        progression=CHART, bpm=BPM, sample_rate=SAMPLE_RATE, repeats=4, **kwargs
    )


class TestRecurrenceStart:
    @pytest.mark.parametrize("pickup", [0, 1, 2, 3])
    def test_intro_and_pickup_are_recovered_from_the_audio(self, pickup):
        performance = the_take(intro=INTRO, pickup=pickup)
        anchor, _ = locate(performance)
        assert anchor.evidence == "recurrence"
        assert anchor.start == pytest.approx(performance.head_start)
        assert anchor.pickup_beats == pickup
        # A pickup takes the last beats of the intro's final bar.
        assert anchor.intro_bars == len(INTRO) - (1 if pickup else 0)
        assert anchor.is_clear

    def test_a_take_that_opens_with_the_head_has_no_intro(self):
        anchor, _ = locate(the_take())
        assert (anchor.evidence, anchor.intro_bars, anchor.pickup_beats) == ("recurrence", 0, 0)
        assert anchor.start == pytest.approx(0.0)

    def test_a_stub_pickup_before_the_first_bar_line_is_found(self):
        performance = the_take(pickup=3)
        anchor, _ = locate(performance)
        assert (anchor.pickup_beats, anchor.intro_bars) == (3, 0)
        assert anchor.start == pytest.approx(performance.head_start)

    def test_it_holds_when_the_solo_choruses_only_roughly_agree_with_the_head(self):
        performance = the_take(intro=INTRO[:3], pickup=1, varied=True)
        anchor, _ = locate(performance)
        assert anchor.evidence == "recurrence"
        assert anchor.start == pytest.approx(performance.head_start)
        assert anchor.pickup_beats == 1

    def test_an_intro_that_echoes_the_tune_does_not_pull_the_start_earlier(self):
        # F, G, F, G7, Em share notes with the chart's last bars one form on; the chart
        # puts the head where it fits, not where the recurrence first opens.
        performance = the_take(intro=("F", "G", "F", "G7", "Em"))
        anchor, _ = locate(performance)
        assert anchor.start == pytest.approx(performance.head_start)
        assert anchor.chart_agrees is True

    def test_a_head_that_differs_from_the_solos_is_still_found_by_the_chart_fit(self):
        # Twelve head bars are related chords, so the recurrence opens a bar late; the chart
        # fits the head's real first bar clearly better.
        related = {"Dm": "F", "G7": "Bdim", "C": "Am", "Am": "C", "Em": "G", "A7": "A"}
        related |= {"Cmaj7": "Em", "Fmaj7": "Dm", "Bdim": "G7", "E7": "E"}
        changes = {0: {i: related.get(CHART[i], "C") for i in range(12)}}
        performance = synthesise_performance(
            progression=CHART, bpm=BPM, sample_rate=SAMPLE_RATE, repeats=4, substitutions=changes
        )
        anchor, _ = locate(performance)
        assert anchor.start == pytest.approx(0.0)
        assert anchor.evidence == "recurrence"
        found = recurrence.measure(
            spectrum_of(performance), beats.grid(performance.beat_events()).downbeats, 4, BARS
        )
        assert found.first_beat > 0

    def test_the_chart_fit_is_reported_with_the_start(self):
        anchor, _ = locate(the_take(intro=INTRO, pickup=2))
        assert anchor.mean_agreement > chroma.ANCHOR_MIN_AGREEMENT
        assert anchor.chart_agrees is True
        assert anchor.ending_bars == 0


class TestTrackerPhase:
    @pytest.mark.parametrize("phase", [1, 2, 3])
    def test_bar_lines_a_few_beats_off_are_moved_to_where_the_chords_change(self, phase):
        performance = the_take(intro=INTRO, pickup=2, varied=True)
        anchor, grid = locate(performance, tracker_phase=phase)
        assert anchor.start == pytest.approx(performance.head_start)
        assert anchor.phase_shift != 0
        assert anchor.pickup_beats == 2
        assert grid.downbeats[anchor.bar_index] == pytest.approx(anchor.start)

    def test_a_correct_tracker_is_left_alone(self):
        performance = the_take(intro=INTRO, pickup=2, varied=True)
        anchor, grid = locate(performance)
        assert anchor.phase_shift == 0
        assert grid.downbeats == beats.grid(performance.beat_events()).downbeats

    def test_a_rival_beat_needs_the_chords_to_change_there_clearly(self):
        performance = the_take(intro=INTRO, pickup=2, varied=True)
        _, lead = recurrence.beat_times(beats.grid(performance.beat_events()).downbeats, 4)
        found = recurrence.measure(
            spectrum_of(performance), beats.grid(performance.beat_events()).downbeats, 4, BARS
        )
        residue, t = recurrence.bar_line_phase(found, lead % 4)
        assert residue == lead % 4
        assert t < recurrence.PHASE_MIN_T
        off, _ = recurrence.bar_line_phase(found, (lead + 1) % 4)
        assert off == lead % 4


class TestKey:
    def test_the_key_comes_from_the_chart_at_the_recurring_start(self):
        performance = _build(
            (("intro", INTRO), ("pickup_beats", 2), ("noise", 0.02), ("transpose", 2)), True
        )
        anchor, _ = locate(performance, shifts=chroma.ANY_KEY)
        assert anchor.evidence == "recurrence"
        assert anchor.start == pytest.approx(performance.head_start)
        assert anchor.shift == 2
        assert anchor.key_margin > chroma.ANCHOR_MIN_MARGIN
        assert anchor.is_clear

    def test_recurrence_does_not_need_the_chart_to_be_in_the_takes_key(self):
        performance = _build((("intro", INTRO), ("pickup_beats", 2), ("transpose", 5)), False)
        anchor, _ = locate(performance, shifts=chroma.AS_WRITTEN)
        assert anchor.start == pytest.approx(performance.head_start)
        assert anchor.pickup_beats == 2
        # The written key does not fit, so the start is found but the chart refuses it.
        assert anchor.evidence == "recurrence"
        assert "chart fits the recurring start badly" in anchor.refusal()

    def test_a_recurring_start_is_held_to_the_plain_fit_not_the_any_key_gate(self):
        fit = (chroma.ANCHOR_MIN_AGREEMENT + chroma.ANCHOR_MIN_AGREEMENT_ANY_KEY) / 2
        clear = chroma.Anchor(0, 0.0, fit, 0.2, 0.3, keys=12, evidence="recurrence", key_margin=0.2)
        assert clear.refusal() is None
        chart_found = chroma.Anchor(0, 0.0, fit, 0.2, 0.3, keys=12)
        assert "under" in chart_found.refusal()

    def test_two_keys_that_fit_alike_are_refused_for_a_recurring_start(self):
        tied = chroma.Anchor(0, 0.0, 0.9, 0.2, 0.7, keys=12, evidence="recurrence", key_margin=0.01)
        assert "another key fits" in tied.refusal()
        assert chroma.Anchor(0, 0.0, 0.9, 0.2, 0.7, evidence="recurrence").refusal() is None


class TestNothingRecurs:
    def test_unrelated_chords_make_no_claim_and_fall_back_to_the_chart(self):
        rng = np.random.default_rng(3)
        names = ["C", "Dm", "Em", "F", "G", "Am", "G7", "A7", "Bdim", "Fmaj7", "Cmaj7", "E7", "D"]
        wandering = tuple(rng.choice(names, size=BARS * 4))
        performance = synthesise_performance(
            progression=wandering, bpm=BPM, sample_rate=SAMPLE_RATE, repeats=1
        )
        found = recurrence.measure(
            spectrum_of(performance), beats.grid(performance.beat_events()).downbeats, 4, BARS
        )
        assert found is not None
        assert not found.is_found

    def test_bars_in_a_random_order_do_not_recur(self):
        performance = the_take(intro=INTRO, pickup=2, varied=True)
        rng = np.random.default_rng(11)
        for _ in range(4):
            shuffled = _shuffle_bars(performance, rng)
            found = recurrence.measure(*shuffled, 4, BARS)
            assert not found.is_found

    def test_a_single_chorus_is_the_chart_match_alone(self):
        performance = synthesise_performance(
            progression=CHART, bpm=BPM, sample_rate=SAMPLE_RATE, repeats=1, pickup_beats=2
        )
        anchor, grid = locate(performance)
        plain = chroma.anchor_bar(
            spectrum_of(performance),
            grid.downbeats,
            chroma.bar_templates(performance.reference, BARS),
            min_bars=form.CHORUS_MIN_BARS,
        )
        assert anchor.evidence == "chart"
        assert (anchor.bar_index, anchor.start) == (plain.bar_index, plain.start)
        assert anchor.pickup_beats is None
        assert anchor.refusal() == plain.refusal()

    def test_a_vamp_on_one_chord_recurs_everywhere_so_it_claims_nothing(self):
        performance = synthesise_performance(
            progression=("C",) * BARS, bpm=BPM, sample_rate=SAMPLE_RATE, repeats=4
        )
        found = recurrence.measure(
            spectrum_of(performance), beats.grid(performance.beat_events()).downbeats, 4, BARS
        )
        assert not found.is_found

    def test_a_recording_shorter_than_two_forms_is_not_measured(self):
        performance = synthesise_performance(
            progression=CHART, bpm=BPM, sample_rate=SAMPLE_RATE, repeats=1
        )
        found = recurrence.measure(
            spectrum_of(performance), beats.grid(performance.beat_events()).downbeats, 4, BARS
        )
        assert found is None


def _shuffle_bars(performance, rng):
    """The chromagram cut at the bar lines and put back in a random order, with its bar lines."""
    spectrum = spectrum_of(performance)
    hop = spectrum.hop
    pieces = []
    for a, b in zip(performance.downbeats, performance.downbeats[1:], strict=False):
        low = int(a / hop)
        pieces.append(spectrum.values[:, low : max(int(b / hop), low + 1)])
    order = rng.permutation(len(pieces))
    downbeats, time = [0.0], 0.0
    for i in order:
        time += pieces[i].shape[1] * hop
        downbeats.append(time)
    values = np.concatenate([pieces[i] for i in order], axis=1)
    return chroma.Chromagram(values, hop), downbeats


class TestRecurrenceInternals:
    def test_a_run_survives_a_bar_that_differs(self):
        level, cap = 0.5, 0.5
        values = np.array([0.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 0.0])
        total, first, end = recurrence._best_run(values, level, cap)
        assert (first, end) == (1, 8)
        assert total == pytest.approx(2.5)

    def test_beats_before_the_first_bar_line_are_extrapolated_up_to_a_bar(self):
        edges, lead = recurrence.beat_times([1.0, 3.0, 5.0], 4)
        assert lead == 2
        assert edges[:3] == pytest.approx([0.0, 0.5, 1.0])
        assert len(edges) == lead + 2 * 4 + 1
        _, none = recurrence.beat_times([0.0, 2.0], 4)
        assert none == 0

    def test_lags_that_are_a_whole_or_half_form_are_not_in_the_null(self):
        lags = recurrence._null_lags(300, 64, 4)
        assert all(abs(lag - 32 * round(lag / 32)) > 1 for lag in lags)
        assert 31 not in lags and 33 not in lags and 64 not in lags
        assert min(lags) == recurrence.MIN_NULL_LAG_BARS * 4


class TestFormCommandShowsHowTheStartWasFound:
    @pytest.fixture
    def files(self, tmp_path):
        performance = the_take(intro=INTRO, pickup=2, varied=True)
        paths = {
            "reference": tmp_path / "reference.lab",
            "beats": tmp_path / "song.beats",
            "wrong": tmp_path / "wrong.beats",
            "audio": tmp_path / "song.wav",
        }
        lab.write(paths["reference"], performance.reference)
        lab.write_beats(paths["beats"], performance.beat_events())
        lab.write_beats(paths["wrong"], performance.beat_events(2))
        sf.write(paths["audio"], performance.audio, SAMPLE_RATE)
        return paths

    def run(self, files, capsys, beats_file="beats"):
        argv = ["form", str(files["reference"]), "--bars", str(BARS)]
        code = cli.main([*argv, "--audio", str(files["audio"]), "--beats", str(files[beats_file])])
        return code, capsys.readouterr().out

    def test_it_prints_the_evidence_intro_pickup_and_the_check(self, files, capsys):
        code, out = self.run(files, capsys)
        assert code == 0
        assert "found by recurrence" in out
        assert "intro 4 bars, pickup 2 beats" in out
        assert "chart match alone agrees" in out
        assert "5.00 s" in out

    def test_it_says_when_the_bar_lines_were_moved(self, files, capsys):
        code, out = self.run(files, capsys, "wrong")
        assert code == 0
        assert "bar lines moved" in out
        assert "5.00 s" in out
        assert [line.split()[1] for line in out.splitlines() if line[:3].strip().isdigit()][:2] == [
            "intro",
            "head",
        ]

    def test_the_json_is_additive(self):
        anchor, _ = locate(the_take(intro=INTRO, pickup=2))
        data = cli._anchor_to_dict(anchor)
        assert {"bar_index", "start", "agreement", "runner_up", "margin"} <= data.keys()
        assert data["evidence"] == "recurrence"
        assert (data["intro_bars"], data["pickup_beats"]) == (4, 2)
