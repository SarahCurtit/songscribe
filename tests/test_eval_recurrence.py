"""Finding the form's start from a chord estimate recurring one form later, checked by the chart."""

from __future__ import annotations

from functools import cache
from itertools import pairwise

import numpy as np
import pytest

from songscribe import beats, lab
from songscribe.demo import synthesise_performance
from songscribe.eval import cli, form, form_start, labels, recurrence
from songscribe.types import ChordEvent

SAMPLE_RATE = 4_000
BPM = 240.0  # a beat every quarter second, a bar every second
CHART = ("Dm", "G7", "C", "Am", "Em", "A7", "Dm", "G7", "Cmaj7", "Fmaj7", "Bdim", "E7")
CHART = (*CHART, "Am", "Dm", "G7", "C")
BARS = len(CHART)
#: Far from the tune's keys, so nothing in it resembles the chart by accident.
INTRO = ("F#", "B", "F#", "B7", "Ebm")
SOLO_CHANGES = {1: {2: "F", 5: "Bb", 9: "Dm"}, 2: {3: "C", 6: "G", 11: "Am"}}


@cache
def the_take(intro=(), pickup=0, varied=False):
    """A cached take of the chart four times, optionally with an intro, pickup and solo changes."""
    kwargs = {"substitutions": SOLO_CHANGES} if varied else {}
    return synthesise_performance(
        progression=CHART,
        bpm=BPM,
        sample_rate=SAMPLE_RATE,
        repeats=4,
        intro=intro,
        pickup_beats=pickup,
        **kwargs,
    )


def locate(performance, estimate=None, tracker_phase=0, shifts=(0,)):
    grid = beats.grid(performance.beat_events(tracker_phase))
    return form_start.locate(
        performance.events if estimate is None else estimate,
        grid,
        performance.reference,
        bars=BARS,
        min_bars=form.CHORUS_MIN_BARS,
        shifts=shifts,
    )


def measure(performance, estimate=None, bars=BARS):
    downbeats = beats.grid(performance.beat_events()).downbeats
    symbols, edges, lead = recurrence.beats_of(
        performance.events if estimate is None else estimate, downbeats, 4
    )
    return recurrence.measure(symbols, edges, lead, 4, bars)


# Ways a chord estimate differs from the chords played, all deterministic.


def relabelled(events, mapping):
    return [ChordEvent(e.start, e.end, mapping.get(e.label, e.label)) for e in events]


def flickering(events, every=2, length=0.1):
    """A wrong chord for ``length`` s in the middle of every ``every``-th chord."""
    out = []
    for i, e in enumerate(events):
        if i % every or e.end - e.start < 3 * length:
            out.append(e)
            continue
        mid = (e.start + e.end) / 2
        out += [
            ChordEvent(e.start, mid, e.label),
            ChordEvent(mid, mid + length, "F#m"),
            ChordEvent(mid + length, e.end, e.label),
        ]
    return out


def dropping(events, every=7):
    return [
        ChordEvent(e.start, e.end, "N" if i % every == 3 else e.label) for i, e in enumerate(events)
    ]


def anticipating(events, by=0.08):
    """Every chord change arrives ``by`` s before the beat it belongs to."""
    out = [ChordEvent(events[0].start, events[0].end - by, events[0].label)]
    for e in events[1:-1]:
        out.append(ChordEvent(e.start - by, e.end - by, e.label))
    out.append(ChordEvent(events[-1].start - by, events[-1].end, events[-1].label))
    return out


#: A different estimator's habits: relative-key swaps, sevenths where the chart has triads
#: and the reverse, a chord it never names.
RELATIVE = {"C": "Am", "Am": "C", "Dm": "F", "Em": "G"}
SEVENTHS = {"Dm": "Dm7", "C": "Cmaj7", "Am": "Am7", "Em": "Em7", "G7": "G", "Cmaj7": "C"}
ROUGH = [
    ("flicker", lambda e: flickering(e)),
    ("dropped chords", lambda e: dropping(e)),
    ("anticipated changes", lambda e: anticipating(e)),
    ("sevenths against triads", lambda e: relabelled(e, SEVENTHS)),
    ("relative swaps", lambda e: relabelled(e, {"Am": "C", "Em": "G"})),
    (
        "all of it",
        lambda e: anticipating(flickering(dropping(relabelled(e, SEVENTHS), every=11), every=3)),
    ),
]


class TestRecurrenceStart:
    @pytest.mark.parametrize("pickup", [0, 1, 2, 3])
    def test_intro_and_pickup_are_recovered_from_the_chords(self, pickup):
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
        # F, G, F, G7, Em share chords with the chart's last bars one form on; the chart
        # puts the head where it fits, not where the recurrence first opens.
        performance = the_take(intro=("F", "G", "F", "G7", "Em"))
        anchor, _ = locate(performance)
        assert anchor.start == pytest.approx(performance.head_start)
        assert anchor.chart_agrees is True

    def test_a_head_that_differs_from_the_solos_is_still_found_by_the_chart_fit(self):
        # The head's first two bars are other chords, so the recurrence opens two bars late;
        # the chart fits the head's real first bar clearly better.
        changes = {0: {0: "F", 1: "Bdim"}}
        performance = synthesise_performance(
            progression=CHART, bpm=BPM, sample_rate=SAMPLE_RATE, repeats=4, substitutions=changes
        )
        anchor, _ = locate(performance)
        assert anchor.start == pytest.approx(0.0)
        assert anchor.evidence == "recurrence"
        assert measure(performance).first_beat > 0

    def test_a_first_chorus_the_chart_does_not_describe_is_left_out_as_an_intro(self):
        # Three bars in four of the first pass are other chords: the chart describes the
        # second, which is where the form starts, and the first is not scored.
        other = {i: "F#" for i in range(BARS) if i % 4}
        performance = synthesise_performance(
            progression=CHART, bpm=BPM, sample_rate=SAMPLE_RATE, repeats=4, substitutions={0: other}
        )
        anchor, _ = locate(performance)
        assert anchor.start == pytest.approx(BARS * 1.0)

    def test_the_chart_fit_is_reported_with_the_start_and_beats_the_shuffled_charts(self):
        anchor, _ = locate(the_take(intro=INTRO, pickup=2))
        assert anchor.label_agreement > anchor.gate() > anchor.chance
        assert anchor.chart_agrees is True
        assert anchor.ending_bars == 0

    @pytest.mark.parametrize(("name", "rough"), ROUGH, ids=[name for name, _ in ROUGH])
    def test_it_holds_on_an_estimate_that_only_roughly_matches_what_was_played(self, name, rough):
        performance = the_take(intro=INTRO, pickup=2, varied=True)
        anchor, _ = locate(performance, rough(performance.events))
        assert anchor.evidence == "recurrence"
        assert anchor.start == pytest.approx(performance.head_start)
        assert anchor.pickup_beats == 2
        assert anchor.is_clear

    def test_seventh_chords_and_triads_are_one_symbol_but_other_thirds_are_not(self):
        performance = the_take()
        full = measure(performance, relabelled(performance.events, SEVENTHS))
        assert full.is_found
        parallel = {"Dm": "D", "Am": "A", "Em": "E"}
        assert measure(performance, relabelled(performance.events, parallel)).is_found
        assert measure(performance, relabelled(performance.events, parallel)).separation < (
            full.separation
        )


class TestTrackerPhase:
    @pytest.mark.parametrize("phase", [1, 2, 3])
    def test_bar_lines_a_few_beats_off_are_moved_to_where_the_chords_change(self, phase):
        performance = the_take(intro=INTRO, pickup=2, varied=True)
        anchor, grid = locate(performance, tracker_phase=phase)
        assert anchor.start == pytest.approx(performance.head_start)
        assert anchor.phase_shift != 0
        assert anchor.pickup_beats == 2
        assert grid.downbeats[anchor.bar_index] == pytest.approx(anchor.start)

    @pytest.mark.parametrize("phase", [1, 3])
    def test_they_are_moved_on_a_rough_estimate_too(self, phase):
        performance = the_take(intro=INTRO, pickup=2, varied=True)
        rough = anticipating(flickering(dropping(performance.events)))
        anchor, _ = locate(performance, rough, tracker_phase=phase)
        assert anchor.start == pytest.approx(performance.head_start)
        assert anchor.phase_shift != 0

    def test_moved_bar_lines_still_close_the_last_chorus(self):
        performance = the_take(intro=INTRO, pickup=2, varied=True)
        anchor, grid = locate(performance, tracker_phase=2)
        blocks = form.choruses(grid, BARS, anchor.bar_index, performance.reference[-1].end)
        assert [b.kind for b in blocks] == ["intro", "head", "chorus", "chorus", "head"]

    def test_a_correct_tracker_is_left_alone(self):
        performance = the_take(intro=INTRO, pickup=2, varied=True)
        anchor, grid = locate(performance)
        assert anchor.phase_shift == 0
        assert grid.downbeats == beats.grid(performance.beat_events()).downbeats

    def test_a_correct_tracker_is_left_alone_on_a_rough_estimate(self):
        performance = the_take(intro=INTRO, pickup=2, varied=True)
        rough = anticipating(flickering(dropping(performance.events)))
        anchor, _ = locate(performance, rough)
        assert anchor.phase_shift == 0

    def test_a_rival_beat_needs_the_chords_to_change_there_clearly(self):
        performance = the_take(intro=INTRO, pickup=2, varied=True)
        found = measure(performance)
        residue, t = recurrence.bar_line_phase(found, found.lead % 4)
        assert residue == found.lead % 4
        assert t < recurrence.PHASE_MIN_T
        off, _ = recurrence.bar_line_phase(found, (found.lead + 1) % 4)
        assert off == found.lead % 4


def shuffled_bars(performance, seed):
    """The estimate cut at the bar lines and put back in a random order, with its bar lines."""
    down = performance.downbeats
    pieces = [
        [
            (max(e.start, a) - a, min(e.end, b) - a, e.label)
            for e in performance.events
            if e.end > a and e.start < b
        ]
        for a, b in pairwise(down)
    ]
    order = np.random.default_rng(seed).permutation(len(pieces))
    events = [
        ChordEvent(k + s, k + t, label) for k, i in enumerate(order) for s, t, label in pieces[i]
    ]
    return events, [float(k) for k in range(len(order) + 1)]


def measure_events(events, downbeats, bars=BARS):
    symbols, edges, lead = recurrence.beats_of(events, downbeats, 4)
    return recurrence.measure(symbols, edges, lead, 4, bars)


class TestNothingRecurs:
    def test_unrelated_chords_make_no_claim_and_are_not_a_form(self):
        rng = np.random.default_rng(3)
        names = ["C", "Dm", "Em", "F", "G", "Am", "G7", "A7", "Bdim", "Fmaj7", "Cmaj7", "E7", "D"]
        wandering = tuple(rng.choice(names, size=BARS * 4))
        performance = synthesise_performance(
            progression=wandering, bpm=BPM, sample_rate=SAMPLE_RATE, repeats=1
        )
        found = measure(performance)
        assert found is not None
        assert not found.is_found
        anchor, _ = locate(performance)
        assert anchor is None or anchor.refusal() is not None

    @pytest.mark.parametrize("seed", range(6))
    def test_bars_in_a_random_order_do_not_recur(self, seed):
        performance = the_take(intro=INTRO, pickup=2, varied=True)
        found = measure_events(*shuffled_bars(performance, seed))
        assert not found.is_found

    @pytest.mark.parametrize("seed", range(3))
    def test_a_chart_with_its_bars_shuffled_is_refused_on_a_real_start(self, seed):
        performance = the_take(intro=INTRO, pickup=2)
        order = np.random.default_rng(seed).permutation(BARS)
        wrong = [ChordEvent(k, k + 1.0, CHART[i]) for k, i in enumerate(order)]
        grid = beats.grid(performance.beat_events())
        anchor, _ = form_start.locate(
            performance.events, grid, wrong, bars=BARS, min_bars=form.CHORUS_MIN_BARS
        )
        assert anchor is not None
        assert anchor.refusal() is not None

    def test_a_chart_of_other_chords_is_refused_on_a_real_start(self):
        performance = the_take(intro=INTRO, pickup=2)
        other = ("F#", "Bb", "Eb", "B", "Ab", "D", "F#", "Bb") * 2
        wrong = [ChordEvent(k, k + 1.0, label) for k, label in enumerate(other)]
        grid = beats.grid(performance.beat_events())
        anchor, _ = form_start.locate(
            performance.events, grid, wrong, bars=BARS, min_bars=form.CHORUS_MIN_BARS
        )
        assert anchor.refusal() is not None

    def test_a_single_chorus_is_the_chart_match_alone(self):
        performance = synthesise_performance(
            progression=CHART, bpm=BPM, sample_rate=SAMPLE_RATE, repeats=1, intro=INTRO
        )
        anchor, _ = locate(performance)
        assert anchor.evidence == "chart"
        assert anchor.start == pytest.approx(performance.head_start)
        assert anchor.pickup_beats is None
        assert anchor.refusal() is None

    def test_a_recording_that_is_exactly_one_chorus_has_only_one_place_to_start(self):
        performance = synthesise_performance(
            progression=CHART, bpm=BPM, sample_rate=SAMPLE_RATE, repeats=1
        )
        anchor, _ = locate(performance)
        assert "only one candidate" in anchor.refusal()

    def test_a_single_noisy_chorus_is_still_found_by_the_chart(self):
        performance = synthesise_performance(
            progression=CHART, bpm=BPM, sample_rate=SAMPLE_RATE, repeats=1, intro=INTRO[:2]
        )
        rough = anticipating(flickering(dropping(relabelled(performance.events, SEVENTHS))))
        anchor, _ = locate(performance, rough)
        assert anchor.evidence == "chart"
        assert anchor.start == pytest.approx(performance.head_start)
        assert anchor.is_clear

    def test_a_vamp_on_one_chord_recurs_everywhere_so_it_claims_nothing(self):
        performance = synthesise_performance(
            progression=("C",) * BARS, bpm=BPM, sample_rate=SAMPLE_RATE, repeats=4
        )
        assert not measure(performance).is_found
        anchor, _ = locate(performance)
        assert anchor.refusal() is not None

    def test_a_two_bar_loop_makes_no_claim_about_where_the_form_starts(self):
        performance = synthesise_performance(
            progression=("C", "Am") * (BARS // 2), bpm=BPM, sample_rate=SAMPLE_RATE, repeats=4
        )
        anchor, _ = locate(performance)
        assert anchor.refusal() is not None

    def test_a_recording_shorter_than_two_forms_is_not_measured(self):
        performance = synthesise_performance(
            progression=CHART, bpm=BPM, sample_rate=SAMPLE_RATE, repeats=1
        )
        assert measure(performance) is None


class TestKeys:
    def test_a_take_in_another_key_needs_the_key_search(self):
        performance = the_take(intro=INTRO, pickup=2)
        moved = form.transpose_events(performance.events, 3)
        as_written, _ = locate(performance, moved)
        assert as_written.refusal() is not None
        anchor, _ = locate(performance, moved, shifts=form_start.chart_fit.ANY_KEY)
        assert anchor.shift == 3
        assert anchor.keys == 12
        assert anchor.start == pytest.approx(performance.head_start)
        assert anchor.is_clear

    def test_more_keys_raise_the_chance_level_instead_of_lowering_the_bar(self):
        performance = the_take(intro=INTRO, pickup=2)
        one, _ = locate(performance)
        twelve, _ = locate(performance, shifts=form_start.chart_fit.ANY_KEY)
        assert twelve.chance >= one.chance
        assert twelve.key_margin > 0.0

    def test_the_chart_in_the_right_key_is_chosen_not_its_relative(self):
        performance = the_take(intro=INTRO, pickup=2)
        anchor, _ = locate(performance, shifts=form_start.chart_fit.ANY_KEY)
        assert anchor.shift == 0


class TestRecurrenceInternals:
    def test_a_run_survives_a_bar_that_differs(self):
        level, cap = 0.5, 0.5
        values = np.array([0.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 0.0])
        total, first, end = recurrence._best_run(values, level, cap)
        assert (first, end) == (1, 8)
        assert total == pytest.approx(2.5)

    def test_the_strongest_run_over_many_lags_is_the_best_of_each(self):
        rng = np.random.default_rng(0)
        lags = [rng.random(n) for n in (30, 50, 40)]
        expected = max(recurrence._best_run(v, 0.5, 0.2)[0] for v in lags)
        assert recurrence._strongest_run(lags, 0.5, 0.2) == pytest.approx(expected)

    def test_beats_before_the_first_bar_line_are_extrapolated_up_to_a_bar(self):
        edges, lead = recurrence.beat_times([1.0, 3.0, 5.0], 4)
        assert lead == 2
        assert edges[:3] == pytest.approx([0.0, 0.5, 1.0])
        assert len(edges) == lead + 2 * 4 + 1
        _, none = recurrence.beat_times([0.0, 2.0], 4)
        assert none == 0

    def test_lags_that_are_a_whole_or_half_form_are_not_in_the_null(self):
        lags = recurrence._null_lags(300, 64, 4)
        assert all(abs(lag - 32 * round(lag / 32)) > 2 for lag in lags)
        assert {30, 31, 33, 34, 62, 64, 66} & set(lags) == set()
        assert min(lags) == recurrence.MIN_NULL_LAG_BARS * 4

    def test_agreement_one_form_apart_is_whole_for_a_clean_repeat(self):
        found = measure(the_take())
        assert found.mean_similarity == pytest.approx(1.0)
        assert found.separation > 3.0


class TestFormCommandShowsHowTheStartWasFound:
    @pytest.fixture
    def files(self, tmp_path):
        performance = the_take(intro=INTRO, pickup=2, varied=True)
        paths = {
            "reference": tmp_path / "reference.lab",
            "beats": tmp_path / "song.beats",
            "wrong": tmp_path / "wrong.beats",
            "estimate": tmp_path / "song.lab",
        }
        lab.write(paths["reference"], performance.reference)
        lab.write_beats(paths["beats"], performance.beat_events())
        lab.write_beats(paths["wrong"], performance.beat_events(2))
        lab.write(paths["estimate"], flickering(performance.events))
        return paths

    def run(self, files, capsys, beats_file="beats"):
        argv = ["form", str(files["reference"]), "--bars", str(BARS)]
        code = cli.main(
            [*argv, "--estimate", str(files["estimate"]), "--beats", str(files[beats_file])]
        )
        return code, capsys.readouterr().out

    def test_it_prints_the_evidence_intro_pickup_and_the_check(self, files, capsys):
        code, out = self.run(files, capsys)
        assert code == 0
        assert "found by recurrence" in out
        assert "intro 4 bars, pickup 2 beats" in out
        assert "chart match alone agrees" in out
        assert "label agreement" in out
        assert "found on the estimate song.lab" in out
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

    def test_the_json_keeps_the_old_keys_and_names_the_new_agreement(self):
        anchor, _ = locate(the_take(intro=INTRO, pickup=2))
        data = cli._anchor_to_dict(anchor)
        assert {"bar_index", "start", "label_agreement", "chance", "runner_up", "margin"} <= (
            data.keys()
        )
        assert "agreement" not in data
        assert data["evidence"] == "recurrence"
        assert (data["intro_bars"], data["pickup_beats"]) == (4, 2)


def test_every_label_the_demo_plays_has_a_symbol():
    assert all(labels.symbol(label) != labels.NO_SYMBOL for label in (*CHART, *INTRO))
