from __future__ import annotations

import json
import random

import pytest
import soundfile as sf

from songscribe import beats, demo, lab
from songscribe.demo import synthesise_performance
from songscribe.eval import chart_fit, cli, dataset, form, form_start, harte, runner
from songscribe.types import ChordEvent

CHART = ("C", "Am", "Dm", "G7", "Em", "A7", "Dm", "G7")
JUNK = ("F#", "Bb", "Eb", "B", "Ab", "D", "F#", "Bb")
BAR = 2.0
SAMPLE_RATE = 11_025
NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def timeline(labels) -> list[ChordEvent]:
    return [ChordEvent(i * BAR, (i + 1) * BAR, label) for i, label in enumerate(labels)]


def anchor_of(labels, chart=CHART, shifts=chart_fit.ANY_KEY):
    grid = beats.BeatGrid(downbeats=tuple(i * BAR for i in range(len(labels) + 1)))
    anchor, _ = form_start.locate(
        timeline(labels), grid, timeline(chart), bars=len(chart), min_bars=4, shifts=shifts
    )
    return anchor


def moved(labels, semitones):
    return [harte.transpose_label(label, semitones) for label in labels]


class TestTransposeLabel:
    @pytest.mark.parametrize(
        ("label", "semitones", "expected"),
        [
            ("C", 2, "D"),
            ("Am", 3, "Cm"),
            ("G7", 5, "C7"),
            ("Bb", 2, "C"),
            ("C:maj", 1, "C#:maj"),
            ("A:min7", -3, "F#:min7"),
            ("E:7(b9)", 4, "G#:7(b9)"),
            ("C:maj/3", 7, "G:maj/3"),
            ("C", 12, "C"),
            ("C", -12, "C"),
        ],
    )
    def test_the_root_moves_and_the_quality_stays(self, label, semitones, expected):
        assert harte.transpose_label(label, semitones) == expected

    @pytest.mark.parametrize("label", ["N", "X", "", "not a chord", "Bogus"])
    def test_no_chord_and_unreadable_labels_are_left_alone(self, label):
        assert harte.transpose_label(label, 5) == label

    def test_the_pitch_content_moves_with_it(self):
        for label in ("C", "Am7", "G:7", "F#:hdim7"):
            before = harte.parse(label)
            after = harte.parse(harte.transpose_label(label, 4))
            assert after.intervals == before.intervals
            assert after.root == (before.root + 4) % 12


class TestDemoTransposition:
    def test_a_transposed_take_plays_in_the_new_key_and_the_chart_stays(self):
        take = synthesise_performance(CHART, repeats=2, sample_rate=SAMPLE_RATE, transpose=2)
        assert [e.label for e in take.events[:3]] == ["D", "Bm", "Em"]
        assert [e.label for e in take.reference[:3]] == ["C", "Am", "Dm"]

    def test_the_cli_flags_reach_the_synthesis(self, tmp_path):
        out = tmp_path / "take.wav"
        argv = ["-o", str(out), "-p", *CHART, "-r", "3", "--sample-rate", "8000"]
        argv += ["--transpose-take", "3"]
        assert demo.main(argv) == 0
        events = lab.read(tmp_path / "take.lab")
        assert [events[i * 8].label for i in range(3)] == ["D#"] * 3
        assert lab.read(tmp_path / "take.reference.lab")[0].label == "C"


class TestAnchorAcrossKeys:
    def test_the_same_key_gives_the_same_answer_as_the_strict_search(self):
        labels = [*JUNK[:3], *CHART, *JUNK, *CHART]
        strict = anchor_of(labels, shifts=chart_fit.AS_WRITTEN)
        free = anchor_of(labels)
        assert (strict.bar_index, strict.shift) == (3, 0)
        assert (free.bar_index, free.shift) == (3, 0)
        assert free.label_agreement == pytest.approx(strict.label_agreement)
        assert free.refusal() is None

    @pytest.mark.parametrize("shift", range(-5, 7))
    def test_a_whole_take_in_any_key_is_found_with_its_shift(self, shift):
        anchor = anchor_of(moved([*JUNK[:3], *CHART * 3], shift))
        assert (anchor.bar_index, anchor.shift) == (3, shift)
        assert anchor.label_agreement > 0.85
        assert anchor.label_agreement - anchor.chance > form_start.ANCHOR_MIN_EXCESS
        assert anchor.refusal() is None

    def test_the_strict_search_refuses_a_take_in_another_key(self):
        anchor = anchor_of(moved(CHART * 3, 2), shifts=chart_fit.AS_WRITTEN)
        assert anchor.refusal() is not None

    def test_a_chart_that_is_not_the_take_is_refused_in_every_key(self):
        anchor = anchor_of(JUNK * 3)
        assert anchor.refusal() is not None

    def test_random_unrelated_charts_do_not_become_right_by_searching_keys(self):
        rng = random.Random(3)

        def random_chart():
            return [rng.choice(NAMES) + rng.choice(["", "m", "7", "m7"]) for _ in range(8)]

        accepted = sum(
            anchor_of(random_chart() * 3, chart=random_chart()).refusal() is None
            for _ in range(300)
        )
        assert accepted == 0

    def test_a_form_that_fits_one_key_at_its_only_start_is_still_refused(self):
        # A take exactly one chorus long has no other phase to be told from.
        assert anchor_of(list(CHART)).refusal() is not None

    def test_the_chance_level_of_the_search_sets_the_bar_the_fit_has_to_clear(self):
        low = form_start.Anchor(0, 0.0, 0.6, 0.2, 0.4, chance=0.3)
        high = form_start.Anchor(0, 0.0, 0.6, 0.2, 0.4, chance=0.58)
        assert low.refusal() is None
        assert "under" in high.refusal()


class TestScoredAcrossKeys:
    def evaluate(
        self, tmp_path, performance, estimate, transpose="auto", dropped_bar=None, chart=None
    ):
        # The start is found on the chords played, whatever estimate is scored.
        audio = tmp_path / "audio"
        audio.mkdir(parents=True, exist_ok=True)
        sf.write(audio / "song.wav", performance.audio, SAMPLE_RATE)
        beats_dir, estimates = tmp_path / "beats", tmp_path / "est"
        anchors = tmp_path / "anchors"
        beat_events = performance.beat_events()
        if dropped_bar is not None:
            lost = performance.downbeats[dropped_bar]
            beat_events = [b for b in beat_events if b.time != lost]
        lab.write_beats(beats_dir / "song.beats", beat_events)
        lab.write(estimates / "song.lab", estimate)
        lab.write(anchors / "song.lab", performance.events)
        lab.write(tmp_path / "ann" / "song.lab", chart or performance.reference)
        manifest = tmp_path / "m.json"
        entry = {"id": "song", "reference": "song.lab", "audio": "song.wav", "bars": len(CHART)}
        dataset.write_manifest(manifest, dataset.build("take", [entry]))
        loaded = dataset.load_manifest(manifest, audio_root=audio, annotation_root=tmp_path / "ann")
        return runner.evaluate(
            loaded,
            estimates_dir=estimates,
            beats_dir=beats_dir,
            anchor_dir=anchors,
            check_alignment=False,
            transpose=transpose,
        )

    @staticmethod
    def take(**kwargs):
        defaults = {
            "progression": CHART,
            "bpm": 240.0,
            "sample_rate": SAMPLE_RATE,
            "repeats": 3,
            "pickup_beats": 3,
        }
        return synthesise_performance(**{**defaults, **kwargs})

    @staticmethod
    def wrong_fifth_bar(events):
        # A pickup is one event before bar 0, so it is skipped when counting bars.
        pickup = len(events) % len(CHART)
        return [
            ChordEvent(e.start, e.end, "A#" if (i - pickup) % len(CHART) == 4 else e.label)
            for i, e in enumerate(events)
        ]

    def test_a_take_a_minor_third_up_scores_as_the_same_take_in_the_written_key(self, tmp_path):
        plain = self.take()
        up = self.take(transpose=3)
        (home,) = self.evaluate(tmp_path / "a", plain, self.wrong_fifth_bar(plain.events)).results
        (away,) = self.evaluate(tmp_path / "b", up, self.wrong_fifth_bar(up.events)).results
        assert (home.anchor.shift, away.anchor.shift) == (0, 3)
        assert [c.chorus.shift for c in away.choruses] == [3, 3, 3]
        assert away.bar == pytest.approx(home.bar) == pytest.approx(7 / 8)
        assert away.scores["majmin"].wcsr == pytest.approx(home.scores["majmin"].wcsr)
        assert away.chart == pytest.approx(home.chart)

    def test_the_pickup_of_a_transposed_take_is_an_unscored_intro_in_the_take_key(self, tmp_path):
        up = self.take(transpose=3)
        # The chart's last chord, G7, a minor third up.
        assert (up.events[0].start, up.events[0].label) == (0.0, "A#7")
        (result,) = self.evaluate(tmp_path, up, up.events).results
        assert (result.anchor.bar_index, result.anchor.shift) == (0, 3)
        assert result.anchor.start == pytest.approx(0.75, abs=1e-3)
        assert result.unscored_seconds == pytest.approx(0.75, abs=1e-3)
        assert result.bar == 1.0

    def test_the_estimate_is_never_moved(self, tmp_path):
        # An estimate in the written key against a take that is a tone up is wrong everywhere.
        up = self.take(transpose=2)
        stale = self.take().events
        (result,) = self.evaluate(tmp_path, up, stale).results
        assert result.anchor.shift == 2
        assert result.bar == 0.0

    def test_one_key_holds_for_every_chorus_even_when_one_is_reharmonised(self, tmp_path):
        performance = self.take(repeats=4, transpose=2, substitutions={1: {0: "F#", 1: "D"}})
        (result,) = self.evaluate(tmp_path, performance, performance.events).results
        assert [c.chorus.shift for c in result.choruses] == [2, 2, 2, 2]
        # Two bars of one chorus differ from the chart, and only those two are wrong.
        assert [c.bar for c in result.choruses] == pytest.approx([1.0, 0.75, 1.0, 1.0])

    def test_a_dropped_bar_still_leaves_out_the_lost_chorus_in_a_transposed_take(self, tmp_path):
        performance = self.take(repeats=4, transpose=2)
        (result,) = self.evaluate(tmp_path, performance, performance.events, dropped_bar=27).results
        assert [c.chorus.shift for c in result.choruses] == [2, 2, 2]
        assert [c.chorus.kind for c in result.choruses] == ["head", "chorus", "head"]
        assert result.unscored_seconds == pytest.approx(0.75 + 8.0, abs=1e-3)

    def test_without_the_key_search_a_transposed_take_is_skipped_as_before(self, tmp_path):
        up = self.take(transpose=3)
        report = self.evaluate(tmp_path, up, up.events, transpose="none")
        assert report.skipped == ("song",)
        assert not report.results

    def test_without_the_key_search_the_written_key_is_scored_as_before(self, tmp_path):
        plain = self.take()
        (result,) = self.evaluate(tmp_path, plain, plain.events, transpose="none").results
        assert result.anchor.shift == 0
        assert result.anchor.keys == 1
        assert result.bar == 1.0

    def test_a_chart_in_no_key_is_still_refused(self, tmp_path):
        stranger = self.take(progression=JUNK)
        chart = self.take().reference
        report = self.evaluate(tmp_path, stranger, stranger.events, chart=chart)
        assert report.skipped == ("song",)

    def test_an_unknown_transpose_mode_is_an_error(self, tmp_path):
        plain = self.take()
        with pytest.raises(ValueError, match="transpose"):
            self.evaluate(tmp_path, plain, plain.events, transpose="up")


class TestCommands:
    @pytest.fixture
    def files(self, tmp_path):
        performance = synthesise_performance(
            CHART,
            bpm=240.0,
            repeats=3,
            sample_rate=SAMPLE_RATE,
            pickup_beats=3,
            transpose=-2,
        )
        paths = {
            "reference": tmp_path / "reference.lab",
            "beats": tmp_path / "song.beats",
            "estimate": tmp_path / "song.lab",
            "audio": tmp_path / "song.wav",
        }
        lab.write(paths["reference"], performance.reference)
        lab.write_beats(paths["beats"], performance.beat_events())
        lab.write(paths["estimate"], performance.events)
        sf.write(paths["audio"], performance.audio, SAMPLE_RATE)
        return paths

    def run_form(self, files, capsys, *extra):
        argv = ["form", str(files["reference"]), "--bars", "8", "--beats", str(files["beats"])]
        code = cli.main([*argv, "--estimate", str(files["estimate"]), *extra])
        return code, capsys.readouterr().out

    def test_the_form_table_has_a_key_column(self, files, capsys):
        code, out = self.run_form(files, capsys)
        assert code == 0
        assert "key -2" in out
        rows = [line.split() for line in out.splitlines() if line[:3].strip().isdigit()]
        assert [(row[1], row[-1]) for row in rows] == [
            ("intro", "--"),
            ("head", "-2"),
            ("chorus", "-2"),
            ("head", "-2"),
        ]

    def test_the_strict_form_refuses_with_exit_two(self, files, capsys):
        code, out = self.run_form(files, capsys, "--transpose", "none")
        assert code == 2
        assert "NOT CLEAR" in out

    def test_score_reports_the_key_on_every_chorus_and_in_the_json(self, files, tmp_path, capsys):
        entry = {"id": "song", "reference": "reference.lab", "audio": "song.wav", "bars": 8}
        manifest = tmp_path / "m.json"
        dataset.write_manifest(manifest, dataset.build("take", [entry]))
        estimates = tmp_path / "est"
        estimates.mkdir()
        (estimates / "song.lab").write_text(files["estimate"].read_text(encoding="utf-8"))
        anchors = tmp_path / "anchors"
        anchors.mkdir()
        (anchors / "song.lab").write_text(files["estimate"].read_text(encoding="utf-8"))
        beats_dir = tmp_path / "beats"
        beats_dir.mkdir()
        (beats_dir / "song.beats").write_text(files["beats"].read_text(encoding="utf-8"))
        output = tmp_path / "out.json"
        argv = ["score", str(manifest), "--estimates", str(estimates), "--beats", str(beats_dir)]
        argv += ["--anchor-estimates", str(anchors)]
        argv += ["--audio", str(tmp_path), "--annotations", str(tmp_path), "--per-chorus"]
        assert cli.main([*argv, "--json", str(output)]) == 0
        out = capsys.readouterr().out
        assert [line.split()[-1] for line in out.splitlines() if " #" in line] == ["-2"] * 3
        (track,) = json.loads(output.read_text(encoding="utf-8"))["tracks"]
        assert [c["shift"] for c in track["choruses"]] == [-2] * 3
        assert track["anchor"]["shift"] == -2
        assert track["bar"] == 1.0

        # The strict run skips the track, exactly as a wrong-key chart always did.
        assert cli.main([*argv, "--transpose", "none"]) == 1
        capsys.readouterr()


def test_transposed_events_leave_the_originals_alone():
    original = [ChordEvent(0.0, 1.0, "Am"), ChordEvent(1.0, 2.0, "N")]
    assert [e.label for e in form.transpose_events(original, 3)] == ["Cm", "N"]
    assert original[0].label == "Am"
