from __future__ import annotations

import json

import pytest
import soundfile as sf

from songscribe import lab
from songscribe.demo import synthesise_performance
from songscribe.eval import cli, dataset

CHART = ("C", "Am", "Dm", "G7", "Em", "A7", "Dm", "G7")
STRANGER = ("F#", "Bb", "Eb", "B", "Ab", "D", "F#", "Bb")
SAMPLE_RATE = 11_025


def take(progression=CHART):
    return synthesise_performance(
        progression=progression,
        bpm=240.0,
        repeats=3,
        sample_rate=SAMPLE_RATE,
        lead_in_seconds=3.0,
        substitutions={1: {2: "Bb"}},
    )


@pytest.fixture
def files(tmp_path):
    performance = take()
    paths = {
        "reference": tmp_path / "reference.lab",
        "beats": tmp_path / "song.beats",
        "estimate": tmp_path / "song.lab",
        "audio": tmp_path / "song.wav",
        "stranger": tmp_path / "stranger.wav",
    }
    lab.write(paths["reference"], performance.reference)
    lab.write_beats(paths["beats"], performance.beat_events())
    lab.write(paths["estimate"], performance.events)
    sf.write(paths["audio"], performance.audio, SAMPLE_RATE)
    sf.write(paths["stranger"], take(STRANGER).audio, SAMPLE_RATE)
    return paths


def run_form(files, capsys, *extra):
    argv = ["form", str(files["reference"]), "--bars", "8", "--beats", str(files["beats"])]
    code = cli.main([*argv, *extra])
    return code, capsys.readouterr()


class TestFormCommand:
    def test_it_prints_the_grid_the_anchor_and_the_choruses(self, files, capsys):
        code, out = run_form(files, capsys, "--audio", str(files["audio"]))
        assert code == 0
        assert "bar grid  24 bars, median 1.000 s (240.0 bpm in 4 beats)" in out.out
        assert "anchor    bar 0 at 3.00 s" in out.out
        kinds = [line.split()[1] for line in out.out.splitlines() if line[:3].strip().isdigit()]
        assert kinds == ["intro", "head", "chorus", "head"]
        assert "3.0 s not scored" in out.out

    def test_anchoring_on_an_estimate_says_it_is_not_for_scoring(self, files, capsys):
        code, out = run_form(files, capsys, "--estimate", str(files["estimate"]))
        assert code == 0
        assert "do not use it for scoring" in out.out

    def test_a_recording_that_is_not_the_chart_is_refused_with_exit_two(self, files, capsys):
        code, out = run_form(files, capsys, "--audio", str(files["stranger"]))
        assert code == 2
        assert "NOT CLEAR" in out.out
        assert "head" not in out.out

    def test_it_wants_exactly_one_source_of_evidence(self, files, capsys):
        assert run_form(files, capsys)[0] == 1
        both = ("--audio", str(files["audio"]), "--estimate", str(files["estimate"]))
        assert run_form(files, capsys, *both)[0] == 1

    def test_a_missing_file_is_an_error_not_a_traceback(self, files, capsys):
        files["beats"].unlink()
        code, out = run_form(files, capsys, "--audio", str(files["audio"]))
        assert code == 1
        assert "songscribe-eval:" in out.err


class TestScoreWithABarGrid:
    @pytest.fixture
    def manifest(self, files, tmp_path):
        beats_dir = tmp_path / "beats"
        beats_dir.mkdir()
        (beats_dir / "song.beats").write_text(files["beats"].read_text(encoding="utf-8"))
        estimates = tmp_path / "est"
        estimates.mkdir()
        (estimates / "song.lab").write_text(files["estimate"].read_text(encoding="utf-8"))
        path = tmp_path / "m.json"
        entry = {"id": "song", "reference": "reference.lab", "audio": "song.wav", "bars": 8}
        dataset.write_manifest(path, dataset.build("take", [entry]))
        return [
            "score",
            str(path),
            "--estimates",
            str(estimates),
            "--beats",
            str(beats_dir),
            "--audio",
            str(tmp_path),
            "--annotations",
            str(tmp_path),
        ]

    def test_per_chorus_rows_and_a_bar_summary(self, manifest, capsys):
        assert cli.main([*manifest, "--per-chorus"]) == 0
        out = capsys.readouterr().out
        assert "#1 head" in out
        assert "#2 chorus" in out
        assert "#3 head" in out
        assert "bars with the right chord" in out
        assert "3 choruses in 1 tracks" in out

    def test_the_json_keeps_every_old_key_and_adds_the_choruses(self, manifest, tmp_path):
        output = tmp_path / "out.json"
        assert cli.main([*manifest, "--json", str(output)]) == 0
        document = json.loads(output.read_text(encoding="utf-8"))
        (track,) = document["tracks"]
        assert {"scores", "chart", "grid", "segmentation", "alignment"} <= set(track)
        assert [c["kind"] for c in track["choruses"]] == ["head", "chorus", "head"]
        assert track["anchor"]["bar_index"] == 0
        # The second chorus plays Bb where the chart has Dm: one bar in 24.
        assert track["bar"] == pytest.approx(23 / 24)
        assert [c["bar"] for c in track["choruses"]] == pytest.approx([1.0, 7 / 8, 1.0])
        assert document["bar"]["choruses"] == 3

    def test_per_chorus_without_a_bar_grid_is_an_error(self, manifest, capsys):
        without = [
            a for a in manifest if a not in ("--beats", manifest[manifest.index("--beats") + 1])
        ]
        assert cli.main([*without, "--per-chorus"]) == 1
        assert "--per-chorus needs" in capsys.readouterr().err

    def test_a_plain_score_has_no_bar_row(self, files, tmp_path, capsys):
        # A single-chorus track scores as it always has.
        estimates = tmp_path / "plain"
        estimates.mkdir()
        (estimates / "short.lab").write_text("0.0 1.0 C\n", encoding="utf-8")
        (tmp_path / "short.lab").write_text("0.0 1.0 C:maj\n", encoding="utf-8")
        path = tmp_path / "plain.json"
        dataset.write_manifest(
            path, dataset.build("p", [{"id": "short", "reference": "short.lab"}])
        )
        argv = ["score", str(path), "--estimates", str(estimates), "--annotations", str(tmp_path)]
        assert cli.main(argv) == 0
        assert "right chord in the right bar" not in capsys.readouterr().out
