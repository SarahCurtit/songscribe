"""Grid viewer: discovery, agreement scoring, and the HTTP API.

The example directory is built per test from invented song names, silent
WAVs and hand-written chord events -- no recordings, no real charts.
"""

from __future__ import annotations

import json
import threading
import time
import wave
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from songscribe import chords, lab
from songscribe import melody as melody_stage
from songscribe.types import BeatEvent, ChordEvent, NoteEvent
from songscribe.viewer import agreement, examples
from songscribe.viewer.server import STAGES, Job, Predictor, Viewer, truth_reply

SAMPLE_RATE = 8_000


def write_silence(path: Path, seconds: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(b"\x00\x00" * int(seconds * SAMPLE_RATE))


def write_grid(
    song_dir: Path, number: int, payload: dict, audio: bool = True, seconds: float = 4.0
) -> None:
    song_dir.mkdir(parents=True, exist_ok=True)
    (song_dir / f"grid{number}_chords.json").write_text(json.dumps(payload), encoding="utf-8")
    if audio:
        write_silence(song_dir / f"grid{number}_audio.wav", seconds)


ANNOTATED = {
    "song": "Paper Lantern Waltz",
    "grid": 2,
    "events": [
        {"start": 0.5, "end": 1.5, "chord": "C:maj7"},
        {"start": 1.5, "end": 2.5, "chord": "A:min"},
        {"start": 2.5, "end": 3.5, "chord": "D:hdim7"},
    ],
    "annotation": {
        "bars": 3,
        "bar_seconds": 1.0,
        "start_seconds": 0.5,
        "bpm_measured": 240.0,
        "audio_seconds": 4.0,
        "sections": [{"bar": 1, "name": "A"}],
        "alternates": [{"bar": 2, "chord": "F:7"}],
    },
}


@pytest.fixture
def root(tmp_path: Path) -> Path:
    write_grid(tmp_path / "paper-lantern-waltz", 2, ANNOTATED)
    plain = {"song": "Quiet Harbour", "events": [{"start": 0.0, "end": 2.0, "chord": "G:maj"}]}
    write_grid(tmp_path / "quiet-harbour", 1, plain)
    write_grid(tmp_path / "quiet-harbour", 5, plain, audio=False)
    return tmp_path


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


def test_discover_lists_grids_with_audio_only(root: Path) -> None:
    grids = examples.discover([root])
    assert list(grids) == ["0/paper-lantern-waltz/2", "0/quiet-harbour/1"]
    assert grids["0/paper-lantern-waltz/2"].song == "Paper Lantern Waltz"
    assert grids["0/paper-lantern-waltz/2"].screenshot is None


def test_discover_rejects_missing_and_empty_roots(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        examples.discover([tmp_path / "nope"])
    with pytest.raises(ValueError, match="no grid"):
        examples.discover([tmp_path])


def test_ground_truth_converts_bars_to_seconds(root: Path) -> None:
    truth = examples.ground_truth(examples.discover([root])["0/paper-lantern-waltz/2"])
    assert truth["bars"] == [0.5, 1.5, 2.5, 3.5]
    assert truth["sections"] == [{"start": 0.5, "bar": 1, "name": "A"}]
    assert truth["alternates"] == [
        {"start": 1.5, "end": 2.5, "bar": 2, "label": "F:7", "short": "F7"}
    ]
    assert [e["short"] for e in truth["events"]] == ["Cmaj7", "Am", "Dm7b5"]
    assert truth["annotated_end"] == 3.5
    assert truth["bpm"] == 240.0


def test_ground_truth_without_annotation(root: Path) -> None:
    truth = examples.ground_truth(examples.discover([root])["0/quiet-harbour/1"])
    assert truth["bars"] == truth["sections"] == truth["alternates"] == []
    assert truth["events"][0]["short"] == "G"


# --------------------------------------------------------------------------- #
# Agreement
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("reference", "estimate", "expected"),
    [
        ("Bb7", "A#7", "exact"),
        ("Bb7", "Bb", "majmin"),
        ("Am7", "Am", "majmin"),
        ("Dm7b5", "Dm", "root"),
        ("D:hdim7", "Ddim", "majmin"),
        ("Am", "C", "miss"),
        ("Am", "N", "miss"),
        ("A:7(#5)", "A", "majmin"),
        ("N", "C", None),
    ],
)
def test_level(reference: str, estimate: str, expected: str | None) -> None:
    assert agreement.level(reference, estimate) == expected


def test_compare_weights_by_time_and_ignores_unscored_spans() -> None:
    reference = [
        {"start": 1.0, "end": 3.0, "short": "C"},
        {"start": 3.0, "end": 4.0, "short": "N"},
        {"start": 4.0, "end": 5.0, "short": "Am7"},
    ]
    estimate = [
        {"start": 0.0, "end": 2.0, "label": "C"},
        {"start": 2.0, "end": 4.5, "label": "G"},
        {"start": 4.5, "end": 9.0, "label": "Am"},
    ]
    result = agreement.compare(reference, estimate)
    assert result["segments"] == [
        {"start": 1.0, "end": 2.0, "level": "exact", "alternate": False},
        {"start": 2.0, "end": 3.0, "level": "miss", "alternate": False},
        {"start": 4.0, "end": 4.5, "level": "miss", "alternate": False},
        {"start": 4.5, "end": 5.0, "level": "majmin", "alternate": False},
    ]
    assert result["scored_seconds"] == pytest.approx(3.0)
    assert result["totals"]["exact"] == pytest.approx(1 / 3)
    assert result["totals"]["majmin"] == pytest.approx(1.5 / 3)
    assert result["totals"]["root"] == pytest.approx(1.5 / 3)


def test_compare_credits_the_alternate_chord() -> None:
    reference = [{"start": 0.0, "end": 4.0, "short": "Dm"}]
    alternates = [{"start": 2.0, "end": 4.0, "short": "D7"}]
    estimate = [{"start": 0.0, "end": 4.0, "label": "D"}]
    result = agreement.compare(reference, estimate, alternates)
    assert result["segments"] == [
        # D over Dm: same root only; the alternate does not apply yet.
        {"start": 0.0, "end": 2.0, "level": "root", "alternate": False},
        # D over Dm-or-D7: same triad as the alternate, so not a mistake.
        {"start": 2.0, "end": 4.0, "level": "majmin", "alternate": True},
    ]
    assert result["totals"]["majmin"] == pytest.approx(0.5)


def test_compare_prefers_the_main_chord_when_it_matches_better() -> None:
    reference = [{"start": 0.0, "end": 2.0, "short": "Dm"}]
    alternates = [{"start": 0.0, "end": 2.0, "short": "D7"}]
    estimate = [{"start": 0.0, "end": 2.0, "label": "Dm"}]
    (segment,) = agreement.compare(reference, estimate, alternates)["segments"]
    assert segment["level"] == "exact"
    assert segment["alternate"] is False


@pytest.mark.parametrize(
    ("reference", "estimate"),
    [("Bb", "A#"), ("Bb7", "A#7"), ("Ebm", "D#m"), ("F#", "Gb"), ("Cb", "B")],
)
def test_compare_treats_enharmonics_as_the_same_chord(reference: str, estimate: str) -> None:
    result = agreement.compare(
        [{"start": 0.0, "end": 1.0, "short": reference}],
        [{"start": 0.0, "end": 1.0, "label": estimate}],
    )
    assert result["totals"]["exact"] == 1.0


def test_respell_follows_the_reference_spelling() -> None:
    reference = [
        {"start": 0.0, "end": 1.0, "short": "Bb7"},
        {"start": 1.0, "end": 2.0, "short": "C#m"},
        {"start": 2.0, "end": 3.0, "short": "N"},
    ]
    estimate = [
        {"start": 0.0, "end": 1.0, "label": "A#"},  # same pitch: respelled
        {"start": 1.0, "end": 2.0, "label": "Dbm"},  # same pitch: respelled
        {"start": 1.2, "end": 1.8, "label": "Eb"},  # different root: untouched
        {"start": 2.0, "end": 3.0, "label": "A#"},  # nothing to follow: untouched
    ]
    assert agreement.respell(estimate, reference) == ["Bb", "C#m", "Eb", "A#"]


def test_compare_with_nothing_scored() -> None:
    result = agreement.compare([{"start": 0.0, "end": 1.0, "short": "N"}], [])
    assert result["segments"] == []
    assert result["totals"]["root"] is None


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #


@pytest.fixture
def server(root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def fake_backend(audio, sample_rate, **kwargs) -> list[ChordEvent]:
        return [ChordEvent(0.5, 2.0, "C"), ChordEvent(2.0, 3.5, "Am")]

    monkeypatch.setitem(chords.BACKENDS, "fake", fake_backend)
    monkeypatch.setitem(chords.BACKEND_SAMPLE_RATES, "fake", SAMPLE_RATE)
    # Real loading would pull in librosa; the fake backend ignores the audio.
    monkeypatch.setattr("songscribe.audio.load", lambda path, rate: (None, rate))

    yield from _serve(examples.discover([root]), tmp_path / "cache")


def _serve(grids: dict[str, examples.Grid], cache: Path):
    """Run a viewer on a free port for the duration of one test."""
    viewer = Viewer(("127.0.0.1", 0), grids, cache)
    threading.Thread(target=viewer.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{viewer.server_address[1]}"
    finally:
        viewer.shutdown()
        viewer.server_close()


def get(url: str, **headers: str):
    with urlopen(Request(url, headers=headers)) as response:
        return response.status, dict(response.headers), response.read()


def get_json(url: str):
    return json.loads(get(url)[2])


def post_json(url: str):
    with urlopen(Request(url, method="POST")) as response:
        return response.status, json.loads(response.read())


def test_api_lists_and_loads_grids(server: str) -> None:
    grids = get_json(f"{server}/api/grids")
    assert [g["song"] for g in grids] == ["Paper Lantern Waltz", "Quiet Harbour"]
    truth = get_json(f"{server}/api/grids/0/paper-lantern-waltz/2")
    assert len(truth["events"]) == 3
    assert "fake" in get_json(f"{server}/api/backends")["chords"]


def test_page_and_static_files_are_served(server: str) -> None:
    assert b"songscribe viewer" in get(f"{server}/")[2]
    status, headers, _ = get(f"{server}/static/app.js")
    assert status == 200
    assert "javascript" in headers["Content-Type"]


def test_audio_supports_range_requests(server: str, root: Path) -> None:
    url = f"{server}/media/0/paper-lantern-waltz/2/audio"
    whole = (root / "paper-lantern-waltz" / "grid2_audio.wav").read_bytes()

    status, headers, body = get(url, Range="bytes=10-19")
    assert status == 206
    assert headers["Content-Range"] == f"bytes 10-19/{len(whole)}"
    assert body == whole[10:20]

    status, _, body = get(url, Range="bytes=-4")
    assert status == 206
    assert body == whole[-4:]

    status, headers, body = get(url)
    assert status == 200
    assert headers["Accept-Ranges"] == "bytes"
    assert body == whole

    with pytest.raises(HTTPError) as error:
        get(url, Range=f"bytes={len(whole)}-")
    assert error.value.code == 416


@pytest.mark.parametrize(
    "path",
    [
        "/api/grids/0/no-such-song/1",
        "/api/grids/0/quiet-harbour/5",  # ground truth but no audio
        "/media/0/quiet-harbour/1/screenshot",
        "/static/../server.py",
        "/nope",
    ],
)
def test_unknown_paths_are_404(server: str, path: str) -> None:
    with pytest.raises(HTTPError) as error:
        get(server + path)
    assert error.value.code == 404


def test_prediction_runs_caches_and_scores(server: str) -> None:
    url = f"{server}/api/grids/0/paper-lantern-waltz/2/predict/chords?backend=fake"
    assert get_json(url)["status"] == "absent"

    status, reply = post_json(url)
    assert status == 202
    deadline = time.monotonic() + 5
    while reply["status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.05)
        reply = get_json(url)

    assert reply["status"] == "done"
    assert [e["label"] for e in reply["events"]] == ["C", "Am"]
    # C over Cmaj7 for 1s, C over Am for 0.5s, Am over Am for 0.5s, Am over Dm7b5 for 1s.
    totals = reply["agreement"]["totals"]
    assert totals["majmin"] == pytest.approx(1.5 / 3)
    assert totals["exact"] == pytest.approx(0.5 / 3)

    # Cached: a second POST answers done straight away.
    assert post_json(url)[1]["status"] == "done"


def test_prediction_failure_is_reported(server: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(audio, sample_rate, **kwargs):
        raise NotImplementedError("not wired up")

    monkeypatch.setitem(chords.BACKENDS, "broken", broken)
    monkeypatch.setitem(chords.BACKEND_SAMPLE_RATES, "broken", SAMPLE_RATE)
    url = f"{server}/api/grids/0/paper-lantern-waltz/2/predict/chords?backend=broken"
    reply = post_json(url)[1]
    deadline = time.monotonic() + 5
    while reply["status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.05)
        reply = get_json(url)
    assert reply["status"] == "error"
    assert "not wired up" in reply["error"]


def test_unknown_backend_is_rejected(server: str) -> None:
    with pytest.raises(HTTPError) as error:
        post_json(f"{server}/api/grids/0/paper-lantern-waltz/2/predict/chords?backend=nope")
    assert error.value.code == 400


# --------------------------------------------------------------------------- #
# Melody
# --------------------------------------------------------------------------- #

# An invented lead line: three notes over the annotated bars, as a chart would
# print them. C5, D5, E5.
WRITTEN = [NoteEvent(0.5, 1.5, 72), NoteEvent(1.5, 2.5, 74), NoteEvent(2.5, 3.5, 76)]


@pytest.fixture
def melody_server(root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    lab.write_notes(root / "paper-lantern-waltz" / "grid2_melody.notes", WRITTEN)

    def fake_melody(audio, sample_rate, **kwargs) -> list[NoteEvent]:
        return [
            NoteEvent(0.5, 1.5, 72),  # the written note
            NoteEvent(1.5, 2.5, 86),  # D6: the written D5 an octave up
            NoteEvent(2.5, 3.5, 77),  # F5 where E5 was written
        ]

    monkeypatch.setitem(melody_stage.BACKENDS, "fakeline", fake_melody)
    monkeypatch.setitem(melody_stage.BACKEND_SAMPLE_RATES, "fakeline", SAMPLE_RATE)
    monkeypatch.setattr("songscribe.audio.load", lambda path, rate: (None, rate))
    yield from _serve(examples.discover([root]), tmp_path / "cache")


def test_melody_backends_are_listed_under_their_own_stage(melody_server: str) -> None:
    backends = get_json(f"{melody_server}/api/backends")
    assert "fakeline" in backends["melody"]
    assert "fakeline" not in backends["chords"]


def test_written_melody_is_engraved_with_the_ground_truth(melody_server: str) -> None:
    truth = get_json(f"{melody_server}/api/grids/0/paper-lantern-waltz/2")
    assert truth["has_melody"] is True
    assert [n["pitch"] for n in truth["melody"]] == [72, 74, 76]
    # 240 bpm, so a beat is 0.25s and each 1s note is a whole note.
    score = truth["melody_score"]
    assert score["clef"] == "treble"
    assert [n["name"] for n in score["notes"]] == ["C5", "D5", "E5"]
    assert {n["value"] for n in score["notes"]} == {"whole"}


def test_melody_prediction_engraves_and_grades(melody_server: str) -> None:
    url = f"{melody_server}/api/grids/0/paper-lantern-waltz/2/predict/melody?backend=fakeline"
    status, reply = post_json(url)
    assert status == 202
    deadline = time.monotonic() + 5
    while reply["status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.05)
        reply = get_json(url)

    assert reply["status"] == "done"
    assert [e["pitch"] for e in reply["events"]] == [72, 86, 77]
    # The written line picks the clef, so both staves read against each other.
    assert reply["score"]["clef"] == "treble"
    assert [n["level"] for n in reply["score"]["notes"]] == ["exact", "octave", "miss"]
    assert [n["name"] for n in reply["score"]["notes"]] == ["C5", "D6", "F5"]
    totals = reply["agreement"]["totals"]
    assert totals["exact"] == pytest.approx(1 / 3)
    assert totals["octave"] == pytest.approx(2 / 3)
    assert totals["extra"] == 0


def test_a_grid_without_a_melody_still_runs_the_stage(server: str) -> None:
    """Nothing written means a staff but no agreement, not an error."""

    def flat(audio, sample_rate, **kwargs) -> list[NoteEvent]:
        return [NoteEvent(0.5, 1.5, 72)]

    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(melody_stage.BACKENDS, "flat", flat)
        patch.setitem(melody_stage.BACKEND_SAMPLE_RATES, "flat", SAMPLE_RATE)
        url = f"{server}/api/grids/0/paper-lantern-waltz/2/predict/melody?backend=flat"
        reply = post_json(url)[1]
        deadline = time.monotonic() + 5
        while reply["status"] == "running" and time.monotonic() < deadline:
            time.sleep(0.05)
            reply = get_json(url)

    assert reply["status"] == "done"
    assert reply["score"]["notes"][0]["name"] == "C5"
    assert "agreement" not in reply
    assert "level" not in reply["score"]["notes"][0]


def test_unknown_melody_backend_is_rejected(melody_server: str) -> None:
    with pytest.raises(HTTPError) as error:
        post_json(f"{melody_server}/api/grids/0/paper-lantern-waltz/2/predict/melody?backend=nope")
    assert error.value.code == 400


def test_an_unknown_stage_is_not_an_endpoint(server: str) -> None:
    with pytest.raises(HTTPError) as error:
        get(f"{server}/api/grids/0/paper-lantern-waltz/2/predict/lyrics?backend=fake")
    assert error.value.code == 404


# --------------------------------------------------------------------------- #
# A single recording, with whatever sits beside it
# --------------------------------------------------------------------------- #


def test_single_audio_file_picks_up_its_sidecars(tmp_path: Path) -> None:
    write_silence(tmp_path / "quiet-harbour.wav", 4.0)
    lab.write(tmp_path / "quiet-harbour.lab", [ChordEvent(0.0, 2.0, "G:maj")])
    lab.write_notes(tmp_path / "quiet-harbour.notes", WRITTEN)

    (grid,) = examples.discover(audio=[tmp_path / "quiet-harbour.wav"]).values()
    assert grid.id == "0/quiet-harbour/1"
    assert grid.song == "quiet-harbour"
    assert (grid.chart, grid.beats) == (None, None)

    truth = examples.ground_truth(grid)
    assert [e["short"] for e in truth["events"]] == ["G"]
    assert [n["pitch"] for n in truth["melody"]] == [72, 74, 76]
    # No tempo, no bar grid and no .beats: the rhythm is left unwritten rather
    # than guessed from the note lengths themselves.
    assert truth["beat_seconds"] is None
    assert truth["bars"] == []


def test_a_beats_sidecar_supplies_the_bar_grid_and_the_beat(tmp_path: Path) -> None:
    write_silence(tmp_path / "lantern-reel.wav", 4.0)
    beats = [BeatEvent(time=index * 0.5, position=index % 4 + 1) for index in range(8)]
    lab.write_beats(tmp_path / "lantern-reel.beats", beats)

    (grid,) = examples.discover(audio=[tmp_path / "lantern-reel.wav"]).values()
    truth = examples.ground_truth(grid)
    assert truth["bars"] == [0.0, 2.0, 4.0]
    assert truth["beat_seconds"] == pytest.approx(0.5)
    assert truth["events"] == []
    assert truth["has_chords"] is False


def test_bare_audio_is_viewable_with_nothing_annotated(tmp_path: Path) -> None:
    write_silence(tmp_path / "untitled.wav", 2.0)
    (grid,) = examples.discover(audio=[tmp_path / "untitled.wav"]).values()
    truth = examples.ground_truth(grid)
    assert (truth["has_chords"], truth["has_melody"]) == (False, False)
    assert truth["melody"] == []
    assert truth["annotated_end"] == 0.0


def test_discover_needs_something_to_show(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nothing to view"):
        examples.discover()
    with pytest.raises(ValueError, match="audio file not found"):
        examples.discover(audio=[tmp_path / "nope.wav"])


def test_a_grid_is_discovered_by_its_audio_alone(root: Path, tmp_path: Path) -> None:
    """Audio is the only required file, so an unannotated grid still lists."""
    write_silence(root / "lantern-reel" / "grid7_audio.wav", 2.0)
    grids = examples.discover([root])
    assert "0/lantern-reel/7" in grids
    assert grids["0/lantern-reel/7"].song == "lantern-reel"
    assert examples.summary(grids["0/lantern-reel/7"])["has_chords"] is False


def test_a_note_the_backend_missed_is_engraved_in_its_place(melody_server: str) -> None:
    def sparse(audio, sample_rate, **kwargs) -> list[NoteEvent]:
        return [WRITTEN[0], WRITTEN[2]]  # the middle note is never found

    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(melody_stage.BACKENDS, "sparse", sparse)
        patch.setitem(melody_stage.BACKEND_SAMPLE_RATES, "sparse", SAMPLE_RATE)
        url = f"{melody_server}/api/grids/0/paper-lantern-waltz/2/predict/melody?backend=sparse"
        reply = post_json(url)[1]
        deadline = time.monotonic() + 5
        while reply["status"] == "running" and time.monotonic() < deadline:
            time.sleep(0.05)
            reply = get_json(url)

    assert reply["status"] == "done"
    assert [n["level"] for n in reply["agreement"]["written"]] == ["exact", "missing", "exact"]
    (missed,) = reply["missed"]["notes"]
    assert (missed["name"], missed["level"], missed["step"]) == ("D5", "missing", 6)


def test_a_flat_chart_spells_the_staff_with_flats(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The chord lane respells to the chart, so the staff must not print A#."""
    song = root / "paper-lantern-waltz"
    flat = {
        "song": "Harbour Lights Waltz",
        "events": [{"start": 0.0, "end": 2.0, "chord": "Bb:maj"}],
        "annotation": {"bars": 1, "bar_seconds": 2.0, "start_seconds": 0.0, "bpm_nominal": 120.0},
    }
    write_grid(song, 9, flat)
    # A# and Bb are one pitch; the chart in Bb decides which is printed.
    lab.write_notes(song / "grid9_melody.notes", [NoteEvent(0.0, 1.0, 70)])

    grids = examples.discover([root])
    truth = truth_reply(grids["0/paper-lantern-waltz/9"])
    assert truth["sharps"] is False
    (note,) = truth["melody_score"]["notes"]
    assert (note["name"], note["accidental"]) == ("Bb4", "b")


def test_a_done_job_without_its_result_is_not_reported_as_done(root: Path, tmp_path: Path) -> None:
    """A poll can read the cache before the run writes it and the job after.

    Answering "done" there hands the page a reply with no events in it, which
    is what it draws from.
    """
    predictor = Predictor(tmp_path / "cache")
    grid = examples.discover([root])["0/paper-lantern-waltz/2"]
    predictor._jobs[(grid.id, "chords", "fake")] = Job("done", time.monotonic())

    reply = predictor.status(grid, STAGES["chords"], "fake")
    assert reply == {"status": "absent"}


def test_a_run_finishing_mid_poll_is_not_reported_as_done(root: Path, tmp_path: Path) -> None:
    """The job must be read once: it can finish between any two reads.

    A status of "done" taken from one read and paired with fields from a later
    one is how a reply ends up saying done with no events in it.
    """

    class FlippingJob:
        """Running the first time it is asked, done every time after."""

        def __init__(self) -> None:
            self.reads = 0
            self.error = None
            self.started = time.monotonic()

        @property
        def status(self) -> str:
            self.reads += 1
            return "running" if self.reads == 1 else "done"

    predictor = Predictor(tmp_path / "cache")
    grid = examples.discover([root])["0/paper-lantern-waltz/2"]
    job = FlippingJob()
    predictor._jobs[(grid.id, "chords", "fake")] = job

    reply = predictor.status(grid, STAGES["chords"], "fake")
    assert job.reads == 1, "the job was read more than once"
    assert reply["status"] == "running"
    assert "events" not in reply


def test_a_missed_note_is_engraved_where_it_sounds(melody_server: str) -> None:
    """Missed notes are not a passage, so they take no octave sign.

    They are drawn to show where the note should have been; moving them an
    octave to tidy them up would put them somewhere it never was.
    """

    def nothing(audio, sample_rate, **kwargs) -> list[NoteEvent]:
        return []

    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(melody_stage.BACKENDS, "silent", nothing)
        patch.setitem(melody_stage.BACKEND_SAMPLE_RATES, "silent", SAMPLE_RATE)
        url = f"{melody_server}/api/grids/0/paper-lantern-waltz/2/predict/melody?backend=silent"
        reply = post_json(url)[1]
        deadline = time.monotonic() + 5
        while reply["status"] == "running" and time.monotonic() < deadline:
            time.sleep(0.05)
            reply = get_json(url)

    assert reply["status"] == "done"
    assert [n["level"] for n in reply["agreement"]["written"]] == ["missing"] * 3
    missed = reply["missed"]
    assert missed["octaves"] == []
    assert all(n["octave"] == 0 for n in missed["notes"])
    assert [n["name"] for n in missed["notes"]] == ["C5", "D5", "E5"]
