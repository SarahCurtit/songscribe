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

from songscribe import chords
from songscribe.types import ChordEvent
from songscribe.viewer import agreement, examples
from songscribe.viewer.server import Viewer

SAMPLE_RATE = 8_000


def write_silence(path: Path, seconds: float) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(b"\x00\x00" * int(seconds * SAMPLE_RATE))


def write_grid(song_dir: Path, number: int, payload: dict, audio: bool = True) -> None:
    song_dir.mkdir(parents=True, exist_ok=True)
    (song_dir / f"grid{number}_chords.json").write_text(json.dumps(payload), encoding="utf-8")
    if audio:
        write_silence(song_dir / f"grid{number}_audio.wav", 4.0)


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
        {"start": 1.0, "end": 2.0, "level": "exact"},
        {"start": 2.0, "end": 3.0, "level": "miss"},
        {"start": 4.0, "end": 4.5, "level": "miss"},
        {"start": 4.5, "end": 5.0, "level": "majmin"},
    ]
    assert result["scored_seconds"] == pytest.approx(3.0)
    assert result["totals"]["exact"] == pytest.approx(1 / 3)
    assert result["totals"]["majmin"] == pytest.approx(1.5 / 3)
    assert result["totals"]["root"] == pytest.approx(1.5 / 3)


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

    viewer = Viewer(("127.0.0.1", 0), examples.discover([root]), tmp_path / "cache")
    thread = threading.Thread(target=viewer.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{viewer.server_address[1]}"
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
    assert "fake" in get_json(f"{server}/api/backends")


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
    url = f"{server}/api/grids/0/paper-lantern-waltz/2/predict?backend=fake"
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
    url = f"{server}/api/grids/0/paper-lantern-waltz/2/predict?backend=broken"
    reply = post_json(url)[1]
    deadline = time.monotonic() + 5
    while reply["status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.05)
        reply = get_json(url)
    assert reply["status"] == "error"
    assert "not wired up" in reply["error"]


def test_unknown_backend_is_rejected(server: str) -> None:
    with pytest.raises(HTTPError) as error:
        post_json(f"{server}/api/grids/0/paper-lantern-waltz/2/predict?backend=nope")
    assert error.value.code == 400
