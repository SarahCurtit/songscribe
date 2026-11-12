"""HTTP server for the grid viewer: a JSON API, the audio, and one static page.

Standard library only, so the viewer adds no dependencies. Backends run in a
background thread and their output is cached on disk, keyed by the audio file's
size and mtime, so a re-recorded grid is recomputed. Chords and melody each sit
behind a :class:`Stage` descriptor holding their registry, so this file names no
backend. Binds to localhost by default: it serves whatever audio it is pointed
at, which is rarely audio that may be shared.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import mimetypes
import re
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from .. import chords as chords_stage
from .. import melody as melody_stage
from ..types import NoteEvent
from . import agreement, examples, melody, score

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_CACHE = Path.home() / ".cache" / "songscribe" / "viewer"
# Bumped when the cached prediction format changes, so old files are ignored.
CACHE_VERSION = 1
# Bytes per write when streaming media: large enough to keep syscalls rare,
# small enough that a seek abandons little already-sent data.
CHUNK_SIZE = 256 * 1024


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Stage:
    """One pluggable stage the viewer can run, and how to talk to its registry.

    ``run`` returns that stage's event type and ``serialise`` turns it into the
    JSON the cache and the page hold. Both come from the stage's own module, so
    adding a backend there is all a new model needs.
    """

    name: str
    backends: dict[str, Callable[..., Any]]
    sample_rate_for: Callable[[str], int]
    run: Callable[[Any, int, str], Sequence[Any]]
    serialise: Callable[[Sequence[Any]], list[dict[str, Any]]]


def _run_chords(audio: Any, rate: int, backend: str) -> Sequence[Any]:
    return chords_stage.recognise(audio, rate, backend=backend)


def _run_melody(audio: Any, rate: int, backend: str) -> Sequence[Any]:
    return melody_stage.transcribe(audio, rate, backend=backend)


STAGES = {
    "chords": Stage(
        name="chords",
        backends=chords_stage.BACKENDS,
        sample_rate_for=chords_stage.sample_rate_for,
        run=_run_chords,
        serialise=lambda events: [
            {"start": e.start, "end": e.end, "label": e.label} for e in events
        ],
    ),
    "melody": Stage(
        name="melody",
        backends=melody_stage.BACKENDS,
        sample_rate_for=melody_stage.sample_rate_for,
        run=_run_melody,
        serialise=lambda notes: [
            {"start": n.start, "end": n.end, "pitch": n.pitch, "confidence": n.confidence}
            for n in notes
        ],
    ),
}

GRID_ID = r"\d+/[^/]+/\d+"
STAGE_NAMES = "|".join(STAGES)
API_GRID = re.compile(rf"/api/grids/({GRID_ID})")
API_PREDICT = re.compile(rf"/api/grids/({GRID_ID})/predict/({STAGE_NAMES})")
MEDIA = re.compile(rf"/media/({GRID_ID})/(audio|screenshot)")
RANGE = re.compile(r"bytes=(\d*)-(\d*)")


# --------------------------------------------------------------------------- #
# Predictions
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Job:
    """One backend run on one grid, as the client polls it."""

    status: str  # "running" | "done" | "error"
    started: float
    error: str | None = None


class Predictor:
    """Runs backends in the background and caches what they return."""

    def __init__(self, cache: Path) -> None:
        self.cache = cache
        self._jobs: dict[tuple[str, str, str], Job] = {}
        self._lock = threading.Lock()

    def start(self, grid: examples.Grid, stage: Stage, backend: str) -> None:
        """Begin a run unless one is going or a fresh result is cached.

        Raises :class:`ValueError` for an unknown backend, from the registry
        itself -- never a fallback to some other model.
        """
        stage.sample_rate_for(backend)  # validates the name
        key = (grid.id, stage.name, backend)
        with self._lock:
            job = self._jobs.get(key)
            if (job and job.status == "running") or self._cached(grid, stage, backend):
                return
            self._jobs[key] = Job("running", time.monotonic())
        threading.Thread(target=self._run, args=(grid, stage, backend), daemon=True).start()

    def status(self, grid: examples.Grid, stage: Stage, backend: str) -> dict[str, Any]:
        cached = self._cached(grid, stage, backend)
        if cached is not None:
            return {"status": "done", **cached}
        # One read of the job, under the lock: a run can finish between any two
        # of them, and a status of "done" paired with a later field read is how
        # a reply ends up saying done with no events for the page to draw.
        with self._lock:
            job = self._jobs.get((grid.id, stage.name, backend))
            seen = None if job is None else (job.status, job.error, job.started)
        if seen is None:
            return {"status": "absent"}
        status, error, started = seen
        if status == "done":
            # The run writes its cache before marking the job done, so the
            # cache read above can be a moment too early. Look once more; if it
            # is still not there the result is gone and the run is offered again.
            again = self._cached(grid, stage, backend)
            return {"status": "done", **again} if again else {"status": "absent"}
        return {
            "status": status,
            "error": error,
            "elapsed": round(time.monotonic() - started, 1),
        }

    def _run(self, grid: examples.Grid, stage: Stage, backend: str) -> None:
        from ..audio import load

        key = (grid.id, stage.name, backend)
        started = time.monotonic()
        try:
            stat = grid.audio.stat()
            audio, rate = load(grid.audio, stage.sample_rate_for(backend))
            events = stage.run(audio, rate, backend)
            payload = {
                "version": CACHE_VERSION,
                "stage": stage.name,
                "backend": backend,
                "audio_size": stat.st_size,
                "audio_mtime": stat.st_mtime,
                "seconds": round(time.monotonic() - started, 1),
                "events": stage.serialise(events),
            }
            path = self._cache_path(grid, stage, backend)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write then rename, so a poll never reads a half-written file.
            partial = path.with_suffix(".tmp")
            partial.write_text(json.dumps(payload), encoding="utf-8")
            partial.replace(path)
            with self._lock:
                self._jobs[key].status = "done"
        except Exception as error:  # surfaced to the page, not swallowed
            log.exception("%s backend %s failed on %s", stage.name, backend, grid.id)
            with self._lock:
                self._jobs[key].status = "error"
                self._jobs[key].error = f"{type(error).__name__}: {error}"

    def _cache_path(self, grid: examples.Grid, stage: Stage, backend: str) -> Path:
        # Hash the absolute path: cache names stay opaque, and two example
        # roots with a same-named song cannot collide.
        digest = hashlib.sha1(str(grid.audio.resolve()).encode()).hexdigest()[:16]
        return self.cache / f"{digest}_{stage.name}_{backend}.json"

    def _cached(self, grid: examples.Grid, stage: Stage, backend: str) -> dict[str, Any] | None:
        path = self._cache_path(grid, stage, backend)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        stat = grid.audio.stat()
        fresh = (
            payload.get("version") == CACHE_VERSION
            and payload.get("audio_size") == stat.st_size
            and payload.get("audio_mtime") == stat.st_mtime
        )
        return payload if fresh else None


# --------------------------------------------------------------------------- #
# Grading a reply
# --------------------------------------------------------------------------- #


def _notes(events: Sequence[dict[str, Any]]) -> list[NoteEvent]:
    return [NoteEvent(e["start"], e["end"], e["pitch"], e.get("confidence", 1.0)) for e in events]


def _grade_chords(reply: dict[str, Any], truth: dict[str, Any]) -> None:
    """Add the agreement strip and the chart's spelling to a chord reply."""
    reply["agreement"] = agreement.compare(truth["events"], reply["events"], truth["alternates"])
    for event, shown in zip(
        reply["events"], agreement.respell(reply["events"], truth["events"]), strict=True
    ):
        event["display"] = shown


def _grade_melody(reply: dict[str, Any], truth: dict[str, Any]) -> None:
    """Add the staff and, where a melody is written, the note-by-note agreement.

    The written line picks the clef and the spelling, so the two staves read
    against each other. Notes the backend did not find are engraved too, under
    ``missed``, so the staff can show where they should have been.
    """
    played = _notes(reply["events"])
    written = _notes(truth["melody"])
    sharps = truth["sharps"]
    clef = score.choose_clef(written or played, sharps)
    beat = truth["beat_seconds"]
    reply["score"] = score.engrave(played, beat_seconds=beat, clef=clef, sharps=sharps)
    if not written:
        return
    reply["agreement"] = melody.compare(written, played)
    for glyph, graded in zip(reply["score"]["notes"], reply["agreement"]["played"], strict=True):
        glyph["level"] = graded["level"]
    missed = [
        note
        for note, graded in zip(written, reply["agreement"]["written"], strict=True)
        if graded["level"] == "missing"
    ]
    reply["missed"] = score.engrave(missed, beat_seconds=beat, clef=clef, sharps=sharps)
    for glyph in reply["missed"]["notes"]:
        glyph["level"] = "missing"


GRADERS = {"chords": _grade_chords, "melody": _grade_melody}


def truth_reply(grid: examples.Grid) -> dict[str, Any]:
    """A grid's ground truth, with the written melody already engraved."""
    truth = examples.ground_truth(grid)
    # Follow the chart's own spelling, as the chord lane does.
    truth["sharps"] = score.spell_with_sharps([e["short"] for e in truth["events"]])
    written = _notes(truth["melody"])
    truth["melody_score"] = (
        score.engrave(written, beat_seconds=truth["beat_seconds"], sharps=truth["sharps"])
        if written
        else None
    )
    return truth


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #


class Viewer(ThreadingHTTPServer):
    """The server, holding what every request handler needs."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], grids: dict[str, examples.Grid], cache: Path):
        super().__init__(address, Handler)
        self.grids = grids
        self.predictor = Predictor(cache)


class Handler(BaseHTTPRequestHandler):
    server: Viewer

    def do_GET(self) -> None:
        path, query = self._route()
        if path in ("/", "/index.html"):
            return self._file(STATIC_DIR / "index.html")
        if path.startswith("/static/"):
            name = path.removeprefix("/static/")
            target = STATIC_DIR / name
            if "/" in name or not target.is_file():
                return self._error(HTTPStatus.NOT_FOUND, "no such file")
            return self._file(target)
        if path == "/api/grids":
            return self._json([examples.summary(g) for g in self.server.grids.values()])
        if path == "/api/backends":
            return self._json({name: sorted(s.backends) for name, s in STAGES.items()})
        if match := API_PREDICT.fullmatch(path):
            return self._predict_reply(match, query)
        if match := API_GRID.fullmatch(path):
            grid = self._grid(match.group(1))
            return None if grid is None else self._json(truth_reply(grid))
        if match := MEDIA.fullmatch(path):
            grid = self._grid(match.group(1))
            if grid is None:
                return None
            target = grid.audio if match.group(2) == "audio" else grid.screenshot
            if target is None:
                return self._error(HTTPStatus.NOT_FOUND, "this grid has no screenshot")
            return self._file(target)
        return self._error(HTTPStatus.NOT_FOUND, "no such endpoint")

    def do_POST(self) -> None:
        path, query = self._route()
        match = API_PREDICT.fullmatch(path)
        if not match:
            return self._error(HTTPStatus.NOT_FOUND, "no such endpoint")
        grid = self._grid(match.group(1))
        if grid is None:
            return None
        stage = STAGES[match.group(2)]
        try:
            self.server.predictor.start(grid, stage, query.get("backend", ""))
        except ValueError as error:
            return self._error(HTTPStatus.BAD_REQUEST, str(error))
        reply = self._prediction(grid, stage, query)
        return None if reply is None else self._json(reply, HTTPStatus.ACCEPTED)

    def log_message(self, format: str, *args: Any) -> None:
        log.debug("%s %s", self.address_string(), format % args)

    # -- helpers ------------------------------------------------------------ #

    def _route(self) -> tuple[str, dict[str, str]]:
        parts = urlsplit(self.path)
        query = {k: v[-1] for k, v in parse_qs(parts.query).items()}
        return unquote(parts.path), query

    def _grid(self, grid_id: str) -> examples.Grid | None:
        grid = self.server.grids.get(grid_id)
        if grid is None:
            self._error(HTTPStatus.NOT_FOUND, f"no grid {grid_id!r}")
        return grid

    def _predict_reply(self, match: re.Match[str], query: dict[str, str]) -> None:
        grid = self._grid(match.group(1))
        if grid is None:
            return None
        reply = self._prediction(grid, STAGES[match.group(2)], query)
        return None if reply is None else self._json(reply)

    def _prediction(
        self, grid: examples.Grid, stage: Stage, query: dict[str, str]
    ) -> dict[str, Any] | None:
        backend = query.get("backend", "")
        if backend not in stage.backends:
            self._error(HTTPStatus.BAD_REQUEST, f"unknown {stage.name} backend {backend!r}")
            return None
        reply = self.server.predictor.status(grid, stage, backend)
        if reply["status"] == "done":
            GRADERS[stage.name](reply, truth_reply(grid))
        return reply

    def _json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json({"error": message}, status)

    def _file(self, path: Path) -> None:
        """Send a file, honouring a single ``Range`` -- browsers seek audio with it."""
        size = path.stat().st_size
        start, end = 0, size - 1
        status = HTTPStatus.OK
        header = self.headers.get("Range")
        if header:
            match = RANGE.fullmatch(header.strip())
            if not match or match.groups() == ("", ""):
                return self._unsatisfiable(size)
            first, last = match.groups()
            if first:
                start = int(first)
                end = min(int(last), size - 1) if last else size - 1
            else:  # "bytes=-N": the last N bytes
                start = max(size - int(last), 0)
            if start > end or start >= size:
                return self._unsatisfiable(size)
            status = HTTPStatus.PARTIAL_CONTENT

        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = end - start + 1
            try:
                while remaining > 0 and (chunk := handle.read(min(CHUNK_SIZE, remaining))):
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the browser seeks by dropping one request for another

    def _unsatisfiable(self, size: int) -> None:
        self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
        self.send_header("Content-Range", f"bytes */{size}")
        self.send_header("Content-Length", "0")
        self.end_headers()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="songscribe-viewer",
        description="Listen to recordings and compare chord and melody backends against them.",
    )
    parser.add_argument(
        "-e",
        "--examples",
        type=Path,
        action="append",
        help="directory of <song>/gridN_{audio.wav,chords.json,melody.notes} (repeatable)",
    )
    parser.add_argument(
        "-a",
        "--audio",
        type=Path,
        action="append",
        help="a single recording, with any .lab/.notes/.beats beside it (repeatable)",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="default: %(default)s")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="default: %(default)s")
    parser.add_argument(
        "--cache", type=Path, default=DEFAULT_CACHE, help="prediction cache (default: %(default)s)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if not args.examples and not args.audio:
        print("songscribe-viewer: pass --examples DIR or --audio FILE", file=sys.stderr)
        return 1
    try:
        grids = examples.discover(args.examples, args.audio)
        server = Viewer((args.host, args.port), grids, args.cache)
    except (ValueError, OSError) as error:
        print(f"songscribe-viewer: {error}", file=sys.stderr)
        return 1
    host, port = server.server_address[:2]
    # Flushed: under a process manager stdout is block-buffered, and this line
    # is the one telling you where to point the browser.
    print(f"{len(grids)} grids; serving on http://{host}:{port}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
