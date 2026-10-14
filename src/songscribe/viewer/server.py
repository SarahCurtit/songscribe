"""HTTP server for the grid viewer: a JSON API, the audio, and one static page.

Standard library only, so the viewer adds no dependencies. Chord backends run
on demand in a background thread and their output is cached on disk, keyed by
the audio file's size and mtime, so re-opening a grid is instant and a
re-recorded grid is recomputed.

Binds to localhost by default: it serves whatever audio it is pointed at, and
that is rarely audio that may be shared.
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
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from ..chords import BACKENDS
from . import agreement, examples

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

GRID_ID = r"\d+/[^/]+/\d+"
API_GRID = re.compile(rf"/api/grids/({GRID_ID})")
API_PREDICT = re.compile(rf"/api/grids/({GRID_ID})/predict")
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
    """Runs chord backends in the background and caches what they return."""

    def __init__(self, cache: Path) -> None:
        self.cache = cache
        self._jobs: dict[tuple[str, str], Job] = {}
        self._lock = threading.Lock()

    def start(self, grid: examples.Grid, backend: str) -> None:
        """Begin a run unless one is going or a fresh result is cached.

        Raises :class:`ValueError` for an unknown backend, from the registry
        itself -- never a fallback to some other model.
        """
        from ..chords import sample_rate_for

        sample_rate_for(backend)  # validates the name
        key = (grid.id, backend)
        with self._lock:
            job = self._jobs.get(key)
            if (job and job.status == "running") or self._cached(grid, backend):
                return
            self._jobs[key] = Job("running", time.monotonic())
        threading.Thread(target=self._run, args=(grid, backend), daemon=True).start()

    def status(self, grid: examples.Grid, backend: str) -> dict[str, Any]:
        cached = self._cached(grid, backend)
        if cached is not None:
            return {"status": "done", **cached}
        job = self._jobs.get((grid.id, backend))
        if job is None:
            return {"status": "absent"}
        return {
            "status": job.status,
            "error": job.error,
            "elapsed": round(time.monotonic() - job.started, 1),
        }

    def _run(self, grid: examples.Grid, backend: str) -> None:
        from ..audio import load
        from ..chords import recognise, sample_rate_for

        key = (grid.id, backend)
        started = time.monotonic()
        try:
            stat = grid.audio.stat()
            audio, rate = load(grid.audio, sample_rate_for(backend))
            events = recognise(audio, rate, backend=backend)
            payload = {
                "version": CACHE_VERSION,
                "backend": backend,
                "audio_size": stat.st_size,
                "audio_mtime": stat.st_mtime,
                "seconds": round(time.monotonic() - started, 1),
                "events": [{"start": e.start, "end": e.end, "label": e.label} for e in events],
            }
            path = self._cache_path(grid, backend)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write then rename, so a poll never reads a half-written file.
            partial = path.with_suffix(".tmp")
            partial.write_text(json.dumps(payload), encoding="utf-8")
            partial.replace(path)
            with self._lock:
                self._jobs[key].status = "done"
        except Exception as error:  # surfaced to the page, not swallowed
            log.exception("backend %s failed on %s", backend, grid.id)
            with self._lock:
                self._jobs[key].status = "error"
                self._jobs[key].error = f"{type(error).__name__}: {error}"

    def _cache_path(self, grid: examples.Grid, backend: str) -> Path:
        # Hash the absolute path: cache names stay opaque, and two example
        # roots with a same-named song cannot collide.
        digest = hashlib.sha1(str(grid.audio.resolve()).encode()).hexdigest()[:16]
        return self.cache / f"{digest}_{backend}.json"

    def _cached(self, grid: examples.Grid, backend: str) -> dict[str, Any] | None:
        path = self._cache_path(grid, backend)
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
            return self._json(sorted(BACKENDS))
        if match := API_PREDICT.fullmatch(path):
            grid = self._grid(match.group(1))
            reply = None if grid is None else self._prediction(grid, query)
            return None if reply is None else self._json(reply)
        if match := API_GRID.fullmatch(path):
            grid = self._grid(match.group(1))
            return None if grid is None else self._json(examples.ground_truth(grid))
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
        backend = query.get("backend", "")
        try:
            self.server.predictor.start(grid, backend)
        except ValueError as error:
            return self._error(HTTPStatus.BAD_REQUEST, str(error))
        reply = self._prediction(grid, query)
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

    def _prediction(self, grid: examples.Grid, query: dict[str, str]) -> dict[str, Any] | None:
        backend = query.get("backend", "")
        if backend not in BACKENDS:
            self._error(HTTPStatus.BAD_REQUEST, f"unknown backend {backend!r}")
            return None
        reply = self.server.predictor.status(grid, backend)
        if reply["status"] == "done":
            reference = examples.ground_truth(grid)["events"]
            reply["agreement"] = agreement.compare(reference, reply["events"])
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
        description="Listen to annotated grids and compare chord backends against them.",
    )
    parser.add_argument(
        "-e",
        "--examples",
        type=Path,
        action="append",
        required=True,
        help="directory of <song>/gridN_{chords.json,audio.wav} (repeatable)",
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
    try:
        grids = examples.discover(args.examples)
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
