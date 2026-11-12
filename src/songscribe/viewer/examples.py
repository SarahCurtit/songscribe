"""Find recordings to view and load whatever is annotated about them.

A grid is a recording plus any annotation sitting beside it; audio is the only
required part. An example directory holds one folder per song and one file per
grid per kind of data, sharing a ``gridN_`` prefix::

    <root>/<song>/gridN_audio.wav        the recording            (required)
    <root>/<song>/gridN_chords.json      the chart it was read from
    <root>/<song>/gridN_melody.notes     the written melody
    <root>/<song>/gridN_beats.beats      the bar grid
    <root>/<song>/gridN_screenshot.png   a picture of the chart

The chart is timed Harte events in seconds from the start of the WAV. Its
optional ``annotation`` object carries bar-indexed sections and the alternate
chords a chart prints above a bar, plus the bar grid (``start_seconds``,
``bar_seconds``) that puts them on the same clock.

A single file passed with ``--audio`` is a grid too, picking up ``song.lab``,
``song.notes`` and ``song.beats`` by stem.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from statistics import median
from typing import Any

from .. import lab
from ..chords import normalise_label
from ..types import DEFAULT_METRE

log = logging.getLogger(__name__)

AUDIO_PATTERN = re.compile(r"grid(\d+)_audio\.wav")

#: Sidecar suffixes a ``--audio`` file picks up, by what they annotate.
SIDECARS = {"chords": ".lab", "notes": ".notes", "beats": ".beats"}


@dataclass(frozen=True, slots=True)
class Grid:
    """One recording, located by the files that annotate it.

    ``chart`` is the rich ``gridN_chords.json``; ``chords`` a plain ``.lab``.
    Every field but ``audio`` may be ``None``: the viewer draws what is there.
    """

    id: str
    song: str
    grid: int
    audio: Path
    chart: Path | None = None
    chords: Path | None = None
    notes: Path | None = None
    beats: Path | None = None
    screenshot: Path | None = None


def discover(roots: list[Path] | None = None, audio: list[Path] | None = None) -> dict[str, Grid]:
    """Every viewable grid under ``roots``, plus each file in ``audio``.

    IDs are ``<source index>/<name>/<grid number>``: stable for a given command
    line, and free of absolute paths so none leak into URLs. Raises when a root
    is missing or holds no recording -- an empty picker would look like a bug in
    the viewer rather than a typo in the path.
    """
    grids: dict[str, Grid] = {}
    index = 0
    for root in roots or []:
        if not root.is_dir():
            raise ValueError(f"examples directory not found: {root}")
        found = 0
        for grid in _in_directory(root, index):
            grids[grid.id] = grid
            found += 1
        if not found:
            raise ValueError(f"no grid with a gridN_audio.wav under {root}")
        index += 1
    for path in audio or []:
        if not path.is_file():
            raise ValueError(f"audio file not found: {path}")
        grid = _single(path, index)
        grids[grid.id] = grid
        index += 1
    if not grids:
        raise ValueError("nothing to view: pass --examples or --audio")
    return dict(sorted(grids.items(), key=lambda item: (item[1].song.lower(), item[1].grid)))


def _in_directory(root: Path, index: int) -> list[Grid]:
    """The grids under one example root, newest convention: audio is required."""
    grids = []
    for audio in sorted(root.glob("*/grid*_audio.wav")):
        match = AUDIO_PATTERN.fullmatch(audio.name)
        if not match:
            continue
        number = int(match.group(1))

        def beside(name: str, number: int = number, audio: Path = audio) -> Path | None:
            path = audio.with_name(f"grid{number}_{name}")
            return path if path.is_file() else None

        chart = beside("chords.json")
        song = (chart and _read_json(chart).get("song")) or audio.parent.name
        grids.append(
            Grid(
                id=f"{index}/{audio.parent.name}/{number}",
                song=str(song),
                grid=number,
                audio=audio,
                chart=chart,
                notes=beside("melody.notes"),
                beats=beside("beats.beats"),
                screenshot=beside("screenshot.png"),
            )
        )
    return grids


def _single(path: Path, index: int) -> Grid:
    """One audio file as a grid, with any sidecar that shares its stem."""
    stem = path.with_suffix("")
    found = {
        kind: sidecar
        for kind, suffix in SIDECARS.items()
        if (sidecar := stem.with_name(stem.name + suffix)).is_file()
    }
    log.info("%s: %s", path.name, ", ".join(sorted(found)) or "no annotation beside it")
    return Grid(
        id=f"{index}/{stem.name}/1",
        song=stem.name,
        grid=1,
        audio=path,
        **found,
    )


def summary(grid: Grid) -> dict[str, Any]:
    """The picker's view of a grid: enough to choose it, nothing more."""
    annotation = _annotation(grid)
    return {
        "id": grid.id,
        "song": grid.song,
        "grid": grid.grid,
        "bpm": annotation.get("bpm_measured") or annotation.get("bpm_nominal"),
        "has_screenshot": grid.screenshot is not None,
        "has_chords": grid.chart is not None or grid.chords is not None,
        "has_melody": grid.notes is not None,
    }


def ground_truth(grid: Grid) -> dict[str, Any]:
    """Everything annotated about a grid, on the recording's own clock.

    Each chord event keeps its Harte ``label`` and gains a ``short`` one in the
    shorthand chord backends emit, so the two timelines can be compared label
    for label. ``melody`` is the written line, and ``beat_seconds`` the beat a
    note value is measured against -- from the stated tempo, the tracked beats
    or the bar length, in that order, and ``None`` when nothing says.
    """
    events = _chord_events(grid)
    annotation = _annotation(grid)
    bars, sections, alternates = _bar_grid(annotation, events)
    beats = _beats(grid)
    if not bars and beats:
        bars = _bars_from_beats(beats)
    melody = [_note(n) for n in (lab.read_notes(grid.notes) if grid.notes else [])]
    ends = [e["end"] for e in events] + [n["end"] for n in melody]

    return {
        **summary(grid),
        "events": events,
        "bars": bars,
        "sections": sections,
        "alternates": alternates,
        "melody": melody,
        "beat_seconds": _beat_seconds(annotation, bars, beats),
        "annotated_end": max(ends, default=0.0),
        "audio_seconds": annotation.get("audio_seconds"),
    }


def _chord_events(grid: Grid) -> list[dict[str, Any]]:
    """The chord timeline, from the chart if there is one, else from a ``.lab``."""
    if grid.chart is not None:
        charted = _read_json(grid.chart)["events"]
        raw = [(float(e["start"]), float(e["end"]), e["chord"]) for e in charted]
    elif grid.chords is not None:
        raw = [(e.start, e.end, e.label) for e in lab.read(grid.chords)]
    else:
        return []
    return [
        {"start": start, "end": end, "label": label, "short": normalise_label(label)}
        for start, end, label in raw
    ]


def _note(note: Any) -> dict[str, Any]:
    return {
        "start": note.start,
        "end": note.end,
        "pitch": note.pitch,
        "confidence": note.confidence,
    }


def _annotation(grid: Grid) -> dict[str, Any]:
    if grid.chart is None:
        return {}
    return _read_json(grid.chart).get("annotation") or {}


def _beats(grid: Grid) -> list[Any]:
    return lab.read_beats(grid.beats) if grid.beats else []


def _bars_from_beats(beats: list[Any]) -> list[float]:
    """Downbeats, plus the boundary that closes the last bar.

    One more boundary than there are bars, as a chart's annotation gives: the
    page reads the last entry as a closing line rather than a bar of its own,
    so without it the final bar goes unnumbered.
    """
    downbeats = [b.time for b in beats if b.position == 1]
    if len(downbeats) > 1:
        downbeats.append(2 * downbeats[-1] - downbeats[-2])
    return downbeats


def _bar_grid(
    annotation: dict[str, Any], events: list[dict[str, Any]]
) -> tuple[list[float], list[dict[str, Any]], list[dict[str, Any]]]:
    """Bar starts, sections and alternates in seconds, from a chart's annotation."""
    if "bar_seconds" not in annotation or "bars" not in annotation:
        return [], [], []
    origin = float(annotation.get("start_seconds", events[0]["start"] if events else 0.0))
    length = float(annotation["bar_seconds"])

    def bar_start(bar: int) -> float:
        # Bars are numbered from 1, as printed on the chart.
        return round(origin + (bar - 1) * length, 3)

    bars = [bar_start(bar) for bar in range(1, int(annotation["bars"]) + 2)]
    sections = [
        {"start": bar_start(s["bar"]), "bar": s["bar"], "name": s["name"]}
        for s in annotation.get("sections", [])
    ]
    alternates = [
        {
            "start": bar_start(a["bar"]),
            "end": bar_start(a["bar"] + 1),
            "bar": a["bar"],
            "label": a["chord"],
            "short": normalise_label(a["chord"]),
        }
        for a in annotation.get("alternates", [])
    ]
    return bars, sections, alternates


def _beat_seconds(annotation: dict[str, Any], bars: list[float], beats: list[Any]) -> float | None:
    """How long one beat is, for turning a note's duration into a note value.

    A stated tempo is the most trustworthy; tracked beats measure it from the
    recording; a bar length only gives it away if the metre is the default.
    """
    bpm = annotation.get("bpm_measured") or annotation.get("bpm_nominal")
    if bpm:
        return 60.0 / float(bpm)
    if len(beats) > 1:
        return median(b.time - a.time for a, b in pairwise(beats))
    if len(bars) > 1:
        return (bars[1] - bars[0]) / DEFAULT_METRE
    return None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
