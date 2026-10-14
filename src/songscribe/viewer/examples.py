"""Find annotated grids in example directories and load their ground truth.

An example directory holds one folder per song and one file per grid per kind
of data, sharing a ``gridN_`` prefix::

    <root>/<song>/gridN_chords.json      ground truth (required)
    <root>/<song>/gridN_audio.wav        the recording it describes (required)
    <root>/<song>/gridN_screenshot.png   the chart it was read from (optional)

The ground truth is timed Harte events in seconds from the start of the WAV.
An optional ``annotation`` object carries bar-indexed metadata -- sections,
and the alternate chords a chart prints above a bar -- plus the bar grid
(``start_seconds``, ``bar_seconds``) needed to put those on the same clock.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..chords import normalise_label

GROUND_TRUTH_PATTERN = re.compile(r"grid(\d+)_chords\.json")


@dataclass(frozen=True, slots=True)
class Grid:
    """One annotated recording, located by the files that make it up."""

    id: str
    song: str
    grid: int
    chords: Path
    audio: Path
    screenshot: Path | None


def discover(roots: list[Path]) -> dict[str, Grid]:
    """Every grid with both ground truth and audio under ``roots``.

    IDs are ``<root index>/<song folder>/<grid number>``: stable for a given
    command line, and free of absolute paths so none leak into URLs. Raises
    when a root is missing or holds no grids -- an empty picker would look
    like a bug in the viewer rather than a typo in the path.
    """
    grids: dict[str, Grid] = {}
    for index, root in enumerate(roots):
        if not root.is_dir():
            raise ValueError(f"examples directory not found: {root}")
        found = 0
        for chords in sorted(root.glob("*/grid*_chords.json")):
            match = GROUND_TRUTH_PATTERN.fullmatch(chords.name)
            if not match:
                continue
            number = int(match.group(1))
            audio = chords.with_name(f"grid{number}_audio.wav")
            if not audio.is_file():
                continue
            screenshot = chords.with_name(f"grid{number}_screenshot.png")
            song = _read_json(chords).get("song") or chords.parent.name
            grid = Grid(
                id=f"{index}/{chords.parent.name}/{number}",
                song=str(song),
                grid=number,
                chords=chords,
                audio=audio,
                screenshot=screenshot if screenshot.is_file() else None,
            )
            grids[grid.id] = grid
            found += 1
        if not found:
            raise ValueError(f"no grid with both _chords.json and _audio.wav under {root}")
    return dict(sorted(grids.items(), key=lambda item: (item[1].song.lower(), item[1].grid)))


def summary(grid: Grid) -> dict[str, Any]:
    """The picker's view of a grid: enough to choose it, nothing more."""
    annotation = _read_json(grid.chords).get("annotation") or {}
    return {
        "id": grid.id,
        "song": grid.song,
        "grid": grid.grid,
        "bpm": annotation.get("bpm_measured") or annotation.get("bpm_nominal"),
        "has_screenshot": grid.screenshot is not None,
    }


def ground_truth(grid: Grid) -> dict[str, Any]:
    """The grid's annotation, with every bar-indexed field converted to seconds.

    Each event keeps its Harte ``label`` and gains a ``short`` one in the
    shorthand chord backends emit, so the two timelines can be compared label
    for label. A file without ``annotation`` still loads; it just has no bar
    grid, sections or alternates.
    """
    data = _read_json(grid.chords)
    events = [
        {
            "start": float(e["start"]),
            "end": float(e["end"]),
            "label": e["chord"],
            "short": normalise_label(e["chord"]),
        }
        for e in data["events"]
    ]
    annotation = data.get("annotation") or {}

    bars: list[float] = []
    sections: list[dict[str, Any]] = []
    alternates: list[dict[str, Any]] = []
    if "bar_seconds" in annotation and "bars" in annotation:
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

    return {
        **summary(grid),
        "events": events,
        "bars": bars,
        "sections": sections,
        "alternates": alternates,
        "annotated_end": events[-1]["end"] if events else 0.0,
        "audio_seconds": annotation.get("audio_seconds"),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
