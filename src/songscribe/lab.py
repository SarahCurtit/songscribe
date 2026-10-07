"""Reading and writing ``.lab`` chord annotation files.

The Isophonics/MIREX convention: one chord per line, ``start end label``,
times in seconds from the start of the recording. It is the lingua franca for
chord ground truth, so it is both what :mod:`songscribe.demo` emits and what
:mod:`songscribe.eval` reads.

Pure Python and dependency-free, like :mod:`songscribe.align` -- a parser is
exactly the sort of thing that should be testable without a model on disk.
"""

from __future__ import annotations

from pathlib import Path

from .types import ChordEvent

#: Isophonics writes space-separated, madmom and :mod:`songscribe.demo` write
#: tab-separated, and some published annotation sets mix both within one file.
#: Splitting on arbitrary whitespace accepts all of them.
WRITE_SEPARATOR = "\t"

#: Times are written with microsecond resolution: enough that a boundary
#: survives a round trip, short enough to stay readable.
TIME_PRECISION = 6


class LabFormatError(ValueError):
    """A ``.lab`` file could not be parsed.

    Raised rather than skipping the offending line: a silently dropped chord
    shifts every boundary after it, which would corrupt an evaluation score in
    a way that is very hard to notice.
    """


def parse(text: str, source: str | None = None) -> list[ChordEvent]:
    """Parse ``.lab`` text into chord events.

    Blank lines and ``#`` comments are skipped. Labels are returned verbatim,
    in whatever notation the file used -- normalising them is the caller's job,
    since a reference file and an estimate file want different treatment.
    """
    events: list[ChordEvent] = []
    where = f"{source}:" if source else "line "

    for number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        fields = line.split()
        if len(fields) < 3:
            raise LabFormatError(f"{where}{number}: expected 'start end label', got {line!r}")

        try:
            start, end = float(fields[0]), float(fields[1])
        except ValueError as error:
            raise LabFormatError(f"{where}{number}: bad timestamp in {line!r}") from error

        if end < start:
            raise LabFormatError(f"{where}{number}: end {end} precedes start {start}")

        # A label may legitimately contain spaces -- ``C:maj (uncertain)`` turns
        # up in hand-corrected annotation sets -- so the remainder is one label,
        # not a fourth field.
        events.append(ChordEvent(start=start, end=end, label=" ".join(fields[2:])))

    return events


def read(path: str | Path) -> list[ChordEvent]:
    """Read a ``.lab`` file into chord events."""
    path = Path(path)
    return parse(path.read_text(encoding="utf-8-sig"), source=path.name)


def format_events(events: list[ChordEvent]) -> str:
    """Render chord events as ``.lab`` text, including the trailing newline."""
    lines = [
        f"{e.start:.{TIME_PRECISION}f}{WRITE_SEPARATOR}"
        f"{e.end:.{TIME_PRECISION}f}{WRITE_SEPARATOR}{e.label}"
        for e in events
    ]
    return "".join(f"{line}\n" for line in lines)


def write(path: str | Path, events: list[ChordEvent]) -> None:
    """Write chord events to ``path`` as a ``.lab`` file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_events(events), encoding="utf-8")
