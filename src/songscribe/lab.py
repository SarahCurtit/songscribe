"""Reading and writing ``.lab`` chord annotation files, and ``.beats`` grids.

The Isophonics/MIREX convention: one chord per line, ``start end label``,
times in seconds from the start of the recording. It is the lingua franca for
chord ground truth, so it is both what :mod:`songscribe.demo` emits and what
:mod:`songscribe.eval` reads. Beat timelines get the same treatment
(``time position``), since this module already owns "simple timed text file".
"""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from pathlib import Path

from .types import BeatEvent, ChordEvent

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

        # Before the ordering test, which nan passes: every comparison against
        # it is False. A nan boundary in a reference or a cached estimate makes
        # every metric over that track nan without anything reporting a problem.
        if not (isfinite(start) and isfinite(end)):
            raise LabFormatError(f"{where}{number}: non-finite timestamp in {line!r}")

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


# A cached grid is what lets the evaluation harness run without a tracker, the
# way .lab fixtures already let it run without a chord model.
def parse_beats(text: str, source: str | None = None) -> list[BeatEvent]:
    """Parse ``time position`` lines into beat events.

    ``position`` is 1-based, as :class:`~songscribe.types.BeatEvent` and
    madmom both have it. A position of 0 is rejected rather than shifted: a
    silently renumbered bar puts every downbeat on the wrong beat.
    """
    beats: list[BeatEvent] = []
    where = f"{source}:" if source else "line "

    for number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        fields = line.split()
        if len(fields) != 2:
            raise LabFormatError(f"{where}{number}: expected 'time position', got {line!r}")

        try:
            time, position = float(fields[0]), int(fields[1])
        except ValueError as error:
            raise LabFormatError(f"{where}{number}: bad beat in {line!r}") from error

        # float() accepts "nan" and "inf". A nan time passes every ordering
        # check downstream, since comparing against it is always False, so it
        # would reach a grid that reports itself healthy and places nothing.
        if not isfinite(time):
            raise LabFormatError(f"{where}{number}: time {time} is not finite")

        if position < 1:
            raise LabFormatError(f"{where}{number}: position {position} is not 1-based")

        beats.append(BeatEvent(time=time, position=position))

    return beats


def read_beats(path: str | Path) -> list[BeatEvent]:
    """Read a ``.beats`` file into beat events."""
    path = Path(path)
    return parse_beats(path.read_text(encoding="utf-8-sig"), source=path.name)


def format_beats(beats: Sequence[BeatEvent]) -> str:
    """Render beat events as ``.beats`` text, including the trailing newline.

    One beat per line: time, a tab, then the 1-based position in the bar::

        0.520000    1
        1.040000    2
    """
    return "".join(f"{b.time:.{TIME_PRECISION}f}{WRITE_SEPARATOR}{b.position}\n" for b in beats)


def write_beats(path: str | Path, beats: Sequence[BeatEvent]) -> None:
    """Write beat events to ``path`` as a ``.beats`` file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_beats(beats), encoding="utf-8")
