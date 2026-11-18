"""Reading and writing ``.lab`` chord annotation files, ``.beats`` grids and ``.notes`` melodies.

The Isophonics/MIREX convention: one chord per line, ``start end label``,
times in seconds from the start of the recording. It is the lingua franca for
chord ground truth, so it is both what :mod:`songscribe.demo` emits and what
:mod:`songscribe.eval` reads. Beat timelines get the same treatment
(``time position``), and melodies theirs (``start end pitch``), since this
module already owns "simple timed text file".
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from math import isfinite
from pathlib import Path

from .types import BeatEvent, ChordEvent, NoteEvent

log = logging.getLogger(__name__)

#: Isophonics writes space-separated, madmom and :mod:`songscribe.demo` write
#: tab-separated, and some published annotation sets mix both within one file.
#: Splitting on arbitrary whitespace accepts all of them.
WRITE_SEPARATOR = "\t"

#: Times are written with microsecond resolution: enough that a boundary
#: survives a round trip, short enough to stay readable.
TIME_PRECISION = 6

#: A reversed segment this narrow is an artefact of how the annotation was
#: produced, not an ordering mistake, and is skipped; the lines around it keep
#: their own timestamps. A wider reversal is a line in the wrong place and is
#: rejected. The width is compared at :data:`TIME_PRECISION`, so a nominal
#: reversal of exactly the tolerance is covered.
MAX_REVERSED_SPAN = 1.0


class LabFormatError(ValueError):
    """A ``.lab`` file could not be parsed.

    Raised rather than skipping the offending line: a silently dropped chord
    leaves its time labelled ``N``, which moves an evaluation score without
    anything reporting a problem. The one exception is the reversal
    :data:`MAX_REVERSED_SPAN` covers, which spans no time and is logged.
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
            if round(start - end, TIME_PRECISION) > MAX_REVERSED_SPAN:
                raise LabFormatError(f"{where}{number}: end {end} precedes start {start}")
            # A survey of a new corpus should name the dropped segment rather
            # than read clean. %g, so a sub-millisecond width isn't shown as 0.
            log.warning(
                "%s%d: skipping segment reversed by %gs (end %s precedes start %s)",
                where,
                number,
                start - end,
                end,
                start,
            )
            continue

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


#: MIDI's own range. A pitch outside it is a conversion gone wrong -- a
#: frequency written where a note number belongs, most likely -- not a note.
MIN_PITCH, MAX_PITCH = 0, 127

#: Confidence of a note whose file left the column off.
DEFAULT_CONFIDENCE = 1.0


def parse_notes(text: str, source: str | None = None) -> list[NoteEvent]:
    """Parse ``start end pitch [confidence]`` lines into note events.

    ``pitch`` is a MIDI note number, as :class:`~songscribe.types.NoteEvent`
    has it. The confidence column is optional: a tracker with nothing to say
    about it leaves it off, and the notes read back at
    :data:`DEFAULT_CONFIDENCE`.
    """
    notes: list[NoteEvent] = []
    where = f"{source}:" if source else "line "

    for number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        fields = line.split()
        if not 3 <= len(fields) <= 4:
            raise LabFormatError(
                f"{where}{number}: expected 'start end pitch [confidence]', got {line!r}"
            )

        try:
            start, end = float(fields[0]), float(fields[1])
            pitch = float(fields[2])
            confidence = float(fields[3]) if len(fields) == 4 else DEFAULT_CONFIDENCE
        except ValueError as error:
            raise LabFormatError(f"{where}{number}: bad number in {line!r}") from error

        # Before the ordering test, which nan passes: every comparison against
        # it is False. See parse() -- a nan boundary spreads silently.
        if not (isfinite(start) and isfinite(end)):
            raise LabFormatError(f"{where}{number}: non-finite timestamp in {line!r}")
        if not isfinite(pitch) or pitch != int(pitch):
            raise LabFormatError(f"{where}{number}: pitch {fields[2]} is not a whole MIDI note")
        if not MIN_PITCH <= pitch <= MAX_PITCH:
            raise LabFormatError(
                f"{where}{number}: pitch {fields[2]} is outside MIDI {MIN_PITCH}-{MAX_PITCH}"
            )
        if not isfinite(confidence):
            raise LabFormatError(f"{where}{number}: non-finite confidence in {line!r}")

        if end < start:
            # Same rule as the chord reader's, for the same reason: see parse().
            if round(start - end, TIME_PRECISION) > MAX_REVERSED_SPAN:
                raise LabFormatError(f"{where}{number}: end {end} precedes start {start}")
            log.warning(
                "%s%d: skipping note reversed by %gs (end %s precedes start %s)",
                where,
                number,
                start - end,
                end,
                start,
            )
            continue

        notes.append(NoteEvent(start=start, end=end, pitch=int(pitch), confidence=confidence))

    return notes


def read_notes(path: str | Path) -> list[NoteEvent]:
    """Read a ``.notes`` file into note events."""
    path = Path(path)
    return parse_notes(path.read_text(encoding="utf-8-sig"), source=path.name)


def format_notes(notes: Sequence[NoteEvent]) -> str:
    """Render note events as ``.notes`` text, including the trailing newline.

    One note per line: start, end, MIDI pitch::

        0.000000    0.625000    72

    The confidence column is written only when some note carries one.
    """
    graded = any(n.confidence != DEFAULT_CONFIDENCE for n in notes)
    lines = []
    for note in notes:
        line = (
            f"{note.start:.{TIME_PRECISION}f}{WRITE_SEPARATOR}"
            f"{note.end:.{TIME_PRECISION}f}{WRITE_SEPARATOR}{note.pitch}"
        )
        if graded:
            line += f"{WRITE_SEPARATOR}{note.confidence:.{TIME_PRECISION}f}"
        lines.append(line)
    return "".join(f"{line}\n" for line in lines)


def write_notes(path: str | Path, notes: Sequence[NoteEvent]) -> None:
    """Write note events to ``path`` as a ``.notes`` file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_notes(notes), encoding="utf-8")
