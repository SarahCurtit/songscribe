"""Staff notation for a melody: which line a note sits on, and how it is written.

Pure Python and pure music theory. This module decides a note's *notation* --
clef, staff position, accidental, note value -- and leaves every pixel to the
page that draws it. That split is what makes a score testable without a
browser, the way :mod:`songscribe.viewer.agreement` is testable without audio.

Positions are in **staff steps**: diatonic steps above the staff's bottom line,
so 0 is the bottom line, 1 the space above it, and 8 the top line. One step is
half a staff space, which is the only vertical unit a renderer needs.

The staff is drawn on the recording's own seconds-from-start clock, not reflowed
into measures of equal width: a note's x is its onset, so the score stays lined
up with the audio, the chord lanes and the playhead. It is proportional
notation, which is what a viewer with a playhead needs; engraved sheet music
would have to break that alignment to justify its bars.

No key signature is printed, so every accidental is written next to its note.
``sharps`` picks the spelling, which also picks the line: C#4 and Db4 are the
same pitch on different lines, as they are on paper.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import floor, log2
from typing import Any

from ..types import NoteEvent

#: Letter names in diatonic order, so an index into this is a staff position.
LETTERS = "CDEFGAB"

#: Pitch class -> (index into :data:`LETTERS`, accidental), spelled with sharps
#: and with flats. Neither table spells B# or Cb, so a note never lands in the
#: octave either side of the one its MIDI number says.
SHARP_SPELLING = (
    (0, ""),
    (0, "#"),
    (1, ""),
    (1, "#"),
    (2, ""),
    (3, ""),
    (3, "#"),
    (4, ""),
    (4, "#"),
    (5, ""),
    (5, "#"),
    (6, ""),
)
FLAT_SPELLING = (
    (0, ""),
    (1, "b"),
    (1, ""),
    (2, "b"),
    (2, ""),
    (3, ""),
    (4, "b"),
    (4, ""),
    (5, "b"),
    (5, ""),
    (6, "b"),
    (6, ""),
)

#: Diatonic index of each clef's bottom line: E4 for treble, G2 for bass.
#: Indices count diatonic steps from C0, so an octave is 7.
CLEF_BOTTOM_LINE = {"treble": 30, "bass": 18}
DEFAULT_CLEF = "treble"

STEPS_PER_OCTAVE = 7

#: The five lines, in staff steps. A note outside them needs ledger lines.
BOTTOM_LINE = 0
MIDDLE_LINE = 4
TOP_LINE = 8

#: Note values as a multiple of one beat, as ``(name, dots, beats)``, assuming
#: the beat is a quarter note -- true of every metre the demo and the chord
#: corpora use. Ordered longest first so a tie resolves to the simpler value.
NOTE_VALUES = (
    ("whole", 0, 4.0),
    ("half", 1, 3.0),
    ("half", 0, 2.0),
    ("quarter", 1, 1.5),
    ("quarter", 0, 1.0),
    ("eighth", 1, 0.75),
    ("eighth", 0, 0.5),
    ("sixteenth", 0, 0.25),
)

#: Each value's head and stem: ``(filled, stem, flags)``.
NOTE_HEADS = {
    "whole": (False, False, 0),
    "half": (False, True, 0),
    "quarter": (True, True, 0),
    "eighth": (True, True, 1),
    "sixteenth": (True, True, 2),
}


def spelling(sharps: bool = True) -> tuple[tuple[int, str], ...]:
    """The pitch-class table ``sharps`` selects."""
    return SHARP_SPELLING if sharps else FLAT_SPELLING


def pitch_name(pitch: int, sharps: bool = True) -> str:
    """``60`` -> ``C4``; ``61`` -> ``C#4``, or ``Db4`` spelled with flats."""
    letter, accidental = spelling(sharps)[pitch % 12]
    return f"{LETTERS[letter]}{accidental}{pitch // 12 - 1}"


def diatonic_index(pitch: int, sharps: bool = True) -> int:
    """Diatonic steps from C0, which is what a staff position is measured in."""
    letter, _ = spelling(sharps)[pitch % 12]
    return (pitch // 12 - 1) * STEPS_PER_OCTAVE + letter


def accidental_of(pitch: int, sharps: bool = True) -> str:
    """``""``, ``"#"`` or ``"b"`` -- what is printed before the note head."""
    return spelling(sharps)[pitch % 12][1]


def staff_step(pitch: int, clef: str = DEFAULT_CLEF, sharps: bool = True) -> int:
    """Staff steps above ``clef``'s bottom line: 0 is that line, 8 the top one.

    Raises :class:`ValueError` for a clef this module does not draw, rather
    than quietly placing the line an octave out.
    """
    if clef not in CLEF_BOTTOM_LINE:
        raise ValueError(f"unknown clef {clef!r}, expected one of {sorted(CLEF_BOTTOM_LINE)}")
    return diatonic_index(pitch, sharps) - CLEF_BOTTOM_LINE[clef]


def staff_step_at(pitch: float, clef: str = DEFAULT_CLEF, sharps: bool = True) -> float:
    """Staff step of a *fractional* pitch, for drawing a pitch contour.

    Interpolated between the two semitones either side, so the curve passes
    exactly through the placement of every whole semitone and stays continuous
    between them. A chromatic glide therefore climbs unevenly -- which is what
    it does on paper, where some semitones change line and some do not.
    """
    low = floor(pitch)
    base = staff_step(low, clef, sharps)
    return base + (pitch - low) * (staff_step(low + 1, clef, sharps) - base)


def choose_clef(pitches: Sequence[int], sharps: bool = True) -> str:
    """The clef that leaves the fewest notes off the staff.

    A lead tracked an octave low -- pYIN's commonest error, and the bass line
    on an unseparated mix -- belongs in the bass clef rather than on eight
    ledger lines. Ties go to the treble, where a sung or played lead sits.
    """
    if not pitches:
        return DEFAULT_CLEF
    return min(
        CLEF_BOTTOM_LINE,
        key=lambda clef: sum(
            not BOTTOM_LINE <= staff_step(p, clef, sharps) <= TOP_LINE for p in pitches
        ),
    )


def ledger_steps(step: int) -> tuple[int, ...]:
    """The ledger lines a note at ``step`` needs, as staff steps.

    Only lines, so a note in the space just outside the staff needs none.
    """
    if step > TOP_LINE:
        return tuple(range(TOP_LINE + 2, step + 1, 2))
    if step < BOTTOM_LINE:
        return tuple(range(BOTTOM_LINE - 2, step - 1, -2))
    return ()


def note_value(duration: float, beat_seconds: float) -> tuple[str, int]:
    """The written value nearest ``duration``, as ``(name, dots)``.

    Compared in log space, because note values are multiplicative: an eighth
    is as far from a quarter as a quarter is from a half. Anything longer than
    a whole note or shorter than a sixteenth is clamped to those, since this is
    a transcription of what was played rather than a score to be performed.
    """
    if beat_seconds <= 0:
        raise ValueError(f"beat_seconds must be positive, got {beat_seconds}")
    if duration <= 0:
        raise ValueError(f"duration must be positive, got {duration}")
    beats = duration / beat_seconds
    name, dots, _ = min(NOTE_VALUES, key=lambda value: abs(log2(beats / value[2])))
    return name, dots


def engrave(
    notes: Sequence[NoteEvent],
    *,
    beat_seconds: float | None = None,
    clef: str | None = None,
    sharps: bool = True,
) -> dict[str, Any]:
    """``notes`` as staff glyphs, ready to draw on the seconds timeline.

    ``clef`` defaults to whichever fits the line; pass one to pin it. Without
    ``beat_seconds`` -- no bar grid and no tempo for this recording -- the
    rhythm is not guessed: every note gets a stemless head, ``value`` is
    ``None``, and the page says the rhythm is unknown rather than printing
    quarter notes nothing measured.
    """
    if clef is None:
        clef = choose_clef([n.pitch for n in notes], sharps)
    elif clef not in CLEF_BOTTOM_LINE:
        raise ValueError(f"unknown clef {clef!r}, expected one of {sorted(CLEF_BOTTOM_LINE)}")
    if beat_seconds is not None and beat_seconds <= 0:
        raise ValueError(f"beat_seconds must be positive, got {beat_seconds}")

    glyphs: list[dict[str, Any]] = []
    for note in notes:
        step = staff_step(note.pitch, clef, sharps)
        if beat_seconds is None or note.duration <= 0:
            value, dots, head = None, 0, (True, False, 0)
        else:
            value, dots = note_value(note.duration, beat_seconds)
            head = NOTE_HEADS[value]
        filled, stem, flags = head
        glyphs.append(
            {
                "start": note.start,
                "end": note.end,
                "pitch": note.pitch,
                "confidence": note.confidence,
                "name": pitch_name(note.pitch, sharps),
                "step": step,
                "accidental": accidental_of(note.pitch, sharps),
                "value": value,
                "dots": dots,
                "filled": filled,
                "stem": stem,
                "flags": flags,
                # Convention: above the middle line the stem hangs down, so it
                # stays inside the staff instead of running off the top.
                "stem_down": step > MIDDLE_LINE,
                "ledger": list(ledger_steps(step)),
            }
        )
    return {
        "clef": clef,
        "sharps": sharps,
        "beat_seconds": beat_seconds,
        "bottom_line": CLEF_BOTTOM_LINE[clef],
        "notes": glyphs,
    }
