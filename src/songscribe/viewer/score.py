"""Staff notation for a melody: which line a note sits on, and how it is written.

Decides clef, staff position, accidental and note value, leaving every pixel to
the page. Positions are in **staff steps**: diatonic steps above the staff's
bottom line, so 0 is that line and 8 the top one. A note's x is its onset in
seconds, so the staff is proportional, not reflowed into bars of equal width.
No key signature is printed, so ``sharps`` picks the accidental and the line
with it: C#4 and Db4 sit a step apart.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import groupby
from math import ceil, floor, log2
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

#: How far outside the staff a note may sit before it is written an octave in
#: and bracketed: five steps is two ledger lines.
OCTAVE_SIGN_MARGIN = 5

#: A displaced run shorter than this keeps its ledger lines instead. A bracket
#: covers a passage; over a short run it is more to read than it saves, and a
#: lone note far from the staff is usually a slip and should look like one.
MIN_OCTAVE_SIGN_NOTES = 4

#: What one octave of displacement is called, before the a or b. Wider
#: displacements are named from the interval, which is 8, 15, 22, ...
OCTAVE_SIGN_NAMES = {1: "8v"}

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


def spell_with_sharps(labels: Sequence[str]) -> bool:
    """Whether a staff should use sharps, following the chart's chord roots.

    A chart in F prints ``Bb``, and its melody should not print ``A#`` beside
    it. One flat root decides it: a key is spelled one way throughout.
    """
    return not any(label[1:2] == "b" for label in labels)


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


def time_in_reach(notes: Sequence[NoteEvent], clef: str, sharps: bool = True) -> float:
    """Seconds of ``notes`` that ``clef`` can carry without an octave sign.

    The five lines and the ledger lines either side of them, since a line
    sitting just under a staff is read there rather than moved.
    """
    return sum(
        note.duration
        for note in notes
        if BOTTOM_LINE - OCTAVE_SIGN_MARGIN
        <= staff_step(note.pitch, clef, sharps)
        <= TOP_LINE + OCTAVE_SIGN_MARGIN
    )


def choose_clef(notes: Sequence[NoteEvent], sharps: bool = True) -> str:
    """The clef that carries the most of the line, by time.

    Counting notes would let a line chopped into many short ones outvote a
    melody of a few long ones. Ties go to the treble.
    """
    if not notes:
        return DEFAULT_CLEF
    return max(CLEF_BOTTOM_LINE, key=lambda clef: time_in_reach(notes, clef, sharps))


def octave_shift(step: int) -> int:
    """Octaves to write a note in by so it lands near the staff; 0 for most.

    Positive means it sounds that many octaves above where it is written, as
    ``8va`` does.
    """
    above = step - TOP_LINE - OCTAVE_SIGN_MARGIN
    if above > 0:
        return ceil(above / STEPS_PER_OCTAVE)
    below = BOTTOM_LINE - OCTAVE_SIGN_MARGIN - step
    if below > 0:
        # Negative: the note is written above where it sounds, as 8vb is.
        return -ceil(below / STEPS_PER_OCTAVE)
    return 0


def passage_shifts(steps: Sequence[int]) -> list[int]:
    """The octave shift for each step, kept only where a run shares one."""
    kept: list[int] = []
    for shift, group in groupby(octave_shift(step) for step in steps):
        run = sum(1 for _ in group)
        kept.extend([shift if run >= MIN_OCTAVE_SIGN_NOTES else 0] * run)
    return kept


def octave_label(shift: int) -> str:
    """``1`` -> ``8va``, ``-2`` -> ``15mb``; ``""`` for no displacement."""
    if not shift:
        return ""
    name = OCTAVE_SIGN_NAMES.get(abs(shift), f"{7 * abs(shift) + 1}m")
    return f"{name}a" if shift > 0 else f"{name}b"


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
    octave_signs: bool = True,
) -> dict[str, Any]:
    """``notes`` as staff glyphs, ready to draw on the seconds timeline.

    ``notes`` are in time order and do not overlap, as a monophonic line is.
    ``clef`` defaults to whichever fits the line; pass one to pin it. Without
    ``beat_seconds`` the rhythm is not guessed: every note gets a stemless
    head and ``value`` is ``None``. ``octave_signs=False`` leaves every note
    where it sounds, for a line drawn beside another that carries the signs.
    """
    if clef is None:
        clef = choose_clef(notes, sharps)
    elif clef not in CLEF_BOTTOM_LINE:
        raise ValueError(f"unknown clef {clef!r}, expected one of {sorted(CLEF_BOTTOM_LINE)}")
    if beat_seconds is not None and beat_seconds <= 0:
        raise ValueError(f"beat_seconds must be positive, got {beat_seconds}")

    steps = [staff_step(n.pitch, clef, sharps) for n in notes]
    shifts = passage_shifts(steps) if octave_signs else [0] * len(steps)
    glyphs: list[dict[str, Any]] = []
    for note, shift in zip(notes, shifts, strict=True):
        step = staff_step(note.pitch, clef, sharps) - shift * STEPS_PER_OCTAVE
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
                "octave": shift,
                "octave_label": octave_label(shift),
            }
        )
    return {
        "clef": clef,
        "sharps": sharps,
        "beat_seconds": beat_seconds,
        "bottom_line": CLEF_BOTTOM_LINE[clef],
        "notes": glyphs,
        "octaves": octave_spans(glyphs),
    }


def octave_spans(glyphs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One bracket per run of neighbouring notes written at the same octave.

    A bracket covers a passage, not a note, so notes that follow one another
    under the same displacement share one.
    """
    spans: list[dict[str, Any]] = []
    # The index of the last note taken into the open span: a bracket may only
    # run over neighbours, or an undisplaced note between two runs would fall
    # under it and be read an octave out.
    previous = None
    for index, glyph in enumerate(glyphs):
        shift = glyph["octave"]
        if not shift:
            previous = None
            continue
        last = spans[-1] if spans else None
        if last and previous == index - 1 and last["shift"] == shift:
            last["end"] = max(last["end"], glyph["end"])
        else:
            spans.append(
                {
                    "start": min(glyph["start"], glyph["end"]),
                    "end": glyph["end"],
                    "shift": shift,
                    "label": octave_label(shift),
                }
            )
        previous = index
    return spans
