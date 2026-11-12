"""The viewer's melody side: staff placement, note values, and note agreement.

Pure music theory and invented notes -- no audio, no model and no recording,
so this runs in CI on the core dependencies.
"""

from __future__ import annotations

import pytest

from songscribe.types import NoteEvent
from songscribe.viewer import melody, score

# MIDI numbers used throughout, named so the expectations read as music.
C4, CS4, D4, E4, F4, A4, AS4, B3, C5, F5, G2, A3 = 60, 61, 62, 64, 65, 69, 70, 59, 72, 77, 43, 57
# Far under the treble staff: sixteen steps, which is where a tracker that has
# slipped onto the bass puts a note.
C2 = 36


def line(*pitches: int, seconds: float = 1.0) -> list[NoteEvent]:
    """Invented notes, one after another, each ``seconds`` long."""
    return [
        NoteEvent(index * seconds, (index + 1) * seconds, pitch)
        for index, pitch in enumerate(pitches)
    ]


# --------------------------------------------------------------------------- #
# Spelling
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("pitch", "sharps", "expected"),
    [
        (C4, True, "C4"),
        (CS4, True, "C#4"),
        (CS4, False, "Db4"),
        (AS4, True, "A#4"),
        (AS4, False, "Bb4"),
        # Either side of the octave boundary: B3 must not become C4's octave.
        (B3, True, "B3"),
        (B3, False, "B3"),
        (C5, True, "C5"),
    ],
)
def test_pitch_name(pitch: int, sharps: bool, expected: str) -> None:
    assert score.pitch_name(pitch, sharps) == expected


def test_accidental_of() -> None:
    assert score.accidental_of(CS4, sharps=True) == "#"
    assert score.accidental_of(CS4, sharps=False) == "b"
    assert score.accidental_of(D4) == ""


# --------------------------------------------------------------------------- #
# Staff placement
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("pitch", "clef", "expected"),
    [
        (E4, "treble", score.BOTTOM_LINE),  # bottom line
        (F5, "treble", score.TOP_LINE),  # top line
        (D4, "treble", -1),  # space below the staff
        (C4, "treble", -2),  # middle C: one ledger line below
        (B3, "treble", -3),  # and B3 below that
        (G2, "bass", score.BOTTOM_LINE),
        (A3, "bass", score.TOP_LINE),
        (C4, "bass", 10),  # middle C: one ledger line above
    ],
)
def test_staff_step(pitch: int, clef: str, expected: int) -> None:
    assert score.staff_step(pitch, clef) == expected


def test_spelling_picks_the_line_not_just_the_accidental() -> None:
    """C#4 and Db4 are one pitch on two lines, as they are on paper."""
    assert score.staff_step(CS4, "treble", sharps=True) == score.staff_step(C4, "treble")
    assert score.staff_step(CS4, "treble", sharps=False) == score.staff_step(D4, "treble")


def test_staff_step_rejects_an_unknown_clef() -> None:
    with pytest.raises(ValueError, match="unknown clef"):
        score.staff_step(C4, "alto")


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        (score.MIDDLE_LINE, ()),
        (-1, ()),  # a space just outside needs no line
        (-2, (-2,)),
        (-4, (-2, -4)),
        (9, ()),
        (10, (10,)),
        (12, (10, 12)),
    ],
)
def test_ledger_steps(step: int, expected: tuple[int, ...]) -> None:
    assert score.ledger_steps(step) == expected


def test_choose_clef_follows_the_range() -> None:
    assert score.choose_clef(line(E4, A4, F5)) == "treble"
    assert score.choose_clef(line(G2, 45, 47)) == "bass"
    assert score.choose_clef([]) == "treble"
    # Middle C is one ledger line off either staff: a tie goes to the treble,
    # where a lead sits.
    assert score.choose_clef(line(C4)) == "treble"


def test_choose_clef_weighs_seconds_not_notes() -> None:
    """A tracker that drops onto the bass must not outvote the melody.

    It chops a wandering line into many short notes while a phrase is a few
    long ones, so counting notes would hand the staff to the wrong voice.
    """
    melody = [NoteEvent(0.0, 4.0, A4), NoteEvent(4.0, 8.0, E4)]
    fragments = [NoteEvent(8.0 + i * 0.2, 8.2 + i * 0.2, 45) for i in range(8)]
    assert score.choose_clef(melody + fragments) == "treble"
    # And where the weight really is in the bass, it goes there.
    assert score.choose_clef([*melody, NoteEvent(8.0, 20.0, 45)]) == "bass"


@pytest.mark.parametrize(
    ("step", "shift"),
    [
        (0, 0),
        (8, 0),
        (13, 0),
        (14, 1),
        (20, 1),
        (21, 2),
        (-5, 0),
        (-6, -1),
        (-12, -1),
        (-16, -2),
    ],
)
def test_octave_shift(step: int, shift: int) -> None:
    assert score.octave_shift(step) == shift
    # However far out it started, it is written within reach of the staff.
    assert -6 <= step - shift * score.STEPS_PER_OCTAVE <= 14


@pytest.mark.parametrize(
    ("shift", "label"),
    [(0, ""), (1, "8va"), (-1, "8vb"), (2, "15ma"), (-2, "15mb")],
)
def test_octave_label(shift: int, label: str) -> None:
    assert score.octave_label(shift) == label


def test_a_lone_note_far_out_keeps_its_ledger_lines() -> None:
    """One note under the staff is a slip, and should look like one."""
    engraved = score.engrave(line(A4, C2, A4, A4), clef="treble", beat_seconds=1.0)
    assert [n["octave"] for n in engraved["notes"]] == [0, 0, 0, 0]
    assert engraved["octaves"] == []
    assert engraved["notes"][1]["step"] == score.staff_step(C2, "treble")


def test_a_passage_far_out_is_written_in_and_bracketed() -> None:
    engraved = score.engrave(line(A4, C2, C2, C2, C2, A4), clef="treble", beat_seconds=1.0)
    assert [n["octave"] for n in engraved["notes"]] == [0, -2, -2, -2, -2, 0]
    (span,) = engraved["octaves"]
    assert (span["label"], span["start"], span["end"]) == ("15mb", 1.0, 5.0)
    # Every head is now within reach of the staff rather than far beneath it.
    assert all(-6 <= n["step"] <= 14 for n in engraved["notes"])


def test_a_bracket_never_covers_a_note_left_where_it_sounds() -> None:
    """Two runs split by a note on the staff stay two brackets.

    One bracket across the gap would have the note under it read an octave out.
    """
    engraved = score.engrave(
        line(C2, C2, C2, C2, A4, C2, C2, C2, C2), clef="treble", beat_seconds=1.0
    )
    first, second = engraved["octaves"]
    assert first["end"] <= 4.0
    assert second["start"] >= 5.0
    assert engraved["notes"][4]["octave"] == 0


def test_staff_step_at_interpolates_between_semitones() -> None:
    assert score.staff_step_at(float(E4)) == score.staff_step(E4)
    # E4 to F4 changes line, so half a semitone is half a step.
    assert score.staff_step_at(E4 + 0.5) == pytest.approx(0.5)
    # C4 to C#4 keeps the letter, so the contour stays on C4's line.
    assert score.staff_step_at(C4 + 0.5) == pytest.approx(score.staff_step(C4))


# --------------------------------------------------------------------------- #
# Note values
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("beats", "expected"),
    [
        (4.0, ("whole", 0)),
        (3.0, ("half", 1)),
        (2.0, ("half", 0)),
        (1.5, ("quarter", 1)),
        (1.0, ("quarter", 0)),
        (0.75, ("eighth", 1)),
        (0.5, ("eighth", 0)),
        (0.25, ("sixteenth", 0)),
        # Nearest in log space: 0.6 of a beat is closer to an eighth than to a
        # dotted eighth, though it is nearer in seconds to the dotted one.
        (0.6, ("eighth", 0)),
        # Clamped at both ends rather than inventing a breve or a 32nd.
        (16.0, ("whole", 0)),
        (0.05, ("sixteenth", 0)),
    ],
)
def test_note_value(beats: float, expected: tuple[str, int]) -> None:
    assert score.note_value(beats * 0.5, 0.5) == expected


@pytest.mark.parametrize(("duration", "beat"), [(1.0, 0.0), (1.0, -1.0), (0.0, 1.0), (-1.0, 1.0)])
def test_note_value_rejects_nonsense(duration: float, beat: float) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        score.note_value(duration, beat)


# --------------------------------------------------------------------------- #
# Engraving
# --------------------------------------------------------------------------- #


def test_engrave_places_and_writes_each_note() -> None:
    notes = [NoteEvent(0.0, 1.0, E4), NoteEvent(1.0, 1.5, F5), NoteEvent(2.0, 2.5, CS4)]
    engraved = score.engrave(notes, beat_seconds=1.0)

    assert engraved["clef"] == "treble"
    assert engraved["beat_seconds"] == 1.0
    first, second, third = engraved["notes"]

    assert (first["step"], first["value"], first["dots"]) == (0, "quarter", 0)
    assert (first["filled"], first["stem"], first["flags"]) == (True, True, 0)
    # On the bottom line the stem points up; above the middle line it hangs down.
    assert first["stem_down"] is False
    assert second["stem_down"] is True
    assert (second["value"], second["flags"]) == ("eighth", 1)
    assert third["accidental"] == "#"
    assert third["name"] == "C#4"
    assert third["ledger"] == [-2]


def test_engrave_without_a_beat_does_not_guess_a_rhythm() -> None:
    engraved = score.engrave([NoteEvent(0.0, 1.0, E4)], beat_seconds=None)
    (note,) = engraved["notes"]
    assert note["value"] is None
    assert note["dots"] == 0
    # A stemless filled head: a note was heard, its written length is unknown.
    assert (note["filled"], note["stem"], note["flags"]) == (True, False, 0)


def test_engrave_follows_a_pinned_clef_and_rejects_a_bad_one() -> None:
    notes = [NoteEvent(0.0, 1.0, G2)]
    assert score.engrave(notes)["clef"] == "bass"
    assert score.engrave(notes, clef="treble")["clef"] == "treble"
    with pytest.raises(ValueError, match="unknown clef"):
        score.engrave(notes, clef="tenor")
    with pytest.raises(ValueError, match="must be positive"):
        score.engrave(notes, beat_seconds=0.0)


def test_engrave_takes_an_empty_line() -> None:
    engraved = score.engrave([])
    assert engraved["notes"] == []
    assert engraved["clef"] == "treble"


# --------------------------------------------------------------------------- #
# Note agreement
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("written", "played", "expected"),
    [
        (C4, C4, "exact"),
        (C4, C5, "octave"),
        (C4, C4 - 12, "octave"),
        (C4, D4, "miss"),
        (C4, CS4, "miss"),
    ],
)
def test_level(written: int, played: int, expected: str) -> None:
    assert melody.level(written, played) == expected


def test_compare_grades_both_lines() -> None:
    written = [NoteEvent(0.0, 1.0, C4), NoteEvent(1.0, 2.0, D4), NoteEvent(2.0, 3.0, E4)]
    played = [
        NoteEvent(0.0, 1.0, C4),  # the written note
        NoteEvent(1.0, 2.0, D4 + 12),  # right note, an octave up
        NoteEvent(2.0, 3.0, F4),  # a different note
    ]
    result = melody.compare(written, played)

    assert [n["level"] for n in result["played"]] == ["exact", "octave", "miss"]
    assert [n["level"] for n in result["written"]] == ["exact", "octave", "miss"]
    totals = result["totals"]
    assert totals["exact"] == pytest.approx(1 / 3)
    assert totals["octave"] == pytest.approx(2 / 3)
    assert totals["found"] == pytest.approx(1.0)
    assert totals["extra"] == 0


def test_compare_names_what_is_missing_and_what_is_extra() -> None:
    written = [NoteEvent(0.0, 1.0, C4), NoteEvent(5.0, 6.0, D4)]
    played = [NoteEvent(0.0, 1.0, C4), NoteEvent(3.0, 4.0, A4)]
    result = melody.compare(written, played)

    assert [n["level"] for n in result["written"]] == ["exact", "missing"]
    assert [n["level"] for n in result["played"]] == ["exact", "extra"]
    assert result["totals"]["found"] == pytest.approx(0.5)
    assert result["totals"]["extra"] == 1


def test_compare_matches_on_midpoints_not_any_overlap() -> None:
    """A clip of the next note is not a reading of this one."""
    written = [NoteEvent(0.0, 1.0, C4)]
    played = [NoteEvent(0.9, 1.9, C4)]
    result = melody.compare(written, played)
    assert result["written"][0]["level"] == "missing"
    assert result["played"][0]["level"] == "extra"


def test_compare_shows_over_segmentation_in_the_counts_not_in_extra() -> None:
    """Notes are not paired off, so a split note is right three times over.

    Deliberate: ``extra`` answers "was anything written here at all", and
    every fragment is inside the written note. What gives the splitting away
    is three played notes against one written one -- and, on the staff, three
    heads under one.
    """
    written = [NoteEvent(0.0, 3.0, C4)]
    played = [NoteEvent(0.0, 1.0, C4), NoteEvent(1.0, 2.0, C4), NoteEvent(2.0, 3.0, C4)]
    result = melody.compare(written, played)

    assert [n["level"] for n in result["played"]] == ["exact", "exact", "exact"]
    assert result["totals"]["exact"] == pytest.approx(1.0)
    assert result["totals"]["extra"] == 0
    assert (result["totals"]["played_notes"], result["totals"]["written_notes"]) == (3, 1)


def test_compare_with_nothing_written_has_no_denominator() -> None:
    result = melody.compare([], [NoteEvent(0.0, 1.0, C4)])
    totals = result["totals"]
    assert totals["exact"] is None
    assert totals["octave"] is None
    assert totals["found"] is None
    assert totals["extra"] == 1
    assert totals["written_notes"] == 0
    assert totals["played_notes"] == 1


@pytest.mark.parametrize(
    ("labels", "sharps"),
    [
        ([], True),
        (["C", "Am", "G7"], True),
        (["F", "Bb", "C7"], False),  # a chart in F
        (["C#m", "F#"], True),
        (["Ebm7", "Ab"], False),
        (["N"], True),  # no chord has no root to follow
    ],
)
def test_spell_with_sharps_follows_the_chart(labels: list[str], sharps: bool) -> None:
    assert score.spell_with_sharps(labels) is sharps
