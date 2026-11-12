"""How far a transcribed melody agrees with a written one, note by note.

Every note of the estimate gets one level against the written line, and every
written note gets one against the estimate:

``exact``    same MIDI pitch
``octave``   same pitch class, wrong octave
``miss``     a different pitch class
``extra``    (estimate only) nothing was written here
``missing``  (written only) nothing was transcribed here

Two notes are compared when one's **midpoint** falls inside the other, not on
any overlap, since a segmenter's boundaries jitter. Notes are not paired off
one-to-one, so a held note split into three is three exact notes; the note
counts are what show that. Totals count notes rather than seconds.
:mod:`songscribe.eval.melody` is what measures a backend.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..types import NoteEvent

#: Worst-to-best is the reverse of this; an earlier level beats a later one.
LEVELS = ("exact", "octave", "miss")

SEMITONES = 12


def level(written: int, played: int) -> str:
    """The agreement level of one played pitch against one written pitch."""
    if played == written:
        return "exact"
    if played % SEMITONES == written % SEMITONES:
        return "octave"
    return "miss"


def _best_level(pitch: int, against: Sequence[NoteEvent]) -> str | None:
    """The best level ``pitch`` reaches against any note in ``against``."""
    levels = [level(other.pitch, pitch) for other in against]
    return min(levels, key=LEVELS.index) if levels else None


def _sounding_at(notes: Sequence[NoteEvent], time: float) -> list[NoteEvent]:
    return [n for n in notes if n.start <= time < n.end]


def _graded(
    notes: Sequence[NoteEvent], against: Sequence[NoteEvent], absent: str
) -> list[dict[str, Any]]:
    """Each of ``notes`` with the level it reaches, ``absent`` where none does."""
    graded = []
    for note in notes:
        midpoint = (note.start + note.end) / 2
        found = _best_level(note.pitch, _sounding_at(against, midpoint))
        graded.append(
            {
                "start": note.start,
                "end": note.end,
                "pitch": note.pitch,
                "level": found or absent,
            }
        )
    return graded


def compare(written: Sequence[NoteEvent], played: Sequence[NoteEvent]) -> dict[str, Any]:
    """Grade ``played`` against ``written``, both ways round.

    Returns each note with its ``level``, and ``totals`` as fractions of the
    written notes: ``exact``, ``octave`` (exact or an octave out) and ``found``
    (anything transcribed there). ``extra`` counts played notes nothing was
    written for. The fractions are ``None`` when nothing was written.
    """
    graded_played = _graded(played, written, "extra")
    graded_written = _graded(written, played, "missing")

    total = len(graded_written)
    totals: dict[str, Any] = {"written_notes": total, "played_notes": len(graded_played)}
    totals["extra"] = sum(n["level"] == "extra" for n in graded_played)
    if not total:
        totals.update(exact=None, octave=None, found=None)
        return {"played": graded_played, "written": graded_written, "totals": totals}

    exact = sum(n["level"] == "exact" for n in graded_written)
    octave = sum(n["level"] in ("exact", "octave") for n in graded_written)
    found = sum(n["level"] != "missing" for n in graded_written)
    totals.update(exact=exact / total, octave=octave / total, found=found / total)
    return {"played": graded_played, "written": graded_written, "totals": totals}
