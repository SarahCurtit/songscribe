"""Turn a :class:`~songscribe.types.Song` into something readable.

Two formats, because they serve different purposes:

``chordpro``
    The interchange format. ChordPro files open in SongBook, OnSong,
    Chordii, and most band/setlist apps.

``text``
    Two-row plain text, chords on the line above the lyrics. This is what
    people paste into a forum post or print and tape to a music stand.
"""

from __future__ import annotations

from .types import Section, Song

RENDERERS = ("text", "chordpro")


def _chordpro_section(section: Section) -> str:
    if section.is_instrumental:
        return " ".join(f"[{p.label}]" for p in section.chords)

    text = section.line.text
    out = []
    cursor = 0
    for placement in section.chords:
        out.append(text[cursor : placement.offset])
        out.append(f"[{placement.label}]")
        cursor = placement.offset
    out.append(text[cursor:])
    return "".join(out)


def _text_section(section: Section) -> str:
    if section.is_instrumental:
        return " ".join(p.label for p in section.chords)

    lyrics = section.line.text
    chord_row = ""
    for placement in section.chords:
        # Never let a long chord name overwrite the one before it; nudge it
        # right instead, keeping at least one space between labels.
        column = max(placement.offset, len(chord_row) + 1 if chord_row else 0)
        chord_row = chord_row.ljust(column) + placement.label
    return f"{chord_row}\n{lyrics}" if chord_row else lyrics


def render(song: Song, fmt: str = "text") -> str:
    """Render ``song`` as ``text`` or ``chordpro``."""
    if fmt not in RENDERERS:
        raise ValueError(f"unknown format {fmt!r}, expected one of {RENDERERS}")

    header: list[str] = []
    body: list[str] = []

    if fmt == "chordpro":
        if song.title:
            header.append(f"{{title: {song.title}}}")
        if song.key:
            header.append(f"{{key: {song.key}}}")
        if song.tempo:
            header.append(f"{{tempo: {song.tempo:.0f}}}")
        body = [_chordpro_section(s) for s in song.sections]
    else:
        if song.title:
            header.append(song.title)
            header.append("=" * len(song.title))
        meta = []
        if song.key:
            meta.append(f"key: {song.key}")
        if song.tempo:
            meta.append(f"tempo: {song.tempo:.0f} bpm")
        if meta:
            header.append("  ".join(meta))
        body = [_text_section(s) for s in song.sections]

    chunks = ["\n".join(header)] if header else []
    chunks.extend(body)
    return "\n\n".join(chunks).rstrip() + "\n"
