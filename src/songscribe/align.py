"""Merge a chord timeline and a lyric timeline into printable sections.

This is the step that turns two independent streams of timed events into the
thing a guitarist actually reads: chord labels sitting above the syllable where
the change happens.
"""

from __future__ import annotations

from itertools import pairwise

from .types import ChordEvent, LyricLine, Placement, Section, Song, Word

# Chords closer together than this are treated as one change; chord trackers
# tend to flicker between enharmonic or relative guesses at boundaries.
MIN_CHORD_DURATION = 0.25

# A gap between lyric lines longer than this is worth printing as an
# instrumental break rather than folding into the next line.
MIN_INSTRUMENTAL_GAP = 2.0


def dedupe(chords: list[ChordEvent]) -> list[ChordEvent]:
    """Collapse runs of the same label and drop sub-threshold flickers."""
    merged: list[ChordEvent] = []
    for chord in sorted(chords, key=lambda c: c.start):
        if merged and merged[-1].label == chord.label:
            previous = merged.pop()
            merged.append(ChordEvent(previous.start, chord.end, chord.label))
        else:
            merged.append(chord)
    return [c for c in merged if c.duration >= MIN_CHORD_DURATION]


def group_words(words: list[Word], max_gap: float = 1.0, max_words: int = 10) -> list[LyricLine]:
    """Break a flat word stream into lines on breath-length pauses.

    ASR models give us words, not lines. A pause longer than ``max_gap``
    usually means the singer finished a phrase; ``max_words`` keeps a
    breathless run from becoming one unreadable line.
    """
    if not words:
        return []

    lines: list[LyricLine] = []
    current: list[Word] = [words[0]]
    for previous, word in pairwise(words):
        too_slow = word.start - previous.end > max_gap
        too_long = len(current) >= max_words
        if too_slow or too_long:
            lines.append(LyricLine(tuple(current)))
            current = []
        current.append(word)
    lines.append(LyricLine(tuple(current)))
    return lines


def place_in_line(line: LyricLine, chords: list[ChordEvent]) -> tuple[Placement, ...]:
    """Pin each chord that sounds during ``line`` to a character offset.

    A chord lands on the word being sung when it changes. A chord that starts
    before the line's first word belongs at the start of the line.
    """
    offsets = line.word_offsets()
    placements: list[Placement] = []
    for chord in chords:
        if not chord.overlaps(line.start, line.end):
            continue
        offset = 0
        for word, word_offset in zip(line.words, offsets, strict=True):
            # The chord change belongs to the last word that has already
            # started by the time the chord sounds.
            if word.start <= chord.start:
                offset = word_offset
            else:
                break
        if placements and placements[-1].offset == offset:
            # Two changes on one syllable: keep the first, it is the downbeat.
            continue
        placements.append(Placement(offset, chord.label))
    return tuple(placements)


def align(chords: list[ChordEvent], words: list[Word], **kwargs) -> Song:
    """Build a :class:`Song` from a chord timeline and a word timeline."""
    chords = dedupe(chords)
    lines = group_words(words, **kwargs)
    sections: list[Section] = []

    cursor = 0.0
    for line in lines:
        gap = [c for c in chords if c.overlaps(cursor, line.start)]
        if gap and line.start - cursor >= MIN_INSTRUMENTAL_GAP:
            sections.append(Section(chords=tuple(Placement(0, c.label) for c in gap), line=None))
        sections.append(Section(chords=place_in_line(line, chords), line=line))
        cursor = line.end

    trailing = [c for c in chords if c.start >= cursor]
    if trailing:
        sections.append(Section(chords=tuple(Placement(0, c.label) for c in trailing), line=None))

    return Song(sections=sections)
