"""Core data types shared by every stage of the pipeline.

Everything is time-based and immutable: each stage takes audio or a previous
stage's events and returns new events on the same seconds-from-start timeline,
so stages can be swapped, cached, or run out of order.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ChordEvent:
    """One chord held over a span of time, e.g. ``Am`` from 4.2s to 6.0s."""

    start: float
    end: float
    label: str

    @property
    def duration(self) -> float:
        return self.end - self.start

    def overlaps(self, start: float, end: float) -> bool:
        return self.start < end and start < self.end


@dataclass(frozen=True, slots=True)
class Word:
    """A single transcribed word with its own timing."""

    start: float
    end: float
    text: str


@dataclass(frozen=True, slots=True)
class LyricLine:
    """A run of words that belong on one printed line."""

    words: tuple[Word, ...]

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    def word_offsets(self) -> list[int]:
        """Character offset of each word within :attr:`text`."""
        offsets = []
        cursor = 0
        for word in self.words:
            offsets.append(cursor)
            cursor += len(word.text) + 1
        return offsets


@dataclass(frozen=True, slots=True)
class Placement:
    """A chord pinned to a character offset inside a lyric line."""

    offset: int
    label: str


@dataclass(frozen=True, slots=True)
class Section:
    """A rendered block: either a lyric line with chords over it, or an
    instrumental run of chords with no words under them."""

    chords: tuple[Placement, ...]
    line: LyricLine | None = None

    @property
    def is_instrumental(self) -> bool:
        return self.line is None


@dataclass(slots=True)
class Song:
    """The finished transcription."""

    sections: list[Section] = field(default_factory=list)
    title: str | None = None
    key: str | None = None
    tempo: float | None = None
