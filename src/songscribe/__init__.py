"""songscribe -- transcribe chords and lyrics from an audio file."""

from .types import ChordEvent, LyricLine, Placement, Section, Song, Word

__version__ = "0.1.0"

__all__ = [
    "ChordEvent",
    "LyricLine",
    "Placement",
    "Section",
    "Song",
    "Word",
    "__version__",
]
