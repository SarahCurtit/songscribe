from __future__ import annotations

import pytest

from songscribe.align import align
from songscribe.render import render
from songscribe.types import LyricLine, Placement, Section, Song, Word


def song_with(*sections: Section, **meta) -> Song:
    return Song(sections=list(sections), **meta)


def line(text: str) -> LyricLine:
    words, cursor = [], 0.0
    for token in text.split():
        words.append(Word(cursor, cursor + 0.4, token))
        cursor += 0.5
    return LyricLine(tuple(words))


class TestChordPro:
    def test_inlines_chords_at_their_offsets(self):
        lyric = line("lantern over water")
        section = Section(
            chords=(Placement(0, "C"), Placement(lyric.text.index("water"), "G")),
            line=lyric,
        )
        assert render(song_with(section), "chordpro") == "[C]lantern over [G]water\n"

    def test_instrumental_section_is_bare_chords(self):
        section = Section(chords=(Placement(0, "Am"), Placement(0, "F")), line=None)
        assert render(song_with(section), "chordpro") == "[Am] [F]\n"

    def test_writes_metadata_directives(self):
        song = song_with(Section(chords=(), line=line("hello")), title="Demo", key="C", tempo=92.4)
        assert render(song, "chordpro").splitlines()[:3] == [
            "{title: Demo}",
            "{key: C}",
            "{tempo: 92}",
        ]


class TestText:
    def test_chord_row_sits_above_the_lyric(self):
        lyric = line("lantern over water")
        offset = lyric.text.index("water")
        section = Section(chords=(Placement(0, "C"), Placement(offset, "G")), line=lyric)
        chord_row, lyric_row = render(song_with(section), "text").splitlines()
        assert lyric_row == "lantern over water"
        # Each chord label starts in the same column as the word it belongs to.
        assert chord_row.index("C") == 0
        assert chord_row.index("G") == offset

    def test_long_chord_names_do_not_overwrite_each_other(self):
        lyric = line("ab cd")
        section = Section(chords=(Placement(0, "Cmaj7"), Placement(3, "G7sus4")), line=lyric)
        chord_row = render(song_with(section), "text").splitlines()[0]
        # Pushed right to column 6 rather than clobbering the tail of Cmaj7.
        assert chord_row == "Cmaj7 G7sus4"

    def test_lyric_line_without_chords_renders_alone(self):
        out = render(song_with(Section(chords=(), line=line("just words"))), "text")
        assert out == "just words\n"

    def test_underlines_the_title(self):
        song = song_with(Section(chords=(), line=line("x")), title="Demo")
        assert render(song, "text").splitlines()[:2] == ["Demo", "===="]


class TestRenderContract:
    def test_rejects_unknown_format(self):
        with pytest.raises(ValueError, match="unknown format"):
            render(Song(), "latex")

    @pytest.mark.parametrize("fmt", ["text", "chordpro"])
    def test_always_ends_with_exactly_one_newline(self, fmt, simple_chords, two_line_words):
        out = render(align(simple_chords, two_line_words), fmt)
        assert out.endswith("\n")
        assert not out.endswith("\n\n")

    @pytest.mark.parametrize("fmt", ["text", "chordpro"])
    def test_empty_song_does_not_crash(self, fmt):
        assert render(Song(), fmt) == "\n"

    @pytest.mark.parametrize("fmt", ["text", "chordpro"])
    def test_every_lyric_survives_rendering(self, fmt, simple_chords, two_line_words):
        out = render(align(simple_chords, two_line_words), fmt)
        for word in two_line_words:
            assert word.text in out
