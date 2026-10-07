from __future__ import annotations

from songscribe.align import align, dedupe, group_words, place_in_line
from songscribe.types import ChordEvent, LyricLine, Word

from .conftest import words_from


class TestDedupe:
    def test_merges_consecutive_identical_labels(self):
        merged = dedupe(
            [
                ChordEvent(0.0, 1.0, "C"),
                ChordEvent(1.0, 2.0, "C"),
                ChordEvent(2.0, 3.0, "G"),
            ]
        )
        assert [(c.start, c.end, c.label) for c in merged] == [
            (0.0, 2.0, "C"),
            (2.0, 3.0, "G"),
        ]

    def test_drops_flickers_below_threshold(self):
        merged = dedupe([ChordEvent(0.0, 0.1, "F#m"), ChordEvent(0.1, 4.0, "A")])
        assert [c.label for c in merged] == ["A"]

    def test_sorts_unordered_input(self):
        merged = dedupe([ChordEvent(2.0, 3.0, "G"), ChordEvent(0.0, 1.0, "C")])
        assert [c.label for c in merged] == ["C", "G"]

    def test_empty_input(self):
        assert dedupe([]) == []


class TestGroupWords:
    def test_splits_on_long_pause(self, two_line_words):
        lines = group_words(two_line_words)
        assert [line.text for line in lines] == [
            "lantern over water",
            "counting every window",
        ]

    def test_splits_on_max_words(self):
        words = words_from([(float(i), i + 0.5, f"w{i}") for i in range(5)])
        lines = group_words(words, max_gap=10.0, max_words=2)
        assert [len(line.words) for line in lines] == [2, 2, 1]

    def test_empty_input(self):
        assert group_words([]) == []


class TestPlaceInLine:
    def test_chord_lands_on_the_word_being_sung(self, simple_chords):
        line = LyricLine(
            tuple(words_from([(0.0, 0.4, "lantern"), (0.5, 0.9, "over"), (1.0, 1.6, "water")]))
        )
        placements = place_in_line(line, simple_chords)
        assert [(p.offset, p.label) for p in placements] == [
            (0, "C"),
            (line.text.index("water"), "G"),
        ]

    def test_chord_starting_before_the_line_pins_to_column_zero(self):
        line = LyricLine((Word(5.0, 5.5, "late"),))
        placements = place_in_line(line, [ChordEvent(1.0, 6.0, "Dm")])
        assert [(p.offset, p.label) for p in placements] == [(0, "Dm")]

    def test_ignores_chords_outside_the_line(self):
        line = LyricLine((Word(0.0, 1.0, "here"),))
        assert place_in_line(line, [ChordEvent(20.0, 24.0, "Bb")]) == ()

    def test_two_changes_on_one_syllable_keeps_the_downbeat(self):
        line = LyricLine((Word(0.0, 2.0, "heldnote"),))
        placements = place_in_line(line, [ChordEvent(0.0, 1.0, "E"), ChordEvent(1.0, 2.0, "B")])
        assert [p.label for p in placements] == ["E"]


class TestAlign:
    def test_produces_one_section_per_line(self, simple_chords, two_line_words):
        song = align(simple_chords, two_line_words)
        lyric_sections = [s for s in song.sections if not s.is_instrumental]
        assert len(lyric_sections) == 2

    def test_emits_an_instrumental_intro(self, two_line_words):
        chords = [ChordEvent(0.0, 4.0, "C"), ChordEvent(4.0, 6.0, "G")]
        words = words_from([(5.0, 5.5, "finally"), (5.6, 6.0, "singing")])
        song = align(chords, words)
        assert song.sections[0].is_instrumental
        assert [p.label for p in song.sections[0].chords] == ["C", "G"]

    def test_emits_a_trailing_outro(self, two_line_words):
        chords = [ChordEvent(0.0, 5.0, "C"), ChordEvent(10.0, 14.0, "G")]
        song = align(chords, two_line_words)
        assert song.sections[-1].is_instrumental
        assert [p.label for p in song.sections[-1].chords] == ["G"]

    def test_no_words_gives_only_instrumental_sections(self, simple_chords):
        song = align(simple_chords, [])
        assert song.sections
        assert all(s.is_instrumental for s in song.sections)

    def test_no_chords_still_gives_the_lyrics(self, two_line_words):
        song = align([], two_line_words)
        assert [s.line.text for s in song.sections] == [
            "lantern over water",
            "counting every window",
        ]
        assert all(s.chords == () for s in song.sections)
