from __future__ import annotations

import pytest

from songscribe.chords import HARTE_QUALITIES, normalise_label
from songscribe.eval.harte import (
    Chord,
    parse,
    parse_degree,
    parse_root,
)


class TestParseRoot:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("C", 0),
            ("D", 2),
            ("E", 4),
            ("F", 5),
            ("G", 7),
            ("A", 9),
            ("B", 11),
        ],
    )
    def test_naturals(self, text, expected):
        assert parse_root(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("C#", 1),
            ("Db", 1),
            ("Bb", 10),
            ("A#", 10),
            ("Ebb", 2),
            ("C##", 2),
            ("Cb", 11),
            ("B#", 0),
        ],
    )
    def test_accidentals_including_enharmonics_and_wraparound(self, text, expected):
        assert parse_root(text) == expected

    @pytest.mark.parametrize("text", ["H", "", "c", "C$", "Cx"])
    def test_rejects_a_non_note(self, text):
        assert parse_root(text) is None


class TestParseDegree:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("1", (0, False)),
            ("3", (4, False)),
            ("b3", (3, False)),
            ("5", (7, False)),
            ("b5", (6, False)),
            ("#5", (8, False)),
            ("7", (11, False)),
            ("b7", (10, False)),
            ("bb7", (9, False)),
            ("*3", (4, True)),
            ("*b3", (3, True)),
        ],
    )
    def test_degrees_and_omissions(self, text, expected):
        assert parse_degree(text) == expected

    def test_compound_degrees_fold_into_one_octave(self):
        # A 9th is a 2nd, an 11th a 4th, a 13th a 6th: comparison is modulo
        # the octave, since no vocabulary here distinguishes voicings.
        assert parse_degree("9") == (2, False)
        assert parse_degree("11") == (5, False)
        assert parse_degree("13") == (9, False)

    @pytest.mark.parametrize("text", ["", "b", "x3", "3b", "99"])
    def test_rejects_a_malformed_degree(self, text):
        assert parse_degree(text) is None


class TestParseLabel:
    def test_bare_root_is_a_major_triad(self):
        chord = parse("C")
        assert chord == Chord(root=0, intervals=frozenset({0, 4, 7}))

    def test_bare_root_and_explicit_maj_agree(self):
        assert parse("C") == parse("C:maj")

    @pytest.mark.parametrize(
        ("label", "intervals"),
        [
            ("C:maj", {0, 4, 7}),
            ("C:min", {0, 3, 7}),
            ("C:dim", {0, 3, 6}),
            ("C:aug", {0, 4, 8}),
            ("C:maj7", {0, 4, 7, 11}),
            ("C:min7", {0, 3, 7, 10}),
            ("C:7", {0, 4, 7, 10}),
            ("C:dim7", {0, 3, 6, 9}),
            ("C:hdim7", {0, 3, 6, 10}),
            ("C:minmaj7", {0, 3, 7, 11}),
            ("C:maj6", {0, 4, 7, 9}),
            ("C:min6", {0, 3, 7, 9}),
            ("C:sus2", {0, 2, 7}),
            ("C:sus4", {0, 5, 7}),
            ("C:9", {0, 4, 7, 10, 2}),
            ("C:13", {0, 4, 7, 10, 2, 5, 9}),
        ],
    )
    def test_harte_shorthands(self, label, intervals):
        assert parse(label).intervals == frozenset(intervals)

    @pytest.mark.parametrize(
        ("label", "intervals"),
        [
            ("Am", {0, 3, 7}),
            ("Am7", {0, 3, 7, 10}),
            ("C7", {0, 4, 7, 10}),
            ("Cmaj7", {0, 4, 7, 11}),
            ("Cm7b5", {0, 3, 6, 10}),
            ("Cmmaj7", {0, 3, 7, 11}),
            ("C6", {0, 4, 7, 9}),
            ("Csus4", {0, 5, 7}),
            ("Cdim", {0, 3, 6}),
        ],
    )
    def test_compact_shorthand_as_backends_emit_it(self, label, intervals):
        assert parse(label).intervals == frozenset(intervals)

    @pytest.mark.parametrize("label", ["Cmin", "Cmin7", "Chdim7", "Cmaj", "Csus2"])
    def test_harte_quality_names_also_work_without_a_colon(self, label):
        # Not a notation anything here emits, but accepting it costs nothing
        # and saves a confusing failure on a hand-written estimate file.
        assert parse(label) == parse(f"C:{label[1:]}")

    def test_added_degree(self):
        assert parse("C:maj(9)").intervals == frozenset({0, 4, 7, 2})

    def test_omitted_degree(self):
        assert parse("C:maj(*5)").intervals == frozenset({0, 4})

    def test_several_degrees(self):
        assert parse("C:min(b7,9)").intervals == frozenset({0, 3, 7, 10, 2})

    def test_explicit_degree_list_with_no_shorthand(self):
        assert parse("C:(1,b3,5)") == parse("C:min")

    def test_root_is_implicit_in_an_explicit_degree_list(self):
        assert 0 in parse("C:(3,5)").intervals

    def test_bass_note_is_recorded_but_does_not_change_the_pitches(self):
        chord = parse("G:7/3")
        assert chord.root == 7
        assert chord.intervals == frozenset({0, 4, 7, 10})
        assert chord.bass == 4

    def test_inversion_of_a_bare_root(self):
        assert parse("C/5").bass == 7

    def test_no_chord(self):
        chord = parse("N")
        assert chord.is_no_chord
        assert not chord.unknown
        assert chord.root is None
        assert chord.pitch_classes == frozenset()

    def test_unknown_chord_is_distinct_from_no_chord(self):
        chord = parse("X")
        assert chord.unknown
        assert not chord.is_no_chord

    @pytest.mark.parametrize(
        "label",
        [
            "",
            "   ",
            "H:maj",
            "C:nonsense",
            "C:",
            "C:maj(bogus)",
            "C:maj(*)",
            "C:maj/x",
            "???",
            "Cwhatever",
        ],
    )
    def test_unparseable_labels_return_none_rather_than_a_guess(self, label):
        assert parse(label) is None

    def test_surrounding_whitespace_is_ignored(self):
        assert parse("  A:min  ") == parse("A:min")


class TestPitchClasses:
    def test_absolute_pitch_classes_of_a_triad(self):
        assert parse("A:min").pitch_classes == frozenset({9, 0, 4})

    def test_enharmonic_spellings_give_the_same_pitch_classes(self):
        assert parse("Db:maj").pitch_classes == parse("C#:maj").pitch_classes

    def test_root_name_is_spelled_with_sharps(self):
        assert parse("Db:maj").root_name == "C#"
        assert parse("N").root_name == "N"


class TestAgreementWithTheProjectVocabulary:
    """The eval parser and ``chords.normalise_label`` must not drift apart.

    ``normalise_label`` turns Harte into this project's compact shorthand on
    the way out of a backend; this parser has to read both and agree. Driving
    the test off ``HARTE_QUALITIES`` means a quality added there is covered
    here automatically.
    """

    @pytest.mark.parametrize("quality", sorted(HARTE_QUALITIES))
    def test_harte_and_its_normalised_form_parse_identically(self, quality):
        harte = f"C:{quality}"
        compact = normalise_label(harte)
        assert parse(harte) == parse(compact), f"{harte} != {compact}"

    @pytest.mark.parametrize("quality", sorted(HARTE_QUALITIES))
    def test_every_quality_parses_at_all(self, quality):
        assert parse(f"C:{quality}") is not None

    def test_an_inversion_normalises_away_and_still_parses(self):
        # normalise_label drops the bass note, so the compact form has none --
        # the pitches must still agree.
        assert parse("C:maj/3").intervals == parse(normalise_label("C:maj/3")).intervals
