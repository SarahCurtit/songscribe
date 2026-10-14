from __future__ import annotations

import json

import pytest

from songscribe.eval import reference
from songscribe.types import Word

# Invented placeholder words: a real transcript may not enter the repo, and
# this format is exercised by structure, not content.
WORDS = [
    Word(0.0, 0.4, "lantern"),
    Word(0.5, 0.9, "over"),
    Word(1.0, 1.6, "water"),
]


class TestRoundTrip:
    def test_it_round_trips(self, tmp_path):
        path = reference.write_words(tmp_path / f"song{reference.SUFFIX}", WORDS)
        assert reference.read_words(path) == WORDS

    def test_it_creates_missing_parents(self, tmp_path):
        path = reference.write_words(tmp_path / "a" / "b" / f"s{reference.SUFFIX}", WORDS)
        assert path.exists()

    def test_the_source_is_recorded(self, tmp_path):
        path = reference.write_words(
            tmp_path / f"s{reference.SUFFIX}", WORDS, source="take3.wav via whisper-base"
        )
        document = json.loads(path.read_text(encoding="utf-8"))
        assert document["source"] == "take3.wav via whisper-base"

    def test_an_empty_timeline_round_trips(self, tmp_path):
        path = reference.write_words(tmp_path / f"s{reference.SUFFIX}", [])
        assert reference.read_words(path) == []

    def test_unicode_text_survives(self, tmp_path):
        words = [Word(0.0, 0.5, "café"), Word(0.6, 1.0, "naïve")]
        path = reference.write_words(tmp_path / f"s{reference.SUFFIX}", words)
        assert reference.read_words(path) == words


class TestTheSuffixIsEnforced:
    """The suffix is the gitignore rule, so writing another name is refused.

    A transcript saved as ``song.json`` would not be ignored and could be
    committed by accident, which hard constraint 5 forbids.
    """

    @pytest.mark.parametrize("name", ["song.json", "song.lab", "words", "song.words.txt"])
    def test_a_non_matching_name_is_refused(self, tmp_path, name):
        with pytest.raises(reference.ReferenceError, match="must end in"):
            reference.write_words(tmp_path / name, WORDS)

    def test_nothing_is_written_when_refused(self, tmp_path):
        with pytest.raises(reference.ReferenceError):
            reference.write_words(tmp_path / "song.json", WORDS)
        assert list(tmp_path.iterdir()) == []


class TestReadValidation:
    def test_words_are_sorted_on_read(self, tmp_path):
        # Hand-edited references are easy to leave out of order, and
        # place_in_line walks words in order.
        path = tmp_path / f"s{reference.SUFFIX}"
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "words": [
                        {"start": 2.0, "end": 2.5, "text": "second"},
                        {"start": 0.0, "end": 0.5, "text": "first"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        assert [w.text for w in reference.read_words(path)] == ["first", "second"]

    def test_malformed_json_raises(self, tmp_path):
        path = tmp_path / f"s{reference.SUFFIX}"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(reference.ReferenceError, match="not valid JSON"):
            reference.read_words(path)

    def test_a_top_level_list_raises(self, tmp_path):
        path = tmp_path / f"s{reference.SUFFIX}"
        path.write_text("[]", encoding="utf-8")
        with pytest.raises(reference.ReferenceError, match="JSON object"):
            reference.read_words(path)

    def test_a_future_version_raises(self, tmp_path):
        path = tmp_path / f"s{reference.SUFFIX}"
        path.write_text(json.dumps({"version": 99, "words": []}), encoding="utf-8")
        with pytest.raises(reference.ReferenceError, match="format version 99"):
            reference.read_words(path)

    def test_words_must_be_a_list(self, tmp_path):
        path = tmp_path / f"s{reference.SUFFIX}"
        path.write_text(json.dumps({"version": 1, "words": {}}), encoding="utf-8")
        with pytest.raises(reference.ReferenceError, match="must be a list"):
            reference.read_words(path)

    @pytest.mark.parametrize(
        "entry",
        [
            {"start": 0.0, "end": 1.0},
            {"start": 0.0, "text": "a"},
            {"end": 1.0, "text": "a"},
            {"start": "x", "end": 1.0, "text": "a"},
        ],
    )
    def test_an_incomplete_word_raises(self, tmp_path, entry):
        path = tmp_path / f"s{reference.SUFFIX}"
        path.write_text(json.dumps({"version": 1, "words": [entry]}), encoding="utf-8")
        with pytest.raises(reference.ReferenceError, match="needs start, end and text"):
            reference.read_words(path)


class TestItFeedsThePlacementMetric:
    def test_a_written_reference_scores_placement(self, tmp_path):
        from songscribe.eval import metrics
        from songscribe.types import ChordEvent

        path = reference.write_words(tmp_path / f"s{reference.SUFFIX}", WORDS)
        words = reference.read_words(path)
        chords = [ChordEvent(0.0, 1.0, "C:maj"), ChordEvent(1.0, 2.0, "G:maj")]
        assert metrics.placement_score(chords, chords, words) == 1.0
