from __future__ import annotations

import json
from pathlib import Path

import pytest

from songscribe.eval import dataset, harte

MANIFEST = Path(__file__).resolve().parent.parent / "eval" / "jaah.json"


class TestCommittedManifest:
    @pytest.fixture
    def loaded(self):
        return dataset.load_manifest(MANIFEST)

    def test_lists_the_113_tracks_of_the_corpus(self, loaded):
        assert loaded.name == "jaah"
        assert len(loaded.tracks) == 113

    def test_holds_references_only_and_no_absolute_paths(self):
        # Read raw: loading resolves each path against the annotation root.
        entries = json.loads(MANIFEST.read_text(encoding="utf-8"))["tracks"]
        for entry in entries:
            assert "audio" not in entry
            assert entry["reference"].startswith("labs/")
            assert not Path(entry["reference"]).is_absolute()


class TestLabelsAsJaahWritesThem:
    # Spelled the way the corpus spells them: a bare triad is "Bb", and an altered
    # dominant is a degree list.
    @pytest.mark.parametrize(
        ("label", "pitch_classes"),
        [
            ("Bb", {10, 2, 5}),
            ("F:7", {5, 9, 0, 3}),
            ("C:min7", {0, 3, 7, 10}),
            ("Bb:maj6", {10, 2, 5, 7}),
            ("F:min6", {5, 8, 0, 2}),
            ("E:dim7", {4, 7, 10, 1}),
            ("G:(3,5,b7,b9)", {7, 11, 2, 5, 8}),
        ],
    )
    def test_parse_to_the_notes_they_name(self, label, pitch_classes):
        chord = harte.parse(label)
        assert chord is not None
        assert set(chord.pitch_classes) == pitch_classes

    def test_no_chord_has_no_root(self):
        chord = harte.parse("N")
        assert chord is not None
        assert chord.root is None
