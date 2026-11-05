from __future__ import annotations

import json
from pathlib import Path

import pytest

from songscribe.eval import split

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
CORPORA = ("jaah", "isophonics-beatles", "backing-track")
IDS = [f"labs/song_{n}" for n in range(200)]


def load(name: str) -> dict:
    return json.loads((EVAL_DIR / f"{name}.json").read_text(encoding="utf-8"))


def manifest(*ids: str) -> dict:
    return {"dataset": "demo", "tracks": [{"id": i, "reference": f"{i}.lab"} for i in ids]}


class TestSongKey:
    def test_performer_in_parentheses_is_dropped(self):
        assert split.song_key("labs/maple_leaf_rag(bechet)") == "maple_leaf_rag"
        assert split.song_key("labs/maple_leaf_rag(hyman)") == "maple_leaf_rag"

    def test_listed_versions_share_a_key(self):
        keys = {split.song_key(f"labs/{s}") for s in ("dinah", "dinah_fats_waller")}
        assert keys == {"dinah"}
        assert split.song_key("labs/new_east_st_louis") == split.song_key("labs/east_st_louis")

    def test_album_track_numbers_are_dropped(self):
        assert split.song_key("a/b/05_-_Dig_It") == "dig_it"
        assert split.song_key("a/b/CD1_-_03_-_Glass_Onion") == "glass_onion"

    def test_a_reprise_is_the_same_song(self):
        title = "a/b/01_-_Some_Song"
        reprise = "a/b/12_-_Some_Song_(Reprise)"
        assert split.song_key(title) == split.song_key(reprise)

    def test_different_songs_stay_apart(self):
        assert split.song_key("a/Revolution_1") != split.song_key("a/Revolution_9")


class TestSplitManifest:
    def test_every_track_lands_in_exactly_one_set_in_order(self):
        train, test = split.split_manifest(manifest(*IDS))
        assert sorted(t["id"] for t in train["tracks"] + test["tracks"]) == sorted(IDS)
        for part in (train, test):
            ids = [t["id"] for t in part["tracks"]]
            assert ids == [i for i in IDS if i in ids]

    def test_sets_are_named_after_the_dataset(self):
        train, test = split.split_manifest(manifest("labs/a"))
        assert (train["dataset"], test["dataset"]) == ("demo-train", "demo-test")

    def test_test_share_is_near_the_fraction(self):
        _, test = split.split_manifest(manifest(*IDS), 0.2)
        assert 0.12 < len(test["tracks"]) / len(IDS) < 0.28

    def test_versions_of_a_song_stay_together(self):
        ids = [f"labs/take_{n}(player{v})" for n in range(60) for v in range(3)]
        train, test = split.split_manifest(manifest(*ids))
        in_train = {split.song_key(t["id"]) for t in train["tracks"]}
        in_test = {split.song_key(t["id"]) for t in test["tracks"]}
        assert test["tracks"]
        assert not in_train & in_test

    def test_a_track_keeps_its_set_when_the_manifest_shrinks(self):
        _, full = split.split_manifest(manifest(*IDS))
        _, half = split.split_manifest(manifest(*IDS[::2]))
        expected = [t for t in full["tracks"] if t["id"] in IDS[::2]]
        assert half["tracks"] == expected

    def test_other_track_fields_are_kept(self):
        document = manifest("labs/a")
        document["tracks"][0].update(audio="a.mp3", bars=12)
        train, test = split.split_manifest(document)
        assert (train["tracks"] + test["tracks"])[0]["bars"] == 12

    @pytest.mark.parametrize("fraction", [0, 1, -0.1, 1.5])
    def test_fraction_outside_zero_and_one_is_rejected(self, fraction):
        with pytest.raises(ValueError, match="test_fraction"):
            split.split_manifest(manifest("labs/a"), fraction)


@pytest.mark.parametrize("name", CORPORA)
class TestCommittedSplits:
    def test_train_and_test_partition_the_manifest(self, name):
        full = load(name)
        train, test = load(f"{name}-train"), load(f"{name}-test")
        ids = [t["id"] for t in train["tracks"] + test["tracks"]]
        assert sorted(ids) == sorted(t["id"] for t in full["tracks"])
        assert len(ids) == len(set(ids))

    def test_no_song_is_in_both_sets(self, name):
        train, test = load(f"{name}-train"), load(f"{name}-test")
        in_train = {split.song_key(t["id"]) for t in train["tracks"]}
        in_test = {split.song_key(t["id"]) for t in test["tracks"]}
        assert test["tracks"]
        assert not in_train & in_test

    def test_files_match_what_the_code_produces(self, name):
        train, test = split.split_manifest(load(name))
        assert (train, test) == (load(f"{name}-train"), load(f"{name}-test"))
