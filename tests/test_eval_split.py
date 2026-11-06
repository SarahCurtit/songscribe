from __future__ import annotations

import json
from pathlib import Path

import pytest

from songscribe.eval import split

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
CORPORA = ("jaah", "isophonics-beatles", "backing-track", "isophonics-zweieck")
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

    def test_a_space_separated_track_number_is_dropped_the_same_way(self):
        assert split.song_key("a/Clara Placeholder/05 Open Window") == "open window"
        assert split.song_key("a/b/05 Dig It") == "dig it"

    def test_a_number_belonging_to_the_title_is_kept(self):
        assert split.song_key("labs/26_2") == "26_2"
        assert split.song_key("labs/500_miles_high") == "500_miles_high"

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


class TestCaroleKingIsHeldOutWhole:
    """Carole King is scored, never tuned on, so it has a test set and no train one.

    Seven songs cannot carry a hash split: the lowest of them hashes at 0.39, so no
    sensible fraction holds one out, and `split` would write an empty test set. The
    annotations are also the set Isophonics says it has not checked carefully, which
    is a reason to keep them off the tuning side rather than to average them in.
    """

    CORPUS = "isophonics-carole-king"

    def test_the_test_manifest_holds_every_track_in_order(self):
        full, test = load(self.CORPUS), load(f"{self.CORPUS}-test")
        assert [t["id"] for t in test["tracks"]] == [t["id"] for t in full["tracks"]]

    def test_it_is_named_as_a_test_set(self):
        assert load(f"{self.CORPUS}-test")["dataset"] == f"{self.CORPUS}-test"

    def test_there_is_deliberately_no_train_manifest(self):
        assert not (EVAL_DIR / f"{self.CORPUS}-train.json").exists()

    def test_splitting_it_by_hash_would_hold_out_nothing(self):
        _, test = split.split_manifest(load(self.CORPUS))
        assert test["tracks"] == []

    def test_no_song_of_it_falls_near_the_test_fraction(self):
        keys = {split.song_key(t["id"]) for t in load(self.CORPUS)["tracks"]}
        assert min(split.position(k) for k in keys) > 0.35


class TestPosition:
    def test_it_decides_what_is_test(self):
        for key in ("open window", "26_2", "beautiful", "dig_it"):
            assert split.is_test(key, 0.2) == (split.position(key) < 0.2)

    def test_it_lands_in_the_unit_interval(self):
        assert all(0.0 <= split.position(f"song_{n}") < 1.0 for n in range(50))


class TestEmptySideWarning:
    """Both manifests are written even when one is empty, so `split` has to say so."""

    @staticmethod
    def _ids(predicate, count: int = 5) -> list[str]:
        return [i for i in IDS if predicate(split.position(split.song_key(i)))][:count]

    def _run(self, tmp_path: Path, ids: list[str]) -> None:
        from songscribe.eval import cli

        path = tmp_path / "corpus.json"
        path.write_text(json.dumps(manifest(*ids)), encoding="utf-8")
        assert cli.main(["split", str(path)]) == 0

    def test_an_empty_test_set_names_the_fraction_that_would_have_worked(self, tmp_path, capsys):
        ids = self._ids(lambda p: p >= split.TEST_FRACTION)
        self._run(tmp_path, ids)
        error = capsys.readouterr().err
        assert "test is empty" in error
        assert f"{len(ids)} songs is too few" in error
        assert f"hashes at {min(split.position(split.song_key(i)) for i in ids):.2f}" in error

    def test_an_empty_train_set_is_called_out_too(self, tmp_path, capsys):
        self._run(tmp_path, self._ids(lambda p: p < split.TEST_FRACTION))
        assert "train is empty" in capsys.readouterr().err

    def test_a_split_with_both_sides_filled_stays_quiet(self, tmp_path, capsys):
        self._run(tmp_path, IDS)
        assert capsys.readouterr().err == ""

    def test_takes_of_one_song_count_as_one_chance(self, tmp_path, capsys):
        song = self._ids(lambda p: p >= split.TEST_FRACTION, count=1)[0]
        self._run(tmp_path, [f"{song}(take{n})" for n in range(4)])
        assert "1 song is too few" in capsys.readouterr().err
