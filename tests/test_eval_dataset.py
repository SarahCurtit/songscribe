from __future__ import annotations

import json

import pytest

from songscribe.eval import dataset, isophonics


def write_manifest(path, tracks, **overrides):
    document = dataset.build(name="test", entries=tracks)
    document.update(overrides)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


class TestResolveRoot:
    def test_explicit_argument_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv(dataset.AUDIO_ROOT_ENV, str(tmp_path / "from-env"))
        resolved = dataset.resolve_root(tmp_path / "explicit", dataset.AUDIO_ROOT_ENV, tmp_path)
        assert resolved == tmp_path / "explicit"

    def test_environment_is_used_when_no_argument(self, tmp_path, monkeypatch):
        monkeypatch.setenv(dataset.AUDIO_ROOT_ENV, str(tmp_path / "from-env"))
        resolved = dataset.resolve_root(None, dataset.AUDIO_ROOT_ENV, tmp_path)
        assert resolved == tmp_path / "from-env"

    def test_fallback_is_used_when_neither_is_set(self, tmp_path, monkeypatch):
        monkeypatch.delenv(dataset.AUDIO_ROOT_ENV, raising=False)
        assert dataset.resolve_root(None, dataset.AUDIO_ROOT_ENV, tmp_path) == tmp_path

    def test_an_empty_environment_variable_is_ignored(self, tmp_path, monkeypatch):
        monkeypatch.setenv(dataset.AUDIO_ROOT_ENV, "")
        assert dataset.resolve_root(None, dataset.AUDIO_ROOT_ENV, tmp_path) == tmp_path

    def test_a_tilde_is_expanded(self, tmp_path):
        resolved = dataset.resolve_root("~/music", dataset.AUDIO_ROOT_ENV, tmp_path)
        assert "~" not in str(resolved)


class TestLoadManifest:
    def test_paths_resolve_against_the_given_roots(self, tmp_path):
        manifest = write_manifest(
            tmp_path / "m.json",
            [{"id": "song", "reference": "song.lab", "audio": "song.flac"}],
        )
        loaded = dataset.load_manifest(
            manifest, audio_root=tmp_path / "audio", annotation_root=tmp_path / "ann"
        )
        (track,) = loaded.tracks
        assert track.reference == tmp_path / "ann" / "song.lab"
        assert track.audio == tmp_path / "audio" / "song.flac"

    def test_roots_fall_back_to_the_manifest_directory(self, tmp_path, monkeypatch):
        monkeypatch.delenv(dataset.AUDIO_ROOT_ENV, raising=False)
        monkeypatch.delenv(dataset.ANNOTATION_ROOT_ENV, raising=False)
        manifest = write_manifest(tmp_path / "m.json", [{"id": "song", "reference": "song.lab"}])
        (track,) = dataset.load_manifest(manifest).tracks
        assert track.reference == tmp_path / "song.lab"

    def test_the_environment_supplies_the_roots(self, tmp_path, monkeypatch):
        monkeypatch.setenv(dataset.AUDIO_ROOT_ENV, str(tmp_path / "library"))
        manifest = write_manifest(
            tmp_path / "m.json",
            [{"id": "song", "reference": "song.lab", "audio": "song.flac"}],
        )
        (track,) = dataset.load_manifest(manifest).tracks
        assert track.audio == tmp_path / "library" / "song.flac"

    def test_a_track_with_no_audio_entry_has_none(self, tmp_path):
        manifest = write_manifest(tmp_path / "m.json", [{"id": "s", "reference": "s.lab"}])
        (track,) = dataset.load_manifest(manifest).tracks
        assert track.audio is None
        assert not track.is_runnable

    def test_an_absolute_path_overrides_its_root(self, tmp_path):
        absolute = tmp_path / "elsewhere" / "song.lab"
        manifest = write_manifest(tmp_path / "m.json", [{"id": "song", "reference": str(absolute)}])
        (track,) = dataset.load_manifest(manifest, annotation_root=tmp_path / "ann").tracks
        assert track.reference == absolute

    def test_missing_files_are_reported_not_raised(self, tmp_path):
        # A manifest covering 180 tracks is still useful with 2 on disk.
        manifest = write_manifest(
            tmp_path / "m.json",
            [
                {"id": "here", "reference": "here.lab", "audio": "here.flac"},
                {"id": "gone", "reference": "gone.lab", "audio": "gone.flac"},
            ],
        )
        (tmp_path / "here.lab").write_text("0.0 1.0 C\n", encoding="utf-8")
        (tmp_path / "here.flac").write_bytes(b"not really audio")

        loaded = dataset.load_manifest(manifest, audio_root=tmp_path, annotation_root=tmp_path)
        assert [t.id for t in loaded.runnable()] == ["here"]
        assert [t.id for t in loaded.missing_audio()] == ["gone"]
        assert [t.id for t in loaded.missing_references()] == ["gone"]

    def test_the_dataset_name_and_notes_survive(self, tmp_path):
        manifest = write_manifest(
            tmp_path / "m.json",
            [{"id": "s", "reference": "s.lab"}],
            dataset="isophonics-beatles",
            notes="annotations only",
        )
        loaded = dataset.load_manifest(manifest)
        assert loaded.name == "isophonics-beatles"
        assert loaded.notes == "annotations only"

    def test_duplicate_track_ids_raise(self, tmp_path):
        manifest = write_manifest(
            tmp_path / "m.json",
            [{"id": "s", "reference": "a.lab"}, {"id": "s", "reference": "b.lab"}],
        )
        with pytest.raises(dataset.ManifestError, match="duplicate track id"):
            dataset.load_manifest(manifest)

    def test_a_track_missing_its_reference_field_raises(self, tmp_path):
        manifest = write_manifest(tmp_path / "m.json", [{"id": "s"}])
        with pytest.raises(dataset.ManifestError, match="needs both"):
            dataset.load_manifest(manifest)

    def test_a_future_manifest_version_raises(self, tmp_path):
        manifest = write_manifest(
            tmp_path / "m.json", [{"id": "s", "reference": "s.lab"}], version=99
        )
        with pytest.raises(dataset.ManifestError, match="manifest version 99"):
            dataset.load_manifest(manifest)

    def test_malformed_json_raises(self, tmp_path):
        path = tmp_path / "m.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(dataset.ManifestError, match="not valid JSON"):
            dataset.load_manifest(path)

    def test_a_top_level_list_raises(self, tmp_path):
        path = tmp_path / "m.json"
        path.write_text("[]", encoding="utf-8")
        with pytest.raises(dataset.ManifestError, match="JSON object"):
            dataset.load_manifest(path)

    def test_tracks_must_be_a_list(self, tmp_path):
        path = tmp_path / "m.json"
        path.write_text(json.dumps({"version": 1, "tracks": {}}), encoding="utf-8")
        with pytest.raises(dataset.ManifestError, match="must be a list"):
            dataset.load_manifest(path)


class TestWriteManifest:
    def test_it_round_trips(self, tmp_path):
        document = dataset.build("corpus", [{"id": "s", "reference": "s.lab"}], notes="hello")
        path = tmp_path / "nested" / "m.json"
        dataset.write_manifest(path, document)
        loaded = dataset.load_manifest(path)
        assert loaded.name == "corpus"
        assert loaded.notes == "hello"
        assert [t.id for t in loaded.tracks] == ["s"]


class TestFoldTitle:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("01_-_Lantern_Over_Water.lab", "lanternoverwater"),
            ("01 Lantern Over Water.flac", "lanternoverwater"),
            ("1-01 lantern over water.mp3", "lanternoverwater"),
            ("Lantern Over Water.wav", "lanternoverwater"),
            ("lantern-over-water.lab", "lanternoverwater"),
            ("Lantern, Over (Water).m4a", "lanternoverwater"),
        ],
    )
    def test_punctuation_case_and_track_numbers_fold_away(self, name, expected):
        assert isophonics.fold_title(name) == expected

    def test_an_all_digit_title_is_not_folded_to_nothing(self):
        assert isophonics.fold_title("1979.flac") == "1979"


class TestBuildManifest:
    @pytest.fixture
    def annotations(self, tmp_path):
        """An Isophonics-shaped annotation tree, plus a decoy key directory."""
        chords = tmp_path / "ann" / "chordlab" / "The Band" / "01_-_First_Album"
        chords.mkdir(parents=True)
        (chords / "01_-_Lantern_Over_Water.lab").write_text("0.0 1.0 C:maj\n", encoding="utf-8")
        (chords / "02_-_Counting_Windows.lab").write_text("0.0 1.0 A:min\n", encoding="utf-8")

        keys = tmp_path / "ann" / "keylab" / "The Band" / "01_-_First_Album"
        keys.mkdir(parents=True)
        (keys / "01_-_Lantern_Over_Water.lab").write_text("0.0 1.0 C\n", encoding="utf-8")
        return tmp_path / "ann"

    def test_only_the_chord_directory_is_read(self, annotations):
        report = isophonics.build_manifest(annotations)
        ids = [t["id"] for t in report.manifest["tracks"]]
        assert len(ids) == 2
        assert all(i.startswith("chordlab/") for i in ids)

    def test_track_ids_are_relative_posix_paths_without_the_suffix(self, annotations):
        report = isophonics.build_manifest(annotations)
        assert "chordlab/The Band/01_-_First_Album/01_-_Lantern_Over_Water" in [
            t["id"] for t in report.manifest["tracks"]
        ]

    def test_references_are_relative_so_the_manifest_is_portable(self, annotations):
        report = isophonics.build_manifest(annotations)
        for track in report.manifest["tracks"]:
            assert not track["reference"].startswith("/")

    def test_audio_is_paired_by_mirrored_path(self, annotations, tmp_path):
        audio = tmp_path / "audio" / "chordlab" / "The Band" / "01_-_First_Album"
        audio.mkdir(parents=True)
        (audio / "01_-_Lantern_Over_Water.flac").write_bytes(b"x")

        report = isophonics.build_manifest(annotations, tmp_path / "audio")
        assert report.paired == 1
        assert report.unpaired == ("chordlab/The Band/01_-_First_Album/02_-_Counting_Windows",)

    def test_audio_is_paired_by_folded_title_anywhere_under_the_root(self, annotations, tmp_path):
        audio = tmp_path / "audio" / "Some Other Layout"
        audio.mkdir(parents=True)
        (audio / "01 Lantern Over Water.mp3").write_bytes(b"x")
        (audio / "02 Counting Windows.mp3").write_bytes(b"x")

        report = isophonics.build_manifest(annotations, tmp_path / "audio")
        assert report.paired == 2
        assert report.unpaired == ()

    def test_a_title_in_two_albums_is_left_ambiguous_rather_than_guessed(
        self, annotations, tmp_path
    ):
        audio = tmp_path / "audio"
        for album in ("Album One", "Album Two"):
            (audio / album).mkdir(parents=True)
            (audio / album / "Lantern Over Water.flac").write_bytes(b"x")

        report = isophonics.build_manifest(annotations, tmp_path / "audio")
        assert "chordlab/The Band/01_-_First_Album/01_-_Lantern_Over_Water" in report.ambiguous
        assert report.paired == 0

    def test_the_album_directory_disambiguates_a_repeated_title(self, annotations, tmp_path):
        audio = tmp_path / "audio"
        (audio / "01_-_First_Album").mkdir(parents=True)
        (audio / "01_-_First_Album" / "Lantern Over Water.flac").write_bytes(b"x")
        (audio / "Live Album").mkdir(parents=True)
        (audio / "Live Album" / "Lantern Over Water.flac").write_bytes(b"x")

        report = isophonics.build_manifest(annotations, tmp_path / "audio")
        paired = {t["id"]: t.get("audio") for t in report.manifest["tracks"]}
        assert paired["chordlab/The Band/01_-_First_Album/01_-_Lantern_Over_Water"] == (
            "01_-_First_Album/Lantern Over Water.flac"
        )

    def test_a_manifest_without_an_audio_root_has_references_only(self, annotations):
        report = isophonics.build_manifest(annotations)
        assert all("audio" not in t for t in report.manifest["tracks"])
        assert report.total == 2

    def test_the_manifest_loads_back_as_a_dataset(self, annotations, tmp_path):
        report = isophonics.build_manifest(annotations)
        path = tmp_path / "m.json"
        dataset.write_manifest(path, report.manifest)

        loaded = dataset.load_manifest(path, annotation_root=annotations)
        assert len(loaded.tracks) == 2
        assert loaded.missing_references() == ()

    def test_a_tree_with_no_chord_directory_is_searched_whole(self, tmp_path):
        flat = tmp_path / "flat"
        flat.mkdir()
        (flat / "song.lab").write_text("0.0 1.0 C\n", encoding="utf-8")
        report = isophonics.build_manifest(flat)
        assert [t["id"] for t in report.manifest["tracks"]] == ["song"]

    def test_a_missing_directory_raises(self, tmp_path):
        with pytest.raises(NotADirectoryError):
            isophonics.build_manifest(tmp_path / "nope")

    def test_a_directory_with_no_annotations_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match=r"no \.lab files"):
            isophonics.build_manifest(empty)
