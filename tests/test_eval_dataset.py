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

    @pytest.mark.parametrize(
        "name",
        [
            "CD1_-_01_-_Lantern_Over_Water.lab",
            "CD2-07 Lantern Over Water.flac",
            "Disc 1 - 01 - Lantern Over Water.flac",
            "disk 2 03 Lantern Over Water.mp3",
            "2-14 Lantern Over Water.flac",
        ],
    )
    def test_a_disc_number_folds_away_with_the_track_number(self, name):
        assert isophonics.fold_title(name) == "lanternoverwater"

    def test_disc_words_inside_a_title_are_kept(self):
        assert isophonics.fold_title("04 Discount Lanterns.flac") == "discountlanterns"

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

    def test_a_references_only_build_does_not_claim_a_pairing(self, annotations):
        report = isophonics.build_manifest(annotations)
        assert report.manifest["notes"] == isophonics.REFERENCES_ONLY_NOTES

    def test_a_build_that_pairs_a_track_keeps_the_paired_notes(self, annotations, tmp_path):
        (tmp_path / "audio").mkdir()
        (tmp_path / "audio" / "Lantern Over Water.flac").write_bytes(b"")
        report = isophonics.build_manifest(annotations, tmp_path / "audio")
        assert report.paired == 1
        assert report.manifest["notes"] == isophonics.DEFAULT_NOTES

    def test_an_audio_root_that_pairs_nothing_writes_references_only_notes(
        self, annotations, tmp_path
    ):
        (tmp_path / "audio").mkdir()
        (tmp_path / "audio" / "Unrelated Song.flac").write_bytes(b"")
        report = isophonics.build_manifest(annotations, tmp_path / "audio")
        assert report.paired == 0
        assert report.manifest["notes"] == isophonics.REFERENCES_ONLY_NOTES

    def test_explicit_notes_win_either_way(self, annotations, tmp_path):
        (tmp_path / "audio").mkdir()
        (tmp_path / "audio" / "Lantern Over Water.flac").write_bytes(b"")
        assert isophonics.build_manifest(annotations, notes="mine").manifest["notes"] == "mine"
        with_audio = isophonics.build_manifest(annotations, tmp_path / "audio", notes="mine")
        assert with_audio.paired == 1
        assert with_audio.manifest["notes"] == "mine"

    def test_audio_is_paired_by_mirrored_path(self, annotations, tmp_path):
        audio = tmp_path / "audio" / "chordlab" / "The Band" / "01_-_First_Album"
        audio.mkdir(parents=True)
        (audio / "01_-_Lantern_Over_Water.flac").write_bytes(b"x")

        report = isophonics.build_manifest(annotations, tmp_path / "audio")
        assert report.paired == 1
        assert report.unpaired == ("chordlab/The Band/01_-_First_Album/02_-_Counting_Windows",)

    @pytest.mark.parametrize("folder", ["Music/the band/Some Rip", "Music/First Album", "."])
    def test_a_lone_title_pairs_under_its_artist_or_album_or_at_the_root(
        self, annotations, tmp_path, folder
    ):
        audio = tmp_path / "audio" / folder
        audio.mkdir(parents=True, exist_ok=True)
        (audio / "01 Lantern Over Water.mp3").write_bytes(b"x")
        (audio / "02 Counting Windows.mp3").write_bytes(b"x")

        report = isophonics.build_manifest(annotations, tmp_path / "audio")
        assert report.paired == 2
        assert report.unpaired == ()

    # With the band's own copy missing, a same-titled song by someone else used to
    # pair and be scored as though it were the annotated recording.
    def test_a_lone_title_under_unrelated_folders_is_reported_unplaced(self, tmp_path):
        chords = tmp_path / "ann" / "chordlab" / "Placeholder Band" / "05_-_Fifth_Album"
        chords.mkdir(parents=True)
        (chords / "13_-_Common_Title.lab").write_text("0.0 1.0 C:maj\n", encoding="utf-8")
        rip = tmp_path / "audio" / "Unrelated Singer" / "Their Album" / "07 Common Title.flac"
        rip.parent.mkdir(parents=True)
        rip.write_bytes(b"")

        report = isophonics.build_manifest(tmp_path / "ann", tmp_path / "audio")
        assert report.paired == 0
        assert report.ambiguous == ()
        assert report.unpaired == ()
        assert report.unplaced == ("chordlab/Placeholder Band/05_-_Fifth_Album/13_-_Common_Title",)
        assert report.total == 1

    # A real library seldom names its folders exactly as Isophonics does; these
    # all used to fail the exact-name check and pair nothing.
    @pytest.mark.parametrize(
        ("root", "folder"),
        [
            ("placeholder band", "2003 - Fifth Album (Remastered)"),
            ("placeholder band", "Fifth Album [2009 Stereo Remaster]"),
            ("Music", "The Placeholder Band - Fifth Album (2009) [FLAC]"),
            ("Music", "Placeholder Band - Fifth Album"),
            ("Music", "Placeholder Band Discography/CD 05"),
            ("Placeholder Band Discography", "CD 05"),
        ],
    )
    def test_a_lone_title_pairs_under_a_decorated_artist_or_album_folder(
        self, tmp_path, root, folder
    ):
        chords = tmp_path / "ann" / "chordlab" / "The Placeholder Band" / "05_-_Fifth_Album"
        chords.mkdir(parents=True)
        (chords / "13_-_Common_Title.lab").write_text("0.0 1.0 C:maj\n", encoding="utf-8")
        rip = tmp_path / root / folder / "13 Common Title.flac"
        rip.parent.mkdir(parents=True)
        rip.write_bytes(b"")

        report = isophonics.build_manifest(tmp_path / "ann", tmp_path / root)
        assert report.paired == 1
        assert report.unplaced == ()

    def test_a_short_name_with_both_artist_and_album_in_one_folder_pairs(self, tmp_path):
        chords = tmp_path / "ann" / "chordlab" / "Zork" / "02_-_Gold"
        chords.mkdir(parents=True)
        (chords / "04_-_Common_Title.lab").write_text("0.0 1.0 C:maj\n", encoding="utf-8")
        rip = tmp_path / "audio" / "Zork - Gold (2001)" / "04 Common Title.flac"
        rip.parent.mkdir(parents=True)
        rip.write_bytes(b"")

        report = isophonics.build_manifest(tmp_path / "ann", tmp_path / "audio")
        assert report.paired == 1

    # Containment alone would let the album "Hold" claim any folder with "hold"
    # in it; "Yourself Sessions" is neither the artist nor an allowed extra.
    @pytest.mark.parametrize(
        ("folder", "paired"),
        [("Hold Yourself Sessions", 0), ("Hold! (Remastered)", 1), ("1999 - Hold", 1)],
    )
    def test_a_short_album_name_does_not_match_a_longer_unrelated_one(
        self, tmp_path, folder, paired
    ):
        chords = tmp_path / "ann" / "chordlab" / "Placeholder Band" / "06_-_Hold!"
        chords.mkdir(parents=True)
        (chords / "03_-_Common_Title.lab").write_text("0.0 1.0 C:maj\n", encoding="utf-8")
        rip = tmp_path / "audio" / "Someone Else" / folder / "03 Common Title.flac"
        rip.parent.mkdir(parents=True)
        rip.write_bytes(b"")

        report = isophonics.build_manifest(tmp_path / "ann", tmp_path / "audio")
        assert report.paired == paired
        assert len(report.unplaced) == 1 - paired
        assert report.total == report.paired + len(report.unpaired) + len(report.unplaced)

    # Each folder names the band or the album, but inside someone else's record:
    # a cover, a tribute, a demo set, or a longer artist name sharing a prefix.
    @pytest.mark.parametrize(
        "folder",
        [
            "Placeholder Band Covers",
            "Various Artists/Songs of The Placeholder Band",
            "The Placeholder Band Tribute Band/Another Album",
            "Some Crooner/Sings the Placeholder Band",
            "Placeholder Band Revival/Live 1999",
            "Placeholder Band Wilkins/Their Album",
            "Other Singer/Fifth Album Demos",
            "Other Singer/Fifth Album Revisited",
            "Other Singer/Album",
            # Both names present is no excuse: the extra word still marks a cover.
            "Various Artists/The Placeholder Band - Fifth Album (Karaoke)",
            "Fifth Album Revisited - A Tribute to The Placeholder Band",
            "Fifth Album - The Placeholder Band Covers",
        ],
    )
    def test_a_lone_title_inside_someone_elses_record_is_unplaced(self, tmp_path, folder):
        chords = tmp_path / "ann" / "chordlab" / "The Placeholder Band" / "05_-_Fifth_Album"
        chords.mkdir(parents=True)
        (chords / "13_-_Common_Title.lab").write_text("0.0 1.0 C:maj\n", encoding="utf-8")
        rip = tmp_path / "audio" / folder / "07 Common Title.flac"
        rip.parent.mkdir(parents=True)
        rip.write_bytes(b"")

        report = isophonics.build_manifest(tmp_path / "ann", tmp_path / "audio")
        assert report.paired == 0
        assert len(report.unplaced) == 1

    # Only the other name or a word from the allowlist may sit beside a contained
    # name, which is what an artist's own "Band - Album" or "Band, The" looks like.
    @pytest.mark.parametrize(
        "folder",
        [
            "Placeholder Band - Fifth Album (2009) [FLAC]",
            "Placeholder Band, The/Some Rip",
            "Placeholder Band Discography/CD 05",
            "The Placeholder Band Complete Box Set/Disc 5",
            "Zork/The Placeholder Band Anthology",
        ],
    )
    def test_a_lone_title_beside_allowed_words_pairs(self, tmp_path, folder):
        chords = tmp_path / "ann" / "chordlab" / "The Placeholder Band" / "05_-_Fifth_Album"
        chords.mkdir(parents=True)
        (chords / "13_-_Common_Title.lab").write_text("0.0 1.0 C:maj\n", encoding="utf-8")
        rip = tmp_path / "audio" / folder / "13 Common Title.flac"
        rip.parent.mkdir(parents=True)
        rip.write_bytes(b"")

        report = isophonics.build_manifest(tmp_path / "ann", tmp_path / "audio")
        assert report.paired == 1
        assert report.unplaced == ()

    # Rips drop or add apostrophes freely; _fold already ignores them, and a flat
    # library with no artist folder used to leave every such album unplaced.
    @pytest.mark.parametrize(
        ("album", "folder"),
        [
            ("03_-_A_Long_Day's_Lantern", "A Long Days Lantern"),
            ("03_-_A_Long_Days_Lantern", "A Long Day's Lantern"),
            ("03_-_A_Long_Day's_Lantern", "A Long Day\u2019s Lantern"),
            ("03_-_Capt._Zork's_Parade", "Capt Zorks Parade"),
            ("03_-_Capt_Zorks_Parade", "Capt. Zork's Parade"),
        ],
    )
    def test_an_album_folder_differing_only_in_apostrophes_pairs(self, tmp_path, album, folder):
        chords = tmp_path / "ann" / "chordlab" / "Placeholder Band" / album
        chords.mkdir(parents=True)
        (chords / "05_-_Common_Title.lab").write_text("0.0 1.0 C:maj\n", encoding="utf-8")
        rip = tmp_path / "Music" / folder / "05 Common Title.flac"
        rip.parent.mkdir(parents=True)
        rip.write_bytes(b"")

        report = isophonics.build_manifest(tmp_path / "ann", tmp_path / "Music")
        assert report.paired == 1
        assert report.unplaced == ()

    # A self-titled album's two names are equal, so "Band - Band" must still count
    # the second as the other name; rip descriptors are decoration, not a cover.
    @pytest.mark.parametrize(
        "folder",
        [
            "Placeholder Band - Placeholder Band (2009) [FLAC]",
            "The Placeholder Band - Placeholder Band [24-96 Vinyl Rip]",
            "Placeholder Band - Placeholder Band (Japanese Edition)",
        ],
    )
    def test_a_band_and_album_folder_pairs_for_a_self_titled_album(self, tmp_path, folder):
        chords = tmp_path / "ann" / "chordlab" / "Placeholder Band" / "07_-_Placeholder_Band"
        chords.mkdir(parents=True)
        (chords / "05_-_Common_Title.lab").write_text("0.0 1.0 C:maj\n", encoding="utf-8")
        rip = tmp_path / "Music" / folder / "05 Common Title.flac"
        rip.parent.mkdir(parents=True)
        rip.write_bytes(b"")

        report = isophonics.build_manifest(tmp_path / "ann", tmp_path / "Music")
        assert report.paired == 1
        assert report.unplaced == ()

    def test_the_manifest_command_reports_unplaced_tracks_with_a_hint(self, tmp_path, capsys):
        from songscribe.eval import cli

        chords = tmp_path / "ann" / "chordlab" / "Placeholder Band" / "05_-_Fifth_Album"
        chords.mkdir(parents=True)
        (chords / "13_-_Common_Title.lab").write_text("0.0 1.0 C:maj\n", encoding="utf-8")
        rip = tmp_path / "audio" / "Unrelated Singer" / "Their Album" / "07 Common Title.flac"
        rip.parent.mkdir(parents=True)
        rip.write_bytes(b"")

        argv = ["manifest", str(tmp_path / "ann"), "--audio", str(tmp_path / "audio")]
        assert cli.main([*argv, "--output", str(tmp_path / "m.json")]) == 0
        out = capsys.readouterr().out
        assert "title found but no folder names the artist or album: 1" in out
        assert cli.UNPLACED_HINT in out
        assert "no audio found" not in out
        assert "chordlab/Placeholder Band/05_-_Fifth_Album/13_-_Common_Title" in out

    @pytest.mark.parametrize(
        "rip",
        [
            "Double Album/Disc 1/01 Paper Kites At Dusk.flac",
            "Double Album/1-01 Paper Kites At Dusk.flac",
        ],
    )
    def test_a_disc_prefixed_annotation_pairs_with_a_rip(self, tmp_path, rip):
        # Isophonics names a two-disc album's files "CD1_-_01_-_...", unlike any rip.
        chords = tmp_path / "ann" / "chordlab" / "The Band" / "03CD1_-_Double_Album"
        chords.mkdir(parents=True)
        (chords / "CD1_-_01_-_Paper_Kites_At_Dusk.lab").write_text("0.0 1.0 C:maj\n", "utf-8")
        (tmp_path / "audio" / rip).parent.mkdir(parents=True)
        (tmp_path / "audio" / rip).write_bytes(b"")

        report = isophonics.build_manifest(tmp_path / "ann", tmp_path / "audio")
        assert report.paired == 1
        assert report.manifest["tracks"][0]["audio"] == rip

    # A self-titled album folds to the artist's name, so the artist folder above a
    # compilation used to pass as the album and pair the track with the wrong copy.
    @pytest.mark.parametrize(
        "library",
        [
            [
                "Fake Band/Greatest Hits/05 Paper Lantern.flac",
                "Cover Band/Tribute/01 Paper Lantern.flac",
            ],
            [
                "Albums/Fake Band White/1-01 Paper Lantern.flac",
                "Fake Band/Greatest Hits/05 Paper Lantern.flac",
            ],
        ],
    )
    def test_a_self_titled_album_does_not_match_the_artist_folder(self, tmp_path, library):
        chords = tmp_path / "ann" / "chordlab" / "Fake Band" / "07CD1_-_Fake_Band"
        chords.mkdir(parents=True)
        (chords / "CD1_-_01_-_Paper_Lantern.lab").write_text("0.0 1.0 C:maj\n", "utf-8")
        for rip in library:
            (tmp_path / "audio" / rip).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / "audio" / rip).write_bytes(b"")

        report = isophonics.build_manifest(tmp_path / "ann", tmp_path / "audio")
        assert report.paired == 0
        assert report.ambiguous == (
            "chordlab/Fake Band/07CD1_-_Fake_Band/CD1_-_01_-_Paper_Lantern",
        )

    def test_a_self_titled_album_still_pairs_with_its_only_copy(self, tmp_path):
        chords = tmp_path / "ann" / "chordlab" / "Fake Band" / "07CD1_-_Fake_Band"
        chords.mkdir(parents=True)
        (chords / "CD1_-_01_-_Paper_Lantern.lab").write_text("0.0 1.0 C:maj\n", "utf-8")
        rip = "Fake Band/Fake Band/01 Paper Lantern.flac"
        (tmp_path / "audio" / rip).parent.mkdir(parents=True)
        (tmp_path / "audio" / rip).write_bytes(b"")

        report = isophonics.build_manifest(tmp_path / "ann", tmp_path / "audio")
        assert report.manifest["tracks"][0]["audio"] == rip

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

    # The second pair sorts its worse copy first and spells the better one in
    # capitals, so neither listing order nor case can make it pass by accident.
    @pytest.mark.parametrize(("better", "worse"), [(".flac", ".mp3"), (".FLAC", ".aac")])
    def test_two_formats_of_one_file_pair_with_the_better_copy(
        self, annotations, tmp_path, better, worse
    ):
        audio = tmp_path / "audio" / "First Album"
        audio.mkdir(parents=True)
        (audio / f"01 Lantern Over Water{better}").write_bytes(b"x")
        (audio / f"01 Lantern Over Water{worse}").write_bytes(b"x")

        report = isophonics.build_manifest(annotations, tmp_path / "audio")
        paired = {t["id"]: t.get("audio") for t in report.manifest["tracks"]}
        assert report.ambiguous == ()
        assert paired["chordlab/The Band/01_-_First_Album/01_-_Lantern_Over_Water"] == (
            f"First Album/01 Lantern Over Water{better}"
        )

    def test_one_title_in_two_albums_stays_ambiguous_whatever_the_formats(
        self, annotations, tmp_path
    ):
        audio = tmp_path / "audio"
        (audio / "Album One").mkdir(parents=True)
        (audio / "Album One" / "Lantern Over Water.flac").write_bytes(b"x")
        (audio / "Album One" / "Lantern Over Water.mp3").write_bytes(b"x")
        (audio / "Album Two").mkdir(parents=True)
        (audio / "Album Two" / "Lantern Over Water.mp3").write_bytes(b"x")

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

    def test_a_title_track_pairs_with_the_copy_in_its_own_album(self, tmp_path):
        chords = tmp_path / "ann" / "chordlab" / "The Band" / "Paper Harbour"
        chords.mkdir(parents=True)
        (chords / "06_-_Paper_Harbour.lab").write_text("0.0 1.0 C:maj\n", encoding="utf-8")
        audio = tmp_path / "audio"
        (audio / "Paper Harbour").mkdir(parents=True)
        (audio / "Paper Harbour" / "06 Paper Harbour.flac").write_bytes(b"x")
        (audio / "Best Of").mkdir(parents=True)
        (audio / "Best Of" / "12 Paper Harbour.flac").write_bytes(b"x")

        report = isophonics.build_manifest(tmp_path / "ann", audio)
        assert report.ambiguous == ()
        assert report.manifest["tracks"][0]["audio"] == "Paper Harbour/06 Paper Harbour.flac"

    # Step 2 takes only a folder named exactly after the album (case, punctuation,
    # a leading year or track number aside): a word rule there kept picking the
    # wrong disc or volume silently, while a decorated folder fails safe.
    @pytest.mark.parametrize(
        ("album", "paired"),
        [
            ("The Band/Paper Harbour", True),
            ("The Band/1971 - Paper Harbour", True),
            ("The Band/paper harbour", True),
            ("The Band/Paper Harbour [FLAC]", False),
            ("The Band/Paper Harbour (Remastered)", False),
            ("The Band - Paper Harbour", False),
        ],
    )
    def test_only_an_exactly_named_album_folder_beats_a_compilation(
        self, tmp_path, capsys, album, paired
    ):
        from songscribe.eval import cli

        chords = tmp_path / "ann" / "chordlab" / "The Band" / "02_-_Paper_Harbour"
        chords.mkdir(parents=True)
        (chords / "03_-_Lantern_Over_Water.lab").write_text("0.0 1.0 C:maj\n", encoding="utf-8")
        audio = tmp_path / "audio"
        for directory in (album, "The Band/Greatest Hits"):
            (audio / directory).mkdir(parents=True)
            (audio / directory / "03 Lantern Over Water.flac").write_bytes(b"")

        report = isophonics.build_manifest(tmp_path / "ann", audio)
        audio_path = report.manifest["tracks"][0].get("audio")
        if paired:
            assert report.ambiguous == ()
            assert audio_path == f"{album}/03 Lantern Over Water.flac"
        else:
            assert audio_path is None
            assert len(report.ambiguous) == 1
            argv = ["manifest", str(tmp_path / "ann"), "--audio", str(audio)]
            assert cli.main([*argv, "--output", str(tmp_path / "m.json")]) == 0
            assert cli.AMBIGUOUS_HINT in capsys.readouterr().out

    @staticmethod
    def _build_album(tmp_path, album, folders):
        chords = tmp_path / "ann" / "chordlab" / "The Band" / album
        chords.mkdir(parents=True)
        (chords / "03_-_Lantern_Over_Water.lab").write_text("0.0 1.0 C:maj\n", encoding="utf-8")
        audio = tmp_path / "audio"
        for directory in folders:
            (audio / directory).mkdir(parents=True)
            (audio / directory / "03 Lantern Over Water.flac").write_bytes(b"")
        return isophonics.build_manifest(tmp_path / "ann", audio)

    # Step 2's exact fold keeps an album's numbers apart, so "Live 1/" is never
    # the album of a "Live_2" annotation; a decorated folder is left ambiguous.
    @pytest.mark.parametrize(
        ("album", "folders", "expected"),
        [
            ("Live_2", ["The Band/Live 1", "The Band/Greatest Hits"], None),
            ("Live_2", ["The Band/Live 1", "The Band/Live 2"], "The Band/Live 2"),
            ("07_-_Live_2", ["The Band/Live 1", "The Band/Live 2"], "The Band/Live 2"),
            ("Vol._2", ["The Band/Vol. 1", "The Band/Greatest Hits"], None),
            ("Vol._2", ["The Band/Vol. 1", "The Band/Vol. 2"], "The Band/Vol. 2"),
            (
                "Anthology_2",
                ["The Band/Anthology 2", "The Band/Anthology 3"],
                "The Band/Anthology 2",
            ),
            ("Live_at_Somewhere_1971", ["The Band/Live at Somewhere 1970", "The Band/Hits"], None),
            (
                "Live_at_Somewhere_1971",
                ["The Band/Live at Somewhere 1970", "The Band/Live at Somewhere 1971"],
                "The Band/Live at Somewhere 1971",
            ),
            ("Paper_Harbour", ["The Band/Paper Harbour (2009 Remaster)", "The Band/Hits"], None),
        ],
    )
    def test_an_album_folder_must_carry_its_numbers(self, tmp_path, album, folders, expected):
        report = self._build_album(tmp_path, album, folders)
        audio_path = report.manifest["tracks"][0].get("audio")
        if expected is None:
            assert audio_path is None
            assert len(report.ambiguous) == 1
        else:
            assert audio_path == f"{expected}/03 Lantern Over Water.flac"

    # Step 3 admits a lone copy by its album folder alone, so the same number rule
    # applies: another volume's folder says nothing about whose recording it is.
    @pytest.mark.parametrize(
        ("album", "folder", "paired"),
        [
            ("Live_2", "Somebody/Live 1", False),
            ("Live_2", "Somebody/Live 2", True),
            ("Vol._2", "Somebody/Vol. 1", False),
            ("Live_at_Somewhere_1971", "Somebody/Live at Somewhere 1970", False),
            ("Paper_Harbour", "Somebody/Paper Harbour (2009 Remaster)", True),
        ],
    )
    def test_a_lone_copy_needs_the_album_number_in_its_folder(
        self, tmp_path, album, folder, paired
    ):
        report = self._build_album(tmp_path, album, [folder])
        assert ("audio" in report.manifest["tracks"][0]) is paired
        assert len(report.unplaced) == (0 if paired else 1)

    def test_the_manifest_command_reports_ambiguous_tracks_with_a_hint(self, tmp_path, capsys):
        from songscribe.eval import cli

        chords = tmp_path / "ann" / "chordlab" / "The Band" / "02_-_Paper_Harbour"
        chords.mkdir(parents=True)
        (chords / "03_-_Lantern_Over_Water.lab").write_text("0.0 1.0 C:maj\n", encoding="utf-8")
        for directory in ("Greatest Hits", "Past Masters"):
            (tmp_path / "audio" / directory).mkdir(parents=True)
            (tmp_path / "audio" / directory / "03 Lantern Over Water.flac").write_bytes(b"")

        argv = ["manifest", str(tmp_path / "ann"), "--audio", str(tmp_path / "audio")]
        assert cli.main([*argv, "--output", str(tmp_path / "m.json")]) == 0
        out = capsys.readouterr().out
        assert "ambiguous: 1" in out
        assert cli.AMBIGUOUS_HINT in out
        assert "chordlab/The Band/02_-_Paper_Harbour/03_-_Lantern_Over_Water" in out

    # The decoy shares everything before the first dot, so a fold that cut album
    # names at a dot left the title ambiguous, or paired it with the decoy.
    @pytest.mark.parametrize("album", ["Mr. Placeholder's Band", "Mr Placeholders Band"])
    def test_an_album_with_a_dot_in_its_name_pairs_with_its_own_copy(self, tmp_path, album):
        chords = tmp_path / "ann" / "chordlab" / "The Band" / "05_-_Mr._Placeholder's_Band"
        chords.mkdir(parents=True)
        (chords / "02_-_Lantern_Over_Water.lab").write_text("0.0 1.0 C:maj\n", encoding="utf-8")
        audio = tmp_path / "audio"
        for directory in (album, "Mr. Decoy Album"):
            (audio / directory).mkdir(parents=True)
            (audio / directory / "02 Lantern Over Water.flac").write_bytes(b"")

        report = isophonics.build_manifest(tmp_path / "ann", audio)
        assert report.ambiguous == ()
        assert report.manifest["tracks"][0]["audio"] == f"{album}/02 Lantern Over Water.flac"

    def test_a_directory_above_the_audio_root_does_not_count_as_the_album(
        self, annotations, tmp_path
    ):
        audio = tmp_path / "01_-_First_Album" / "library"
        for album in ("01_-_First_Album", "Live Album"):
            (audio / album).mkdir(parents=True)
            (audio / album / "Lantern Over Water.flac").write_bytes(b"x")

        report = isophonics.build_manifest(annotations, audio)
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

    def test_a_chord_directory_below_the_root_still_excludes_key_and_segment_labels(self, tmp_path):
        archive = tmp_path / "data" / "The Band Annotations"
        for sub in ("chordlab", "keylab", "seglab"):
            album = archive / sub / "01_-_First_Album"
            album.mkdir(parents=True)
            (album / "01_-_Lantern_Over_Water.lab").write_text("0.0 1.0 C\n", encoding="utf-8")

        report = isophonics.build_manifest(tmp_path / "data")
        assert [t["id"] for t in report.manifest["tracks"]] == [
            "The Band Annotations/chordlab/01_-_First_Album/01_-_Lantern_Over_Water"
        ]

    def test_every_archive_with_its_own_chord_directory_is_read(self, tmp_path):
        for band in ("The Band", "Other Band"):
            album = tmp_path / "data" / band / "chordlab" / "Album"
            album.mkdir(parents=True)
            (album / "Lantern_Over_Water.lab").write_text("0.0 1.0 C\n", encoding="utf-8")
            (tmp_path / "data" / band / "keylab").mkdir()
            (tmp_path / "data" / band / "keylab" / "Lantern_Over_Water.lab").touch()

        report = isophonics.build_manifest(tmp_path / "data")
        assert [t["id"] for t in report.manifest["tracks"]] == [
            "Other Band/chordlab/Album/Lantern_Over_Water",
            "The Band/chordlab/Album/Lantern_Over_Water",
        ]

    def test_a_chord_directory_nested_in_another_is_not_counted_twice(self, tmp_path):
        nested = tmp_path / "ann" / "chordlab" / "The Band" / "chords"
        nested.mkdir(parents=True)
        (nested / "Lantern_Over_Water.lab").write_text("0.0 1.0 C\n", encoding="utf-8")

        report = isophonics.build_manifest(tmp_path / "ann")
        assert report.total == 1

    def test_a_flat_tree_beside_an_archive_is_still_read(self, tmp_path):
        data = tmp_path / "data"
        (data / "Other Band" / "Album").mkdir(parents=True)
        (data / "Other Band" / "Album" / "Lantern_Over_Water.lab").touch()
        for sub in ("chordlab", "keylab"):
            (data / "The Band" / sub / "Album").mkdir(parents=True)
            (data / "The Band" / sub / "Album" / "Counting_Windows.lab").touch()

        report = isophonics.build_manifest(data)
        assert [t["id"] for t in report.manifest["tracks"]] == [
            "Other Band/Album/Lantern_Over_Water",
            "The Band/chordlab/Album/Counting_Windows",
        ]

    def test_an_album_folder_named_chords_does_not_drop_its_siblings(self, tmp_path):
        band = tmp_path / "ann" / "The Band"
        for album in ("Album A", "Chords"):
            (band / album).mkdir(parents=True)
        (band / "Album A" / "01.lab").touch()
        (band / "Chords" / "02.lab").touch()

        report = isophonics.build_manifest(tmp_path / "ann")
        assert [t["id"] for t in report.manifest["tracks"]] == [
            "The Band/Album A/01",
            "The Band/Chords/02",
        ]

    def test_a_stray_label_file_inside_an_archive_is_skipped(self, tmp_path):
        archive = tmp_path / "ann"
        for sub in ("chordlab", "seglab"):
            (archive / sub).mkdir(parents=True)
            (archive / sub / "Lantern_Over_Water.lab").touch()
        (archive / "Lantern_Over_Water.lab").touch()

        report = isophonics.build_manifest(archive)
        assert [t["id"] for t in report.manifest["tracks"]] == ["chordlab/Lantern_Over_Water"]

    def test_a_missing_directory_raises(self, tmp_path):
        with pytest.raises(NotADirectoryError):
            isophonics.build_manifest(tmp_path / "nope")

    def test_a_missing_audio_directory_raises(self, annotations, tmp_path):
        with pytest.raises(NotADirectoryError, match="no audio directory"):
            isophonics.build_manifest(annotations, tmp_path / "musci")

    def test_the_manifest_command_keeps_the_old_output_on_a_missing_audio_root(
        self, annotations, tmp_path, capsys
    ):
        from songscribe.eval import cli

        output = tmp_path / "m.json"
        output.write_text("previous manifest", encoding="utf-8")
        argv = ["manifest", str(annotations), "--audio", str(tmp_path / "musci")]
        assert cli.main([*argv, "--output", str(output)]) == 1
        assert "no audio directory at" in capsys.readouterr().err
        assert output.read_text(encoding="utf-8") == "previous manifest"

    def test_a_directory_with_no_annotations_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match=r"no \.lab files"):
            isophonics.build_manifest(empty)
