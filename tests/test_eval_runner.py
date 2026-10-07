from __future__ import annotations

from math import isnan

import pytest

from songscribe.eval import dataset, runner
from songscribe.eval.cli import format_report, report_to_dict


@pytest.fixture
def corpus(tmp_path):
    """Two tracks on disk: references, and estimates in a separate directory.

    The short track is labelled perfectly and the long one entirely wrong, so
    the two corpus aggregates come out different -- which is the point of
    reporting both.
    """
    annotations = tmp_path / "ann"
    estimates = tmp_path / "est"
    annotations.mkdir()
    estimates.mkdir()

    (annotations / "short.lab").write_text("0.0 1.0 C:maj\n", encoding="utf-8")
    (estimates / "short.lab").write_text("0.0 1.0 C\n", encoding="utf-8")

    (annotations / "long.lab").write_text("0.0 9.0 C:maj\n", encoding="utf-8")
    (estimates / "long.lab").write_text("0.0 9.0 F\n", encoding="utf-8")

    manifest = tmp_path / "m.json"
    dataset.write_manifest(
        manifest,
        dataset.build(
            "tiny",
            [
                {"id": "short", "reference": "short.lab"},
                {"id": "long", "reference": "long.lab"},
            ],
        ),
    )
    return dataset.load_manifest(manifest, annotation_root=annotations), estimates


class TestCachePath:
    def test_it_is_keyed_by_backend(self, tmp_path):
        madmom = runner.cache_path(tmp_path, "madmom", "song")
        template = runner.cache_path(tmp_path, "template", "song")
        assert madmom != template

    def test_a_track_id_with_directories_flattens(self, tmp_path):
        path = runner.cache_path(tmp_path, "madmom", "Artist/Album/Track")
        assert path.parent == tmp_path / "madmom"
        assert "/" not in path.name

    @pytest.mark.parametrize(
        "track_id", ["song", "Artist/Album/Track", "Artist\\Album\\Track", "a/b\\c"]
    )
    def test_what_the_cache_writes_is_what_the_reader_looks_for(self, tmp_path, track_id):
        # If the two flattened differently every lookup would miss and the
        # cache would silently re-run the model on every track.
        written = runner.cache_path(tmp_path, "madmom", track_id)
        written.parent.mkdir(parents=True, exist_ok=True)
        written.write_text("0.0 1.0 C\n", encoding="utf-8")

        events = runner.read_estimate(tmp_path / "madmom", track_id)
        assert events is not None, f"cache miss for {track_id!r}"
        assert [e.label for e in events] == ["C"]


class TestReadEstimate:
    def test_it_reads_a_flattened_filename(self, tmp_path):
        (tmp_path / "Artist__Album__Track.lab").write_text("0.0 1.0 C\n", encoding="utf-8")
        events = runner.read_estimate(tmp_path, "Artist/Album/Track")
        assert [e.label for e in events] == ["C"]

    def test_it_reads_a_mirrored_tree(self, tmp_path):
        nested = tmp_path / "Artist" / "Album"
        nested.mkdir(parents=True)
        (nested / "Track.lab").write_text("0.0 1.0 G\n", encoding="utf-8")
        events = runner.read_estimate(tmp_path, "Artist/Album/Track")
        assert [e.label for e in events] == ["G"]

    def test_a_missing_estimate_is_none(self, tmp_path):
        assert runner.read_estimate(tmp_path, "absent") is None


class TestEvaluate:
    def test_it_scores_every_track_from_a_directory_of_estimates(self, corpus):
        data, estimates = corpus
        report = runner.evaluate(data, estimates_dir=estimates)
        assert {r.track_id for r in report.results} == {"short", "long"}
        assert report.skipped == ()

    def test_mean_counts_every_song_once(self, corpus):
        data, estimates = corpus
        report = runner.evaluate(data, estimates_dir=estimates)
        assert report.mean("majmin") == pytest.approx(0.5)

    def test_weighted_counts_every_second_once(self, corpus):
        data, estimates = corpus
        report = runner.evaluate(data, estimates_dir=estimates)
        assert report.weighted("majmin") == pytest.approx(0.1)

    def test_total_duration_is_the_annotated_time(self, corpus):
        data, estimates = corpus
        report = runner.evaluate(data, estimates_dir=estimates)
        assert report.total_duration == pytest.approx(10.0)

    def test_a_track_with_no_estimate_is_skipped_not_scored_as_zero(self, corpus):
        data, estimates = corpus
        (estimates / "long.lab").unlink()
        report = runner.evaluate(data, estimates_dir=estimates)
        assert report.skipped == ("long",)
        # A missing measurement is absent, not wrong: the surviving track's
        # perfect score must not be dragged down by it.
        assert report.mean("majmin") == pytest.approx(1.0)

    def test_a_track_with_no_reference_is_skipped(self, corpus, tmp_path):
        data, estimates = corpus
        (tmp_path / "ann" / "long.lab").unlink()
        report = runner.evaluate(data, estimates_dir=estimates)
        assert report.skipped == ("long",)

    def test_a_malformed_estimate_is_skipped_rather_than_crashing_the_run(self, corpus):
        data, estimates = corpus
        (estimates / "long.lab").write_text("this is not a lab file\n", encoding="utf-8")
        report = runner.evaluate(data, estimates_dir=estimates)
        assert report.skipped == ("long",)
        assert [r.track_id for r in report.results] == ["short"]

    def test_limit_truncates_the_run(self, corpus):
        data, estimates = corpus
        report = runner.evaluate(data, estimates_dir=estimates, limit=1)
        assert len(report.results) == 1

    def test_the_callback_sees_each_result_as_it_lands(self, corpus):
        data, estimates = corpus
        seen = []
        runner.evaluate(data, estimates_dir=estimates, on_track=seen.append)
        assert [r.track_id for r in seen] == ["short", "long"]

    def test_it_records_the_chord_counts_on_both_sides(self, corpus):
        data, estimates = corpus
        report = runner.evaluate(data, estimates_dir=estimates)
        for result in report.results:
            assert result.reference_chords == 1
            assert result.estimate_chords == 1

    def test_requiring_exactly_one_source_of_estimates(self, corpus):
        data, estimates = corpus
        with pytest.raises(ValueError, match="exactly one"):
            runner.evaluate(data)
        with pytest.raises(ValueError, match="exactly one"):
            runner.evaluate(data, backend="template", estimates_dir=estimates)

    def test_an_unknown_vocabulary_raises_before_any_work(self, corpus):
        data, estimates = corpus
        with pytest.raises(ValueError, match="unknown vocabulary"):
            runner.evaluate(data, estimates_dir=estimates, vocabularies=("nope",))

    def test_an_aggregate_over_nothing_is_nan(self, corpus, tmp_path):
        data, estimates = corpus
        for name in ("short.lab", "long.lab"):
            (estimates / name).unlink()
        report = runner.evaluate(data, estimates_dir=estimates)
        assert report.results == ()
        assert isnan(report.mean("majmin"))
        assert isnan(report.weighted("majmin"))
        assert isnan(report.mean_segmentation())


class TestReportFormatting:
    def test_the_summary_names_the_dataset_and_every_vocabulary(self, corpus):
        data, estimates = corpus
        report = runner.evaluate(data, estimates_dir=estimates)
        text = format_report(report)
        assert "tiny" in text
        for vocabulary in report.vocabularies:
            assert vocabulary in text

    def test_percentages_are_rendered(self, corpus):
        data, estimates = corpus
        report = runner.evaluate(data, estimates_dir=estimates)
        assert "50.0%" in format_report(report)

    def test_json_output_is_serialisable_and_nan_free(self, corpus):
        import json

        data, estimates = corpus
        report = runner.evaluate(data, estimates_dir=estimates)
        document = report_to_dict(report)
        text = json.dumps(document)
        # json.dumps emits bare NaN, which is not valid JSON: unscoreable
        # values must already have become nulls.
        assert "NaN" not in text
        assert document["summary"]["majmin"]["mean"] == pytest.approx(0.5)
        assert len(document["tracks"]) == 2
