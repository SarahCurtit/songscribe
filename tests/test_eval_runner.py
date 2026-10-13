from __future__ import annotations

import subprocess
from math import isnan

import pytest

from songscribe import separate
from songscribe.eval import dataset, runner
from songscribe.eval.cli import format_report, report_to_dict
from songscribe.types import ChordEvent


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

    def test_it_is_keyed_by_input_variant(self, tmp_path):
        full_mix = runner.cache_path(tmp_path, "madmom", "song")
        separated = runner.cache_path(tmp_path, "madmom", "song", separated=True)
        assert full_mix != separated


class TestEstimateCache:
    """The cache must never serve one input variant's estimate as the other's."""

    @pytest.fixture
    def track(self, tmp_path):
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"")
        return dataset.Track(id="song", reference=tmp_path / "song.lab", audio=audio)

    @pytest.fixture
    def model(self, monkeypatch):
        """A fake chord branch that labels by input variant and counts runs."""
        calls = []

        def fake_recognise(audio_path, backend, use_separation):
            calls.append(use_separation)
            label = "G" if use_separation else "C"
            return [ChordEvent(0.0, 1.0, label)], use_separation

        monkeypatch.setattr(separate, "available", lambda: True)
        monkeypatch.setattr(runner, "_recognise", fake_recognise)
        return calls

    def test_separated_and_full_mix_runs_do_not_share_entries(self, tmp_path, track, model):
        cache = tmp_path / "cache"
        full_mix = runner.estimate(track, "madmom", cache_dir=cache)
        separated = runner.estimate(track, "madmom", cache_dir=cache, use_separation=True)

        assert [e.label for e in full_mix] == ["C"]
        assert [e.label for e in separated] == ["G"]
        assert model == [False, True], "the second variant must run the model, not hit cache"

    def test_each_variant_still_hits_its_own_entry(self, tmp_path, track, model):
        cache = tmp_path / "cache"
        runner.estimate(track, "madmom", cache_dir=cache, use_separation=True)
        again = runner.estimate(track, "madmom", cache_dir=cache, use_separation=True)
        assert [e.label for e in again] == ["G"]
        assert model == [True]

    def test_a_full_mix_fallback_is_not_cached_as_separated(self, tmp_path, track, monkeypatch):
        # demucs missing: separation was asked for but the full mix was scored.
        from songscribe import audio, chords, separate

        monkeypatch.setattr(separate, "available", lambda: False)
        monkeypatch.setattr(audio, "load", lambda source, rate: ([], rate))
        monkeypatch.setattr(
            chords,
            "recognise",
            lambda samples, rate, backend: [ChordEvent(0.0, 1.0, "C")],
        )

        cache = tmp_path / "cache"
        runner.estimate(track, "madmom", cache_dir=cache, use_separation=True)

        assert not runner.cache_path(cache, "madmom", "song", separated=True).exists()
        assert runner.cache_path(cache, "madmom", "song").exists()

    def test_a_full_mix_fallback_hits_its_own_entry_on_rerun(
        self, tmp_path, track, monkeypatch, caplog
    ):
        # Looking up the separated key while writing the full-mix one would
        # miss forever and rerun the model in full on every invocation.
        from songscribe import audio, chords

        runs = []

        def recognise(samples, rate, backend):
            runs.append(backend)
            return [ChordEvent(0.0, 1.0, "C")]

        monkeypatch.setattr(separate, "available", lambda: False)
        monkeypatch.setattr(audio, "load", lambda source, rate: ([], rate))
        monkeypatch.setattr(chords, "recognise", recognise)

        cache = tmp_path / "cache"
        runner.estimate(track, "madmom", cache_dir=cache, use_separation=True)
        caplog.clear()
        again = runner.estimate(track, "madmom", cache_dir=cache, use_separation=True)

        assert [e.label for e in again] == ["C"]
        assert len(runs) == 1, "the rerun must hit the full-mix entry"
        # Still warned on the hit: the report would otherwise read as separated.
        assert "demucs not installed" in caplog.text


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

    @pytest.mark.parametrize(
        "content",
        [b"this is not a lab file\n", b"2.0 1.0 C:maj\n", b"0.0 9.0 C:maj \xff\xfe\n"],
        ids=["too-few-fields", "end-before-start", "not-utf8"],
    )
    def test_a_malformed_reference_is_skipped_rather_than_crashing_the_run(
        self, corpus, tmp_path, content
    ):
        # Raising out of evaluate() would discard every track already scored,
        # which with a model backend and no cache is hours of model time.
        data, estimates = corpus
        (tmp_path / "ann" / "long.lab").write_bytes(content)
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

    def test_scoring_in_mirex_alone_scores_every_track(self, corpus):
        # mirex has no per-chord projection, so the chart metrics cannot be
        # built in it; they must fall back rather than abort the whole run.
        data, estimates = corpus
        report = runner.evaluate(data, estimates_dir=estimates, vocabularies=("mirex",))
        assert [r.track_id for r in report.results] == ["short", "long"]
        assert report.mean("mirex") == pytest.approx(0.5)
        assert report.results[0].chart == pytest.approx(1.0)
        assert report.results[1].chart == pytest.approx(0.0)

    def test_an_aggregate_over_nothing_is_nan(self, corpus, tmp_path):
        data, estimates = corpus
        for name in ("short.lab", "long.lab"):
            (estimates / name).unlink()
        report = runner.evaluate(data, estimates_dir=estimates)
        assert report.results == ()
        assert isnan(report.mean("majmin"))
        assert isnan(report.weighted("majmin"))
        assert isnan(report.mean_segmentation())


class TestEvaluateWithSeparation:
    """A demucs failure is one track's problem, not the whole run's."""

    @pytest.fixture
    def data(self, tmp_path):
        tracks = []
        for name in ("short", "long"):
            (tmp_path / f"{name}.lab").write_text("0.0 1.0 C:maj\n", encoding="utf-8")
            (tmp_path / f"{name}.wav").write_bytes(b"")
            tracks.append(
                dataset.Track(
                    id=name, reference=tmp_path / f"{name}.lab", audio=tmp_path / f"{name}.wav"
                )
            )
        return dataset.Dataset(name="tiny", tracks=tuple(tracks))

    @pytest.fixture
    def model(self, monkeypatch):
        """Stubs every stage, demucs included, so nothing heavy ever runs."""
        from songscribe import audio, chords

        monkeypatch.setattr(separate, "available", lambda: True)
        monkeypatch.setattr(separate, "isolate_vocals", lambda path, out_dir: (path, path))
        monkeypatch.setattr(audio, "load", lambda source, rate: ([], rate))
        monkeypatch.setattr(
            chords, "recognise", lambda samples, rate, backend: [ChordEvent(0.0, 1.0, "C")]
        )

    def _fail_on(self, monkeypatch, error):
        def isolate_vocals(path, out_dir):
            if path.stem == "long":
                raise error
            return path, path

        monkeypatch.setattr(separate, "isolate_vocals", isolate_vocals)

    @pytest.mark.parametrize(
        "error",
        [
            subprocess.CalledProcessError(1, ["demucs"]),
            separate.SeparationError("demucs produced no no_vocals stem"),
        ],
        ids=["demucs-exit-status", "missing-stem"],
    )
    def test_a_failed_separation_skips_the_track(self, data, model, monkeypatch, error):
        self._fail_on(monkeypatch, error)
        report = runner.evaluate(data, backend="madmom", use_separation=True)
        assert report.skipped == ("long",)
        assert [r.track_id for r in report.results] == ["short"]

    def test_an_unwired_backend_still_aborts_the_run(self, data, model, monkeypatch):
        # NotImplementedError is a RuntimeError; skipping it per track would
        # turn "this backend does not exist yet" into an empty report.
        from songscribe import chords

        def unwired(samples, rate, backend):
            raise NotImplementedError("not wired up")

        monkeypatch.setattr(chords, "recognise", unwired)
        with pytest.raises(NotImplementedError):
            runner.evaluate(data, backend="btc", use_separation=True)

    def test_a_missing_model_package_aborts_even_with_a_partial_cache(
        self, data, model, monkeypatch, tmp_path
    ):
        # Skipping it per track would report the cached tracks alone as the
        # corpus figure, with exit status 0.
        from songscribe import chords, lab

        cache = tmp_path / "cache"
        lab.write(runner.cache_path(cache, "madmom", "short"), [ChordEvent(0.0, 1.0, "C")])

        def uninstalled(samples, rate, backend):
            raise ModuleNotFoundError("No module named 'madmom'", name="madmom")

        monkeypatch.setattr(chords, "recognise", uninstalled)
        with pytest.raises(ModuleNotFoundError, match="madmom"):
            runner.evaluate(data, backend="madmom", cache_dir=cache)

    def test_an_unknown_backend_raises_before_any_work(self, data):
        with pytest.raises(ValueError, match="unknown chord backend"):
            runner.evaluate(data, backend="definitely-not-a-model")


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
        assert "50.00%" in format_report(report)

    def test_scores_render_to_two_decimals(self):
        # One decimal put the rendering right on a rounding boundary for
        # round-numbered corpora: a 1e-15 summation difference between a fresh
        # run and the same estimates read back from cache flipped the printed
        # digit, making two identical runs look like they disagreed.
        from songscribe.eval.cli import _percent

        assert _percent(0.9325000000000001) == _percent(0.9324999999999999)
        assert _percent(0.9325) == "93.25%"

    def test_an_uncomputable_score_renders_as_a_dash(self):
        from math import nan

        from songscribe.eval.cli import _percent

        assert _percent(nan) == "--"

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
