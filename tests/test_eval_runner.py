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

    @pytest.mark.parametrize("vocabularies", [("triads", "sevenths"), ("root",)])
    def test_chart_and_grid_do_not_follow_the_reported_vocabularies(self, tmp_path, vocabularies):
        # Their rows carry no vocabulary name, so a re-score under other `-v`
        # rows must print the same chart and grid as a default run. The data
        # scores differently in each vocabulary: triads reduces 7sus4 to sus4,
        # majmin keeps it distinct, root also hears C and Cm as one chord.
        reference = tmp_path / "ref.lab"
        reference.write_text(
            "0.0 2.0 C:sus4(b7)\n2.0 4.0 C:maj\n4.0 6.0 C:sus4(b7)\n6.0 8.0 C:maj\n",
            encoding="utf-8",
        )
        track = dataset.Track(id="song", reference=reference)
        estimated = [
            ChordEvent(0.0, 2.0, "C:sus4"),
            ChordEvent(2.0, 4.0, "C:min"),
            ChordEvent(4.0, 6.0, "C:sus4"),
            ChordEvent(6.0, 8.0, "C:min"),
        ]

        default = runner.score_track(track, estimated, check_alignment=False)
        narrowed = runner.score_track(track, estimated, vocabularies, check_alignment=False)

        assert narrowed.chart == pytest.approx(default.chart)
        assert narrowed.reference_chart_chords == default.reference_chart_chords
        assert narrowed.estimate_chart_chords == default.estimate_chart_chords
        assert narrowed.grid == pytest.approx(default.grid, nan_ok=True)
        assert narrowed.harmonic == default.harmonic

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

    def _fail_to_decode(self, monkeypatch, error):
        from songscribe import audio

        def load(source, rate):
            if source.stem == "long":
                raise error
            return [], rate

        monkeypatch.setattr(audio, "load", load)

    @pytest.mark.parametrize(
        "error",
        # Stand-ins for what librosa surfaced on a non-audio and an empty .wav:
        # audioread's NoBackendError subclasses plain Exception.
        [type("NoBackendError", (Exception,), {})(), EOFError()],
        ids=["no-backend", "empty-file"],
    )
    def test_an_undecodable_file_skips_the_track(self, data, model, monkeypatch, caplog, error):
        self._fail_to_decode(monkeypatch, error)
        report = runner.evaluate(data, backend="madmom")
        assert report.skipped == ("long",)
        assert [r.track_id for r in report.results] == ["short"]
        assert "long: cannot decode" in caplog.text

    def test_the_decode_error_keeps_its_cause(self, data, model, monkeypatch):
        cause = EOFError()
        self._fail_to_decode(monkeypatch, cause)
        with pytest.raises(runner.AudioDecodeError) as raised:
            runner.estimate(data.tracks[1], "madmom")
        assert raised.value.__cause__ is cause

    def test_a_missing_decoder_package_still_aborts_the_run(self, data, model, monkeypatch):
        self._fail_to_decode(monkeypatch, ModuleNotFoundError("No module named 'librosa'"))
        with pytest.raises(ModuleNotFoundError, match="librosa"):
            runner.evaluate(data, backend="madmom")

    def test_an_unknown_backend_raises_before_any_work(self, data):
        with pytest.raises(ValueError, match="unknown chord backend"):
            runner.evaluate(data, backend="definitely-not-a-model")


class TestReportInputVariant:
    """Separated and full-mix scores are not comparable, so the report says which."""

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
        """A chord branch that separates whatever it is asked to and counts runs."""
        runs = []

        def fake_recognise(audio_path, backend, use_separation):
            runs.append(audio_path.stem)
            return [ChordEvent(0.0, 1.0, "C")], use_separation

        monkeypatch.setattr(runner, "_recognise", fake_recognise)
        return runs

    def test_a_full_mix_run_says_so(self, data, model, monkeypatch):
        monkeypatch.setattr(separate, "available", lambda: True)
        report = runner.evaluate(data, backend="madmom")
        assert (report.separated, report.fallbacks) == (False, 0)
        assert "input     full mix\n" in format_report(report)

    def test_a_separated_run_says_so(self, data, model, monkeypatch):
        monkeypatch.setattr(separate, "available", lambda: True)
        report = runner.evaluate(data, backend="madmom", use_separation=True)
        assert (report.separated, report.fallbacks) == (True, 0)
        assert "input     separated stems\n" in format_report(report)
        document = report_to_dict(report)
        assert document["separated"] is True
        assert document["separation_fallbacks"] == 0

    def test_every_full_mix_fallback_is_counted(self, data, model, monkeypatch):
        monkeypatch.setattr(separate, "available", lambda: False)
        report = runner.evaluate(data, backend="madmom", use_separation=True)
        assert (report.separated, report.fallbacks) == (True, 2)
        assert "separated stems (2 of 2 fell back to the full mix)" in format_report(report)
        assert report_to_dict(report)["separation_fallbacks"] == 2

    def test_a_cache_hit_on_a_fallback_still_counts(self, data, model, monkeypatch, tmp_path):
        # The rerun reads the full-mix entry without running anything, but it
        # is still a full-mix score in a run that asked for separation.
        monkeypatch.setattr(separate, "available", lambda: False)
        cache = tmp_path / "cache"
        runner.evaluate(data, backend="madmom", cache_dir=cache, use_separation=True)
        model.clear()
        report = runner.evaluate(data, backend="madmom", cache_dir=cache, use_separation=True)
        assert model == [], "the rerun must be served from cache"
        assert report.fallbacks == 2

    def test_a_skipped_track_is_not_counted_as_a_fallback(self, data, model, monkeypatch):
        def recognise(audio_path, backend, use_separation):
            if audio_path.stem == "long":
                raise runner.AudioDecodeError("cannot decode long.wav")
            return [ChordEvent(0.0, 1.0, "C")], use_separation

        monkeypatch.setattr(separate, "available", lambda: False)
        monkeypatch.setattr(runner, "_recognise", recognise)
        report = runner.evaluate(data, backend="madmom", use_separation=True)
        assert report.skipped == ("long",)
        assert report.fallbacks == 1

    def test_precomputed_estimates_have_no_input_variant(self, corpus):
        data, estimates = corpus
        report = runner.evaluate(data, estimates_dir=estimates)
        assert report.separated is None
        assert "input     not applicable (pre-computed estimates)" in format_report(report)
        document = report_to_dict(report)
        assert document["separated"] is None
        assert document["separation_fallbacks"] == 0


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


class TestGradedReport:
    """The theory row: partial credit beside WCSR, never inside it.

    In the ``corpus`` fixture the one-second track is labelled exactly right and
    the nine-second one is a plain wrong chord, so the two aggregates differ and
    no relation but ``same`` and ``none`` is in play.
    """

    def test_every_track_is_graded(self, corpus):
        data, estimates = corpus
        report = runner.evaluate(data, estimates_dir=estimates)
        assert all(result.graded is not None for result in report.results)

    def test_the_mean_is_per_track_and_the_weighted_by_duration(self, corpus):
        data, estimates = corpus
        report = runner.evaluate(data, estimates_dir=estimates)
        assert report.mean_graded() == pytest.approx(0.5)
        assert report.weighted_graded() == pytest.approx(0.1)

    def test_relations_account_for_the_compared_time(self, corpus):
        data, estimates = corpus
        report = runner.evaluate(data, estimates_dir=estimates)
        assert report.graded_relations() == {
            "none": pytest.approx(9.0),
            "same": pytest.approx(1.0),
        }

    def test_relations_are_listed_largest_share_first(self, corpus):
        data, estimates = corpus
        report = runner.evaluate(data, estimates_dir=estimates)
        shares = list(report.graded_relations().values())
        assert shares == sorted(shares, reverse=True)

    def test_wcsr_is_untouched_by_the_graded_score(self, corpus):
        # The no-regression pin: a duration of partial credit must never reach
        # Score.correct, which is what WCSR is a ratio of.
        data, estimates = corpus
        report = runner.evaluate(data, estimates_dir=estimates)
        assert report.weighted("majmin") == pytest.approx(0.1)
        assert report.results[0].scores["majmin"].correct == pytest.approx(1.0)
        assert report.results[1].scores["majmin"].correct == pytest.approx(0.0)

    def test_the_summary_prints_a_theory_row(self, corpus):
        data, estimates = corpus
        report = runner.evaluate(data, estimates_dir=estimates)
        text = format_report(report)
        assert "theory" in text
        assert "theory by relation (share of compared time):" in text
        assert "none 90.0%" in text

    def test_json_carries_the_corpus_and_per_track_numbers(self, corpus):
        import json

        data, estimates = corpus
        report = runner.evaluate(data, estimates_dir=estimates)
        document = report_to_dict(report)
        json.dumps(document)
        assert document["theory"]["mean"] == pytest.approx(0.5)
        assert document["theory"]["weighted"] == pytest.approx(0.1)
        assert document["theory"]["by_relation"]["none"] == pytest.approx(9.0)
        first = document["tracks"][0]["theory"]
        assert first["score"] == pytest.approx(1.0)
        assert first["by_relation"] == {"same": pytest.approx(1.0)}

    def test_choruses_without_a_graded_score_do_not_sink_the_others(self):
        from songscribe.eval.substitution import GradedScore

        present = GradedScore(credit=1.8, compared=2.0, ignored=0.5, by_relation={"same": 2.0})
        assert runner._sum_graded([None, present, None]) == present
        assert runner._sum_graded([None, None]) is None

    def test_summing_choruses_adds_durations_rather_than_averaging(self):
        from songscribe.eval.substitution import GradedScore

        parts = [
            GradedScore(credit=1.0, compared=1.0, ignored=0.0, by_relation={"same": 1.0}),
            GradedScore(credit=0.0, compared=9.0, ignored=1.0, by_relation={"none": 9.0}),
        ]
        total = runner._sum_graded(parts)
        assert (total.credit, total.compared, total.ignored) == (1.0, 10.0, 1.0)
        assert total.by_relation == {"same": 1.0, "none": 9.0}
        assert total.score == pytest.approx(0.1)

    def test_an_unscoreable_track_renders_as_a_dash_not_a_zero(self, tmp_path):
        annotations = tmp_path / "ann"
        estimates = tmp_path / "est"
        annotations.mkdir()
        estimates.mkdir()
        # An all-X reference: nothing compared, so there is no score to report.
        (annotations / "only.lab").write_text("0.0 4.0 X\n", encoding="utf-8")
        (estimates / "only.lab").write_text("0.0 4.0 C\n", encoding="utf-8")
        manifest = tmp_path / "m.json"
        dataset.write_manifest(
            manifest,
            dataset.build("x-only", [{"id": "only", "reference": "only.lab"}]),
        )
        data = dataset.load_manifest(manifest, annotation_root=annotations)

        report = runner.evaluate(data, estimates_dir=estimates)
        assert isnan(report.results[0].graded.score)
        assert isnan(report.mean_graded())
        assert report.graded_relations() == {}
        assert report_to_dict(report)["theory"]["mean"] is None
        assert "theory by relation" not in format_report(report)


class TestPairwiseVocabulariesInTheReport:
    def test_a_substitutions_run_scores_and_prints(self, corpus):
        data, estimates = corpus
        report = runner.evaluate(
            data, estimates_dir=estimates, vocabularies=("majmin", "substitutions")
        )
        assert report.weighted("substitutions") == pytest.approx(0.1)
        assert "substitutions" in format_report(report)

    def test_no_bar_row_is_printed_for_a_pairwise_vocabulary(self, corpus):
        # bar compares projected label strings, which a pairwise vocabulary has
        # none of. Printing the row would mean computing it, which raises.
        data, estimates = corpus
        report = runner.evaluate(
            data, estimates_dir=estimates, vocabularies=("majmin", "substitutions")
        )
        assert "bar substitutions" not in format_report(report)
        assert "substitutions" not in report_to_dict(report)["bar"]["by_vocabulary"]


@pytest.fixture
def chart_corpus(tmp_path):
    """A reference printing two chords for one of its two bars.

    Bar 1 is labelled right; bar 2 is labelled with the chart's *alternative*,
    so strict scoring calls it wrong and consensus scoring calls it right. The
    second track prints one chord per bar, which is what makes this a mixed
    corpus -- the case where the consensus column has to cover every track or
    stop being comparable with the strict one beside it.
    """
    annotations = tmp_path / "ann"
    estimates = tmp_path / "est"
    annotations.mkdir()
    estimates.mkdir()

    (annotations / "chart.lab").write_text("0.0 2.0 C:maj\n2.0 4.0 D:min|D:7\n", encoding="utf-8")
    (estimates / "chart.lab").write_text("0.0 2.0 C\n2.0 4.0 D7\n", encoding="utf-8")

    (annotations / "plain.lab").write_text("0.0 4.0 C:maj\n", encoding="utf-8")
    (estimates / "plain.lab").write_text("0.0 4.0 C\n", encoding="utf-8")

    manifest = tmp_path / "m.json"
    dataset.write_manifest(
        manifest,
        dataset.build(
            "charts",
            [
                {"id": "chart", "reference": "chart.lab"},
                {"id": "plain", "reference": "plain.lab"},
            ],
        ),
    )
    return dataset.load_manifest(manifest, annotation_root=annotations), estimates


class TestConsensusReport:
    """Strict and consensus side by side, over a reference carrying alternatives."""

    def test_only_the_track_with_alternatives_carries_a_consensus_score(self, chart_corpus):
        data, estimates = chart_corpus
        report = runner.evaluate(data, estimates_dir=estimates)
        by_id = {r.track_id: r for r in report.results}
        assert by_id["chart"].consensus
        assert by_id["plain"].consensus == {}
        assert by_id["plain"].consensus_graded is None

    def test_the_report_knows_the_corpus_has_alternatives(self, chart_corpus):
        data, estimates = chart_corpus
        report = runner.evaluate(data, estimates_dir=estimates)
        assert report.has_consensus

    def test_an_ordinary_corpus_has_no_consensus_at_all(self, corpus):
        # The discipline: a corpus without hand-entered alternatives reports
        # exactly what it reported before this existed.
        data, estimates = corpus
        report = runner.evaluate(data, estimates_dir=estimates)
        assert not report.has_consensus
        assert all(result.consensus == {} for result in report.results)

    def test_strict_calls_the_alternative_wrong_and_consensus_calls_it_right(self, chart_corpus):
        data, estimates = chart_corpus
        report = runner.evaluate(data, estimates_dir=estimates)
        chart = next(r for r in report.results if r.track_id == "chart")
        assert chart.scores["majmin"].wcsr == pytest.approx(0.5)
        assert chart.consensus["majmin"].wcsr == pytest.approx(1.0)

    def test_the_corpus_aggregate_covers_the_track_without_alternatives_too(self, chart_corpus):
        # Both tracks are 4 s. Strict: 2 s + 4 s correct of 8 s. Consensus: 8/8.
        data, estimates = chart_corpus
        report = runner.evaluate(data, estimates_dir=estimates)
        assert report.weighted("majmin") == pytest.approx(0.75)
        assert report.weighted_consensus("majmin") == pytest.approx(1.0)
        assert report.mean_consensus("majmin") == pytest.approx(1.0)

    def test_the_theory_row_gets_a_consensus_figure_too(self, chart_corpus):
        data, estimates = chart_corpus
        report = runner.evaluate(data, estimates_dir=estimates)
        chart = next(r for r in report.results if r.track_id == "chart")
        # Bar 2 is D:min against D:7 strictly -- parallel, 0.40 -- and an exact
        # match against the alternative.
        assert chart.graded.score == pytest.approx(0.7)
        assert chart.consensus_graded.score == pytest.approx(1.0)
        assert report.weighted_consensus_graded() == pytest.approx(1.0)

    def test_wcsr_is_untouched_by_the_consensus_reading(self, chart_corpus):
        # The same no-regression pin as the graded score: consensus credit must
        # never reach the strict Score the published numbers come from.
        data, estimates = chart_corpus
        report = runner.evaluate(data, estimates_dir=estimates)
        chart = next(r for r in report.results if r.track_id == "chart")
        assert chart.scores["majmin"].correct == pytest.approx(2.0)

    def test_the_summary_prints_a_consensus_column(self, chart_corpus):
        data, estimates = chart_corpus
        report = runner.evaluate(data, estimates_dir=estimates)
        text = format_report(report)
        assert "consensus" in text
        assert "is how much of the apparent error is chart choice" in text

    def test_an_ordinary_corpus_prints_no_consensus_column(self, corpus):
        data, estimates = corpus
        report = runner.evaluate(data, estimates_dir=estimates)
        assert "consensus" not in format_report(report)

    def test_json_carries_the_consensus_numbers(self, chart_corpus):
        import json

        data, estimates = chart_corpus
        report = runner.evaluate(data, estimates_dir=estimates)
        document = report_to_dict(report)
        json.dumps(document)
        assert document["summary"]["majmin"]["consensus_weighted"] == pytest.approx(1.0)
        assert document["theory"]["consensus_weighted"] == pytest.approx(1.0)
        chart = next(t for t in document["tracks"] if t["id"] == "chart")
        assert chart["consensus"]["majmin"]["wcsr"] == pytest.approx(1.0)
        assert chart["consensus_theory"]["score"] == pytest.approx(1.0)

    def test_json_omits_the_consensus_keys_on_an_ordinary_corpus(self, corpus):
        data, estimates = corpus
        report = runner.evaluate(data, estimates_dir=estimates)
        document = report_to_dict(report)
        assert "consensus_weighted" not in document["summary"]["majmin"]
        assert "consensus_weighted" not in document["theory"]
        assert document["tracks"][0]["consensus"] == {}


class TestSummingScoresAcrossChoruses:
    def _score(self, correct, compared, ignored):
        return runner.metrics.Score(
            vocabulary="majmin", correct=correct, compared=compared, ignored=ignored
        )

    def test_durations_add_rather_than_average(self):
        parts = [
            {"majmin": self._score(1.0, 2.0, 0.5)},
            {"majmin": self._score(3.0, 6.0, 1.5)},
        ]
        summed = runner._sum_scores(parts, parts[0])
        assert summed["majmin"].correct == pytest.approx(4.0)
        assert summed["majmin"].compared == pytest.approx(8.0)
        assert summed["majmin"].ignored == pytest.approx(2.0)

    def test_an_empty_vocabulary_set_sums_to_nothing(self):
        # A reference with no alternatives has no consensus entries to sum, and
        # must not acquire any here.
        assert runner._sum_scores([{}, {}], {}) == {}
