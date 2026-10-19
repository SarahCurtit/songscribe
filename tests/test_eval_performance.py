from __future__ import annotations

from math import isnan

import pytest
import soundfile as sf

from songscribe import beats, lab
from songscribe.beats import BeatGrid
from songscribe.demo import synthesise_performance
from songscribe.eval import chroma, dataset, form, runner
from songscribe.types import ChordEvent

CHART = ("C", "Am", "Dm", "G7", "Em", "A7", "Dm", "G7")
SAMPLE_RATE = 11_025
BPM = 240.0  # one second to the bar


def take(**kwargs):
    defaults = {"progression": CHART, "bpm": BPM, "sample_rate": SAMPLE_RATE, "repeats": 4}
    return synthesise_performance(**{**defaults, **kwargs})


def track_for(tmp_path, performance):
    reference = tmp_path / "reference.lab"
    lab.write(reference, performance.reference)
    return dataset.Track(id="song", reference=reference, bars=len(CHART))


def sure(performance) -> chroma.Anchor:
    """An anchor at the first bar, for tests that do not test anchor finding."""
    return chroma.Anchor(0, performance.downbeats[0], 0.9, 0.4, 0.5)


def score(tmp_path, performance, estimated, **kwargs):
    return runner.score_performance(
        track_for(tmp_path, performance),
        estimated,
        grid=beats.grid(performance.beat_events()),
        bars_per_chorus=len(CHART),
        anchor=sure(performance),
        **kwargs,
    )


def wrong_fifth_bar(events):
    """An estimate that has the wrong chord in the fifth bar of every chorus."""
    # A pickup is one event before bar 0, so it is skipped when counting bars.
    pickup = len(events) % len(CHART)
    return [
        ChordEvent(e.start, e.end, "F#" if (i - pickup) % len(CHART) == 4 else e.label)
        for i, e in enumerate(events)
    ]


class TestScorePerformance:
    def test_each_chorus_is_scored_on_its_own(self, tmp_path):
        performance = take(pickup_beats=3)
        result = score(tmp_path, performance, performance.events)
        assert [c.chorus.kind for c in result.choruses] == ["head", "chorus", "chorus", "head"]
        assert all(c.bar == 1.0 for c in result.choruses)
        assert result.scores["majmin"].wcsr == pytest.approx(1.0)
        assert result.bar == 1.0

    def test_the_pickup_is_an_unscored_intro(self, tmp_path):
        performance = take(pickup_beats=3)
        result = score(tmp_path, performance, performance.events)
        assert result.unscored_seconds == pytest.approx(0.75, abs=1e-3)
        assert result.choruses[0].chorus.first_bar == 0
        assert result.choruses[0].chorus.start == pytest.approx(0.75, abs=1e-3)

    def test_a_chorus_that_goes_wrong_is_seen_in_its_own_row(self, tmp_path):
        performance = take()
        wrong = [
            ChordEvent(e.start, e.end, "F#" if 8 <= i < 16 else e.label)
            for i, e in enumerate(performance.events)
        ]
        result = score(tmp_path, performance, wrong)
        assert [c.bar for c in result.choruses] == [1.0, 0.0, 1.0, 1.0]
        # The track score combines all four choruses. It is not the best chorus's score.
        assert result.bar == pytest.approx(0.75, abs=1e-3)
        assert result.scores["majmin"].wcsr == pytest.approx(0.75, abs=0.02)

    def test_a_pickup_with_tempo_drift_is_an_intro_and_the_rest_scores_exactly(self, tmp_path):
        performance = take(pickup_beats=3, tempo_drift=0.05)
        grid = beats.grid(performance.beat_events())
        found = form.choruses(grid, len(CHART), 0, len(CHART) * 1.0)
        assert [(c.kind, c.bars) for c in found][:2] == [("intro", 0), ("head", 8)]
        assert found[0].end == pytest.approx(found[1].start) == pytest.approx(0.75, abs=1e-3)
        result = score(tmp_path, performance, wrong_fifth_bar(performance.events))
        assert result.bar == pytest.approx(7 / 8)

    def test_a_wrong_pickup_changes_no_score(self, tmp_path):
        plain, picked = take(), take(pickup_beats=3)
        near = score(tmp_path, plain, wrong_fifth_bar(plain.events))
        events = [ChordEvent(0.0, picked.downbeats[0], "B"), *picked.events[1:]]
        far = score(tmp_path, picked, wrong_fifth_bar(events))
        assert far.bar == pytest.approx(near.bar)
        for name, expected in near.scores.items():
            assert far.scores[name].correct == pytest.approx(expected.correct)
            assert far.scores[name].compared == pytest.approx(expected.compared)
        assert far.chart == pytest.approx(near.chart)
        assert far.unscored_seconds == pytest.approx(0.75, abs=1e-3)

    def test_tempo_drift_leaves_the_bar_score_exact(self, tmp_path):
        performance = take(tempo_drift=0.05)
        result = score(tmp_path, performance, wrong_fifth_bar(performance.events))
        assert result.bar == pytest.approx(7 / 8)
        assert result.scores["majmin"].wcsr == pytest.approx(7 / 8, abs=0.03)

    def test_a_chorus_with_a_dropped_bar_is_named_and_left_out(self, tmp_path):
        performance = take(repeats=3)
        downbeats = [d for i, d in enumerate(performance.downbeats) if i != 11]
        result = runner.score_performance(
            track_for(tmp_path, performance),
            performance.events,
            grid=BeatGrid(downbeats=tuple(downbeats)),
            bars_per_chorus=len(CHART),
            anchor=sure(performance),
        )
        # The chorus with the missing bar and the one after it are left out, not scored as 0.
        assert [c.chorus.kind for c in result.choruses] == ["head"]
        assert result.unscored_seconds == pytest.approx(16.0)
        assert result.scores["majmin"].wcsr == pytest.approx(1.0)

    def test_a_partial_last_pass_is_reported_but_not_aggregated(self, tmp_path):
        performance = take(repeats=3)
        cut = len(CHART) * 3 - 3  # leaves five bars of the last chorus
        events = performance.events[:cut]
        events[-1] = ChordEvent(events[-1].start, events[-1].end, "F#")
        grid = beats.grid(
            [b for b in performance.beat_events() if b.time <= performance.downbeats[cut]]
        )
        result = runner.score_performance(
            track_for(tmp_path, performance),
            events,
            grid=grid,
            bars_per_chorus=len(CHART),
            anchor=sure(performance),
        )
        kinds = [c.chorus.kind for c in result.choruses]
        assert kinds == ["head", "head", "partial"]
        assert result.choruses[-1].bar < 1.0
        assert result.bar == 1.0
        assert result.scores["majmin"].wcsr == pytest.approx(1.0)

    def test_a_refused_anchor_raises_rather_than_scoring_a_guess(self, tmp_path):
        performance = take()
        weak = chroma.Anchor(0, performance.downbeats[0], 0.9, 0.88, 0.02)
        with pytest.raises(ValueError, match="not clear"):
            runner.score_performance(
                track_for(tmp_path, performance),
                performance.events,
                grid=beats.grid(performance.beat_events()),
                bars_per_chorus=len(CHART),
                anchor=weak,
            )

    def test_a_recording_with_nothing_scoreable_raises(self, tmp_path):
        performance = take(repeats=1)
        grid = beats.grid([b for b in performance.beat_events() if b.time < 2.0])
        with pytest.raises(ValueError, match="no block"):
            runner.score_performance(
                track_for(tmp_path, performance),
                performance.events,
                grid=grid,
                bars_per_chorus=len(CHART),
                anchor=sure(performance),
            )


class TestSingleChorusIsUnchanged:
    def test_one_chorus_at_scale_one_scores_exactly_as_score_track_does(self, tmp_path):
        performance = take(repeats=1, tempo_drift=0.0)
        # Bar lines every 1.0 s from 0, the same as the chart, so anchor 0 and scale 1.
        estimate = wrong_fifth_bar(performance.reference)
        track = track_for(tmp_path, performance)
        plain = runner.score_track(track, estimate)

        framed = runner.score_performance(
            track,
            estimate,
            grid=BeatGrid(downbeats=tuple(float(i) for i in range(len(CHART) + 1))),
            bars_per_chorus=len(CHART),
            anchor=chroma.Anchor(0, 0.0, 1.0, 0.0, 1.0),
        )
        assert framed.scores == plain.scores
        assert framed.segmentation == plain.segmentation
        assert framed.chart == plain.chart
        assert (isnan(framed.grid) and isnan(plain.grid)) or framed.grid == plain.grid
        assert framed.duration == plain.duration
        assert framed.unscored_seconds == 0.0
        assert len(framed.choruses) == 1


class TestEvaluateWithABarGrid:
    @pytest.fixture
    def setup(self, tmp_path):
        performance = take(repeats=3, pickup_beats=3, tempo_drift=0.02)
        audio = tmp_path / "audio"
        audio.mkdir()
        sf.write(audio / "song.wav", performance.audio, SAMPLE_RATE)
        grids, estimates = tmp_path / "beats", tmp_path / "est"
        lab.write_beats(grids / "song.beats", performance.beat_events())
        lab.write(estimates / "song.lab", wrong_fifth_bar(performance.events))
        lab.write(tmp_path / "ann" / "song.lab", performance.reference)
        manifest = tmp_path / "m.json"
        dataset.write_manifest(
            manifest,
            dataset.build(
                "take",
                [{"id": "song", "reference": "song.lab", "audio": "song.wav", "bars": len(CHART)}],
            ),
        )
        loaded = dataset.load_manifest(manifest, audio_root=audio, annotation_root=tmp_path / "ann")
        return performance, loaded, grids, estimates

    def test_the_form_is_found_in_the_audio_and_scored_per_chorus(self, setup):
        _, loaded, grids, estimates = setup
        report = runner.evaluate(
            loaded, estimates_dir=estimates, beats_dir=grids, check_alignment=False
        )
        (result,) = report.results
        assert result.anchor.bar_index == 0
        assert result.anchor.refusal() is None
        assert [c.chorus.kind for c in result.choruses] == ["head", "chorus", "head"]
        assert result.bar == pytest.approx(7 / 8)
        assert report.mean_bar() == pytest.approx(7 / 8)
        assert report.chorus_count() == 3
        assert result.unscored_seconds == pytest.approx(0.75, abs=1e-3)

    def test_a_manifest_track_without_bars_is_skipped_and_named(self, setup):
        _, loaded, grids, estimates = setup
        bare = dataset.Dataset(
            name="x",
            tracks=tuple(
                dataset.Track(id=t.id, reference=t.reference, audio=t.audio) for t in loaded.tracks
            ),
        )
        report = runner.evaluate(bare, estimates_dir=estimates, beats_dir=grids)
        assert report.skipped == ("song",)
        assert not report.results

    def test_a_missing_beats_file_skips_the_track(self, setup, tmp_path):
        _, loaded, _, estimates = setup
        empty = tmp_path / "nowhere"
        empty.mkdir()
        report = runner.evaluate(loaded, estimates_dir=estimates, beats_dir=empty)
        assert report.skipped == ("song",)

    def test_a_recording_that_is_not_the_chart_is_refused_not_scored(self, setup, tmp_path):
        _, loaded, grids, estimates = setup
        other = ["F#", "Bb", "Eb", "B", "Ab", "D", "F#", "Bb"]
        stranger = take(progression=other, repeats=3, pickup_beats=3)
        sf.write(loaded.tracks[0].audio, stranger.audio, SAMPLE_RATE)
        report = runner.evaluate(loaded, estimates_dir=estimates, beats_dir=grids)
        assert report.skipped == ("song",)

    def test_a_beats_dir_and_a_backend_together_are_a_mistake(self, setup, tmp_path):
        _, loaded, grids, estimates = setup
        with pytest.raises(ValueError, match="at most one"):
            runner.evaluate(
                loaded, estimates_dir=estimates, beats_dir=grids, beat_backend="librosa"
            )

    def test_an_unknown_beat_backend_aborts_rather_than_skipping_every_track(self, setup):
        _, loaded, _, estimates = setup
        with pytest.raises(ValueError):
            runner.evaluate(loaded, estimates_dir=estimates, beat_backend="nope")

    def test_without_a_bar_grid_nothing_changes(self, setup):
        _, loaded, _, estimates = setup
        report = runner.evaluate(loaded, estimates_dir=estimates)
        (result,) = report.results
        assert result.choruses == ()
        assert result.anchor is None
        assert isnan(result.bar)
        assert isnan(report.mean_bar())


class TestManifestBars:
    def test_bars_must_be_a_positive_integer(self, tmp_path):
        for bad in (0, -3, 2.5, "32", True):
            manifest = tmp_path / "m.json"
            dataset.write_manifest(
                manifest, dataset.build("x", [{"id": "a", "reference": "a.lab", "bars": bad}])
            )
            with pytest.raises(dataset.ManifestError, match="bars"):
                dataset.load_manifest(manifest)

    def test_bars_default_to_none(self, tmp_path):
        manifest = tmp_path / "m.json"
        dataset.write_manifest(manifest, dataset.build("x", [{"id": "a", "reference": "a.lab"}]))
        assert dataset.load_manifest(manifest).tracks[0].bars is None
