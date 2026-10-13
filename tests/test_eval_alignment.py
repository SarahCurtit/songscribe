from __future__ import annotations

from math import isnan

import pytest

from songscribe.eval import metrics
from songscribe.types import ChordEvent


def timeline(*spec: tuple[float, float, str]) -> list[ChordEvent]:
    return [ChordEvent(start, end, label) for start, end, label in spec]


def shift(events: list[ChordEvent], offset: float) -> list[ChordEvent]:
    return [ChordEvent(e.start + offset, e.end + offset, e.label) for e in events]


@pytest.fixture
def progression() -> list[ChordEvent]:
    """Four distinct chords, so a shift cannot fit by coincidence."""
    return timeline(
        (0.0, 4.0, "C:maj"),
        (4.0, 8.0, "G:maj"),
        (8.0, 12.0, "A:min"),
        (12.0, 16.0, "F:maj"),
        (16.0, 20.0, "D:min"),
        (20.0, 24.0, "E:min"),
    )


class TestBestOffset:
    def test_an_aligned_estimate_needs_no_shift(self, progression):
        alignment = metrics.best_offset(progression, progression)
        assert alignment.offset == 0.0
        assert alignment.measured == 1.0
        assert alignment.gain == 0.0

    @pytest.mark.parametrize("offset", [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
    def test_it_recovers_an_injected_offset(self, progression, offset):
        # The estimate is the reference moved later by `offset`, so the shift
        # that undoes it is -offset.
        alignment = metrics.best_offset(progression, shift(progression, offset))
        assert alignment.offset == pytest.approx(-offset, abs=0.051)
        assert alignment.shifted == pytest.approx(1.0, abs=0.01)

    def test_the_gain_is_against_the_measured_score(self, progression):
        alignment = metrics.best_offset(progression, shift(progression, 2.0))
        assert alignment.measured < 0.6
        assert alignment.gain > 0.3

    def test_it_never_changes_the_measured_score(self, progression):
        estimate = shift(progression, 1.5)
        alignment = metrics.best_offset(progression, estimate)
        # The reported score must still be the unshifted one.
        assert alignment.measured == metrics.score(progression, estimate, "majmin").wcsr

    def test_an_offset_beyond_the_search_window_is_not_found(self, progression):
        alignment = metrics.best_offset(progression, shift(progression, 50.0), search=3.0)
        assert alignment.shifted < 0.5

    def test_an_empty_estimate_is_handled(self, progression):
        alignment = metrics.best_offset(progression, [])
        assert alignment.offset == 0.0

    def test_an_empty_reference_is_handled(self):
        alignment = metrics.best_offset([], timeline((0.0, 1.0, "C")))
        assert alignment.offset == 0.0
        assert isnan(alignment.measured)

    def test_it_respects_the_requested_vocabulary(self, progression):
        # Under `root` the quality is ignored, so an estimate with every
        # quality wrong still aligns perfectly.
        wrong_qualities = [
            ChordEvent(e.start, e.end, e.label.split(":")[0] + ":min") for e in progression
        ]
        alignment = metrics.best_offset(progression, wrong_qualities, vocabulary="root")
        assert alignment.measured == 1.0


class TestIsSuspicious:
    def test_a_large_gain_at_a_large_offset_is_suspicious(self):
        alignment = metrics.Alignment(offset=-1.7, shifted=0.80, measured=0.50)
        assert alignment.is_suspicious(min_gain=0.05, min_offset=0.15)

    def test_a_large_gain_at_a_tiny_offset_is_not(self):
        # Ordinary boundary slack: the model is approximately right already.
        alignment = metrics.Alignment(offset=0.05, shifted=0.80, measured=0.50)
        assert not alignment.is_suspicious(min_gain=0.05, min_offset=0.15)

    def test_a_tiny_gain_at_a_large_offset_is_not(self):
        alignment = metrics.Alignment(offset=-2.0, shifted=0.51, measured=0.50)
        assert not alignment.is_suspicious(min_gain=0.05, min_offset=0.15)

    def test_a_nan_score_is_not_suspicious(self):
        from math import nan

        alignment = metrics.Alignment(offset=-2.0, shifted=nan, measured=nan)
        assert not alignment.is_suspicious(min_gain=0.05, min_offset=0.15)

    def test_the_cli_and_the_notebook_share_one_set_of_thresholds(self):
        # The notebook titles its offset plot from metrics; `score` warns from
        # cli. Separate copies would let the two verdicts drift apart.
        from songscribe.eval import cli

        assert cli.ALIGNMENT_WARN_GAIN is metrics.ALIGNMENT_WARN_GAIN
        assert cli.ALIGNMENT_WARN_OFFSET is metrics.ALIGNMENT_WARN_OFFSET

    def test_a_small_lag_on_long_chords_is_slack_not_misalignment(self, progression):
        # A 0.25 s lag on 8 s chords is past the offset threshold but recovers
        # only ~3 points: a rule on the offset alone would call it misaligned.
        long_chords = [ChordEvent(e.start * 2, e.end * 2, e.label) for e in progression]
        alignment = metrics.best_offset(long_chords, shift(long_chords, 0.25))
        assert abs(alignment.offset) >= metrics.ALIGNMENT_WARN_OFFSET
        assert alignment.gain < metrics.ALIGNMENT_WARN_GAIN
        assert not alignment.is_suspicious(
            metrics.ALIGNMENT_WARN_GAIN, metrics.ALIGNMENT_WARN_OFFSET
        )


class TestRunnerIntegration:
    @pytest.fixture
    def misaligned(self, tmp_path):
        """A corpus whose single estimate is shifted off the reference."""
        from songscribe import lab
        from songscribe.eval import dataset

        annotations = tmp_path / "ann"
        estimates = tmp_path / "est"
        reference = timeline(
            (0.0, 4.0, "C:maj"),
            (4.0, 8.0, "G:maj"),
            (8.0, 12.0, "A:min"),
            (12.0, 16.0, "F:maj"),
        )
        lab.write(annotations / "a.lab", reference)
        lab.write(estimates / "a.lab", shift(reference, 2.0))

        manifest = tmp_path / "m.json"
        dataset.write_manifest(manifest, dataset.build("tiny", [{"id": "a", "reference": "a.lab"}]))
        return dataset.load_manifest(manifest, annotation_root=annotations), estimates

    def test_the_result_carries_an_alignment(self, misaligned):
        from songscribe.eval import runner

        data, estimates = misaligned
        report = runner.evaluate(data, estimates_dir=estimates)
        (result,) = report.results
        assert result.alignment is not None
        assert result.alignment.offset == pytest.approx(-2.0, abs=0.051)

    def test_the_check_can_be_turned_off(self, misaligned):
        from songscribe.eval import runner

        data, estimates = misaligned
        report = runner.evaluate(data, estimates_dir=estimates, check_alignment=False)
        (result,) = report.results
        assert result.alignment is None

    def test_an_empty_vocabulary_list_skips_the_sweep(self, misaligned):
        # Nothing is being reported, so there is no number to compare a shift
        # against; this must skip rather than index off the end of the list.
        from songscribe.eval import runner

        data, estimates = misaligned
        report = runner.evaluate(data, estimates_dir=estimates, vocabularies=())
        (result,) = report.results
        assert result.alignment is None

    def test_the_reported_score_is_unaffected_by_the_check(self, misaligned):
        from songscribe.eval import runner

        data, estimates = misaligned
        checked = runner.evaluate(data, estimates_dir=estimates)
        unchecked = runner.evaluate(data, estimates_dir=estimates, check_alignment=False)
        assert checked.weighted("majmin") == unchecked.weighted("majmin")

    def test_the_report_warns_and_names_the_shift(self, misaligned):
        from songscribe.eval import runner
        from songscribe.eval.cli import format_report

        data, estimates = misaligned
        text = format_report(runner.evaluate(data, estimates_dir=estimates))
        assert "WARNING" in text
        assert "-2.00s" in text
        assert "fitting to the reference" in text

    def test_an_aligned_corpus_produces_no_warning(self, tmp_path):
        from songscribe import lab
        from songscribe.eval import dataset, runner
        from songscribe.eval.cli import format_report

        reference = timeline((0.0, 4.0, "C:maj"), (4.0, 8.0, "G:maj"))
        lab.write(tmp_path / "ann" / "a.lab", reference)
        lab.write(tmp_path / "est" / "a.lab", reference)
        manifest = tmp_path / "m.json"
        dataset.write_manifest(manifest, dataset.build("tiny", [{"id": "a", "reference": "a.lab"}]))
        data = dataset.load_manifest(manifest, annotation_root=tmp_path / "ann")

        text = format_report(runner.evaluate(data, estimates_dir=tmp_path / "est"))
        assert "WARNING" not in text

    def test_json_carries_the_alignment(self, misaligned):
        import json

        from songscribe.eval import runner
        from songscribe.eval.cli import report_to_dict

        data, estimates = misaligned
        document = report_to_dict(runner.evaluate(data, estimates_dir=estimates))
        alignment = document["tracks"][0]["alignment"]
        assert alignment["suspicious"] is True
        assert alignment["best_offset"] == pytest.approx(-2.0, abs=0.051)
        assert "NaN" not in json.dumps(document)
