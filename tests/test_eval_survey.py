from __future__ import annotations

import pytest

from songscribe.eval import dataset
from songscribe.eval.cli import format_survey, survey_to_dict
from songscribe.eval.survey import survey


@pytest.fixture
def corpus(tmp_path):
    """A manifest over three hand-written annotations, references only."""

    def build(tracks: dict[str, str]):
        annotations = tmp_path / "ann"
        annotations.mkdir(exist_ok=True)
        for name, text in tracks.items():
            (annotations / f"{name}.lab").write_text(text, encoding="utf-8")
        manifest = tmp_path / "m.json"
        dataset.write_manifest(
            manifest,
            dataset.build("tiny", [{"id": n, "reference": f"{n}.lab"} for n in tracks]),
        )
        return dataset.load_manifest(manifest, annotation_root=annotations)

    return build


class TestSurvey:
    def test_it_counts_tracks_segments_and_time(self, corpus):
        data = corpus(
            {
                "a": "0.0 10.0 C:maj\n10.0 20.0 G:maj\n",
                "b": "0.0 30.0 A:min\n",
            }
        )
        result = survey(data)
        assert result.tracks == 2
        assert result.segments == 3
        assert result.duration == pytest.approx(50.0)

    def test_labels_are_tallied_by_time_descending(self, corpus):
        data = corpus({"a": "0.0 5.0 C:maj\n5.0 30.0 G:maj\n30.0 35.0 C:maj\n"})
        result = survey(data)
        assert [s.label for s in result.labels] == ["G:maj", "C:maj"]
        assert result.labels[0].duration == pytest.approx(25.0)
        assert result.labels[1].segments == 2

    def test_every_label_parsing_leaves_unparseable_empty(self, corpus):
        data = corpus({"a": "0.0 1.0 C:maj\n1.0 2.0 E:7/3\n2.0 3.0 A:min/b3\n3.0 4.0 N\n"})
        assert survey(data).unparseable == ()

    def test_an_unreadable_label_is_reported_with_its_cost(self, corpus):
        data = corpus({"a": "0.0 1.0 C:maj\n1.0 21.0 C:gibberish\n"})
        result = survey(data)
        (bad,) = result.unparseable
        assert bad.label == "C:gibberish"
        assert bad.duration == pytest.approx(20.0)
        assert bad.parses is False

    def test_a_malformed_file_is_reported_not_raised(self, corpus):
        data = corpus({"a": "0.0 1.0 C:maj\n", "b": "this is not a lab file\n"})
        result = survey(data)
        assert result.tracks == 1
        assert [track_id for track_id, _ in result.unreadable] == ["b"]

    def test_a_missing_file_is_reported_not_raised(self, corpus):
        data = corpus({"a": "0.0 1.0 C:maj\n"})
        data = dataset.Dataset(
            name=data.name,
            tracks=(
                *data.tracks,
                dataset.Track(id="gone", reference=data.tracks[0].reference.parent / "gone.lab"),
            ),
        )
        result = survey(data)
        assert [track_id for track_id, _ in result.unreadable] == ["gone"]


class TestScoreableTime:
    def test_a_plain_triad_corpus_is_fully_scoreable(self, corpus):
        data = corpus({"a": "0.0 10.0 C:maj\n10.0 20.0 A:min\n"})
        result = survey(data)
        for vocabulary in result.scoreable.values():
            assert vocabulary == pytest.approx(1.0)

    def test_a_suspension_is_scoreable_in_triads_but_not_majmin(self, corpus):
        data = corpus({"a": "0.0 10.0 C:sus4\n10.0 20.0 C:maj\n"})
        result = survey(data, vocabularies=("majmin", "triads"))
        assert result.scoreable["majmin"] == pytest.approx(0.5)
        assert result.scoreable["triads"] == pytest.approx(1.0)

    def test_a_seventh_is_scoreable_in_both_majmin_and_sevenths(self, corpus):
        # It reduces to its triad for majmin and is named exactly for sevenths,
        # so neither vocabulary has to exclude it.
        data = corpus({"a": "0.0 10.0 C:maj7\n"})
        result = survey(data, vocabularies=("majmin", "sevenths"))
        assert result.scoreable["majmin"] == pytest.approx(1.0)
        assert result.scoreable["sevenths"] == pytest.approx(1.0)

    def test_minmaj7_is_excluded_from_sevenths_only(self, corpus):
        data = corpus({"a": "0.0 10.0 C:minmaj7\n"})
        result = survey(data, vocabularies=("majmin", "sevenths"))
        assert result.scoreable["majmin"] == pytest.approx(1.0)
        assert result.scoreable["sevenths"] == pytest.approx(0.0)

    def test_unknown_reference_time_is_scoreable_nowhere(self, corpus):
        # Including under root, which excludes nothing else, and mirex.
        data = corpus({"a": "0.0 10.0 X\n10.0 20.0 C:maj\n"})
        result = survey(data, vocabularies=("root", "mirex"))
        assert result.scoreable["root"] == pytest.approx(0.5)
        assert result.scoreable["mirex"] == pytest.approx(0.5)

    def test_a_power_chord_is_excluded_from_mirex_only(self, corpus):
        # Two pitch classes can never share three, so mirex cannot score it,
        # while root still can.
        data = corpus({"a": "0.0 10.0 A:(1,5)\n10.0 20.0 C:maj\n"})
        result = survey(data, vocabularies=("root", "mirex"))
        assert result.scoreable["root"] == pytest.approx(1.0)
        assert result.scoreable["mirex"] == pytest.approx(0.5)

    def test_no_chord_time_is_scoreable_everywhere(self, corpus):
        data = corpus({"a": "0.0 10.0 N\n10.0 20.0 C:maj\n"})
        result = survey(data)
        for fraction in result.scoreable.values():
            assert fraction == pytest.approx(1.0)


class TestSurveyFormatting:
    def test_the_report_names_the_corpus_and_its_totals(self, corpus):
        data = corpus({"a": "0.0 3600.0 C:maj\n"})
        text = format_survey(survey(data))
        assert "tiny" in text
        assert "1.00 h" in text
        assert "every label parses" in text

    def test_an_unparseable_label_is_called_out(self, corpus):
        data = corpus({"a": "0.0 10.0 C:gibberish\n"})
        text = format_survey(survey(data))
        assert "UNPARSEABLE" in text
        assert "C:gibberish" in text

    def test_the_label_table_can_be_suppressed(self, corpus):
        data = corpus({"a": "0.0 10.0 C:maj\n"})
        assert "most common labels" not in format_survey(survey(data), label_limit=0)

    def test_a_corpus_of_zero_total_duration_still_renders(self, corpus):
        # The parser accepts zero-length segments because published annotation
        # sets contain them; a file of nothing but those gives a zero
        # denominator, which must not take the report down with it.
        data = corpus({"a": "1.0 1.0 C:maj\n"})
        result = survey(data)
        assert result.duration == 0.0
        text = format_survey(result)
        assert "C:maj" in text

    def test_json_output_lists_every_label(self, corpus):
        import json

        data = corpus({"a": "0.0 10.0 C:maj\n10.0 20.0 A:min\n"})
        document = survey_to_dict(survey(data))
        assert {entry["label"] for entry in document["labels"]} == {"C:maj", "A:min"}
        assert "NaN" not in json.dumps(document)
