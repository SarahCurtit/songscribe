from __future__ import annotations

import pytest

from songscribe import beats, lab
from songscribe.types import BeatEvent, ChordEvent


class TestParse:
    def test_reads_space_separated_isophonics_style(self):
        events = lab.parse("0.000000 1.912154 N\n1.912154 4.123000 C:maj\n")
        assert events == [
            ChordEvent(0.0, 1.912154, "N"),
            ChordEvent(1.912154, 4.123, "C:maj"),
        ]

    def test_reads_tab_separated_demo_style(self):
        events = lab.parse("0.0\t2.5\tC\n2.5\t5.0\tAm\n")
        assert [e.label for e in events] == ["C", "Am"]

    def test_accepts_mixed_separators_within_one_file(self):
        events = lab.parse("0.0 1.0\tC\n1.0\t2.0 G\n")
        assert [e.label for e in events] == ["C", "G"]

    def test_skips_blank_lines_and_comments(self):
        events = lab.parse("# annotated by hand\n\n0.0 1.0 C\n\n   \n1.0 2.0 G\n")
        assert len(events) == 2

    def test_keeps_a_label_containing_spaces_as_one_label(self):
        # Hand-corrected annotation sets do this; splitting it into a fourth
        # field would silently truncate the label.
        (event,) = lab.parse("0.0 1.0 C:maj (uncertain)")
        assert event.label == "C:maj (uncertain)"

    def test_tolerates_a_missing_trailing_newline(self):
        assert len(lab.parse("0.0 1.0 C")) == 1

    def test_zero_length_segment_is_allowed(self):
        # Some published annotations contain them; it is the scorer's job to
        # drop them, not the parser's to reject the file.
        (event,) = lab.parse("1.0 1.0 C")
        assert event.duration == 0.0

    @pytest.mark.parametrize(
        "text",
        [
            "0.0 1.0",
            "0.0",
            "C",
        ],
    )
    def test_too_few_fields_raises(self, text):
        with pytest.raises(lab.LabFormatError, match="expected 'start end label'"):
            lab.parse(text)

    def test_non_numeric_timestamp_raises(self):
        with pytest.raises(lab.LabFormatError, match="bad timestamp"):
            lab.parse("start end C")

    def test_end_before_start_raises(self):
        with pytest.raises(lab.LabFormatError, match="precedes start"):
            lab.parse("2.0 1.0 C")

    @pytest.mark.parametrize("text", ["nan nan C:maj", "0.0 inf C:maj", "-inf 1.0 C"])
    def test_rejects_a_non_finite_timestamp(self, text):
        # float() accepts these and `end < start` is False for nan, so without
        # this a nan boundary reaches the scorer and makes every metric over
        # that track nan with nothing reporting a problem.
        with pytest.raises(lab.LabFormatError, match="non-finite"):
            lab.parse(text)

    def test_error_names_the_line_number(self):
        with pytest.raises(lab.LabFormatError, match="line 3"):
            lab.parse("0.0 1.0 C\n1.0 2.0 G\nbroken\n")

    def test_error_names_the_source_file_when_given(self):
        with pytest.raises(lab.LabFormatError, match=r"song\.lab:1"):
            lab.parse("broken", source="song.lab")


class TestWrite:
    def test_round_trips_through_a_file(self, tmp_path):
        events = [ChordEvent(0.0, 2.5, "C"), ChordEvent(2.5, 5.0, "A:min7")]
        path = tmp_path / "nested" / "song.lab"
        lab.write(path, events)

        read_back = lab.read(path)
        assert len(read_back) == len(events)
        for written, event in zip(read_back, events, strict=True):
            assert written.start == pytest.approx(event.start)
            assert written.end == pytest.approx(event.end)
            assert written.label == event.label

    def test_creates_missing_parent_directories(self, tmp_path):
        path = tmp_path / "a" / "b" / "c.lab"
        lab.write(path, [ChordEvent(0.0, 1.0, "C")])
        assert path.exists()

    def test_ends_with_exactly_one_newline(self):
        text = lab.format_events([ChordEvent(0.0, 1.0, "C")])
        assert text.endswith("\n")
        assert not text.endswith("\n\n")

    def test_empty_timeline_writes_nothing(self):
        assert lab.format_events([]) == ""

    def test_survives_a_byte_order_mark(self, tmp_path):
        path = tmp_path / "bom.lab"
        path.write_text("0.0 1.0 C\n", encoding="utf-8-sig")
        assert lab.read(path) == [ChordEvent(0.0, 1.0, "C")]


class TestParseBeats:
    def test_reads_time_and_position(self):
        parsed = lab.parse_beats("0.0\t1\n0.5\t2\n1.0\t3\n1.5\t4\n")
        assert parsed == [
            BeatEvent(0.0, 1),
            BeatEvent(0.5, 2),
            BeatEvent(1.0, 3),
            BeatEvent(1.5, 4),
        ]

    def test_accepts_spaces_as_well_as_tabs(self):
        assert lab.parse_beats("0.0 1\n0.5 2\n") == [BeatEvent(0.0, 1), BeatEvent(0.5, 2)]

    def test_skips_blank_lines_and_comments(self):
        assert len(lab.parse_beats("# tracked by madmom\n\n0.0 1\n   \n0.5 2\n")) == 2

    @pytest.mark.parametrize(
        "text",
        ["0.0", "0.0 1 extra", "beat 1", "0.0 one"],
    )
    def test_rejects_a_malformed_line(self, text):
        with pytest.raises(lab.LabFormatError):
            lab.parse_beats(text)

    @pytest.mark.parametrize("time", ["nan", "inf", "-inf"])
    def test_rejects_a_non_finite_time(self, time):
        # float() accepts these, and a nan time survives every ordering check
        # downstream, reaching a grid that reports itself healthy.
        with pytest.raises(lab.LabFormatError, match="not finite"):
            lab.parse_beats(f"{time} 1")

    @pytest.mark.parametrize("position", [0, -1])
    def test_rejects_a_position_that_is_not_one_based(self, position):
        # Renumbering it silently would put every downbeat on the wrong beat.
        with pytest.raises(lab.LabFormatError, match="not 1-based"):
            lab.parse_beats(f"0.0 {position}")

    def test_error_names_the_line_number(self):
        with pytest.raises(lab.LabFormatError, match="line 3"):
            lab.parse_beats("0.0 1\n0.5 2\nbroken\n")


class TestWriteBeats:
    def test_round_trips_through_a_file(self, tmp_path):
        tracked = [BeatEvent(0.0, 1), BeatEvent(0.5, 2), BeatEvent(1.0, 3)]
        path = tmp_path / "nested" / "song.beats"
        lab.write_beats(path, tracked)

        read_back = lab.read_beats(path)
        assert len(read_back) == len(tracked)
        for written, beat in zip(read_back, tracked, strict=True):
            assert written.time == pytest.approx(beat.time)
            assert written.position == beat.position

    def test_round_trip_preserves_a_grid(self, tmp_path):
        # The point of the format: a tracked grid can be cached and re-read, so
        # the harness runs without a tracker. Compares the bar lines themselves,
        # not just how many there are.
        tracked = [BeatEvent(14.0 + i * 0.4226, i % 4 + 1) for i in range(16)]
        path = tmp_path / "song.beats"
        lab.write_beats(path, tracked)

        reloaded = beats.grid(lab.read_beats(path))
        original = beats.grid(tracked)
        assert reloaded.downbeats == pytest.approx(original.downbeats)
        assert reloaded.beats_per_bar == original.beats_per_bar

    def test_empty_timeline_writes_nothing(self):
        assert lab.format_beats([]) == ""

    def test_ends_with_exactly_one_newline(self):
        text = lab.format_beats([BeatEvent(0.0, 1)])
        assert text.endswith("\n")
        assert not text.endswith("\n\n")
