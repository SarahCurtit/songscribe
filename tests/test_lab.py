from __future__ import annotations

import logging

import pytest

from songscribe import beats, lab
from songscribe.types import BeatEvent, ChordEvent, NoteEvent


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
            lab.parse("20.0 1.0 C")

    def test_a_tiny_reversal_is_skipped_and_keeps_the_rest_of_the_file(self):
        # A trailing reversed no-chord span, as some published annotations end on.
        events = lab.parse("0.0 281.907 D\n281.907 284.352 N\n284.352 284.307 N\n")
        assert [e.label for e in events] == ["D", "N"]
        assert events[-1].end == 284.352

    def test_a_tiny_reversal_mid_file_leaves_the_surrounding_times_alone(self):
        # The kept events overlap by the width of the skipped segment; neither
        # of them moves.
        events = lab.parse("0.0 10.0 C\n10.0 9.96 N\n9.96 20.0 G\n")
        assert [e.label for e in events] == ["C", "G"]
        assert (events[0].start, events[0].end) == (0.0, 10.0)
        assert (events[1].start, events[1].end) == (9.96, 20.0)

    def test_a_reversal_inside_the_tolerance_is_skipped_whatever_the_tolerance_is(self):
        half = lab.MAX_REVERSED_SPAN / 2
        events = lab.parse(f"0.0 100.0 C\n100.0 {100.0 - half} N\n{100.0 - half} 120.0 G\n")
        assert [e.label for e in events] == ["C", "G"]
        assert (events[1].start, events[1].end) == (100.0 - half, 120.0)

    def test_a_reversal_of_exactly_the_tolerance_is_skipped(self):
        assert lab.parse(f"100.0 {100.0 - lab.MAX_REVERSED_SPAN} N") == []

    def test_a_reversal_wider_than_the_tolerance_still_raises(self):
        with pytest.raises(lab.LabFormatError, match="precedes start"):
            lab.parse(f"100.0 {100.0 - lab.MAX_REVERSED_SPAN * 2} C")

    def test_times_a_hair_over_the_tolerance_are_skipped(self):
        # Written times, not derived ones, and tied to the tolerance's current
        # value: the first assert fails loudly if it moves and this pair stops
        # being the case the test is for.
        width = 2.003 - 1.003
        assert width > lab.MAX_REVERSED_SPAN
        assert lab.parse("2.003 1.003 N") == []

    def test_a_skipped_reversal_is_logged(self, caplog):
        with caplog.at_level(logging.WARNING, logger="songscribe.lab"):
            lab.parse("284.352 284.307 N", source="song.lab")
        assert "song.lab:1" in caplog.text
        assert "0.045" in caplog.text

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


class TestParseNotes:
    def test_reads_start_end_pitch(self):
        notes = lab.parse_notes("0.0\t0.5\t72\n0.5\t1.0\t74\n")
        assert notes == [NoteEvent(0.0, 0.5, 72), NoteEvent(0.5, 1.0, 74)]

    def test_a_missing_confidence_means_certain(self):
        assert lab.parse_notes("0.0 0.5 72")[0].confidence == lab.DEFAULT_CONFIDENCE

    def test_reads_an_optional_confidence(self):
        assert lab.parse_notes("0.0 0.5 72 0.25")[0].confidence == pytest.approx(0.25)

    def test_skips_blank_lines_and_comments(self):
        assert len(lab.parse_notes("# tracked\n\n0.0 0.5 72\n\n0.5 1.0 74\n")) == 2

    @pytest.mark.parametrize("line", ["0.0 0.5", "0.0 0.5 72 0.5 extra"])
    def test_rejects_the_wrong_number_of_fields(self, line):
        with pytest.raises(lab.LabFormatError, match="start end pitch"):
            lab.parse_notes(line)

    def test_rejects_a_fractional_pitch(self):
        # A tracker's f0 belongs in a .notes file as a note, not as a pitch
        # with a fraction left on it: rounding is the segmenter's decision.
        with pytest.raises(lab.LabFormatError, match="not a whole MIDI note"):
            lab.parse_notes("0.0 0.5 72.4")

    @pytest.mark.parametrize("pitch", ["-1", "128", "440"])
    def test_rejects_a_pitch_outside_midi(self, pitch):
        # 440 is the giveaway: a frequency written where a note number goes.
        with pytest.raises(lab.LabFormatError, match="outside MIDI"):
            lab.parse_notes(f"0.0 0.5 {pitch}")

    @pytest.mark.parametrize("line", ["nan 0.5 72", "0.0 inf 72", "0.0 0.5 nan"])
    def test_rejects_non_finite_numbers(self, line):
        with pytest.raises(lab.LabFormatError):
            lab.parse_notes(line)

    def test_rejects_a_reversed_note(self):
        with pytest.raises(lab.LabFormatError, match="precedes start"):
            lab.parse_notes("5.0 1.0 72")

    def test_skips_a_hairline_reversal(self, caplog):
        # The same tolerance as the chord reader's, for the same reason.
        with caplog.at_level(logging.WARNING):
            notes = lab.parse_notes("0.0 0.5 72\n1.0 0.999 74\n")
        assert [n.pitch for n in notes] == [72]
        assert "skipping note reversed" in caplog.text

    def test_error_names_the_line_number(self):
        with pytest.raises(lab.LabFormatError, match="line 3"):
            lab.parse_notes("0.0 0.5 72\n0.5 1.0 74\nbroken\n")

    def test_names_the_file_when_it_has_one(self):
        with pytest.raises(lab.LabFormatError, match=r"lead\.notes:1"):
            lab.parse_notes("broken", source="lead.notes")


class TestWriteNotes:
    def test_round_trips_through_a_file(self, tmp_path):
        line = [NoteEvent(0.0, 0.625, 72), NoteEvent(0.625, 1.25, 76)]
        path = tmp_path / "nested" / "song.notes"
        lab.write_notes(path, line)
        assert lab.read_notes(path) == line

    def test_a_certain_line_stays_three_columns(self):
        text = lab.format_notes([NoteEvent(0.0, 0.5, 72)])
        assert len(text.split("\n")[0].split()) == 3

    def test_a_tracked_line_keeps_its_confidence(self, tmp_path):
        tracked = [NoteEvent(0.0, 0.5, 72, 0.4), NoteEvent(0.5, 1.0, 74, 1.0)]
        path = tmp_path / "song.notes"
        lab.write_notes(path, tracked)
        assert [n.confidence for n in lab.read_notes(path)] == pytest.approx([0.4, 1.0])

    def test_empty_timeline_writes_nothing(self):
        assert lab.format_notes([]) == ""

    def test_ends_with_exactly_one_newline(self):
        text = lab.format_notes([NoteEvent(0.0, 0.5, 72)])
        assert text.endswith("\n")
        assert not text.endswith("\n\n")
