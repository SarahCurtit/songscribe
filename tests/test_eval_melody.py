"""Melody scoring: bars and beats, octaves, transposition, heads against solos.

Every test here is on invented notes -- no audio, no tracker -- which is the
point of keeping the metric and the backend apart.
"""

from __future__ import annotations

from math import isnan

import numpy as np
import pytest

from songscribe.beats import BeatGrid
from songscribe.eval import form
from songscribe.eval.melody import (
    MELODY_ONSET_TOLERANCE_BEATS,
    ChorusMelody,
    MelodyScore,
    best_transposition,
    head_score,
    matching_transposition,
    melody_score,
    melody_sequence,
    novelty,
    raw_pitch_accuracy,
    score_choruses,
)
from songscribe.melody import midi_to_hz
from songscribe.types import NoteEvent

BAR = 2.0
METRE = 4
#: One note on each bar line of a four-bar chart: the written melody.
CHART_PITCHES = (60, 62, 64, 65)
CHART_BARS = tuple(bar * BAR for bar in range(len(CHART_PITCHES) + 1))


def line(pitches, bar: float = BAR, start: float = 0.0, step: float | None = None):
    """One note per bar (or per ``step`` seconds) from ``start``."""
    step = bar if step is None else step
    return [
        NoteEvent(start + index * step, start + index * step + step / 2, pitch)
        for index, pitch in enumerate(pitches)
    ]


CHART = line(CHART_PITCHES)


class TestMelodySequence:
    def test_onsets_land_on_their_bar_lines(self):
        placed = melody_sequence(CHART, CHART_BARS, METRE)
        assert [beat for beat, _ in placed] == [0.0, 4.0, 8.0, 12.0]
        assert [pitch for _, pitch in placed] == list(CHART_PITCHES)

    def test_a_note_halfway_through_a_bar_is_on_the_middle_beat(self):
        placed = melody_sequence([NoteEvent(1.0, 1.5, 60)], CHART_BARS, METRE)
        assert placed == ((2.0, 60),)

    def test_a_faster_chorus_lands_on_the_same_beats(self):
        # The same line played in bars of 1.2 s rather than 2 s: bars and beats
        # do not notice, which is the whole reason the metric works in them.
        quick_bars = tuple(bar * 1.2 for bar in range(len(CHART_PITCHES) + 1))
        quick = line(CHART_PITCHES, bar=1.2)
        assert melody_sequence(quick, quick_bars, METRE) == melody_sequence(
            CHART, CHART_BARS, METRE
        )

    def test_notes_outside_the_grid_are_dropped(self):
        notes = [NoteEvent(-1.0, -0.5, 59), *CHART, NoteEvent(100.0, 100.5, 72)]
        assert len(melody_sequence(notes, CHART_BARS, METRE)) == len(CHART)

    def test_a_note_on_the_last_bar_line_is_outside_the_form(self):
        assert melody_sequence([NoteEvent(CHART_BARS[-1], 9.0, 60)], CHART_BARS, METRE) == ()

    def test_the_result_is_in_time_order(self):
        placed = melody_sequence(list(reversed(CHART)), CHART_BARS, METRE)
        assert [beat for beat, _ in placed] == sorted(beat for beat, _ in placed)

    def test_an_unusable_grid_is_refused(self):
        with pytest.raises(ValueError, match="strictly ascending"):
            melody_sequence(CHART, (0.0, 2.0, 1.0), METRE)
        with pytest.raises(ValueError, match="beats_per_bar"):
            melody_sequence(CHART, CHART_BARS, 0)

    def test_one_bar_line_places_nothing(self):
        assert melody_sequence(CHART, (0.0,), METRE) == ()


class TestMelodyScore:
    def test_the_same_line_scores_one(self):
        scored = melody_score(CHART, CHART, CHART_BARS, METRE)
        assert (scored.matched, scored.reference_notes, scored.estimate_notes) == (4, 4, 4)
        assert scored.recall == scored.precision == scored.f1 == 1.0

    def test_an_onset_inside_the_tolerance_still_matches(self):
        late = 0.9 * MELODY_ONSET_TOLERANCE_BEATS * BAR / METRE
        moved = [NoteEvent(n.start + late, n.end + late, n.pitch) for n in CHART]
        assert melody_score(CHART, moved, CHART_BARS, METRE).matched == 4

    def test_an_onset_past_the_tolerance_does_not(self):
        late = 1.1 * MELODY_ONSET_TOLERANCE_BEATS * BAR / METRE
        moved = [NoteEvent(n.start + late, n.end + late, n.pitch) for n in CHART]
        assert melody_score(CHART, moved, CHART_BARS, METRE).matched == 0

    def test_a_wrong_pitch_on_the_right_beat_does_not_match(self):
        wrong = [NoteEvent(n.start, n.end, n.pitch + 1) for n in CHART]
        assert melody_score(CHART, wrong, CHART_BARS, METRE, transposition=0).matched == 0

    def test_the_octave_is_ignored_by_default(self):
        down = [NoteEvent(n.start, n.end, n.pitch - 12) for n in CHART]
        assert melody_score(CHART, down, CHART_BARS, METRE).matched == 4
        strict = melody_score(CHART, down, CHART_BARS, METRE, octave_blind=False)
        assert strict.matched == 0
        assert strict.octave_blind is False

    def test_a_played_note_matches_at_most_one_written_note(self):
        # A tremolo on one pitch must not match the whole chart.
        tremolo = [NoteEvent(0.0 + k * 0.05, 0.05 + k * 0.05, 60) for k in range(8)]
        scored = melody_score(CHART, tremolo, CHART_BARS, METRE, transposition=0)
        assert scored.matched == 1
        assert scored.precision == pytest.approx(1 / 8)
        assert scored.recall == pytest.approx(1 / 4)

    def test_a_missing_note_costs_recall_not_precision(self):
        scored = melody_score(CHART, CHART[:2], CHART_BARS, METRE, transposition=0)
        assert scored.recall == 0.5
        assert scored.precision == 1.0
        assert scored.f1 == pytest.approx(2 / 3)

    def test_nothing_played_is_nan_rather_than_zero(self):
        scored = melody_score(CHART, [], CHART_BARS, METRE, transposition=0)
        assert scored.recall == 0.0
        assert isnan(scored.precision)
        assert isnan(scored.f1)

    def test_nothing_written_is_nan(self):
        assert isnan(melody_score([], CHART, CHART_BARS, METRE, transposition=0).recall)

    def test_a_negative_tolerance_is_refused(self):
        with pytest.raises(ValueError, match="tolerance_beats"):
            melody_score(CHART, CHART, CHART_BARS, METRE, tolerance_beats=-1)

    def test_the_transposition_used_is_reported(self):
        up = [NoteEvent(n.start, n.end, n.pitch + 3) for n in CHART]
        scored = melody_score(CHART, up, CHART_BARS, METRE, octave_blind=False)
        assert scored.transposition == 3
        assert scored.matched == 4


class TestBestTransposition:
    def test_the_same_key_is_no_shift(self):
        assert best_transposition(CHART, CHART) == 0

    def test_a_transposed_line_is_found(self):
        for shift in (-5, -2, 3, 6):
            up = [NoteEvent(n.start, n.end, n.pitch + shift) for n in CHART]
            assert best_transposition(CHART, up) == shift

    def test_the_octave_is_not_a_transposition(self):
        up = [NoteEvent(n.start, n.end, n.pitch + 12) for n in CHART]
        assert best_transposition(CHART, up) == 0

    def test_it_is_duration_weighted_not_note_counted(self):
        # One long tonic against three passing notes: pitch content is time,
        # not a note count, or a flurry of quavers would pick the key. A fifth
        # up is a fourth down -- pitch content names a rotation, not a
        # direction -- and the candidate range spells it as the nearer move.
        written = [NoteEvent(0.0, 8.0, 60), NoteEvent(8.0, 8.2, 61), NoteEvent(8.2, 8.4, 62)]
        played = [NoteEvent(0.0, 8.0, 67), NoteEvent(8.0, 8.2, 61), NoteEvent(8.2, 8.4, 62)]
        assert best_transposition(written, played) == -5

    def test_a_tie_goes_to_no_shift_rather_than_to_float_noise(self):
        # Half the take in the chart's key and half a minor third up: 0 and 3 fit the
        # content equally, and they differ by about 1e-16, which must not pick the key.
        played = [*CHART, *[NoteEvent(n.start + 8.0, n.end + 8.0, n.pitch + 3) for n in CHART]]
        assert best_transposition(CHART, played) == 0

    def test_an_empty_side_means_no_shift(self):
        assert best_transposition(CHART, []) == 0
        assert best_transposition([], CHART) == 0

    def test_it_ignores_the_match_count(self):
        # The played line is the chart a fourth up, but every onset is wrong:
        # pitch content still names the key, which is the point -- the shift
        # must not be chosen by what scores best.
        up = [NoteEvent(n.start + 0.9, n.end + 0.9, n.pitch + 5) for n in CHART]
        assert best_transposition(CHART, up) == 5
        assert melody_score(CHART, up, CHART_BARS, METRE).matched == 0


class TestMatchingTransposition:
    def test_it_finds_the_shift_that_matches_most(self):
        up = [NoteEvent(n.start, n.end, n.pitch + 2) for n in CHART]
        assert matching_transposition(CHART, up, CHART_BARS, METRE, octave_blind=False) == 2

    def test_it_can_disagree_with_the_pitch_content(self):
        # Two of the four notes a tone up, on the beat; the rest a fourth up,
        # off the beat. Content says a fourth, matching says a tone -- a
        # disagreement worth printing, and the reason the score uses content.
        played = [
            NoteEvent(0.0, 1.0, CHART_PITCHES[0] + 2),
            NoteEvent(BAR, BAR + 1.0, CHART_PITCHES[1] + 2),
            NoteEvent(2 * BAR + 0.9, 2 * BAR + 1.9, CHART_PITCHES[2] + 5),
            NoteEvent(3 * BAR + 0.9, 3 * BAR + 1.9, CHART_PITCHES[3] + 5),
        ]
        content = best_transposition(CHART, played)
        matching = matching_transposition(CHART, played, CHART_BARS, METRE, octave_blind=False)
        assert matching == 2
        assert content != matching


class TestRawPitchAccuracy:
    def test_an_exact_contour_scores_one(self):
        f0 = midi_to_hz(np.array([60.0, 62.0, 64.0, 65.0]))
        assert raw_pitch_accuracy(f0, f0) == 1.0

    def test_within_half_a_semitone_counts(self):
        written = midi_to_hz(np.full(4, 60.0))
        assert raw_pitch_accuracy(written, midi_to_hz(np.full(4, 60.49))) == 1.0
        assert raw_pitch_accuracy(written, midi_to_hz(np.full(4, 60.51))) == 0.0

    def test_unvoiced_reference_frames_are_not_counted(self):
        written = midi_to_hz(np.array([60.0, 60.0, 60.0, 60.0]))
        written[2:] = np.nan
        played = midi_to_hz(np.full(4, 60.0))
        assert raw_pitch_accuracy(written, played) == 1.0

    def test_an_unvoiced_estimate_frame_is_a_miss(self):
        written = midi_to_hz(np.full(4, 60.0))
        played = midi_to_hz(np.full(4, 60.0))
        played[:2] = np.nan
        assert raw_pitch_accuracy(written, played) == 0.5

    def test_zero_hertz_is_unvoiced_too(self):
        written = midi_to_hz(np.full(4, 60.0))
        played = np.array([0.0, 0.0, *midi_to_hz(np.full(2, 60.0))])
        assert raw_pitch_accuracy(written, played) == 0.5

    def test_an_octave_error_is_wrong_unless_asked_to_ignore_it(self):
        written = midi_to_hz(np.full(4, 60.0))
        played = midi_to_hz(np.full(4, 48.0))
        assert raw_pitch_accuracy(written, played) == 0.0
        assert raw_pitch_accuracy(written, played, octave_blind=True) == 1.0

    def test_folding_the_octave_away_still_measures_the_cents(self):
        # An octave down and 40 cents flat is 40 cents out, not 1240; an octave
        # down and 60 cents flat is still a miss.
        written = midi_to_hz(np.full(4, 60.0))
        assert raw_pitch_accuracy(written, midi_to_hz(np.full(4, 47.6)), octave_blind=True) == 1.0
        assert raw_pitch_accuracy(written, midi_to_hz(np.full(4, 47.4)), octave_blind=True) == 0.0

    def test_an_all_unvoiced_reference_is_nan(self):
        assert isnan(raw_pitch_accuracy(np.full(4, np.nan), midi_to_hz(np.full(4, 60.0))))

    def test_different_lengths_are_refused(self):
        with pytest.raises(ValueError, match="same hop"):
            raw_pitch_accuracy(np.full(4, 440.0), np.full(5, 440.0))

    def test_a_nonsense_tolerance_is_refused(self):
        with pytest.raises(ValueError, match="tolerance_cents"):
            raw_pitch_accuracy(np.full(4, 440.0), np.full(4, 440.0), tolerance_cents=0)


class TestNovelty:
    def test_playing_the_head_is_no_novelty(self):
        assert novelty(CHART, CHART, CHART_BARS, METRE) == 0.0

    def test_playing_none_of_it_is_total_novelty(self):
        away = [NoteEvent(n.start + 0.9, n.end + 0.9, n.pitch + 1) for n in CHART]
        assert novelty(CHART, away, CHART_BARS, METRE, transposition=0) == 1.0

    def test_half_the_head_is_half_the_novelty(self):
        assert novelty(CHART, CHART[:2], CHART_BARS, METRE, transposition=0) == 0.5

    def test_nothing_written_is_nan(self):
        assert isnan(novelty([], CHART, CHART_BARS, METRE, transposition=0))


def _performance(repeats: int, bars_per_chorus: int = 4):
    """A grid of ``repeats`` choruses, and the blocks ``form`` cuts it into."""
    bars = repeats * bars_per_chorus
    grid = BeatGrid(tuple(bar * BAR for bar in range(bars + 1)), METRE)
    blocks = form.choruses(grid, bars_per_chorus, 0, bars_per_chorus * BAR)
    return grid, blocks


class TestScoreChoruses:
    def test_the_heads_are_scored_and_the_solo_is_not(self):
        _, blocks = _performance(3)
        kinds = [b.kind for b in blocks]
        assert kinds == ["head", "chorus", "head"]
        played = [
            NoteEvent(n.start + chorus * 4 * BAR, n.end + chorus * 4 * BAR, n.pitch)
            for chorus in range(3)
            for n in CHART
        ]
        results = score_choruses(CHART, played, blocks, CHART_BARS, METRE, transposition=0)
        assert [r.is_head for r in results] == [True, False, True]
        assert [r.score.recall for r in results if r.is_head] == [1.0, 1.0]
        # The demo-style take plays the head every chorus, so the solo's
        # novelty is 0 -- the number is descriptive, not an accuracy.
        assert results[1].novelty == 0.0

    def test_an_intro_is_left_out(self):
        bars = 2 * 4 + 2
        grid = BeatGrid(tuple(bar * BAR for bar in range(bars + 1)), METRE)
        blocks = form.choruses(grid, 4, 2, 4 * BAR)
        assert blocks[0].kind == "intro"
        played = [NoteEvent(n.start + 2 * BAR, n.end + 2 * BAR, n.pitch) for n in CHART]
        results = score_choruses(CHART, played, blocks, CHART_BARS, METRE, transposition=0)
        assert [r.chorus.kind for r in results] == ["head", "head"]
        assert results[0].score.recall == 1.0

    def test_a_note_is_scored_in_the_chorus_it_starts_in(self):
        # A note held over the chorus line belongs to the chorus it began in.
        # Cutting it and re-onsetting it at the next chorus's start would
        # invent an onset on beat 1 that nobody played.
        _, blocks = _performance(2)
        spilling = [NoteEvent(3 * BAR + 1.0, 4 * BAR + 1.0, CHART_PITCHES[0])]
        results = score_choruses(CHART, spilling, blocks, CHART_BARS, METRE, transposition=0)
        assert results[0].score.estimate_notes == 1
        assert results[1].score.estimate_notes == 0

    def test_a_chorus_played_faster_is_mapped_back_onto_the_chart(self):
        # The second chorus 5% quicker: by its last bar that is 0.6 of a beat
        # of drift, far past the onset tolerance, so this only scores because
        # the block is projected onto the chart. More than ~10% and `form`
        # calls the blocks unaligned and declines to score them at all.
        quick = BAR * 0.95
        downbeats = [bar * BAR for bar in range(5)]
        downbeats += [downbeats[-1] + (bar + 1) * quick for bar in range(4)]
        grid = BeatGrid(tuple(downbeats), METRE)
        blocks = form.choruses(grid, 4, 0, 4 * BAR)
        assert [b.kind for b in blocks] == ["head", "head"]
        played = [
            *CHART,
            *[
                NoteEvent(downbeats[4] + index * quick, downbeats[4] + index * quick + 0.5, pitch)
                for index, pitch in enumerate(CHART_PITCHES)
            ],
        ]
        results = score_choruses(CHART, played, blocks, CHART_BARS, METRE, transposition=0)
        assert [r.score.recall for r in results] == [1.0, 1.0]

    def test_one_key_is_chosen_for_the_whole_take(self):
        # A key fitted per block would give these two heads different shifts, and
        # head_score would then refuse to add them.
        _, blocks = _performance(2)
        played = [*CHART, *[NoteEvent(n.start + 8.0, n.end + 8.0, n.pitch + 3) for n in CHART]]
        results = score_choruses(CHART, played, blocks, CHART_BARS, METRE)
        assert len({r.score.transposition for r in results}) == 1
        assert head_score(results) is not None

    def test_a_partial_block_is_scored_against_the_bars_it_reached(self):
        # Four bars of an eight-bar chart, played as written: novelty 0, not 0.5.
        eight = line(range(60, 68))
        eight_bars = tuple(bar * BAR for bar in range(9))
        grid = BeatGrid(tuple(bar * BAR for bar in range(8 + 4 + 1)), METRE)
        blocks = form.choruses(grid, 8, 0, 8 * BAR)
        assert [(b.kind, b.bars) for b in blocks] == [("head", 8), ("partial", 4)]
        played = [*eight, *[NoteEvent(n.start + 16.0, n.end + 16.0, n.pitch) for n in eight[:4]]]
        results = score_choruses(eight, played, blocks, eight_bars, METRE, transposition=0)
        assert results[1].chorus.kind == "partial"
        assert results[1].novelty == 0.0

    def test_a_partial_block_that_played_nothing_written_is_all_novelty(self):
        eight = line(range(60, 68))
        eight_bars = tuple(bar * BAR for bar in range(9))
        grid = BeatGrid(tuple(bar * BAR for bar in range(8 + 4 + 1)), METRE)
        blocks = form.choruses(grid, 8, 0, 8 * BAR)
        played = [NoteEvent(16.0 + i * BAR, 16.5 + i * BAR, 71) for i in range(4)]
        results = score_choruses(eight, played, blocks, eight_bars, METRE, transposition=0)
        assert results[-1].novelty == 1.0

    def test_nothing_scored_gives_nothing_back(self):
        grid = BeatGrid((0.0, BAR), METRE)
        blocks = form.choruses(grid, 4, 0, 4 * BAR)
        assert all(not b.is_scored for b in blocks)
        assert score_choruses(CHART, CHART, blocks, CHART_BARS, METRE) == ()


class TestHeadScore:
    def test_the_heads_are_summed_from_their_counts(self):
        blocks = _performance(3)[1]
        results = (
            ChorusMelody(blocks[0], MelodyScore(3, 4, 5, 0, True)),
            ChorusMelody(blocks[1], novelty=0.5),
            ChorusMelody(blocks[2], MelodyScore(1, 4, 3, 0, True)),
        )
        total = head_score(results)
        assert (total.matched, total.reference_notes, total.estimate_notes) == (4, 8, 8)
        assert total.recall == 0.5

    def test_without_a_head_there_is_no_score(self):
        blocks = _performance(3)[1]
        assert head_score((ChorusMelody(blocks[1], novelty=0.5),)) is None

    def test_heads_scored_differently_are_not_added(self):
        blocks = _performance(3)[1]
        results = (
            ChorusMelody(blocks[0], MelodyScore(3, 4, 5, 0, True)),
            ChorusMelody(blocks[2], MelodyScore(1, 4, 3, 2, True)),
        )
        with pytest.raises(ValueError, match="different settings"):
            head_score(results)
