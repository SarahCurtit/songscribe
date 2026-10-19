from __future__ import annotations

import numpy as np
import pytest

from songscribe.demo import synthesise_performance
from songscribe.eval import chroma
from songscribe.types import ChordEvent

CHART = ("C", "Am", "Dm", "G7", "Em", "A7", "Dm", "G7")
JUNK = ("F#", "Bb", "Eb", "B", "Ab", "D", "F#", "Bb")
BAR = 2.0


def timeline(labels, start=0.0, length=BAR) -> list[ChordEvent]:
    return [
        ChordEvent(start + i * length, start + (i + 1) * length, label)
        for i, label in enumerate(labels)
    ]


def downbeats(bars: int, length=BAR) -> list[float]:
    return [i * length for i in range(bars + 1)]


def take(intro: int, choruses: int = 3, solo: bool = True) -> list[ChordEvent]:
    """Intro bars of unrelated chords, then choruses. The middle chorus has different chords."""
    labels = [JUNK[i % len(JUNK)] for i in range(intro)]
    for index in range(choruses):
        labels += list(JUNK if solo and index == 1 else CHART)
    return timeline(labels)


def anchor_of(events, bars, reference=None):
    reference = reference or timeline(CHART)
    templates = chroma.bar_templates(reference, len(reference))
    return chroma.anchor_bar(
        chroma.chroma_from_events(events), downbeats(bars), templates, min_bars=4
    )


class TestBarTemplates:
    def test_one_template_per_bar_weighted_by_time(self):
        reference = [ChordEvent(0, 3, "C"), ChordEvent(3, 4, "G"), ChordEvent(4, 8, "A:min")]
        first, second = chroma.bar_templates(reference, 2)
        assert first[0] == pytest.approx(3.0)  # C's root, 3 of the bar's 4 seconds
        assert first[7] == pytest.approx(1.0 + 3.0)  # G's root, and C's fifth
        assert second[9] == pytest.approx(4.0)  # A, the whole bar

    def test_no_chord_and_unreadable_labels_add_nothing(self):
        reference = [ChordEvent(0, 2, "N"), ChordEvent(2, 4, "not a chord")]
        assert not any(t.any() for t in chroma.bar_templates(reference, 2))

    def test_an_empty_reference_raises(self):
        with pytest.raises(ValueError):
            chroma.bar_templates([], 4)


class TestBarAgreement:
    def test_the_chart_agrees_with_itself(self):
        agreement = chroma.bar_agreement(
            chroma.chroma_from_events(timeline(CHART * 2)),
            downbeats(16),
            chroma.bar_templates(timeline(CHART), 8),
        )
        diagonal = [agreement[i, i % 8] for i in range(16)]
        assert min(diagonal) > 0.7

    def test_a_shuffled_chart_fits_worse_by_more_than_the_margin(self):
        events = timeline(CHART * 2)
        right = chroma.bar_templates(timeline(CHART), 8)
        shuffled = chroma.bar_templates(timeline([CHART[i] for i in (3, 6, 1, 4, 0, 7, 2, 5)]), 8)
        args = (chroma.chroma_from_events(events), downbeats(16))
        good = np.mean([chroma.bar_agreement(*args, right)[i, i % 8] for i in range(16)])
        bad = np.mean([chroma.bar_agreement(*args, shuffled)[i, i % 8] for i in range(16)])
        assert good - bad > chroma.ANCHOR_MIN_MARGIN

    def test_a_transposed_chart_fits_worse_still(self):
        events = timeline(CHART * 2)
        args = (chroma.chroma_from_events(events), downbeats(16))
        right = chroma.bar_templates(timeline(CHART), 8)
        # A tritone (6 semitones) up has almost no notes in common with these chords.
        up = ("F#", "D#m", "G#m", "C#7", "A#m", "D#7", "G#m", "C#7")
        moved = chroma.bar_templates(timeline(up), 8)
        good = np.mean([chroma.bar_agreement(*args, right)[i, i % 8] for i in range(16)])
        far = np.mean([chroma.bar_agreement(*args, moved)[i, i % 8] for i in range(16)])
        assert good - far > 2 * chroma.ANCHOR_MIN_MARGIN

    def test_a_flat_recording_agrees_with_nothing(self):
        flat = chroma.Chromagram(values=np.ones((12, 400)), hop=0.1)
        agreement = chroma.bar_agreement(
            flat, downbeats(8), chroma.bar_templates(timeline(CHART), 8)
        )
        assert np.allclose(agreement, 0.0)

    def test_bars_past_the_recording_score_zero(self):
        short = chroma.chroma_from_events(timeline(CHART), duration=16.0)
        agreement = chroma.bar_agreement(
            short, downbeats(12), chroma.bar_templates(timeline(CHART), 8)
        )
        assert np.allclose(agreement[9:], 0.0)


class TestAnchorBar:
    @pytest.mark.parametrize("intro", [0, 3, 7, 11])
    def test_it_finds_the_first_chorus_after_any_intro(self, intro):
        events = take(intro)
        anchor = anchor_of(events, intro + 24)
        assert anchor is not None
        assert anchor.bar_index == intro
        assert anchor.start == pytest.approx(intro * BAR)
        assert anchor.refusal() is None

    def test_forty_bars_of_junk_do_not_hide_the_form(self):
        anchor = anchor_of(take(40, solo=False), 40 + 24)
        assert anchor.bar_index == 40
        assert anchor.is_clear

    def test_a_reharmonised_first_chorus_is_passed_over(self):
        # The first chorus has different chords, so the anchor is the start of the next chorus.
        labels = list(JUNK) + list(CHART) * 2
        anchor = anchor_of(timeline(labels), 24)
        assert anchor.bar_index == 8

    def test_it_refuses_a_recording_that_is_not_the_chart(self):
        anchor = anchor_of(timeline(JUNK * 3), 24)
        assert anchor is not None
        assert anchor.refusal() is not None

    def test_it_refuses_a_chart_that_fits_every_phase_alike(self):
        # With one chord throughout, every start scores the same, so no start can be told apart.
        anchor = anchor_of(timeline(["C"] * 24), 24, reference=timeline(["C"] * 8))
        assert anchor.margin < chroma.ANCHOR_MIN_MARGIN
        assert "margin" in anchor.refusal() or "agreement" in anchor.refusal()

    def test_a_recording_shorter_than_a_chorus_has_nothing_to_fit(self):
        assert anchor_of(timeline(CHART[:3]), 3) is None

    def test_a_recording_that_is_only_one_phase_wide_is_refused(self):
        # There is only one possible start, so there is no other position to compare it with.
        anchor = anchor_of(timeline(CHART), 8)
        assert anchor is not None
        assert anchor.refusal() is not None


class TestChromagram:
    @pytest.mark.parametrize("pickup_beats", [0, 3])
    @pytest.mark.parametrize("chart", [CHART, (*CHART[:-1], CHART[0])], ids=["plain", "ends-on-c"])
    def test_audio_chroma_anchors_a_synthetic_take(self, chart, pickup_beats):
        # The second chart ends on its first chord, so the pickup (the last chord) sounds
        # like the start of a chorus.
        sample_rate = 11_025
        performance = synthesise_performance(
            progression=chart,
            bpm=240.0,
            repeats=3,
            sample_rate=sample_rate,
            pickup_beats=pickup_beats,
            tempo_drift=0.03,
            substitutions={1: {0: "Gm", 1: "Bb", 2: "Eb", 5: "C"}},
        )
        spectrum = chroma.chromagram(performance.audio, sample_rate)
        # Use the demo's exact bar lines, so no beat tracker is needed.
        grid = list(performance.downbeats)
        templates = chroma.bar_templates(performance.reference, 8)
        anchor = chroma.anchor_bar(spectrum, grid, templates, min_bars=4)
        assert anchor is not None
        assert anchor.bar_index == 0
        assert anchor.start == pytest.approx(grid[0])
        assert anchor.refusal() is None

    @pytest.mark.parametrize("last", ["G7", "C"])
    def test_a_tracker_that_counts_the_pickup_as_a_bar_does_not_shift_the_anchor(self, last):
        sample_rate = 11_025
        performance = synthesise_performance(
            progression=(*CHART[:-1], last),
            bpm=240.0,
            repeats=3,
            sample_rate=sample_rate,
            pickup_beats=3,
            substitutions={1: {0: "Gm", 1: "Bb", 2: "Eb", 5: "C"}},
        )
        # A short bar line at the start of the pickup.
        full = [0.0, *performance.downbeats]
        spectrum = chroma.chromagram(performance.audio, sample_rate)
        templates = chroma.bar_templates(performance.reference, 8)
        anchor = chroma.anchor_bar(spectrum, full, templates, min_bars=4)
        assert anchor is not None
        assert anchor.start == pytest.approx(performance.downbeats[0])
        assert anchor.refusal() is None
