"""Melody scoring: the lead line as played against the lead sheet's.

Four rules, with the reasoning in ``EVALUATION.md``'s *Melody* section:

- Onsets are placed on the bar grid and the tolerance is in beats, so a different tempo
  and any time offset cost nothing, as with the ``chart`` and ``grid`` chord metrics.
- Pitches compare by pitch class by default; the strict score is reported beside that one
  rather than folded into a penalty.
- The transposition comes from pitch content (:func:`best_transposition`). The shift that
  matches the most notes is a diagnostic (:func:`matching_transposition`), never the one
  scored with.
- Only ``head`` blocks are scored. Any other scored block is reported as
  :func:`novelty`, which is descriptive rather than an accuracy.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import isnan, nan

import numpy as np

from ..melody import CENTS_PER_SEMITONE, pitch_profile, transpose
from ..types import DEFAULT_METRE, NoteEvent
from . import form
from .chart_fit import ANY_KEY

#: How far an onset may sit from the written one and still be the same note, in beats.
#: A sixteenth at the bar level: tighter would fail a swung eighth, looser would start
#: matching the next beat's note.
MELODY_ONSET_TOLERANCE_BEATS = 0.25

#: Half a semitone, the MIREX standard for raw pitch accuracy.
MELODY_PITCH_TOLERANCE_CENTS = 50.0

#: Decimal places two key fits are compared at. The profiles are unit vectors, so a
#: difference below this is float noise, and noise must not be what picks a key: two
#: keys the pitch content really does support equally have to reach the tie-break.
TRANSPOSITION_FIT_PRECISION = 12

SEMITONES = 12
CENTS_PER_OCTAVE = SEMITONES * CENTS_PER_SEMITONE


@dataclass(frozen=True, slots=True)
class MelodyScore:
    """How much of the written line was played, and how much of the playing was written.

    Counts rather than ratios, so a corpus total can be summed from them -- the same
    reason :class:`~songscribe.eval.metrics.Score` keeps its durations.
    """

    matched: int
    reference_notes: int
    estimate_notes: int
    transposition: int
    octave_blind: bool

    @property
    def recall(self) -> float:
        """Fraction of written notes that were played; ``nan`` with nothing written."""
        return self.matched / self.reference_notes if self.reference_notes else nan

    @property
    def precision(self) -> float:
        """Fraction of played notes that are in the chart; ``nan`` with nothing played."""
        return self.matched / self.estimate_notes if self.estimate_notes else nan

    @property
    def f1(self) -> float:
        """Harmonic mean of the two; ``nan`` if either is ``nan``."""
        precision, recall = self.precision, self.recall
        if isnan(precision) or isnan(recall) or precision + recall == 0:
            return nan
        return 2 * precision * recall / (precision + recall)


# --------------------------------------------------------------------------- #
# Onto the bar grid
# --------------------------------------------------------------------------- #


def melody_sequence(
    notes: Sequence[NoteEvent],
    downbeats: Sequence[float],
    beats_per_bar: int = DEFAULT_METRE,
) -> tuple[tuple[float, int], ...]:
    """``(beat, pitch)`` for each note, beats counted from the first downbeat.

    A bar is ``beats_per_bar`` beats however long it lasted, so a chorus played faster
    than the chart lands on the same beats. Notes starting outside the grid are dropped:
    what sounds before bar 1 or after the last bar line is not in the form. Only the
    onset is placed -- a player holds or clips a note at will, so its length says little
    about whether it is the written one.
    """
    if beats_per_bar < 1:
        raise ValueError(f"beats_per_bar must be at least 1, got {beats_per_bar}")
    bars = list(downbeats)
    if any(b <= a for a, b in pairwise(bars)):
        raise ValueError("downbeats must be strictly ascending")
    if len(bars) < 2:
        return ()

    placed: list[tuple[float, int]] = []
    for note in notes:
        index = bisect_right(bars, note.start) - 1
        if not 0 <= index < len(bars) - 1:
            continue
        low, high = bars[index], bars[index + 1]
        beat = (index + (note.start - low) / (high - low)) * beats_per_bar
        placed.append((beat, note.pitch))
    return tuple(sorted(placed))


def best_transposition(
    reference: Sequence[NoteEvent],
    estimate: Sequence[NoteEvent],
    shifts: Sequence[int] = ANY_KEY,
) -> int:
    """Semitones the played line sounds above the written one, from pitch content alone.

    The duration-weighted pitch-class profile of each side, correlated at each candidate
    shift: which key the playing is *in*, not which shift scores best, so the number
    stays independent of what it is used to measure. A tie goes to the smallest move and
    to 0 above all, since a tie means the content does not distinguish the keys. Returns
    0 when either side is empty.
    """
    if not shifts:
        raise ValueError("shifts is empty: there is no transposition to choose")
    written, played = pitch_profile(reference), pitch_profile(estimate)
    if not written.any() or not played.any():
        return 0
    # Normalised, so the correlation compares shapes rather than how much was played.
    written = written / np.linalg.norm(written)
    played = played / np.linalg.norm(played)
    fit = {
        shift: round(
            float(np.roll(written, shift % SEMITONES) @ played), TRANSPOSITION_FIT_PRECISION
        )
        for shift in shifts
    }
    return min(fit, key=lambda shift: (-fit[shift], abs(shift), shift))


def matching_transposition(
    reference: Sequence[NoteEvent],
    estimate: Sequence[NoteEvent],
    downbeats: Sequence[float],
    beats_per_bar: int = DEFAULT_METRE,
    *,
    octave_blind: bool = True,
    tolerance_beats: float = MELODY_ONSET_TOLERANCE_BEATS,
    shifts: Sequence[int] = ANY_KEY,
) -> int:
    """The shift that matches the most notes: a **diagnostic**, never the scored one.

    Report it beside :func:`best_transposition`; a disagreement usually means the lead was
    mistracked, or that this is not the tune the chart says it is. Do not average the two,
    and do not score with this one -- a key chosen to maximise a score makes the score
    partly a measure of itself.
    """
    if not shifts:
        raise ValueError("shifts is empty: there is no transposition to choose")
    matched = {
        shift: melody_score(
            reference,
            estimate,
            downbeats,
            beats_per_bar,
            transposition=shift,
            octave_blind=octave_blind,
            tolerance_beats=tolerance_beats,
        ).matched
        for shift in shifts
    }
    return min(matched, key=lambda shift: (-matched[shift], abs(shift), shift))


def melody_score(
    reference: Sequence[NoteEvent],
    estimate: Sequence[NoteEvent],
    downbeats: Sequence[float],
    beats_per_bar: int = DEFAULT_METRE,
    *,
    transposition: int | None = None,
    octave_blind: bool = True,
    tolerance_beats: float = MELODY_ONSET_TOLERANCE_BEATS,
) -> MelodyScore:
    """Match the played line against the written one on one bar grid.

    Both sides are placed on ``downbeats`` and must already share a timeline;
    ``form.project_estimate`` puts a chorus of a recording there. The written line moves up
    by ``transposition``, from :func:`best_transposition` when not given. Notes match on
    pitch -- by pitch class if ``octave_blind`` -- within ``tolerance_beats``; each written
    note claims the nearest unclaimed played one, so a tremolo cannot match a whole chart.
    """
    if tolerance_beats < 0:
        raise ValueError(f"tolerance_beats must not be negative, got {tolerance_beats}")
    shift = best_transposition(reference, estimate) if transposition is None else transposition

    written = melody_sequence(transpose(reference, shift), downbeats, beats_per_bar)
    played = melody_sequence(estimate, downbeats, beats_per_bar)

    taken = [False] * len(played)
    matched = 0
    for beat, pitch in written:
        best, distance = None, nan
        for index, (other_beat, other_pitch) in enumerate(played):
            if taken[index] or not _same_pitch(pitch, other_pitch, octave_blind):
                continue
            gap = abs(other_beat - beat)
            # Sorted by beat, so once a candidate is past the window the rest are too.
            if other_beat > beat + tolerance_beats:
                break
            # Strictly closer, so equally close candidates go to the earlier note.
            if gap <= tolerance_beats and (best is None or gap < distance):
                best, distance = index, gap
        if best is not None:
            taken[best] = True
            matched += 1

    return MelodyScore(
        matched=matched,
        reference_notes=len(written),
        estimate_notes=len(played),
        transposition=shift,
        octave_blind=octave_blind,
    )


def _same_pitch(written: int, played: int, octave_blind: bool) -> bool:
    return (written - played) % SEMITONES == 0 if octave_blind else written == played


# --------------------------------------------------------------------------- #
# The tracker on its own
# --------------------------------------------------------------------------- #


def raw_pitch_accuracy(
    reference_f0: Sequence[float] | np.ndarray,
    estimate_f0: Sequence[float] | np.ndarray,
    *,
    tolerance_cents: float = MELODY_PITCH_TOLERANCE_CENTS,
    octave_blind: bool = False,
) -> float:
    """Fraction of voiced reference frames whose pitch the estimate got right.

    MIREX's raw pitch accuracy, and with ``octave_blind`` its raw chroma accuracy. Both
    sides are frequencies in hertz, one per frame, on the same hop and the same length; a
    frame with no pitch is ``nan`` or zero. It measures the tracker alone, before any note
    segmentation -- without it a tracker that wanders and a segmenter that chops are one
    number.
    """
    if tolerance_cents <= 0:
        raise ValueError(f"tolerance_cents must be positive, got {tolerance_cents}")
    written = np.asarray(reference_f0, dtype=float)
    played = np.asarray(estimate_f0, dtype=float)
    if written.shape != played.shape:
        raise ValueError(
            f"both sides need one frame each on the same hop, got {written.shape} "
            f"and {played.shape}"
        )

    voiced = np.isfinite(written) & (written > 0)
    if not voiced.any():
        return nan
    comparable = voiced & np.isfinite(played) & (played > 0)
    cents = np.zeros(written.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        cents[comparable] = CENTS_PER_OCTAVE * np.log2(played[comparable] / written[comparable])
    if octave_blind:
        # Fold to the nearest octave: -1150 cents is 50 cents flat, an octave down.
        cents = (cents + CENTS_PER_OCTAVE / 2) % CENTS_PER_OCTAVE - CENTS_PER_OCTAVE / 2
    return float((comparable & (np.abs(cents) <= tolerance_cents)).sum() / voiced.sum())


# --------------------------------------------------------------------------- #
# Over a whole recording
# --------------------------------------------------------------------------- #


def novelty(
    reference: Sequence[NoteEvent],
    estimate: Sequence[NoteEvent],
    downbeats: Sequence[float],
    beats_per_bar: int = DEFAULT_METRE,
    *,
    transposition: int | None = None,
    octave_blind: bool = True,
    tolerance_beats: float = MELODY_ONSET_TOLERANCE_BEATS,
) -> float:
    """How little of the written melody this chorus played: ``1 - recall``.

    **Descriptive, not an accuracy.** On an improvised chorus a high number is what a
    working soloist produces; a low one says the player stayed on the head -- or that the
    block was mislabelled a solo. Never average it with a head's score, and never report it
    without the word "novelty" attached: 0.18 recall on a solo is not 18% accuracy.
    """
    return _novelty_of(
        melody_score(
            reference,
            estimate,
            downbeats,
            beats_per_bar,
            transposition=transposition,
            octave_blind=octave_blind,
            tolerance_beats=tolerance_beats,
        )
    )


def _novelty_of(scored: MelodyScore) -> float:
    return nan if isnan(scored.recall) else 1.0 - scored.recall


@dataclass(frozen=True, slots=True)
class ChorusMelody:
    """What the melody branch has to say about one scored block of a recording.

    Exactly one of the two is set: ``score`` on a head, ``novelty`` on anything else that
    was scored. They are not the same measurement and must not be averaged together.
    """

    chorus: form.Chorus
    score: MelodyScore | None = None
    novelty: float = nan

    @property
    def is_head(self) -> bool:
        return self.score is not None


def score_choruses(
    reference: Sequence[NoteEvent],
    estimate: Sequence[NoteEvent],
    blocks: Sequence[form.Chorus],
    downbeats: Sequence[float],
    beats_per_bar: int = DEFAULT_METRE,
    *,
    transposition: int | None = None,
    octave_blind: bool = True,
    tolerance_beats: float = MELODY_ONSET_TOLERANCE_BEATS,
) -> tuple[ChorusMelody, ...]:
    """Score each block of a recording against the chart's melody.

    ``reference`` is one chorus of the written line on the chart's timeline, whose bar lines
    are ``downbeats``; ``estimate`` is the whole recording's notes. Heads get a
    :class:`MelodyScore`, every other scored block a :func:`novelty`; intros, tails and
    unaligned blocks are left out, as they are for chords. A short block is scored against
    the bars it reached. One key for the whole take, as on the chord side.
    """
    origin = downbeats[0] if len(downbeats) else 0.0
    bars_per_chorus = max(len(downbeats) - 1, 1)
    # Resolved once over the whole take, not per block: a key fitted to each block would
    # give two heads two different keys, which is not a thing a band does, and would read
    # a solo in a key of its own.
    shift = best_transposition(reference, estimate) if transposition is None else transposition
    results: list[ChorusMelody] = []
    for block in blocks:
        if not block.is_scored:
            continue
        # By onset, before projection: a note spilling over a chorus line would otherwise
        # be cut and re-onset at the block's start, an onset nobody played.
        inside = [n for n in estimate if block.start <= n.start < block.end]
        played = form.project_estimate(inside, block, origin)
        # A partial block only reached part of the chart, so only that part is its
        # reference. The chart's own bar lines are to hand here, so the cut is a bar line
        # rather than form.truncate_reference's fraction of the extent.
        written = (
            reference
            if block.bars >= bars_per_chorus
            else [n for n in reference if n.start < downbeats[block.bars]]
        )
        scored = melody_score(
            written,
            played,
            downbeats,
            beats_per_bar,
            transposition=shift,
            octave_blind=octave_blind,
            tolerance_beats=tolerance_beats,
        )
        head = block.kind == "head"
        results.append(
            ChorusMelody(
                chorus=block,
                score=scored if head else None,
                novelty=nan if head else _novelty_of(scored),
            )
        )
    return tuple(results)


def head_score(results: Sequence[ChorusMelody]) -> MelodyScore | None:
    """The heads' scores added together, or ``None`` if no head was scored.

    Summed from the counts rather than averaged from the per-chorus f1s: two heads of
    different lengths are not two equal votes.
    """
    heads = [r.score for r in results if r.score is not None]
    if not heads:
        return None
    first = heads[0]
    if len({(h.transposition, h.octave_blind) for h in heads}) > 1:
        raise ValueError("heads scored under different settings cannot be added together")
    return MelodyScore(
        matched=sum(h.matched for h in heads),
        reference_notes=sum(h.reference_notes for h in heads),
        estimate_notes=sum(h.estimate_notes for h in heads),
        transposition=first.transposition,
        octave_blind=first.octave_blind,
    )
