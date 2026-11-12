"""Melody scoring: the lead line as played against the lead sheet's.

The comparison is in **bars and beats**, never seconds: every onset is placed on the bar
grid Stage 1 already produces, so the score survives a different tempo, a rubato head and
any time offset -- the same reasons the chord branch's ``chart`` and ``grid`` metrics work
in bars. The tolerance is therefore in beats too.

It is **octave-blind by default**: the recording's instrument may sit an octave from the
written range, and a pitch tracker's own commonest error is the octave. The strict number
is reported beside it rather than folded into a penalty, so the two causes stay separable.

Transposition comes from *pitch content* -- which twelve-tone rotation of the written line
the played one looks like -- and never from whichever shift happens to score best. The
match-best shift is available as a diagnostic (:func:`matching_transposition`) and is worth
printing next to the chosen one, but fitting the key to the score is how a metric starts
measuring itself.

Only the **head** choruses are scored (``form.HEAD_CHORUSES``). On a solo chorus the same
comparison is reported as :func:`novelty`, a deliberately *descriptive* number: a soloist
who matched the written melody would be the anomaly, so a high novelty is the expected
reading and a low one is the thing worth looking at.
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
    whatever sounds before bar 1 or after the last bar line is not in the form.

    Only the onset is placed. A note's length says little about whether it is the written
    note -- a player holds or clips at will -- while its start is what the ear hears as
    the melody's rhythm.
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
    shift: it asks which key the playing is *in*, not which shift scores best, so the
    number stays independent of the thing it is used to measure. Ties go to the smallest
    move, and to 0 above all -- a tie means the content does not distinguish the keys, and
    inventing a modulation out of that would be worse than reporting none.

    Returns 0 when either side is empty.
    """
    if not shifts:
        raise ValueError("shifts is empty: there is no transposition to choose")
    written, played = pitch_profile(reference), pitch_profile(estimate)
    if not written.any() or not played.any():
        return 0
    # Normalised, so the correlation compares shapes rather than how much was played.
    written = written / np.linalg.norm(written)
    played = played / np.linalg.norm(played)
    fit = {shift: float(np.roll(written, shift % SEMITONES) @ played) for shift in shifts}
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

    Report it beside :func:`best_transposition` and look into a disagreement -- it usually
    means the lead was mistracked, or that this is not the tune the chart says it is. Do
    not average the two, and do not score with this one: a key chosen to maximise the score
    makes the score partly a measure of itself.
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

    Both sides are placed on ``downbeats`` by :func:`melody_sequence`, so they must already
    be on the same timeline -- ``form.project_estimate`` is what puts a chorus of a
    recording there. The written line is moved up by ``transposition`` semitones, taken
    from :func:`best_transposition` when not given.

    A written note matches a played one when their pitches agree -- by pitch class if
    ``octave_blind``, exactly otherwise -- and their onsets are within ``tolerance_beats``.
    Taken in written order, each written note claims the nearest unclaimed played note, so
    one played note can answer only one written note: without that, a tremolo on the tonic
    would match the whole chart and score 1.0. A greedy pass rather than an optimal
    assignment, because an optimal matching pairs notes a listener would never hear as the
    same one.
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
    sides are frequencies in hertz, one per frame, on the **same hop** and the same length;
    a frame with no pitch is ``nan`` or zero.

    This measures the tracker alone, before any note segmentation. Without it a tracker
    that wanders and a segmenter that chops are one number, and neither can be fixed first.
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
    scored = melody_score(
        reference,
        estimate,
        downbeats,
        beats_per_bar,
        transposition=transposition,
        octave_blind=octave_blind,
        tolerance_beats=tolerance_beats,
    )
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

    ``reference`` is one chorus of the written line on the chart's own timeline, whose bar
    lines are ``downbeats``; ``estimate`` is the whole recording's notes. Each scored block
    is mapped onto the chart by ``form.project_estimate``. Heads get a
    :class:`MelodyScore`, every other scored block a :func:`novelty`; intros, tails and
    unaligned blocks are left out, as they are for chords.

    A note is kept for the block its **onset** falls in, before projection: a note spilling
    over a chorus line would otherwise be cut and re-onset at the block's start, which is
    an onset nobody played.
    """
    origin = downbeats[0] if len(downbeats) else 0.0
    results: list[ChorusMelody] = []
    for block in blocks:
        if not block.is_scored:
            continue
        inside = [n for n in estimate if block.start <= n.start < block.end]
        played = form.project_estimate(inside, block, origin)
        if block.kind == "head":
            results.append(
                ChorusMelody(
                    chorus=block,
                    score=melody_score(
                        reference,
                        played,
                        downbeats,
                        beats_per_bar,
                        transposition=transposition,
                        octave_blind=octave_blind,
                        tolerance_beats=tolerance_beats,
                    ),
                )
            )
        else:
            results.append(
                ChorusMelody(
                    chorus=block,
                    novelty=novelty(
                        reference,
                        played,
                        downbeats,
                        beats_per_bar,
                        transposition=transposition,
                        octave_blind=octave_blind,
                        tolerance_beats=tolerance_beats,
                    ),
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
