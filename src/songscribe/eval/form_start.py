"""Decide where a recording's form starts: by recurrence, checked against the chart.

``recurrence.measure`` finds the stretch of a chord estimate that repeats one form later and
where it opens, which is the pickup into the head or the head itself. The chart then checks
it and settles which bar within a form of the opening is the head (an intro can echo the
tune, a head can differ from the solos). The chords' timing moves bar lines the tracker has a
few beats off. With no recurrence it is the chart match alone.

The estimate is a model's output, so a start found on it can be the one that suits it. Anchor
on an estimate from a different backend than the one scored; ``score`` warns when they match.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import isnan, nan

import numpy as np

from ..beats import BeatGrid
from ..types import ChordEvent
from . import chart_fit, recurrence
from .chart_fit import AS_WRITTEN

#: The chart must beat the best bar-shuffled chart (``Anchor.chance``, over the same starts
#: and keys) by this much. The chance level rises as more starts and keys are searched, so
#: twelve keys need no separate bar. Over 1000 shuffled and unrelated charts the best excess
#: was 0.03; true charts under heavy perturbation had at least 0.07.
ANCHOR_MIN_EXCESS = 0.05
#: Between two positions in the form (or two keys), the one fitting better must lead by this
#: much, to move the start off the recurrence's bar or to accept a chart-only start. Loops
#: and repeated phrases, where the position is genuinely ambiguous, reached 0.14; true charts
#: at least 0.28.
ANCHOR_MIN_MARGIN = 0.20


@dataclass(frozen=True, slots=True)
class Anchor:
    """The recording's bar that matches the first bar of the chart.

    ``bar_index`` is that bar's position in the recording, and ``start`` its time in seconds.
    ``label_agreement`` is the share of beats of one chorus from there whose chord agrees with
    the estimate's, in the key ``shift``. ``chance`` is the best any bar-shuffled chart reached
    over the same starts and keys. ``runner_up`` is the best agreement at a different position
    in the form, and ``margin`` is ``label_agreement`` minus it.
    """

    bar_index: int
    start: float
    label_agreement: float
    runner_up: float
    margin: float
    #: Semitones the chart is moved up to fit, and how many keys were searched.
    shift: int = 0
    keys: int = 1
    #: What found the start: ``chart`` (matching the chart's chords) or ``recurrence``
    #: (the estimate repeating one form later, see ``recurrence.py``).
    evidence: str = "chart"
    #: Whole bars before the pickup, and the pickup's beats; ``None`` if not measured.
    intro_bars: int | None = None
    pickup_beats: int | None = None
    #: Beats the tracker's bar lines were moved to put the start on a downbeat.
    phase_shift: int = 0
    #: Bars after the last stretch that recurs (a tag or ending), by recurrence.
    ending_bars: int | None = None
    #: Strength of the recurrence over the best other lag; ``nan`` if none was claimed.
    recurrence_ratio: float = nan
    #: Whether the chart match alone picks the same start (a whole form apart counts);
    #: ``None`` when it is not clear enough to say.
    chart_agrees: bool | None = None
    #: With a start found by recurrence: the fit in the chosen key over the best other key.
    key_margin: float = nan
    chance: float = nan

    def gate(self) -> float:
        """The agreement the chart has to reach: the shuffled charts' best, and a little more."""
        return self.chance + ANCHOR_MIN_EXCESS

    def refusal(self, min_margin: float = ANCHOR_MIN_MARGIN) -> str | None:
        """Return the reason this anchor is unreliable, or ``None`` if it is reliable.

        A start found by recurrence does not hinge on the chart, so only the chart's fit
        there is checked: a recurring stretch the chart does not describe is not its form.
        """
        gate = self.gate()
        if self.evidence == "recurrence":
            if self.label_agreement < gate:
                return (
                    f"chart fits the recurring start badly: {self.label_agreement:.2f}, "
                    f"shuffled charts reach {self.chance:.2f}"
                )
            return None
        if isnan(self.margin):
            return "only one candidate position in the form, nothing to compare it with"
        if self.label_agreement < gate:
            return (
                f"agreement {self.label_agreement:.2f} under {gate:.2f} "
                f"(shuffled charts reach {self.chance:.2f})"
            )
        if self.margin < min_margin:
            return f"margin {self.margin:.2f} under {min_margin:.2f}: another bar fits as well"
        return None

    @property
    def is_clear(self) -> bool:
        return self.refusal() is None


def _chart_only(
    symbols: np.ndarray,
    lead: int,
    grid: BeatGrid,
    chart: np.ndarray,
    shifts: Sequence[int],
    min_bars: int,
) -> Anchor | None:
    """Find the downbeat that opens the chart by matching it alone, or ``None``.

    Each downbeat is scored as a start on the one chorus after it, so an intro or a
    different-chord solo does not matter. The chart's bar count is the form length.
    The anchor is the earliest start, at the best-scoring position in the form, whose
    agreement reaches the gate.
    """
    size, per_bar = chart.shape[1], chart.shape[2]
    bars = len(grid.downbeats) - 1
    # A whole chorus is the evidence; a short block only when no whole one fits.
    need = size if bars >= size else min(min_bars, size)
    if size < 1 or bars < need:
        return None
    first_bars = range(bars - need + 1)
    starts = [lead + k * per_bar for k in first_bars]
    fitted = chart_fit.fit_chart(symbols, starts, chart, shifts, need=need)
    fits = {k: fitted.best(i) for i, k in enumerate(first_bars)}
    phase_best: dict[int, float] = {}
    for k, value in fits.items():
        phase_best[k % size] = max(phase_best.get(k % size, -np.inf), value)
    best_phase = max(phase_best, key=lambda p: phase_best[p])

    on_phase = [k for k in first_bars if k % size == best_phase]
    gate = fitted.chance + ANCHOR_MIN_EXCESS
    chosen = next((k for k in on_phase if fits[k] >= gate), None)
    if chosen is None:
        # Not hidden: the refusal reports the best fit it found.
        chosen = max(on_phase, key=lambda k: fits[k])
    row = first_bars.index(chosen)
    other_phases = [v for k, v in fits.items() if k % size != best_phase]
    best_key = int(np.nanargmax(fitted.fit[row]))
    other_keys = [v for j, v in enumerate(fitted.fit[row]) if j != best_key]
    # Without another phase to compare, the start is forced, not found.
    runner_up = max([*other_phases, *other_keys]) if other_phases else nan
    return Anchor(
        bar_index=chosen,
        start=float(grid.downbeats[chosen]),
        label_agreement=fits[chosen],
        runner_up=runner_up,
        margin=fits[chosen] - runner_up,
        shift=fitted.key(row),
        keys=len(shifts),
        chance=fitted.chance,
    )


def _pickup(found: recurrence.Recurrence, start: int) -> int:
    """Beats just before ``start`` that recur one form later, 0 to a bar less one.

    The pickup plays the form's tail, so it resembles the tail of the form before it.
    The count is where the running sum of agreement over the beat level peaks.
    """
    best, beats, running = 0.0, 0, 0.0
    for back in range(1, found.beats_per_bar):
        index = start - back
        if index < 0 or index >= len(found.beat_similarity):
            break
        running += float(found.beat_similarity[index]) - found.beat_level
        if running > best:
            best, beats = running, back
    return beats


def locate(
    estimate: Sequence[ChordEvent],
    grid: BeatGrid,
    chart: Sequence[ChordEvent],
    *,
    bars: int,
    min_bars: int,
    shifts: Sequence[int] = AS_WRITTEN,
) -> tuple[Anchor | None, BeatGrid]:
    """Find the form's first bar, returning the anchor and the grid it indexes.

    The grid is the tracker's unless the start fell off its bar lines, in which case
    the bar lines are moved by whole beats and ``Anchor.phase_shift`` says by how many.
    Recurrence does not depend on the key; the chart's key is the best fit at the start.
    """
    shifts = tuple(shifts)
    per_bar = grid.beats_per_bar
    symbols, times, lead = recurrence.beats_of(estimate, grid.downbeats, per_bar)
    chart_symbols = chart_fit.chart_beats(chart, bars, per_bar, shifts)
    alone = _chart_only(symbols, lead, grid, chart_symbols, shifts, min_bars)
    found = recurrence.measure(symbols, times, lead, per_bar, bars)
    if found is None or not found.is_found:
        ratio = nan if found is None else found.ratio
        return (None if alone is None else replace(alone, recurrence_ratio=ratio)), grid

    opening = found.first_beat
    # The tracker's downbeat unless the chords change on another beat, bar after bar.
    phase, _ = recurrence.bar_line_phase(found, lead % per_bar)
    form_beats = found.form_beats
    starts = list(range(phase % per_bar, opening + form_beats + 1, per_bar))
    try:
        fitted = chart_fit.fit_chart(
            symbols, starts, chart_symbols, shifts, need=min(min_bars, bars)
        )
    except ValueError:
        return alone, grid
    fits = {s: fitted.best(i) for i, s in enumerate(starts) if np.isfinite(fitted.best(i))}
    opens = next((s for s in fits if s >= opening), None)
    if opens is None:
        return alone, grid
    gate = fitted.chance + ANCHOR_MIN_EXCESS
    # Starts a whole form apart are the same bar of the tune. The recurrence's opening
    # keeps its bar unless the chart fits another clearly better, and well (an intro that
    # echoes the tune); within the bar, the earliest pass that fits is the head, even if it
    # differs enough from the solos that the recurrence opens a pass later.
    by_bar: dict[int, float] = {}
    for s, v in fits.items():
        by_bar[s % form_beats] = max(by_bar.get(s % form_beats, -np.inf), v)
    bar_of_tune = max(by_bar, key=lambda b: by_bar[b])
    clearly_better = by_bar[bar_of_tune] - by_bar[opens % form_beats] >= ANCHOR_MIN_MARGIN
    if not (clearly_better and by_bar[bar_of_tune] >= gate):
        bar_of_tune = opens % form_beats
    passes = [s for s in fits if s % form_beats == bar_of_tune]
    # With no pass the chart describes, the recurrence's own opening stands; picking the best
    # of several poor fits would be picking noise.
    start = next((s for s in passes if fits[s] >= gate), opens)
    value = fits[start]
    rivals = [v for b, v in by_bar.items() if b != bar_of_tune]
    runner_up = max(rivals) if rivals else nan
    pickup = _pickup(found, start)
    row = starts.index(start)

    moved = (start - lead) % per_bar
    shift = ((moved + per_bar // 2) % per_bar) - per_bar // 2
    if moved:
        first = start % per_bar
        marks = [float(t) for t in times[first::per_bar]]
        if (len(times) - 1 - first) % per_bar:
            # The audio ends inside a bar of the moved grid; close it at the last tempo.
            marks.append(marks[-1] + per_bar * float(times[-1] - times[-2]))
        grid = BeatGrid(downbeats=tuple(marks), beats_per_bar=per_bar)
        bar = (start - first) // per_bar
    else:
        bar = (start - lead) // per_bar

    beats = len(times) - 1
    agrees = None
    if alone is not None and alone.refusal() is None:
        agrees = (lead + alone.bar_index * per_bar - start) % form_beats == 0
    anchor = Anchor(
        bar_index=bar,
        start=float(times[start]),
        label_agreement=value,
        runner_up=runner_up,
        margin=value - runner_up,
        shift=fitted.key(row),
        keys=len(shifts),
        evidence="recurrence",
        intro_bars=max(start - pickup, 0) // per_bar,
        pickup_beats=pickup,
        phase_shift=shift if moved else 0,
        ending_bars=max(0, beats - (found.end_beat + form_beats)) // per_bar,
        recurrence_ratio=found.ratio,
        chart_agrees=agrees,
        key_margin=fitted.key_margin(row),
        chance=fitted.chance,
    )
    return anchor, grid
