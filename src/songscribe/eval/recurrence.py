"""Find where a recording's form starts from a chord estimate of the recording.

A tune is played as a head and then choruses that repeat its changes, so the chords one
form later resemble the chords now. Comparing each bar of the estimate with the bar one
form later finds a stretch where that holds: it begins at the head (or at the pickup into
it) and ends one form before the last pass. The chart supplies only the form's length in
bars, never searched. The estimate is rough, so bars are compared by root and third.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import floor

import numpy as np

from ..types import ChordEvent
from .labels import NO_SYMBOL, agreement, beat_symbols

#: Windows closer than this many bars overlap or share a sustained chord, which
#: makes any short lag look like a recurrence.
MIN_NULL_LAG_BARS = 2
#: A window counts as recurring above this many standard deviations over the null's mean;
#: a run's steps are limited to the same distance either way. On synthetic controls none of
#: 450 shuffled, 450 unrelated and 450 one-chord takes claimed a form at 0.75 or at 1.0, and
#: 0.75 keeps more of the heavily perturbed takes that do recur (78% against 67%).
LEVEL_SIGMAS = 0.75
#: The run must beat the strongest run at any null lag. Alone that passed unrelated takes
#: (up to 1.27x), so it must also span this much of a form: shuffled takes reached 0.56 of
#: one, one-chord vamps 0.66, and takes that recur reach about 3 (5th percentile 2.7 under
#: moderate perturbation).
MIN_STRENGTH_RATIO = 1.0
MIN_RUN_FORMS = 0.75
#: A beat other than the tracker's downbeat takes over only if chords change into it more
#: often than into the tracker's, bar after bar, with a paired t at least this large. Correct
#: trackers gave a best rival at -3.2 or lower (180 takes), ones off by 1-3 beats 3.2 or more
#: (720 takes), so the bar sits in the empty gap, nearer the wrong ones.
PHASE_MIN_T = 3.0


@dataclass(frozen=True, slots=True, eq=False)
class Recurrence:
    """Where the recording repeats itself one form later.

    Beats index ``beat_times`` (the bar grid cut into beats). ``first_beat`` opens the first
    stretch that recurs; ``end_beat`` is one past the last beat with a later copy.
    ``strength`` is the run's summed agreement over the level, ``null_best`` the strongest
    run at any other lag; ``beat_similarity[b]`` recurs above ``beat_level``.
    """

    beat_times: np.ndarray
    lead: int
    beats_per_bar: int
    form_beats: int
    first_beat: int
    end_beat: int
    strength: float
    null_best: float
    mean_similarity: float
    null_mean: float
    null_spread: float
    symbols: np.ndarray
    beat_similarity: np.ndarray
    beat_level: float

    @property
    def ratio(self) -> float:
        return self.strength / self.null_best if self.null_best > 0 else float("inf")

    @property
    def run_bars(self) -> float:
        return (self.end_beat - self.first_beat) / self.beats_per_bar

    @property
    def separation(self) -> float:
        """How many null spreads the form lag's mean agreement in the run is over the null."""
        return (self.mean_similarity - self.null_mean) / max(self.null_spread, 1e-9)

    @property
    def is_found(self) -> bool:
        long_enough = self.end_beat - self.first_beat >= MIN_RUN_FORMS * self.form_beats
        return long_enough and self.ratio > MIN_STRENGTH_RATIO


def beat_times(downbeats: Sequence[float], beats_per_bar: int) -> tuple[np.ndarray, int]:
    """Edges of every beat: each bar split evenly, plus the beats a bar line back to 0 s.

    Returns the edges (one more than the beats) and ``lead``, the number of beats
    added before the first downbeat, at most one bar's worth.
    """
    down = np.asarray(downbeats, dtype=float)
    if len(down) < 2:
        return down, 0
    steps = np.linspace(0.0, 1.0, beats_per_bar, endpoint=False)
    inner = (down[:-1, None] + np.diff(down)[:, None] * steps).ravel()
    first = (down[1] - down[0]) / beats_per_bar
    lead = int(min(beats_per_bar - 1, floor(down[0] / first + 1e-9)))
    before = down[0] - first * np.arange(lead, 0, -1)
    return np.concatenate([before, inner, down[-1:]]), lead


def _windowed(values: np.ndarray, width: int) -> np.ndarray:
    """Mean of every ``width`` consecutive values, one per starting beat."""
    total = np.concatenate([[0.0], np.cumsum(values)])
    return (total[width:] - total[:-width]) / width


def _lag_agreement(symbols: np.ndarray, lag: int) -> np.ndarray:
    """Agreement of every beat with the beat ``lag`` later."""
    return agreement(symbols[:-lag], symbols[lag:])


def _best_run(values: np.ndarray, level: float, cap: float) -> tuple[float, int, int]:
    """Contiguous stretch with the largest sum of ``values - level``: (sum, first, end).

    Each step is limited to ``cap`` either way, so one bar that differs (a substituted chord)
    costs no more than one matching bar earns, instead of ending the run.
    """
    best, best_first, best_end = 0.0, 0, 0
    running, first = 0.0, 0
    for i, value in enumerate(values):
        if running <= 0:
            running, first = 0.0, i
        running += min(max(value - level, -cap), cap)
        if running > best:
            best, best_first, best_end = running, first, i + 1
    return best, best_first, best_end


def _strongest_run(runs: Sequence[np.ndarray], level: float, cap: float) -> float:
    """The largest sum :func:`_best_run` gives for any of ``runs``, all lags at once."""
    steps = np.zeros((len(runs), max((len(r) for r in runs), default=0)))
    for row, values in enumerate(runs):
        steps[row, : len(values)] = np.clip(values - level, -cap, cap)
    running, best = np.zeros(len(runs)), np.zeros(len(runs))
    for column in steps.T:
        running = np.maximum(running + column, 0.0)
        best = np.maximum(best, running)
    return float(best.max(initial=0.0))


def _null_lags(count: int, form_beats: int, beats_per_bar: int) -> list[int]:
    """Lags that are not a whole or half form, nor within half a bar of one.

    A sustained chord still agrees with itself over half its bar when shifted by half the
    bar, which is above the level a run counts at, so such lags recur as the form does.
    """
    half = form_beats / 2
    near = beats_per_bar // 2
    keep = []
    for lag in range(MIN_NULL_LAG_BARS * beats_per_bar, count - MIN_NULL_LAG_BARS * beats_per_bar):
        nearest = round(lag / half) * half
        if nearest == 0 or abs(lag - nearest) > near:
            keep.append(lag)
    return keep


def _refine(
    beat_sim: np.ndarray, level: float, around: int, reach: int, rising: bool, span: int
) -> int:
    """Beat where the single-beat agreement crosses ``level``, searched within ``reach``.

    Picks the split of a ``span``-beat window either side that best separates
    recurring (above ``level``) from not; beats with no partner count for neither side.
    """
    count = len(beat_sim)
    best, where = -np.inf, around
    for split in range(max(0, around - reach), min(count, around + reach) + 1):
        left = beat_sim[max(0, split - span) : split] - level
        right = beat_sim[split : split + span] - level
        score = (-left.sum() + right.sum()) if rising else (left.sum() - right.sum())
        if score > best + 1e-12:
            best, where = score, split
    return where


def bar_line_phase(found: Recurrence, tracker: int) -> tuple[int, float]:
    """Which beat of the bar the chords change on, as ``(beat residue, t statistic)``.

    Over the recurring music, compares how often the chord changes into each beat of the
    bar with how often it changes into the tracker's downbeat (``tracker``, a residue
    modulo the metre), bar by bar. The statistic is the best rival's paired t over the
    tracker's: a rival wins only with ``t`` of at least ``PHASE_MIN_T``.
    """
    per_bar, symbols = found.beats_per_bar, found.symbols
    low = max(found.first_beat, 1)
    high = min(found.end_beat + found.form_beats, len(symbols))
    changed = (symbols[low:high] != NO_SYMBOL) & (symbols[low:high] != symbols[low - 1 : high - 1])
    change = changed.astype(float)
    first_bar = low // per_bar + 1
    last_bar = high // per_bar
    best, best_t = tracker, -np.inf
    for residue in range(per_bar):
        if residue == tracker:
            continue
        bars = np.arange(first_bar, last_bar)
        mine, theirs = bars * per_bar + residue, bars * per_bar + tracker
        keep = (mine >= low) & (mine < high) & (theirs >= low) & (theirs < high)
        diff = change[mine[keep] - low] - change[theirs[keep] - low]
        if len(diff) < 2 or diff.std() < 1e-9:
            continue
        t = float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))))
        if t > best_t:
            best, best_t = residue, t
    return (best, best_t) if best_t >= PHASE_MIN_T else (tracker, best_t)


def beats_of(
    estimate: Sequence[ChordEvent], downbeats: Sequence[float], beats_per_bar: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """The estimate as one symbol per beat of the bar grid: (symbols, beat edges, lead)."""
    edges, lead = beat_times(downbeats, beats_per_bar)
    return beat_symbols(estimate, edges), edges, lead


def measure(
    symbols: np.ndarray, edges: np.ndarray, lead: int, beats_per_bar: int, bars: int
) -> Recurrence | None:
    """Find the stretch that recurs ``bars`` bars later, or ``None`` if the take is too short.

    Each bar-long window is compared with the one a form later; the run is the stretch
    above the null's level with the largest total, and its edges are refined on single
    beats. The null is every other lag, scored the same way, so a vamp raises the bar.
    """
    form_beats = bars * beats_per_bar
    count = len(symbols) - beats_per_bar + 1
    if bars < 1 or count <= form_beats + beats_per_bar * MIN_NULL_LAG_BARS:
        return None

    lags = _null_lags(count, form_beats, beats_per_bar)
    beat_by_lag = {lag: _lag_agreement(symbols, lag) for lag in [*lags, form_beats]}
    by_lag = {lag: _windowed(values, beats_per_bar) for lag, values in beat_by_lag.items()}
    pooled = np.concatenate([by_lag[lag] for lag in lags]) if lags else np.zeros(1)
    null_mean, spread = float(pooled.mean()), float(pooled.std())
    level, cap = null_mean + LEVEL_SIGMAS * spread, LEVEL_SIGMAS * spread
    strength, first, end = _best_run(by_lag[form_beats], level, cap)
    null_best = _strongest_run([by_lag[lag] for lag in lags], level, cap)
    mean = float(by_lag[form_beats][first:end].mean()) if end > first else float("nan")

    # Edges on single beats: a window straddling the start is already half inside.
    beat_sim = beat_by_lag[form_beats]
    beat_null = np.concatenate([beat_by_lag[lag] for lag in lags] or [[0.0]])
    inside = beat_sim[first : end + beats_per_bar - 1]
    # Equally many standard deviations from the null and from what recurs, but no stricter
    # than the windows' level: a sharp run would otherwise reject its own pickup.
    low, high = float(beat_null.mean()), float(inside.mean()) if len(inside) else 0.0
    wide, tight = float(beat_null.std()), float(inside.std()) if len(inside) else 0.0
    beat_level = min(low + wide / (wide + tight + 1e-9) * (high - low), low + LEVEL_SIGMAS * wide)
    span = 2 * beats_per_bar
    start = _refine(beat_sim, beat_level, first, beats_per_bar, True, span)
    stop = _refine(beat_sim, beat_level, end + beats_per_bar - 1, beats_per_bar, False, span)
    return Recurrence(
        beat_times=edges,
        lead=lead,
        beats_per_bar=beats_per_bar,
        form_beats=form_beats,
        first_beat=start,
        end_beat=max(stop, start),
        strength=strength,
        null_best=null_best,
        mean_similarity=mean,
        null_mean=null_mean,
        null_spread=spread,
        symbols=symbols,
        beat_similarity=beat_sim,
        beat_level=beat_level,
    )
