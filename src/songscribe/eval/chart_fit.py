"""How well the chart's chords describe a chord estimate, from each candidate start.

The chart is cut into equal bars and beats and compared with the estimate's beats from each
start, in each key asked for (the chart is transposed, not the estimate). Every fit comes
with the best fit of bar-shuffled copies of the chart over the same starts and keys, as the
level a wrong chart reaches. Pure numpy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..types import ChordEvent
from .form import transpose_events
from .labels import NO_SYMBOL, agreement, beat_symbols
from .metrics import MIN_SEGMENT, extent

#: Keys to try, as semitones above the chart's: only the written key, or all twelve (-5..+6).
AS_WRITTEN = (0,)
ANY_KEY = tuple(range(-5, 7))

#: Bar-shuffled charts used for the chance level, which is their best fit over all starts and
#: keys. The seed is fixed so a rerun gives the same level.
NULL_SHUFFLES = 200
NULL_SEED = 0


@dataclass(frozen=True, slots=True, eq=False)
class ChartFit:
    """Agreement of the chart with the estimate at each candidate start and key.

    ``fit[i, k]`` is the mean per-beat agreement from ``starts[i]`` (a beat index), with the
    chart ``shifts[k]`` semitones up; ``nan`` where fewer than ``need`` bars remain.
    ``chance`` is the best fit any shuffled chart reached.
    """

    starts: tuple[int, ...]
    shifts: tuple[int, ...]
    fit: np.ndarray
    chance: float

    def best(self, index: int) -> float:
        """Agreement at ``starts[index]`` in the best key."""
        return float(np.nanmax(self.fit[index])) if not np.isnan(self.fit[index]).all() else -np.inf

    def key(self, index: int) -> int:
        """The key (semitones above the chart's) that fits ``starts[index]`` best."""
        return self.shifts[int(np.nanargmax(self.fit[index]))]

    def key_margin(self, index: int) -> float:
        """Fit in the best key minus the next best at ``starts[index]``; ``nan`` for one key."""
        row = np.sort(self.fit[index])[::-1]
        return float(row[0] - row[1]) if len(row) > 1 else float("nan")


def chart_beats(
    chart: Sequence[ChordEvent], bars: int, beats_per_bar: int, shifts: Sequence[int]
) -> np.ndarray:
    """The chart as symbols, shape (shifts, bars, beats per bar), in equal bars and beats."""
    if bars < 1:
        raise ValueError(f"bars must be at least 1, got {bars}")
    start, end = extent(list(chart))
    if end - start <= MIN_SEGMENT:
        raise ValueError("the reference covers no time")
    edges = start + (end - start) * np.arange(bars * beats_per_bar + 1) / (bars * beats_per_bar)
    return np.array(
        [
            beat_symbols(transpose_events(chart, shift), edges).reshape(bars, beats_per_bar)
            for shift in shifts
        ]
    )


def _bar_matches(chart: np.ndarray, played: np.ndarray) -> np.ndarray:
    """Agreement of chart bar j with played bar k, shape (starts, shifts, j, k)."""
    total = 0.0
    for beat in range(chart.shape[2]):
        total = total + agreement(
            chart[None, :, :, None, beat], played[:, None, None, :, beat]
        ).astype(np.float32)
    return total / chart.shape[2]


def fit_chart(
    symbols: np.ndarray,
    starts: Sequence[int],
    chart: np.ndarray,
    shifts: Sequence[int],
    *,
    need: int,
) -> ChartFit:
    """Fit ``chart`` (from :func:`chart_beats`) to ``symbols`` from each of ``starts``.

    A start with fewer than ``need`` bars left is not fitted; ``ValueError`` if none can be.
    """
    size, per_bar = chart.shape[1], chart.shape[2]
    left = np.array([(len(symbols) - s) // per_bar if s >= 0 else 0 for s in starts])
    usable = np.flatnonzero(left >= need)
    if not len(usable):
        raise ValueError("the recording is too short to hold the chart")
    played = np.full((len(usable), size, per_bar), NO_SYMBOL, dtype=np.int16)
    for row, i in enumerate(usable):
        rows = min(size, left[i])
        window = symbols[starts[i] : starts[i] + rows * per_bar]
        played[row, :rows] = window.reshape(rows, per_bar)
    bars = np.arange(size)
    matches = _bar_matches(chart, played)
    mask = bars[None, :] < left[usable][:, None]
    counts = np.maximum(mask.sum(axis=1), 1)[:, None]
    fit = np.full((len(starts), len(shifts)), np.nan)
    fit[usable] = (matches[:, :, bars, bars] * mask[:, None, :]).sum(axis=2) / counts

    rng = np.random.default_rng(NULL_SEED)
    chance = -np.inf
    for _ in range(NULL_SHUFFLES):
        order = rng.permutation(size)
        shuffled = (matches[:, :, order, bars] * mask[:, None, :]).sum(axis=2) / counts
        chance = max(chance, float(shuffled.max()))
    return ChartFit(tuple(starts), tuple(shifts), fit, chance)
