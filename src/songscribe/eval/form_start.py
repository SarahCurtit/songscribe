"""Decide where a recording's form starts: by recurrence, checked against the chart.

``recurrence.measure`` finds the stretch of the audio that repeats one form later and
where it opens, which is the pickup into the head or the head itself. The chart then
checks it and settles which bar within a form of the opening is the head (an intro can
echo the tune, a head can differ from the solos). The chords' timing moves bar lines the
tracker has a few beats off. With no recurrence it is the chart match alone.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from math import nan

import numpy as np

from ..beats import BeatGrid
from . import chroma as chroma_mod
from . import recurrence
from .chroma import ANCHOR_MIN_MARGIN, Anchor, Chromagram


def window_fit(profiles: np.ndarray, per_bar: int, templates: Sequence[np.ndarray]) -> np.ndarray:
    """Agreement of the bar starting at each beat with each chart bar, shape (beats, bars).

    Same cosine as ``chroma.bar_agreement``, for bars that may start on any beat.
    """
    bars = recurrence.bar_windows(profiles, per_bar)
    wanted = np.array([np.asarray(t, dtype=float) for t in templates]).reshape(-1, 12)
    wanted = np.where(
        wanted.any(axis=1, keepdims=True), wanted - wanted.mean(axis=1, keepdims=True), 0
    )
    norms = np.linalg.norm(bars, axis=1)[:, None] * np.linalg.norm(wanted, axis=1)[None, :]
    return np.divide(bars @ wanted.T, norms, out=np.zeros_like(norms), where=norms > 1e-9)


def _pickup(found: recurrence.Recurrence, start: int) -> int:
    """Beats just before ``start`` that recur one form later, 0 to a bar less one.

    The pickup plays the form's tail, so it resembles the tail of the form before it.
    The count is where the running sum of similarity over the beat level peaks.
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
    spectrum: Chromagram,
    grid: BeatGrid,
    templates: Sequence[np.ndarray],
    *,
    min_bars: int,
    shifts: Sequence[int] = chroma_mod.AS_WRITTEN,
) -> tuple[Anchor | None, BeatGrid]:
    """Find the form's first bar, returning the anchor and the grid it indexes.

    The grid is the tracker's unless the start fell off its bar lines, in which case
    the bar lines are moved by whole beats and ``Anchor.phase_shift`` says by how many.
    Recurrence does not depend on the key; the chart's key is the best fit at the start.
    """
    chart = chroma_mod.anchor_bar(
        spectrum, grid.downbeats, templates, min_bars=min_bars, shifts=shifts
    )
    found = recurrence.measure(spectrum, grid.downbeats, grid.beats_per_bar, len(templates))
    if found is None or not found.is_found:
        ratio = nan if found is None else found.ratio
        return (None if chart is None else replace(chart, recurrence_ratio=ratio)), grid

    per_bar, lead, size = grid.beats_per_bar, found.lead, len(templates)
    times = found.beat_times
    keyed = {
        k: window_fit(found.profiles, per_bar, [np.roll(t, k) for t in templates]) for k in shifts
    }
    need = min(min_bars, size)
    gate = chroma_mod.agreement_gate(len(shifts))

    def fit_in(start: int, key: int) -> float:
        agreement = keyed[key]
        rows = np.arange(min(size, (len(agreement) - start + per_bar - 1) // per_bar))
        if len(rows) < need or start < 0:
            return -np.inf
        return float(agreement[start + rows * per_bar, rows].mean())

    def fit(start: int) -> float:
        return max(fit_in(start, k) for k in shifts)

    opening = found.first_beat
    # The tracker's downbeat unless the chords change on another beat, bar after bar.
    phase, _ = recurrence.bar_line_phase(found, lead % per_bar)
    form_beats = found.form_beats
    fits = {s: fit(s) for s in range(phase % per_bar, opening + form_beats + 1, per_bar)}
    fits = {s: v for s, v in fits.items() if v > -np.inf}
    opens = next((s for s in fits if s >= opening), None)
    if opens is None:
        return chart, grid
    # Starts a whole form apart are the same bar of the tune. The recurrence's opening
    # keeps its bar unless the chart fits another clearly better, and well (an intro that echoes
    # the tune); within the bar, the earliest pass that fits is the head, even if it
    # differs enough from the solos that the recurrence opens a pass later.
    by_bar: dict[int, float] = {}
    for s, v in fits.items():
        by_bar[s % form_beats] = max(by_bar.get(s % form_beats, -np.inf), v)
    bar_of_tune = max(by_bar, key=lambda b: by_bar[b])
    clearly_better = by_bar[bar_of_tune] - by_bar[opens % form_beats] >= ANCHOR_MIN_MARGIN
    if not (clearly_better and by_bar[bar_of_tune] >= gate):
        bar_of_tune = opens % form_beats
    passes = [s for s in fits if s % form_beats == bar_of_tune]
    start = next((s for s in passes if fits[s] >= gate), None)
    if start is None:
        start = max(passes, key=lambda s: fits[s])
    value = fits[start]
    rivals = [v for b, v in by_bar.items() if b != bar_of_tune]
    runner_up = max(rivals) if rivals else nan
    pickup = _pickup(found, start)
    key = max(shifts, key=lambda k: fit_in(start, k))
    others = [fit_in(start, k) for k in shifts if k != key]
    key_margin = fit_in(start, key) - max(others) if others else nan

    moved = (start - lead) % per_bar
    shift = ((moved + per_bar // 2) % per_bar) - per_bar // 2
    if moved:
        first = start % per_bar
        marks = tuple(float(t) for t in times[first::per_bar])
        grid = BeatGrid(downbeats=marks, beats_per_bar=per_bar)
        bar = (start - first) // per_bar
    else:
        bar = (start - lead) // per_bar

    beats = len(times) - 1
    agrees = None
    if chart is not None and chart.refusal() is None:
        agrees = (lead + chart.bar_index * per_bar - start) % found.form_beats == 0
    anchor = Anchor(
        bar_index=bar,
        start=float(times[start]),
        mean_agreement=value,
        runner_up=runner_up,
        margin=value - runner_up,
        evidence="recurrence",
        intro_bars=max(start - pickup, 0) // per_bar,
        pickup_beats=pickup,
        phase_shift=shift if moved else 0,
        ending_bars=max(0, beats - (found.end_beat + found.form_beats)) // per_bar,
        recurrence_ratio=found.ratio,
        chart_agrees=agrees,
        shift=key,
        keys=len(shifts),
        key_margin=key_margin,
    )
    return anchor, grid
