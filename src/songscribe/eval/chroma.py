"""Find which bar of a recording is the first bar of the chart.

The chart's chords become 12 pitch-class weights per bar and are compared with the
recording's per-bar chroma (energy per pitch class, octaves folded) at every possible
start. The comparison uses the audio, never the chord estimate being evaluated, which
would pick the start that flatters it. Self-similarity gives the cycle length but not
its start, and a wider offset search finds false matches in repetitive chords.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import ceil, floor, isnan, nan

import numpy as np

from ..types import ChordEvent
from .harte import parse
from .metrics import MIN_SEGMENT, extent

#: Frequency range analysed: from 55 Hz (A1, low bass) upward over 5 octaves
#: (about 1.7 kHz). Higher partials are ignored.
CHROMA_FMIN = 55.0
CHROMA_OCTAVES = 5

#: librosa's constant-Q transform needs a hop length that is a multiple of
#: 2**(CHROMA_OCTAVES - 1) samples.
_HOP_MULTIPLE = 2 ** (CHROMA_OCTAVES - 1)
CHROMA_HOP_SECONDS = 0.023

#: An anchor is only reported as clear if both thresholds are met. The margin
#: is the difference between the best starting bar and the best starting bar
#: that falls at a different position in the form. It is needed because a
#: chart in the wrong key, or with its bars shuffled, can still reach the
#: minimum agreement. Agreement is the cosine similarity of standardised
#: chroma. Both values are guesses, not yet checked on real recordings.
ANCHOR_MIN_AGREEMENT = 0.50
ANCHOR_MIN_MARGIN = 0.08


@dataclass(frozen=True, slots=True, eq=False)
class Chromagram:
    """Energy per pitch class over time.

    ``values`` has shape (12, frames), with row 0 = C. ``hop`` is the time
    between frames, in seconds.
    """

    values: np.ndarray
    hop: float

    def standardised(self) -> np.ndarray:
        """Rescale each pitch class to mean 0 and standard deviation 1.

        Heavily compressed recordings have every pitch class loud all the time,
        so without this every candidate start scores about the same.
        """
        centred = self.values - self.values.mean(axis=1, keepdims=True)
        spread = self.values.std(axis=1, keepdims=True)
        # A pitch class with no variation stays at 0 (avoids dividing by zero).
        return np.divide(centred, spread, out=np.zeros_like(centred), where=spread > 1e-9)


def chromagram(audio: np.ndarray, sample_rate: int, hop: float = CHROMA_HOP_SECONDS) -> Chromagram:
    """Compute the chromagram of mono ``audio`` with librosa's constant-Q chroma.

    ``hop`` is rounded to the nearest length the transform accepts, so the
    returned ``Chromagram.hop`` can differ slightly from the one requested.
    """
    import librosa

    hop_length = max(_HOP_MULTIPLE, round(hop * sample_rate / _HOP_MULTIPLE) * _HOP_MULTIPLE)
    values = librosa.feature.chroma_cqt(
        y=audio, sr=sample_rate, hop_length=hop_length, fmin=CHROMA_FMIN, n_octaves=CHROMA_OCTAVES
    )
    return Chromagram(values=values, hop=hop_length / sample_rate)


def chroma_from_events(
    events: Sequence[ChordEvent], hop: float = CHROMA_HOP_SECONDS, duration: float | None = None
) -> Chromagram:
    """Build a chromagram from chord events: 1 for the notes of the chord, else 0.

    This is for inspecting a chart without audio. Do not use it to anchor an
    evaluation: the start found would be the one that suits the estimate.
    """
    end = max((e.end for e in events), default=0.0) if duration is None else duration
    values = np.zeros((12, max(1, ceil(end / hop))))
    for event in events:
        chord = parse(event.label)
        if chord is None:
            continue
        low, high = floor(event.start / hop), min(values.shape[1], ceil(event.end / hop))
        for pitch_class in chord.pitch_classes:
            values[pitch_class, max(low, 0) : high] = 1.0
    return Chromagram(values=values, hop=hop)


def bar_templates(reference: Sequence[ChordEvent], bars: int) -> tuple[np.ndarray, ...]:
    """Split the chart into ``bars`` equal bars and return 12 pitch-class weights for each.

    Each note of a chord is weighted by how long the chord sounds in the bar.
    A bar with two chords therefore contains the notes of both. Labels that
    cannot be parsed, and ``N`` (no chord), add nothing.
    """
    if bars < 1:
        raise ValueError(f"bars must be at least 1, got {bars}")
    start, end = extent(list(reference))
    if end - start <= MIN_SEGMENT:
        raise ValueError("the reference covers no time")
    length = (end - start) / bars
    templates = [np.zeros(12) for _ in range(bars)]
    for event in reference:
        chord = parse(event.label)
        if chord is None:
            continue
        first = max(0, floor((event.start - start) / length + 1e-9))
        for bar in range(first, bars):
            low = start + bar * length
            if low >= event.end - MIN_SEGMENT:
                break
            overlap = min(event.end, low + length) - max(event.start, low)
            for pitch_class in chord.pitch_classes:
                templates[bar][pitch_class] += max(overlap, 0.0)
    return tuple(templates)


def bar_agreement(
    chroma: Chromagram, downbeats: Sequence[float], templates: Sequence[np.ndarray]
) -> np.ndarray:
    """Compare each bar of the recording with each template bar.

    Returns an array of shape (recording bars, templates). Each value is the
    cosine between the bar's mean standardised chroma and the template with
    its own mean subtracted. It ranges from -1 to 1, and is 0 when the bar's
    chroma is flat or the bar lies outside the recording.
    """
    standard = chroma.standardised()
    frames = standard.shape[1]
    profiles = np.zeros((max(len(downbeats) - 1, 0), 12))
    for bar, (start, end) in enumerate(pairwise(downbeats)):
        low = max(0, floor(start / chroma.hop))
        high = min(frames, max(ceil(end / chroma.hop), low + 1))
        if high > low:
            profiles[bar] = standard[:, low:high].mean(axis=1)

    wanted = np.array([np.asarray(t, dtype=float) for t in templates]).reshape(-1, 12)
    wanted = np.where(
        wanted.any(axis=1, keepdims=True), wanted - wanted.mean(axis=1, keepdims=True), 0
    )
    norms = np.linalg.norm(profiles, axis=1)[:, None] * np.linalg.norm(wanted, axis=1)[None, :]
    return np.divide(profiles @ wanted.T, norms, out=np.zeros_like(norms), where=norms > 1e-9)


@dataclass(frozen=True, slots=True)
class Anchor:
    """The recording's bar that matches the first bar of the chart.

    ``bar_index`` is that bar's position in the recording, and ``start`` its
    time in seconds. ``mean_agreement`` is the average agreement over one
    chorus starting there. ``runner_up`` is the best such average for a start
    at a different position in the form. ``margin`` is ``mean_agreement`` minus
    ``runner_up``.
    """

    bar_index: int
    start: float
    mean_agreement: float
    runner_up: float
    margin: float

    def refusal(
        self, min_agreement: float = ANCHOR_MIN_AGREEMENT, min_margin: float = ANCHOR_MIN_MARGIN
    ) -> str | None:
        """Return the reason this anchor is unreliable, or ``None`` if it is reliable."""
        if isnan(self.margin):
            return "only one candidate position in the form, nothing to compare it with"
        if self.mean_agreement < min_agreement:
            return f"agreement {self.mean_agreement:.2f} under {min_agreement:.2f}"
        if self.margin < min_margin:
            return f"margin {self.margin:.2f} under {min_margin:.2f}: another bar fits as well"
        return None

    @property
    def is_clear(self) -> bool:
        return self.refusal() is None


def anchor_bar(
    chroma: Chromagram,
    downbeats: Sequence[float],
    templates: Sequence[np.ndarray],
    *,
    min_bars: int,
) -> Anchor | None:
    """Find the recording's downbeat that matches the first bar of the chart, or ``None``.

    Each downbeat is scored as a start on the one chorus after it, so an intro or a
    different-chord solo does not matter. The chart's bar count is the form length.
    The anchor is the earliest start, at the best-scoring position in the form, whose
    agreement reaches ``ANCHOR_MIN_AGREEMENT``.
    """
    # ``None`` means the recording is too short to compare.
    size = len(templates)
    agreement = bar_agreement(chroma, downbeats, templates)
    bars = agreement.shape[0]
    # Compare a full chorus if the recording has one. Otherwise accept a
    # shorter block of at least ``min_bars``.
    need = size if bars >= size else min(min_bars, size)
    if size < 1 or bars < need:
        return None

    def fit(first: int) -> float:
        rows = np.arange(first, min(first + size, bars))
        return float(agreement[rows, rows - first].mean())

    starts = range(bars - need + 1)
    fits = {s: fit(s) for s in starts}
    # "Phase" is a start's position in the form (start % size). Starts with the
    # same phase are whole choruses apart, so keep the best score for each.
    phase_best: dict[int, float] = {}
    for s, value in fits.items():
        phase_best[s % size] = max(phase_best.get(s % size, -np.inf), value)
    best_phase = max(phase_best, key=lambda p: phase_best[p])
    runner_up = max((v for p, v in phase_best.items() if p != best_phase), default=nan)

    on_phase = [s for s in starts if s % size == best_phase]
    chosen = next((s for s in on_phase if fits[s] >= ANCHOR_MIN_AGREEMENT), None)
    if chosen is None:
        # No start reaches the minimum. Return the best one anyway, so that
        # refusal() can report its score.
        chosen = max(on_phase, key=lambda s: fits[s])
    return Anchor(
        bar_index=chosen,
        start=float(downbeats[chosen]),
        mean_agreement=fits[chosen],
        runner_up=runner_up,
        margin=fits[chosen] - runner_up,
    )
