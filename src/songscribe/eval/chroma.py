"""Where the form starts: the chart's pitch content against a recording's chroma.

Self-similarity finds the harmonic period but never the phase, and a widened
offset sweep finds spurious fits in repetitive changes. This slides the chart's
per-bar pitch classes along the recording's bars instead, in each of the twelve
keys, and says so when the fit is not clear. The audio is the evidence, never
the estimate under test: anchoring on it would pick the start that flatters the model.
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

#: Bass fundamental up to where harmony blurs, in the CQT behind the chromagram.
CHROMA_FMIN = 55.0
CHROMA_OCTAVES = 5

#: The CQT needs a hop that is a multiple of 2**(CHROMA_OCTAVES - 1) samples.
_HOP_MULTIPLE = 2 ** (CHROMA_OCTAVES - 1)
CHROMA_HOP_SECONDS = 0.023

#: Gates on a start the fit may stand behind. The margin is the one that
#: matters: it is how far the best phase beats the best *other* phase, and a
#: shuffled or wrongly keyed chart can still clear a bare agreement bar.
#: Agreement here is a cosine of standardised chroma, so these are provisional
#: until measured on real recordings.
ANCHOR_MIN_AGREEMENT = 0.50
ANCHOR_MIN_MARGIN = 0.08
#: Twelve keys are twelve chances to fit by luck: random 8-bar charts cleared 0.50
#: 1.4% of the time against 0.2% in one key, and never cleared 0.65 (n=1500).
ANCHOR_MIN_AGREEMENT_ANY_KEY = 0.65

#: Keys a take may be in, as semitones up from the chart's: ``AS_WRITTEN`` is the
#: strict search, ``ANY_KEY`` the twelve. Reported in -5..+6, a tone down being -2.
AS_WRITTEN = (0,)
ANY_KEY = tuple(range(-5, 7))


@dataclass(frozen=True, slots=True, eq=False)
class Chromagram:
    """Pitch-class energy over time: ``values`` is 12 x frames, C first."""

    values: np.ndarray
    hop: float

    def standardised(self) -> np.ndarray:
        """Each pitch class scaled to zero mean, unit spread over the recording.

        Without this every lag of a compressed master scores alike, because a
        pitch class that is always loud tells nothing about which bar it is.
        """
        centred = self.values - self.values.mean(axis=1, keepdims=True)
        spread = self.values.std(axis=1, keepdims=True)
        # A pitch class that never moves carries no evidence either way.
        return np.divide(centred, spread, out=np.zeros_like(centred), where=spread > 1e-9)


def chromagram(audio: np.ndarray, sample_rate: int, hop: float = CHROMA_HOP_SECONDS) -> Chromagram:
    """Constant-Q chroma of mono ``audio``; the hop snaps to what the CQT allows."""
    import librosa

    hop_length = max(_HOP_MULTIPLE, round(hop * sample_rate / _HOP_MULTIPLE) * _HOP_MULTIPLE)
    values = librosa.feature.chroma_cqt(
        y=audio, sr=sample_rate, hop_length=hop_length, fmin=CHROMA_FMIN, n_octaves=CHROMA_OCTAVES
    )
    return Chromagram(values=values, hop=hop_length / sample_rate)


def chroma_from_events(
    events: Sequence[ChordEvent], hop: float = CHROMA_HOP_SECONDS, duration: float | None = None
) -> Chromagram:
    """The chroma a chord timeline would give, for diagnostics without audio.

    Anchoring on an estimate fits the start to the model's own output; use it
    to look at a form, never to score one.
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
    """The chart's pitch classes in each of its ``bars`` equal bars, time-weighted.

    A bar holding two chords gets both, in proportion; an unreadable label or
    ``N`` contributes nothing, so a bar of nothing matches nothing.
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
    """Cosine of every recorded bar's chroma with every template: bars x templates.

    The cosine of centred vectors, so it runs from -1 to 1 and a flat chroma
    scores 0 against anything. A bar outside the recording scores 0.
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


def keyed_agreement(
    chroma: Chromagram,
    downbeats: Sequence[float],
    templates: Sequence[np.ndarray],
    shifts: Sequence[int] = AS_WRITTEN,
) -> dict[int, np.ndarray]:
    """:func:`bar_agreement` with the chart moved up each of ``shifts`` semitones.

    Moving the chart up is a circular rotation of its pitch classes; the
    recording is untouched, so one standardisation serves every key.
    """
    rotated = [np.roll(np.asarray(t, dtype=float), shift) for shift in shifts for t in templates]
    every = bar_agreement(chroma, downbeats, rotated)
    size = len(templates)
    return {shift: every[:, k * size : (k + 1) * size] for k, shift in enumerate(shifts)}


def _agreement_gate(keys: int) -> float:
    return ANCHOR_MIN_AGREEMENT if keys == 1 else ANCHOR_MIN_AGREEMENT_ANY_KEY


@dataclass(frozen=True, slots=True)
class Anchor:
    """The downbeat the chart's first bar sits on, and how clearly it was found.

    ``mean_agreement`` is the fit of the one chorus starting at ``bar_index``
    with the chart ``shift`` semitones up, ``runner_up`` the best fit of any
    rival (another phase of the form in any key, or this start in another key),
    and ``margin`` the gap between them.
    """

    bar_index: int
    start: float
    mean_agreement: float
    runner_up: float
    margin: float
    shift: int = 0
    #: How many keys were searched; more than one raises the agreement gate.
    keys: int = 1

    def refusal(
        self, min_agreement: float | None = None, min_margin: float = ANCHOR_MIN_MARGIN
    ) -> str | None:
        """Why this start should not be trusted, or ``None`` if it can be."""
        if min_agreement is None:
            min_agreement = _agreement_gate(self.keys)
        if isnan(self.margin):
            return "only one candidate phase, so there is nothing to tell it from"
        if self.mean_agreement < min_agreement:
            return f"agreement {self.mean_agreement:.2f} under {min_agreement:.2f}"
        if self.margin < min_margin:
            return (
                f"margin {self.margin:.2f} under {min_margin:.2f}: "
                "another start or key fits as well"
            )
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
    shifts: Sequence[int] = AS_WRITTEN,
) -> Anchor | None:
    """Find which downbeat opens the chart, and in which key, or ``None``.

    Each start is scored on its own chorus in each of ``shifts``, so an intro
    or a soloist's different chorus costs nothing. The anchor is the first start
    on the best phase that fits in some key. Rivals are every other phase in
    every key and the chosen start in every other key, so more keys never make
    a wrong chart look right. The form length is the chart's, never searched.
    """
    size = len(templates)
    by_shift = keyed_agreement(chroma, downbeats, templates, shifts)
    bars = len(downbeats) - 1
    # A whole chorus is the evidence; a short block only when no whole one fits.
    need = size if bars >= size else min(min_bars, size)
    if size < 1 or bars < need:
        return None

    def fit(first: int, shift: int) -> float:
        rows = np.arange(first, min(first + size, bars))
        return float(by_shift[shift][rows, rows - first].mean())

    starts = range(bars - need + 1)
    fits = {(s, k): fit(s, k) for s in starts for k in shifts}
    # Phase = position within the form; "mod size" groups the repeats of it.
    phase_best: dict[int, float] = {}
    for (s, _), value in fits.items():
        phase_best[s % size] = max(phase_best.get(s % size, -np.inf), value)
    best_phase = max(phase_best, key=lambda p: phase_best[p])

    on_phase = [s for s in starts if s % size == best_phase]
    best_key = {s: max(shifts, key=lambda k: fits[(s, k)]) for s in on_phase}
    gate = _agreement_gate(len(shifts))
    chosen = next((s for s in on_phase if fits[(s, best_key[s])] >= gate), None)
    if chosen is None:
        # Not hidden: the refusal reports the best fit it found.
        chosen = max(on_phase, key=lambda s: fits[(s, best_key[s])])
    key = best_key[chosen]

    other_phases = [v for (s, _), v in fits.items() if s % size != best_phase]
    other_keys = [fits[(chosen, k)] for k in shifts if k != key]
    # Without another phase to compare, the start is forced, not found.
    runner_up = max([*other_phases, *other_keys]) if other_phases else nan
    return Anchor(
        bar_index=chosen,
        start=float(downbeats[chosen]),
        mean_agreement=fits[(chosen, key)],
        runner_up=runner_up,
        margin=fits[(chosen, key)] - runner_up,
        shift=key,
        keys=len(shifts),
    )
