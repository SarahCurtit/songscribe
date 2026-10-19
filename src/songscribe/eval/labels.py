"""Chord labels reduced to one rough symbol per beat, and how well two sequences agree.

A symbol is the root and whether the chord has a minor third, so ``C``, ``C7`` and ``Cmaj7``
are one symbol and ``Cm``, ``Cm7`` and ``Cdim`` another. Estimates disagree about
extensions and flicker; they rarely do about a bar's root and third. Pure numpy.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import numpy as np

from ..types import ChordEvent
from .harte import parse

#: Symbol of a beat that has no chord (``N``, unparseable, or nothing sounding).
NO_SYMBOL = -1
#: Interval of a minor third; a chord without a major or minor third (sus, power) counts as major.
MINOR_THIRD = 3
#: Credit for the same root with the other third (``A`` against ``Am``), against 1 for a
#: match and 0 for a different root. Estimates confuse a chord with its parallel often.
ROOT_CREDIT = 0.5


@lru_cache(maxsize=4096)
def symbol(label: str) -> int:
    """``2 * root + minor`` for a chord label, ``NO_SYMBOL`` for anything without a root."""
    chord = parse(label)
    if chord is None or chord.root is None:
        return NO_SYMBOL
    return 2 * chord.root + (MINOR_THIRD in chord.intervals)


def beat_symbols(events: Sequence[ChordEvent], edges: np.ndarray) -> np.ndarray:
    """The symbol sounding for most of each beat between consecutive ``edges``.

    A chord that moves a fraction of a beat early or flickers for a moment does not change
    the beat's symbol, which is what makes the comparison tolerant of both. A beat that is
    mostly ``N`` or uncovered has no symbol.
    """
    count = max(len(edges) - 1, 0)
    out = np.full(count, NO_SYMBOL, dtype=np.int16)
    ordered = sorted(events, key=lambda e: (e.start, e.end))
    first = 0
    for beat in range(count):
        low, high = float(edges[beat]), float(edges[beat + 1])
        while first < len(ordered) and ordered[first].end <= low:
            first += 1
        held: dict[int, float] = {}
        for event in ordered[first:]:
            if event.start >= high:
                break
            sym = symbol(event.label)
            overlap = min(event.end, high) - max(event.start, low)
            if sym != NO_SYMBOL and overlap > 0:
                held[sym] = held.get(sym, 0.0) + overlap
        if held:
            # Ties go to the chord that arrived last.
            best = max(reversed(held), key=lambda s: held[s])
            if held[best] > (high - low) - sum(held.values()):
                out[beat] = best
    return out


def agreement(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-beat agreement of two symbol arrays: 1 equal, ``ROOT_CREDIT`` same root, else 0."""
    known = (a != NO_SYMBOL) & (b != NO_SYMBOL)
    same = known & (a == b)
    root = known & ~same & ((a >> 1) == (b >> 1))
    return same + ROOT_CREDIT * root
