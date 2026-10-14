"""How far a predicted chord timeline agrees with the ground truth, over time.

Both timelines are in the shorthand chord backends emit (``Am``, ``Bb7``,
``F#m7b5``), with any quality :func:`~songscribe.chords.normalise_label` does
not know passed through as Harte (``A:7(#5)``). Every instant inside the
annotated span gets one level:

``exact``   same root and same chord
``majmin``  same root and same triad family -- a triad model saying ``Bb``
            over a ``Bb7`` is as right as its vocabulary allows
``root``    same root only
``miss``    different root, or no chord predicted where one is sounding

Roots compare as pitch classes, so ``A#`` and ``Bb`` are the same chord. On a
bar with an alternate chord, the better of the two matches counts.

Instants where the reference says ``N``, or that lie outside the annotated
span, score nothing: what plays there was deliberately left unlabelled.
This is a viewer's at-a-glance score, not the evaluation metric.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Sequence
from typing import Any

from ..chords import NO_CHORD

LEVELS = ("exact", "majmin", "root", "miss")

_CHORD = re.compile(r"([A-G])([#b]?)(.*)")
_NATURALS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_ACCIDENTALS = {"": 0, "#": 1, "b": -1}

# Interval below which two event boundaries count as the same instant: the
# reference is rounded to milliseconds, so finer slivers are rounding, not
# disagreement.
EPSILON = 1e-3


def parse(label: str) -> tuple[int, str] | None:
    """``Bb7`` -> ``(10, "7")``; ``N`` or an unparseable label -> ``None``.

    The root becomes a pitch class so ``A#`` and ``Bb`` compare equal, and a
    slash bass is dropped, as a chord chart would.
    """
    match = _CHORD.fullmatch(label.strip())
    if not match:
        return None
    root, accidental, quality = match.groups()
    quality = quality.removeprefix(":").partition("/")[0]
    return (_NATURALS[root] + _ACCIDENTALS[accidental]) % 12, quality


def family(quality: str) -> str:
    """The triad a quality is built on: ``maj``, ``min``, ``dim``, ``aug`` or ``sus``.

    Accepts both shorthand (``m7``, ``m7b5``) and Harte (``min7``, ``hdim7``)
    spellings. Order matters: ``m7b5`` and ``hdim7`` start like a minor and a
    major chord respectively but are diminished.
    """
    if "dim" in quality or quality.startswith(("m7b5", "o", "ø")):
        return "dim"
    if quality.startswith(("aug", "+")):
        return "aug"
    if quality.startswith("sus"):
        return "sus"
    if quality.startswith("m") and not quality.startswith("maj"):
        return "min"
    return "maj"


def level(reference: str, estimate: str) -> str | None:
    """The agreement level of one estimate label against one reference label."""
    ref = parse(reference)
    if ref is None:
        return None  # reference N: not scored
    est = parse(estimate)
    if est is None or est[0] != ref[0]:
        return "miss"
    if est[1] == ref[1]:
        return "exact"
    if family(est[1]) == family(ref[1]):
        return "majmin"
    return "root"


def compare(
    reference: list[dict[str, Any]],
    estimate: list[dict[str, Any]],
    alternates: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Segment-by-segment agreement and its time-weighted totals.

    Events are dicts with ``start``, ``end`` and a ``short`` (reference and
    alternates) or ``label`` (estimate) key. Where a bar carries an alternate
    chord, an estimate is scored against whichever of the two it matches
    better: the alternate is a substitution a player may take, so hearing it
    is not a mistake. Such segments carry ``"alternate": True``.

    Returns ``segments`` -- ``{start, end, level, alternate}`` for drawing --
    and ``totals``: for each level, the fraction of scored time at *that level
    or better*, so ``totals["root"]`` reads as root accuracy.
    """
    boundaries = sorted(
        {e["start"] for e in [*reference, *alternates, *estimate]}
        | {e["end"] for e in [*reference, *alternates, *estimate]}
    )
    span_start = min((e["start"] for e in reference), default=0.0)
    span_end = max((e["end"] for e in reference), default=0.0)

    segments: list[dict[str, Any]] = []
    for start, end in itertools.pairwise(boundaries):
        if end - start < EPSILON or end <= span_start or start >= span_end:
            continue
        middle = (start + end) / 2
        ref = _label_at(reference, middle, "short")
        if ref is None:
            continue
        guess = _label_at(estimate, middle, "label") or NO_CHORD
        found = level(ref, guess)
        if found is None:
            continue
        via_alternate = False
        alt = _label_at(alternates, middle, "short")
        if alt is not None:
            alt_found = level(alt, guess)
            if alt_found is not None and LEVELS.index(alt_found) < LEVELS.index(found):
                found, via_alternate = alt_found, True
        last = segments[-1] if segments else None
        if (
            last
            and last["level"] == found
            and last["alternate"] == via_alternate
            and last["end"] == start
        ):
            last["end"] = end
        else:
            segments.append(
                {"start": start, "end": end, "level": found, "alternate": via_alternate}
            )

    scored = sum(s["end"] - s["start"] for s in segments)
    totals: dict[str, float | None] = {}
    for rank, name in enumerate(LEVELS[:-1]):
        good = sum(s["end"] - s["start"] for s in segments if LEVELS.index(s["level"]) <= rank)
        totals[name] = good / scored if scored else None
    return {"segments": segments, "totals": totals, "scored_seconds": scored}


def respell(estimate: list[dict[str, Any]], reference: list[dict[str, Any]]) -> list[str]:
    """Each estimate label, its root spelled the way the reference spells it.

    Models pick sharps or flats arbitrarily (madmom always says ``A#``); a
    chart in F says ``Bb``. Same pitch class, so the score already treats
    them as one chord -- this only stops the display looking like a mismatch.
    The reference chord at each estimate's midpoint decides the spelling.
    """
    spelled: list[str] = []
    for event in estimate:
        label = event["label"]
        ref = _label_at(reference, (event["start"] + event["end"]) / 2, "short")
        ours, theirs = parse(label), parse(ref) if ref else None
        if ours and theirs and ours[0] == theirs[0]:
            root, accidental, _ = _CHORD.fullmatch(ref.strip()).groups()
            label = root + accidental + _CHORD.fullmatch(label.strip()).group(3)
        spelled.append(label)
    return spelled


def _label_at(events: Sequence[dict[str, Any]], time: float, key: str) -> str | None:
    for event in events:
        if event["start"] <= time < event["end"]:
            return event[key]
    return None
