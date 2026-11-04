"""Corpus health check from annotations alone -- no audio, no model.

Reports annotated time, whether every label parses, and how much time each
vocabulary can score. Worth knowing before a model run: a vocabulary that
silently excludes a third of the reference just looks like a confident number.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ..types import ChordEvent
from .consensus import Disagreements, disagreements
from .dataset import Dataset
from .harte import ALTERNATIVE_SEPARATOR, parse
from .metrics import DEFAULT_VOCABULARIES, compare

#: Labels reported individually in the summary before the tail is collapsed.
TOP_LABELS = 15


@dataclass(frozen=True, slots=True)
class LabelStats:
    """How much of a corpus one chord label accounts for."""

    label: str
    segments: int
    duration: float
    parses: bool


@dataclass(frozen=True, slots=True)
class Survey:
    """What :func:`survey` found in a corpus's references."""

    dataset: str
    tracks: int
    segments: int
    duration: float
    labels: tuple[LabelStats, ...]
    scoreable: dict[str, float]
    unreadable: tuple[tuple[str, str], ...] = ()
    #: What the corpus's sources disagree about, where a label prints more than
    #: one chord for a bar. Empty for a corpus that prints one chord per segment.
    disagreements: Disagreements | None = None

    @property
    def unparseable(self) -> tuple[LabelStats, ...]:
        """Labels this project cannot read, worst offender first.

        Any entry here is a gap in ``harte.py`` or in
        ``chords.HARTE_QUALITIES``, not a property of the corpus -- the time
        it accounts for is time no vocabulary can score.
        """
        return tuple(sorted((s for s in self.labels if not s.parses), key=_by_duration))


def _by_duration(stats: LabelStats) -> float:
    return -stats.duration


def survey(
    dataset: Dataset,
    vocabularies: tuple[str, ...] = DEFAULT_VOCABULARIES,
) -> Survey:
    """Tally labels and scoreable time across ``dataset``'s references."""
    from .. import lab

    segment_counts: Counter[str] = Counter()
    durations: Counter[str] = Counter()
    unreadable: list[tuple[str, str]] = []
    # Only the segments whose label prints more than one chord are kept, which on
    # a hand-entered corpus is a few bars a tune, so this stays small where
    # holding every event would not.
    contested: list[ChordEvent] = []
    tracks = 0

    for track in dataset.tracks:
        try:
            events = lab.read(track.reference)
        except (OSError, ValueError) as error:
            unreadable.append((track.id, str(error)))
            continue
        tracks += 1
        for event in events:
            segment_counts[event.label] += 1
            durations[event.label] += event.duration
            if ALTERNATIVE_SEPARATOR in event.label:
                contested.append(event)

    total = sum(durations.values())
    labels = tuple(
        sorted(
            (
                LabelStats(
                    label=label,
                    segments=segment_counts[label],
                    duration=durations[label],
                    parses=parse(label) is not None,
                )
                for label in segment_counts
            ),
            key=_by_duration,
        )
    )

    # Scoreability is asked of the comparison itself rather than reimplemented
    # here: a label is scoreable in a vocabulary exactly when comparing it
    # against itself is not "skip". That keeps this honest for `mirex` too,
    # which has no per-chord projection to interrogate.
    scoreable: dict[str, float] = {}
    for vocabulary in vocabularies:
        kept = sum(
            durations[s.label]
            for s in labels
            if compare(parse(s.label), parse(s.label), vocabulary) is not None
        )
        scoreable[vocabulary] = kept / total if total else 0.0

    return Survey(
        dataset=dataset.name,
        tracks=tracks,
        segments=sum(segment_counts.values()),
        duration=total,
        labels=labels,
        scoreable=scoreable,
        unreadable=tuple(unreadable),
        disagreements=disagreements(contested, annotated=total) if contested else None,
    )
