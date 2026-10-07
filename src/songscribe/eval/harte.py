"""Chord labels parsed into pitch content, for comparison.

Scoring cannot be done on label strings. ``C:maj`` and ``C`` name the same
chord, ``C#`` and ``Db`` name the same root, ``C:maj7`` and ``C:maj`` agree on
their triad and differ only in an extension some vocabularies ignore. All of
that has to be decided on the notes a label denotes, not on its spelling.

So both sides of an evaluation are parsed into a :class:`Chord` -- a root
pitch class plus the set of semitone intervals above it -- and the comparison
happens there. Harte notation (``C:min7(b5)/b3``, what reference annotations
use) and this project's compact shorthand (``Cm7b5``, what backends emit) both
parse; :func:`parse` accepts either, so a reference and an estimate go through
the same door.

Reference: Harte et al., "Symbolic representation of musical chords: a
proposed syntax for text annotations", ISMIR 2005.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..chords import HARTE_QUALITIES, NO_CHORD, PITCH_CLASSES

#: Semitone of each natural note. Accidentals are applied on top, so ``Abb``
#: and ``G`` come out as the same pitch class -- which is the point: an
#: enharmonic respelling is not a chord error.
NATURALS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

#: Semitone above the root for each scale degree. Compound degrees fold into
#: one octave (a 9th is a 2nd, an 11th a 4th): comparison is modulo the octave
#: throughout, since no vocabulary in use distinguishes voicings.
DEGREES = {1: 0, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9, 7: 11, 8: 12, 9: 14, 10: 16, 11: 17, 13: 21}

#: Harte shorthands, as the degrees they stand for. The root (degree 1) is
#: implicit in every one of them and added separately.
SHORTHANDS: dict[str, tuple[str, ...]] = {
    "maj": ("3", "5"),
    "min": ("b3", "5"),
    "dim": ("b3", "b5"),
    "aug": ("3", "#5"),
    "maj7": ("3", "5", "7"),
    "min7": ("b3", "5", "b7"),
    "7": ("3", "5", "b7"),
    "dim7": ("b3", "b5", "bb7"),
    "hdim7": ("b3", "b5", "b7"),
    "minmaj7": ("b3", "5", "7"),
    "maj6": ("3", "5", "6"),
    "min6": ("b3", "5", "6"),
    "9": ("3", "5", "b7", "9"),
    "maj9": ("3", "5", "7", "9"),
    "min9": ("b3", "5", "b7", "9"),
    "11": ("3", "5", "b7", "9", "11"),
    "maj11": ("3", "5", "7", "9", "11"),
    "min11": ("b3", "5", "b7", "9", "11"),
    "13": ("3", "5", "b7", "9", "11", "13"),
    "maj13": ("3", "5", "7", "9", "11", "13"),
    "min13": ("b3", "5", "b7", "9", "11", "13"),
    "sus2": ("2", "5"),
    "sus4": ("4", "5"),
}

#: This project's shorthand, inverted back to Harte. ``chords.normalise_label``
#: is the forward direction, so inverting its table means the two notations can
#: never drift apart: a quality added there is understood here for free.
COMPACT_TO_HARTE = {compact: harte for harte, compact in HARTE_QUALITIES.items()}

#: Label for "no chord sounding". Shared with the rest of the pipeline.
NO_CHORD_LABEL = NO_CHORD

#: Harte's label for "a chord is sounding but the annotator could not name it".
#: Distinct from ``N``, and excluded from scoring rather than counted as a
#: miss -- penalising a model for disagreeing with an admitted non-answer would
#: measure the annotation, not the model.
UNKNOWN_LABEL = "X"

_ROOT = re.compile(r"([A-G])([b#]*)")
_QUALITY = re.compile(r"^([a-zA-Z0-9]*)(?:\(([^)]*)\))?$")
_DEGREE = re.compile(r"^(\*?)([b#]*)(\d+)$")

# The root is matched greedily, so "Bb" is B-flat rather than B with a quality
# of "b". That is unambiguous because no quality in either notation begins with
# an accidental.
_ROOT_PREFIX = re.compile(r"^[A-G][b#]*")


@dataclass(frozen=True, slots=True)
class Chord:
    """The pitch content a chord label denotes.

    ``root`` is a pitch class 0-11, or ``None`` for ``N`` and ``X``.
    ``intervals`` are semitones above the root, modulo the octave, and always
    contain 0. ``bass`` is the semitone of the bass note above the root, which
    every vocabulary here ignores -- it is kept so an inversion-sensitive
    comparison can be added without reparsing.
    """

    root: int | None
    intervals: frozenset[int] = field(default_factory=frozenset)
    bass: int = 0
    unknown: bool = False

    @property
    def is_no_chord(self) -> bool:
        return self.root is None and not self.unknown

    @property
    def pitch_classes(self) -> frozenset[int]:
        """Absolute pitch classes sounding, for overlap-based comparison."""
        if self.root is None:
            return frozenset()
        return frozenset((self.root + i) % 12 for i in self.intervals)

    @property
    def root_name(self) -> str:
        """The root spelled with sharps, so enharmonics compare equal."""
        return NO_CHORD_LABEL if self.root is None else PITCH_CLASSES[self.root]


#: Singletons, since these two turn up on a large fraction of all segments.
NO_CHORD_CHORD = Chord(root=None)
UNKNOWN_CHORD = Chord(root=None, unknown=True)


def parse_root(text: str) -> int | None:
    """Pitch class of a note name such as ``F#`` or ``Ebb``, else ``None``."""
    match = _ROOT.fullmatch(text)
    if match is None:
        return None
    natural, accidentals = match.groups()
    shift = accidentals.count("#") - accidentals.count("b")
    return (NATURALS[natural] + shift) % 12


def parse_degree(text: str) -> tuple[int, bool] | None:
    """Parse one Harte degree into ``(semitone, is_omission)``.

    ``b7`` -> ``(10, False)``, ``*3`` -> ``(4, True)``. Returns ``None`` for
    anything unparseable, including degrees outside the tabulated range.
    """
    match = _DEGREE.fullmatch(text.strip())
    if match is None:
        return None
    omit, accidentals, number = match.groups()
    degree = DEGREES.get(int(number))
    if degree is None:
        return None
    shift = accidentals.count("#") - accidentals.count("b")
    return (degree + shift) % 12, bool(omit)


def parse(label: str) -> Chord | None:
    """Parse a chord label into its pitch content.

    Accepts Harte notation and this project's compact shorthand. Returns
    ``None`` for a label that cannot be parsed, rather than guessing at it --
    the same rule ``chords.normalise_label`` follows. The caller decides what
    an unparseable label means, which differs by side: an unparseable
    reference is excluded from scoring, an unparseable estimate is wrong.
    """
    label = label.strip()
    if not label:
        return None
    if label == NO_CHORD_LABEL:
        return NO_CHORD_CHORD
    if label == UNKNOWN_LABEL:
        return UNKNOWN_CHORD

    head, slash, bass_text = label.partition("/")

    match = _ROOT_PREFIX.match(head)
    if match is None:
        return None
    root = parse_root(match.group())
    if root is None:
        return None

    rest = head[match.end() :]
    if rest.startswith(":"):
        # Harte: the quality is spelled out after a colon, and may carry a
        # parenthesised list of added or omitted degrees.
        intervals = _parse_quality(rest[1:])
    elif rest:
        # Compact shorthand, as backends emit it: "Am7", "Cm7b5".
        intervals = _parse_quality(rest)
    else:
        intervals = _parse_quality(None)
    if intervals is None:
        return None

    bass = 0
    if slash:
        parsed_bass = parse_degree(bass_text)
        if parsed_bass is None or parsed_bass[1]:
            return None
        bass = parsed_bass[0]

    return Chord(root=root, intervals=frozenset(intervals), bass=bass)


def _parse_quality(quality: str | None) -> set[int] | None:
    """Intervals named by the part of a label after the root."""
    # A bare root is a major triad in both notations: "C" == "C:maj".
    if quality is None:
        return {0, *_degrees_to_semitones(SHORTHANDS["maj"])}

    match = _QUALITY.fullmatch(quality.strip())
    if match is None:
        return None
    name, degree_text = match.groups()

    if name:
        shorthand = SHORTHANDS.get(name) or SHORTHANDS.get(COMPACT_TO_HARTE.get(name, ""))
        if shorthand is None:
            return None
        intervals = {0, *_degrees_to_semitones(shorthand)}
    elif degree_text is not None:
        # "C:(1,b3,5)" -- an explicit degree list with no shorthand.
        intervals = {0}
    else:
        # A trailing colon with nothing after it names no quality at all.
        return None

    for item in (degree_text or "").split(","):
        item = item.strip()
        if not item:
            continue
        parsed = parse_degree(item)
        if parsed is None:
            return None
        semitone, omit = parsed
        if omit:
            intervals.discard(semitone)
        else:
            intervals.add(semitone)

    return intervals


def _degrees_to_semitones(degrees: tuple[str, ...]) -> set[int]:
    semitones = set()
    for degree in degrees:
        parsed = parse_degree(degree)
        if parsed is not None:
            semitones.add(parsed[0])
    return semitones
