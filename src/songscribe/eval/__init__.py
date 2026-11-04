"""Chord recognition evaluation.

Measures a chord backend against reference annotations, so "the model got better" is a number
rather than an impression. The baseline it exists to beat is the ``template`` backend; the corpus
it is wired up for is Isophonics. See ``EVALUATION.md`` for how to set up a corpus and read the
scores. Only ``runner`` touches audio or models, so scoring is tested with no weights.
"""

from __future__ import annotations

from . import consensus
from .consensus import Disagreement, Disagreements, disagreements
from .dataset import ANNOTATION_ROOT_ENV, AUDIO_ROOT_ENV, Dataset, Track, load_manifest
from .harte import ALTERNATIVE_SEPARATOR, Chord, alternatives, parse
from .metrics import (
    DEFAULT_VOCABULARIES,
    PAIRWISE_VOCABULARIES,
    VOCABULARIES,
    Score,
    Segmentation,
    score,
    score_all,
    segmentation,
)
from .substitution import RELATIONS, Grade, GradedScore, Relation, grade, graded_score

__all__ = [
    "ALTERNATIVE_SEPARATOR",
    "ANNOTATION_ROOT_ENV",
    "AUDIO_ROOT_ENV",
    "DEFAULT_VOCABULARIES",
    "PAIRWISE_VOCABULARIES",
    "RELATIONS",
    "VOCABULARIES",
    "Chord",
    "Dataset",
    "Disagreement",
    "Disagreements",
    "Grade",
    "GradedScore",
    "Relation",
    "Score",
    "Segmentation",
    "Track",
    "alternatives",
    # The module, not its score/graded_score: those names are taken here by the
    # strict ones, and `consensus.score` beside `score` is the distinction.
    "consensus",
    "disagreements",
    "grade",
    "graded_score",
    "load_manifest",
    "parse",
    "score",
    "score_all",
    "segmentation",
]
