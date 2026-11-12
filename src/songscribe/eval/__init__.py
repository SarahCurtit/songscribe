"""Chord recognition evaluation.

Measures a chord backend against reference annotations, so "the model got better" is a number
rather than an impression. The baseline it exists to beat is the ``template`` backend; the corpus
it is wired up for is Isophonics. See ``EVALUATION.md`` for how to set up a corpus and read the
scores. Only ``runner`` touches audio or models, so scoring is tested with no weights.
"""

from __future__ import annotations

from .dataset import ANNOTATION_ROOT_ENV, AUDIO_ROOT_ENV, Dataset, Track, load_manifest
from .harte import Chord, parse
from .melody import (
    ChorusMelody,
    MelodyScore,
    best_transposition,
    head_score,
    matching_transposition,
    melody_score,
    melody_sequence,
    novelty,
    raw_pitch_accuracy,
    score_choruses,
)
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
    "ANNOTATION_ROOT_ENV",
    "AUDIO_ROOT_ENV",
    "DEFAULT_VOCABULARIES",
    "PAIRWISE_VOCABULARIES",
    "RELATIONS",
    "VOCABULARIES",
    "Chord",
    "ChorusMelody",
    "Dataset",
    "Grade",
    "GradedScore",
    "MelodyScore",
    "Relation",
    "Score",
    "Segmentation",
    "Track",
    "best_transposition",
    "grade",
    "graded_score",
    "head_score",
    "load_manifest",
    "matching_transposition",
    "melody_score",
    "melody_sequence",
    "novelty",
    "parse",
    "raw_pitch_accuracy",
    "score",
    "score_all",
    "score_choruses",
    "segmentation",
]
