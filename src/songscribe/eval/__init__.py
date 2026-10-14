"""Chord recognition evaluation.

Measures a chord backend against reference annotations, so "the model got better" is a number
rather than an impression. The baseline it exists to beat is the ``template`` backend; the corpus
it is wired up for is Isophonics. See ``EVALUATION.md`` for how to obtain a corpus, and what may
be redistributed. Only ``runner`` touches audio or models, so scoring is tested with no weights.
"""

from __future__ import annotations

from .dataset import ANNOTATION_ROOT_ENV, AUDIO_ROOT_ENV, Dataset, Track, load_manifest
from .harte import Chord, parse
from .metrics import (
    DEFAULT_VOCABULARIES,
    VOCABULARIES,
    Score,
    Segmentation,
    score,
    score_all,
    segmentation,
)

__all__ = [
    "ANNOTATION_ROOT_ENV",
    "AUDIO_ROOT_ENV",
    "DEFAULT_VOCABULARIES",
    "VOCABULARIES",
    "Chord",
    "Dataset",
    "Score",
    "Segmentation",
    "Track",
    "load_manifest",
    "parse",
    "score",
    "score_all",
    "segmentation",
]
