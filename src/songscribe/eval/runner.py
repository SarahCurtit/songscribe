"""Run a chord backend over a dataset and aggregate the scores.

Estimates are cached as ``.lab`` files keyed by backend, input variant (full
mix or separated stem) and track. ``madmom``
runs at roughly 2x realtime on one CPU, so the 8-hour Isophonics Beatles set
takes about 4 hours -- long enough that a run will sometimes be interrupted,
and far too long to repeat because a scoring rule changed. Re-scoring cached
estimates under a different vocabulary is instant, and it is the scoring rules
that get iterated on, not usually the model.

Two corpus-level aggregates are reported, because they answer different
questions and MIREX reports both:

* **mean** -- the unweighted mean of per-track WCSR. Every song counts once.
* **weighted** -- total correct duration over total compared duration. Every
  second counts once, so long songs pull harder.

Heavy imports stay inside functions, per the project's rule: importing this
module must not drag in librosa or madmom.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from math import isnan, nan
from pathlib import Path

from ..types import ChordEvent
from . import metrics
from .dataset import Dataset, Track

log = logging.getLogger(__name__)

#: Cache filenames are derived from the track id, which contains directory
#: separators. Flattening them keeps the cache one directory deep and avoids
#: having to recreate the corpus tree.
_CACHE_SEPARATOR = "__"
#: Appended to the backend's cache directory for estimates made on the
#: separated stem. No registered backend name contains "+", so it cannot collide.
_SEPARATED_SUFFIX = "+separated"


@dataclass(frozen=True, slots=True)
class TrackResult:
    """Scores for one track."""

    track_id: str
    duration: float
    scores: dict[str, metrics.Score]
    segmentation: metrics.Segmentation
    reference_chords: int
    estimate_chords: int
    alignment: metrics.Alignment | None = None
    #: Sequence similarity of the chart each side would print: no timing in it
    #: at all, so a uniformly late transcription still scores well. This is the
    #: number that tracks "is the chart playable", where WCSR tracks "is each
    #: second labelled right".
    chart: float = nan
    reference_chart_chords: int = 0
    estimate_chart_chords: int = 0
    #: Chart order *and* harmonic rhythm, in bars rather than seconds, so it
    #: survives a tempo difference. ``nan`` when the reference's own chords do
    #: not fall on a grid -- see metrics.HarmonicRhythm.is_reliable.
    grid: float = nan
    harmonic: metrics.HarmonicRhythm | None = None


@dataclass(frozen=True, slots=True)
class Report:
    """Everything one evaluation run produced."""

    dataset: str
    backend: str
    vocabularies: tuple[str, ...]
    results: tuple[TrackResult, ...]
    skipped: tuple[str, ...] = ()

    def mean(self, vocabulary: str) -> float:
        """Unweighted mean of per-track WCSR, skipping unscoreable tracks."""
        values = [
            r.scores[vocabulary].wcsr
            for r in self.results
            if vocabulary in r.scores and not isnan(r.scores[vocabulary].wcsr)
        ]
        return sum(values) / len(values) if values else nan

    def weighted(self, vocabulary: str) -> float:
        """Duration-weighted WCSR across the whole corpus."""
        correct = sum(r.scores[vocabulary].correct for r in self.results if vocabulary in r.scores)
        compared = sum(
            r.scores[vocabulary].compared for r in self.results if vocabulary in r.scores
        )
        return correct / compared if compared > metrics.MIN_SEGMENT else nan

    def mean_segmentation(self) -> float:
        values = [r.segmentation.score for r in self.results if not isnan(r.segmentation.score)]
        return sum(values) / len(values) if values else nan

    def mean_grid(self) -> float:
        """Unweighted mean grid similarity over the tracks that have a grid.

        Tracks whose reference has no recoverable harmonic rhythm are absent,
        not zero -- the same rule skipped tracks follow. :meth:`graded_grids`
        says how many contributed.
        """
        values = [r.grid for r in self.results if not isnan(r.grid)]
        return sum(values) / len(values) if values else nan

    def graded_grids(self) -> int:
        """How many tracks the grid score could actually be computed for."""
        return sum(1 for r in self.results if not isnan(r.grid))

    def mean_chart(self) -> float:
        """Unweighted mean chart-sequence similarity.

        Unweighted on purpose, and not duration-weighted like WCSR: a chart is
        a chart whether the song is two minutes or seven, so every song counts
        once here regardless of length.
        """
        values = [r.chart for r in self.results if not isnan(r.chart)]
        return sum(values) / len(values) if values else nan

    @property
    def total_duration(self) -> float:
        return sum(r.duration for r in self.results)


# --------------------------------------------------------------------------- #
# Estimates
# --------------------------------------------------------------------------- #


def flatten_track_id(track_id: str) -> str:
    """Collapse a track id's directory separators into one filename.

    Shared by the writer and the reader on purpose: if they flattened
    differently, every lookup would miss and the cache would silently re-run
    the model on every track.
    """
    return track_id.replace("/", _CACHE_SEPARATOR).replace("\\", _CACHE_SEPARATOR)


def cache_path(cache_dir: Path, backend: str, track_id: str, separated: bool = False) -> Path:
    """Where a cached estimate for one track, backend and input variant lives.

    Separated and full-mix estimates are kept apart: separation is the largest
    accuracy factor in the pipeline, so reusing one as the other would report
    a score for an input the model never saw.
    """
    variant = f"{backend}{_SEPARATED_SUFFIX}" if separated else backend
    return cache_dir / variant / f"{flatten_track_id(track_id)}.lab"


def estimate(
    track: Track,
    backend: str,
    cache_dir: Path | None = None,
    use_separation: bool = False,
    refresh: bool = False,
) -> list[ChordEvent]:
    """Chord estimate for one track, from cache when available.

    ``N`` spans are kept, unlike in the rendering pipeline: "no chord here" is
    a label the reference also uses, and discarding it would turn a correct
    silence into a gap the scorer has to infer.
    """
    from .. import lab

    cached = cache_path(cache_dir, backend, track.id, use_separation) if cache_dir else None
    if cached is not None and cached.exists() and not refresh:
        log.debug("cache hit %s", cached)
        return lab.read(cached)

    if track.audio is None:
        raise FileNotFoundError(f"{track.id}: manifest has no audio path")
    if not track.audio.exists():
        raise FileNotFoundError(f"{track.id}: no audio at {track.audio}")

    events, separated = _recognise(track.audio, backend, use_separation=use_separation)
    if cache_dir is not None:
        # Keyed on what actually ran, not what was asked for: a full-mix
        # fallback cached as separated would be served as such on every rerun.
        lab.write(cache_path(cache_dir, backend, track.id, separated), events)
    return events


def _recognise(
    audio_path: Path, backend: str, use_separation: bool
) -> tuple[list[ChordEvent], bool]:
    """Run the chord branch of the pipeline on one file.

    Also returns whether separation actually ran, which differs from
    ``use_separation`` when demucs is missing.
    """
    import tempfile

    from .. import audio, chords, separate

    source: str | Path = audio_path
    temporary: tempfile.TemporaryDirectory | None = None

    if use_separation:
        if separate.available():
            temporary = tempfile.TemporaryDirectory(prefix="songscribe-eval-")
            _, source = separate.isolate_vocals(audio_path, temporary.name)
        else:
            # Loud, because a run silently done on the full mix is not
            # comparable with one done on the accompaniment stem, and the gap
            # is the single largest accuracy factor in the pipeline.
            log.warning("demucs not installed; scoring %s on the full mix", audio_path.name)

    try:
        samples, sample_rate = audio.load(source, chords.sample_rate_for(backend))
        return chords.recognise(samples, sample_rate, backend=backend), temporary is not None
    finally:
        if temporary is not None:
            temporary.cleanup()


def read_estimate(estimates_dir: Path, track_id: str) -> list[ChordEvent] | None:
    """Load a pre-computed estimate for ``track_id``, if one is there.

    Accepts the flattened cache layout and a mirrored corpus tree, so a
    directory written by another tool can be scored without rearranging it.
    """
    from .. import lab

    flattened = flatten_track_id(track_id)
    for candidate in (estimates_dir / f"{flattened}.lab", estimates_dir / f"{track_id}.lab"):
        if candidate.exists():
            return lab.read(candidate)
    return None


# --------------------------------------------------------------------------- #
# Scoring a dataset
# --------------------------------------------------------------------------- #


def score_track(
    track: Track,
    estimated: list[ChordEvent],
    vocabularies: tuple[str, ...] = metrics.DEFAULT_VOCABULARIES,
    check_alignment: bool = True,
    reference: list[ChordEvent] | None = None,
) -> TrackResult:
    """Score one track's estimate against its reference annotation.

    ``check_alignment`` also sweeps a time offset, which costs no model time
    and catches a recording that does not share the reference's timeline. It
    only ever reports; the score itself is never shifted.

    ``reference`` takes an already-parsed annotation, so a caller can read it
    under its own error handling; otherwise it is read from ``track.reference``.
    """
    from .. import lab

    if reference is None:
        reference = lab.read(track.reference)
    start, end = metrics.extent(reference)
    scores = metrics.score_all(reference, estimated, vocabularies)

    alignment = None
    if check_alignment and scores:
        # Sweep in whichever vocabulary is being reported: asking for a
        # vocabulary the caller excluded would report a gain against a number
        # that appears nowhere in the output. Keyed off `scores` rather than
        # the argument, so an empty vocabulary list skips the sweep instead of
        # indexing off the end of it.
        vocabulary = "majmin" if "majmin" in scores else next(iter(scores))
        alignment = metrics.best_offset(reference, estimated, vocabulary)

    # The chart metrics project each chord on its own, which mirex cannot do
    # (it is a pairwise pitch-class overlap), so it is never picked here. The
    # alignment sweep above needs no such guard: it goes through score().
    chart_vocabulary = (
        "majmin" if "majmin" in scores else next((v for v in scores if v != "mirex"), "majmin")
    )
    reference_chart = metrics.chart_sequence(reference, chart_vocabulary)
    estimate_chart = metrics.chart_sequence(estimated, chart_vocabulary)
    harmonic = metrics.harmonic_rhythm(metrics.dedupe_chart(reference, chart_vocabulary))

    return TrackResult(
        track_id=track.id,
        duration=end - start,
        scores=scores,
        segmentation=metrics.segmentation(reference, estimated),
        reference_chords=len(reference),
        estimate_chords=len(estimated),
        alignment=alignment,
        chart=metrics.sequence_similarity(reference_chart, estimate_chart),
        reference_chart_chords=len(reference_chart),
        estimate_chart_chords=len(estimate_chart),
        grid=metrics.grid_score(reference, estimated, chart_vocabulary),
        harmonic=harmonic,
    )


def evaluate(
    dataset: Dataset,
    backend: str | None = None,
    estimates_dir: Path | None = None,
    vocabularies: tuple[str, ...] = metrics.DEFAULT_VOCABULARIES,
    cache_dir: Path | None = None,
    use_separation: bool = False,
    refresh: bool = False,
    limit: int | None = None,
    on_track: Callable[[TrackResult], None] | None = None,
    check_alignment: bool = True,
) -> Report:
    """Score every usable track in ``dataset``.

    Exactly one source of estimates is required: ``backend`` runs a model,
    ``estimates_dir`` reads ``.lab`` files somebody else produced. A track
    whose reference or estimate is missing is skipped and named in the report
    rather than scored as zero -- an absent measurement is not a wrong one.
    """
    from .. import lab

    if (backend is None) == (estimates_dir is None):
        raise ValueError("pass exactly one of backend= or estimates_dir=")
    for vocabulary in vocabularies:
        if vocabulary not in metrics.VOCABULARIES:
            raise ValueError(
                f"unknown vocabulary {vocabulary!r}, expected one of {sorted(metrics.VOCABULARIES)}"
            )

    tracks = dataset.tracks if estimates_dir else dataset.runnable()
    if limit is not None:
        tracks = tracks[:limit]

    results: list[TrackResult] = []
    skipped: list[str] = []

    for track in tracks:
        if not track.reference.exists():
            log.warning("%s: no reference at %s", track.id, track.reference)
            skipped.append(track.id)
            continue

        # Read before estimating, so a bad annotation costs no model time, and
        # under its own narrow catch: one unparseable file must not discard the
        # tracks already scored, but an error from the metrics code still should.
        try:
            reference = lab.read(track.reference)
        except (OSError, ValueError) as error:
            log.warning("%s: unusable reference: %s", track.id, error)
            skipped.append(track.id)
            continue

        try:
            if estimates_dir is not None:
                estimated = read_estimate(estimates_dir, track.id)
                if estimated is None:
                    log.warning("%s: no estimate under %s", track.id, estimates_dir)
                    skipped.append(track.id)
                    continue
            else:
                estimated = estimate(
                    track,
                    backend,
                    cache_dir=cache_dir,
                    use_separation=use_separation,
                    refresh=refresh,
                )
        except (OSError, ValueError, ImportError) as error:
            log.warning("%s: %s", track.id, error)
            skipped.append(track.id)
            continue

        result = score_track(
            track, estimated, vocabularies, check_alignment=check_alignment, reference=reference
        )
        results.append(result)
        if on_track is not None:
            on_track(result)

    return Report(
        dataset=dataset.name,
        backend=backend or f"estimates:{estimates_dir}",
        vocabularies=tuple(vocabularies),
        results=tuple(results),
        skipped=tuple(skipped),
    )
