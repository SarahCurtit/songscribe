"""Run a chord backend over a dataset and aggregate the scores.

Estimates are cached as ``.lab`` files keyed by backend, input variant (full mix or
separated stem) and track; the report gives per-track mean and duration-weighted WCSR.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from math import isnan, nan
from pathlib import Path

from ..beats import BeatGrid
from ..types import ChordEvent
from . import chart_fit, form, form_start, metrics
from .dataset import Dataset, Track

log = logging.getLogger(__name__)

#: Cache filenames are derived from the track id, which contains directory
#: separators. Flattening them keeps the cache one directory deep and avoids
#: having to recreate the corpus tree.
_CACHE_SEPARATOR = "__"
#: Appended to the backend's cache directory for estimates made on the
#: separated stem. No registered backend name contains "+", so it cannot collide.
_SEPARATED_SUFFIX = "+separated"


class AudioDecodeError(Exception):
    """A track's audio exists but could not be decoded."""


@dataclass(frozen=True, slots=True)
class ChorusResult:
    """Scores for one scored block (see ``form.Chorus``) of a recording."""

    chorus: form.Chorus
    scores: dict[str, metrics.Score]
    segmentation: metrics.Segmentation
    chart: float = nan
    grid: float = nan
    #: Fraction of bars with the right chord (``metrics.bar_score``).
    bar: float = nan
    #: ``bar`` per scored vocabulary; ``bar`` itself is the majmin entry.
    bar_by_vocabulary: dict[str, float] = field(default_factory=dict)
    alignment: metrics.Alignment | None = None


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
    #: For a recording of the chart played several times: the results for each
    #: scored block. The track-level scores above are computed from the full
    #: choruses among them.
    choruses: tuple[ChorusResult, ...] = ()
    bar: float = nan
    bar_by_vocabulary: dict[str, float] = field(default_factory=dict)
    anchor: form_start.Anchor | None = None
    #: Seconds of the recording not in any scored block (intro, tail, unaligned).
    unscored_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class Report:
    """Everything one evaluation run produced."""

    dataset: str
    backend: str
    vocabularies: tuple[str, ...]
    results: tuple[TrackResult, ...]
    skipped: tuple[str, ...] = ()
    #: Whether separation was asked for; ``None`` when the estimates were read
    #: from disk, whose input nobody here can vouch for. Separated and full-mix
    #: scores are not comparable, so the report has to say which this was.
    separated: bool | None = None
    #: Scored tracks that ran on the full mix although separation was asked
    #: for (demucs missing). Without it a fallback leaves no trace in the output.
    fallbacks: int = 0
    #: With a bar grid: whose estimate the form's start was found on.
    anchor: str | None = None

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

    def mean_bar(self, vocabulary: str | None = None) -> float:
        """Unweighted mean of the per-track bar score, ignoring tracks without one.

        ``vocabulary`` picks an entry of ``bar_by_vocabulary``; the default is majmin.
        """
        values = [
            r.bar if vocabulary is None else r.bar_by_vocabulary.get(vocabulary, nan)
            for r in self.results
        ]
        values = [v for v in values if not isnan(v)]
        return sum(values) / len(values) if values else nan

    def chorus_count(self) -> int:
        """Number of scored choruses over all tracks."""
        return sum(len(r.choruses) for r in self.results)

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
    return _estimate(track, backend, cache_dir, use_separation, refresh)[0]


def _estimate(
    track: Track,
    backend: str,
    cache_dir: Path | None,
    use_separation: bool,
    refresh: bool,
) -> tuple[list[ChordEvent], bool]:
    """:func:`estimate`, also returning whether the estimate is of the separated stem."""
    from .. import lab, separate

    # Resolved before the lookup, so a full-mix fallback reads the entry it
    # writes; keyed on the request, every rerun without demucs would miss.
    separated = use_separation and separate.available()
    if use_separation and not separated:
        # Loud, and on a cache hit too, because a run silently done on the
        # full mix is not comparable with one done on the accompaniment stem,
        # and the gap is the single largest accuracy factor in the pipeline.
        log.warning("demucs not installed; scoring %s on the full mix", track.id)

    cached = cache_path(cache_dir, backend, track.id, separated) if cache_dir else None
    if cached is not None and cached.exists() and not refresh:
        log.debug("cache hit %s", cached)
        return lab.read(cached), separated

    if track.audio is None:
        raise FileNotFoundError(f"{track.id}: manifest has no audio path")
    if not track.audio.exists():
        raise FileNotFoundError(f"{track.id}: no audio at {track.audio}")

    events, separated = _recognise(track.audio, backend, use_separation=separated)
    if cache_dir is not None:
        # Keyed on what actually ran, not what was asked for: a full-mix
        # fallback cached as separated would be served as such on every rerun.
        lab.write(cache_path(cache_dir, backend, track.id, separated), events)
    return events, separated


def _recognise(
    audio_path: Path, backend: str, use_separation: bool
) -> tuple[list[ChordEvent], bool]:
    """Run the chord branch of the pipeline on one file.

    Also returns whether separation actually ran, which differs from
    ``use_separation`` when demucs is missing. The caller warns about that.
    """
    import tempfile

    from .. import audio, chords, separate

    source: str | Path = audio_path
    temporary: tempfile.TemporaryDirectory | None = None

    try:
        if use_separation and separate.available():
            # Inside the try, so a failed split still removes its stems:
            # evaluate() skips the track and carries on to the next one.
            temporary = tempfile.TemporaryDirectory(prefix="songscribe-eval-")
            _, source = separate.isolate_vocals(audio_path, temporary.name)
        try:
            samples, sample_rate = audio.load(source, chords.sample_rate_for(backend))
        except ImportError:
            # A missing decoder package fails every track alike; let it abort.
            raise
        except Exception as error:
            # The decoders raise unrelated types for a bad file (audioread's
            # NoBackendError, EOFError on an empty one), so wrap them all.
            raise AudioDecodeError(f"cannot decode {source}: {error!r}") from error
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
    alignment_search: float = metrics.ALIGNMENT_SEARCH_SECONDS,
) -> TrackResult:
    """Score one track's estimate against its reference annotation.

    ``check_alignment`` also tries time offsets of up to ``alignment_search``
    seconds, to report the best one. The score itself is not shifted.
    ``reference`` is an already-parsed annotation; if not given, it is read
    from ``track.reference``.
    """
    from .. import lab

    # A pre-parsed reference lets the caller read it under its own error handling.
    if reference is None:
        reference = lab.read(track.reference)
    start, end = metrics.extent(reference)
    scores = metrics.score_all(reference, estimated, vocabularies)

    alignment = None
    if check_alignment and scores:
        # The sweep costs no model time and catches a recording that does not
        # share the reference's timeline.
        #
        # Sweep in whichever vocabulary is being reported: asking for a
        # vocabulary the caller excluded would report a gain against a number
        # that appears nowhere in the output. Keyed off `scores` rather than
        # the argument, so an empty vocabulary list skips the sweep instead of
        # indexing off the end of it.
        vocabulary = "majmin" if "majmin" in scores else next(iter(scores))
        alignment = metrics.best_offset(reference, estimated, vocabulary, search=alignment_search)

    # Always majmin, whatever WCSR rows were asked for: the chart and grid rows
    # carry no vocabulary in their name, so following `-v` would make a
    # `-v root` run's chart incomparable with a default one's, silently. It
    # also projects each chord on its own, which mirex cannot.
    chart_vocabulary = "majmin"
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


def _mean(values: list[float]) -> float:
    present = [v for v in values if not isnan(v)]
    return sum(present) / len(present) if present else nan


def score_performance(
    track: Track,
    estimated: list[ChordEvent],
    *,
    grid: BeatGrid,
    bars_per_chorus: int,
    anchor: form_start.Anchor,
    vocabularies: tuple[str, ...] = metrics.DEFAULT_VOCABULARIES,
    check_alignment: bool = True,
    reference: list[ChordEvent] | None = None,
) -> TrackResult:
    """Score a recording of the chart played several times, one chorus at a time.

    Each scored block is mapped to the chart's timeline and scored with :func:`score_track`
    against the chart transposed to the anchor's key. Track-level scores combine all full
    choruses, not the best; a partial chorus is left out unless it is the only block.
    Raises ``ValueError`` if the anchor is not clear or no block can be scored.
    """
    from .. import lab

    refusal = anchor.refusal()
    if refusal is not None:
        raise ValueError(f"{track.id}: the form's start is not clear ({refusal})")
    if reference is None:
        reference = lab.read(track.reference)
    start, end = metrics.extent(reference)
    found = form.choruses(grid, bars_per_chorus, anchor.bar_index, end - start)
    # One key for the whole take; a tune that modulates is not handled.
    blocks = form.in_key(found, anchor.shift)
    reference_bar = (end - start) / bars_per_chorus
    # mirex scores pairs of chords, so a bar has no mirex class.
    bar_vocabularies = tuple(name for name in vocabularies if name != "mirex")

    results: list[ChorusResult] = []
    inner: list[TrackResult] = []
    for chorus in blocks:
        if not chorus.is_scored:
            continue
        part = reference
        if chorus.bars < bars_per_chorus:
            part = form.truncate_reference(reference, bars_per_chorus, chorus.bars)
        part = form.transpose_events(part, chorus.shift)
        result = score_track(
            track,
            form.project_estimate(estimated, chorus, origin=start),
            vocabularies,
            check_alignment,
            reference=part,
            # One bar is the widest offset worth trying; repeating chords match further out.
            alignment_search=reference_bar,
        )
        chart_lines = [start + k * reference_bar for k in range(chorus.bars + 1)]
        played_lines = grid.downbeats[chorus.first_bar : chorus.first_bar + chorus.bars + 1]
        results.append(
            ChorusResult(
                chorus=chorus,
                scores=result.scores,
                segmentation=result.segmentation,
                chart=result.chart,
                grid=result.grid,
                bar=metrics.bar_score(part, estimated, chart_lines, "majmin", played_lines),
                bar_by_vocabulary={
                    name: metrics.bar_score(part, estimated, chart_lines, name, played_lines)
                    for name in bar_vocabularies
                },
                alignment=result.alignment,
            )
        )
        inner.append(result)
    if not results:
        raise ValueError(f"{track.id}: no block of the performance could be scored")

    counted = [i for i, c in enumerate(results) if c.chorus.kind != "partial"] or [0]
    chosen = [results[i] for i in counted]
    first = inner[counted[0]]
    scores = {
        name: metrics.Score(
            vocabulary=name,
            correct=sum(c.scores[name].correct for c in chosen),
            compared=sum(c.scores[name].compared for c in chosen),
            ignored=sum(c.scores[name].ignored for c in chosen),
        )
        for name in first.scores
    }
    return TrackResult(
        track_id=track.id,
        duration=sum(inner[i].duration for i in counted),
        scores=scores,
        segmentation=metrics.Segmentation(
            over=_mean([c.segmentation.over for c in chosen]),
            under=_mean([c.segmentation.under for c in chosen]),
        ),
        reference_chords=sum(inner[i].reference_chords for i in counted),
        estimate_chords=sum(inner[i].estimate_chords for i in counted),
        alignment=chosen[0].alignment if len(results) == 1 else None,
        chart=_mean([c.chart for c in chosen]),
        reference_chart_chords=sum(inner[i].reference_chart_chords for i in counted),
        estimate_chart_chords=sum(inner[i].estimate_chart_chords for i in counted),
        grid=_mean([c.grid for c in chosen]),
        harmonic=first.harmonic,
        choruses=tuple(results),
        bar=_mean([c.bar for c in chosen]),
        bar_by_vocabulary={
            name: _mean([c.bar_by_vocabulary[name] for c in chosen]) for name in bar_vocabularies
        },
        anchor=anchor,
        unscored_seconds=form.unscored_seconds(blocks),
    )


def bar_grid(
    track: Track, beats_dir: Path | None, beat_backend: str | None, cache_dir: Path | None
) -> BeatGrid:
    """Get the bar grid of the recording, either from a ``.beats`` file or a beat tracker.

    With ``beats_dir``, read ``<track>.beats`` from that directory. Otherwise
    run ``beat_backend`` on the audio, and cache the result in ``cache_dir``
    if given.
    """
    from .. import audio, beats, lab

    if beats_dir is not None:
        flattened = flatten_track_id(track.id)
        for candidate in (beats_dir / f"{flattened}.beats", beats_dir / f"{track.id}.beats"):
            if candidate.exists():
                return beats.grid(lab.read_beats(candidate))
        raise FileNotFoundError(f"{track.id}: no .beats file under {beats_dir}")

    if beat_backend is None:
        raise ValueError("no bar grid source: pass beats_dir or beat_backend")
    cached = None
    if cache_dir is not None:
        cached = cache_dir / "beats" / beat_backend / f"{flatten_track_id(track.id)}.beats"
        if cached.exists():
            return beats.grid(lab.read_beats(cached))
    if track.audio is None or not track.audio.exists():
        raise FileNotFoundError(f"{track.id}: no audio to track beats in")
    samples, rate = audio.load(track.audio, beats.sample_rate_for(beat_backend))
    tracked = beats.track(samples, rate, backend=beat_backend)
    if cached is not None:
        lab.write_beats(cached, tracked)
    return beats.grid(tracked)


def anchor_estimate(
    track: Track,
    backend: str | None = None,
    cache_dir: Path | None = None,
    estimates_dir: Path | None = None,
    refresh: bool = False,
) -> list[ChordEvent]:
    """The chord estimate the form's start is found on.

    It is the track's ``.lab`` in ``estimates_dir``, or else ``backend`` (default
    ``chords.DEFAULT_BACKEND``) run on the full mix. The backend's estimate is cached like
    any other, so a scored run of the same backend reuses it. Prefer a backend other than
    the one being scored. An unknown backend raises.
    """
    from .. import chords

    if estimates_dir is not None:
        found = read_estimate(estimates_dir, track.id)
        if found is None:
            raise FileNotFoundError(f"{track.id}: no anchor estimate under {estimates_dir}")
        return found
    backend = backend or chords.DEFAULT_BACKEND
    chords.sample_rate_for(backend)  # raises on a name no backend has
    return estimate(track, backend, cache_dir, False, refresh)


def _check_anchor(
    backend: str | None,
    estimates_dir: Path | None,
    anchor_backend: str | None,
    anchor_dir: Path | None,
) -> str:
    """Name the anchoring estimate; warn if it is the one under test; unknown backends raise."""
    from .. import chords

    if anchor_backend is not None and anchor_dir is not None:
        raise ValueError("pass at most one of anchor_backend= or anchor_dir=")
    if anchor_dir is not None:
        same = estimates_dir is not None and anchor_dir.resolve() == estimates_dir.resolve()
        name = f"estimates:{anchor_dir}"
    else:
        name = anchor_backend or chords.DEFAULT_BACKEND
        chords.sample_rate_for(name)
        same = backend == name
    if same:
        log.warning(
            "the form's start is found on the estimate being scored (%s): it can pick the start "
            "that flatters it. Use a different --anchor-backend or --anchor-estimates.",
            name,
        )
    return name


def _score_form(
    track: Track,
    estimated: list[ChordEvent],
    reference: list[ChordEvent],
    vocabularies: tuple[str, ...],
    check_alignment: bool,
    beats_dir: Path | None,
    beat_backend: str | None,
    cache_dir: Path | None,
    anchor_backend: str | None,
    anchor_dir: Path | None,
    refresh: bool,
    shifts: Sequence[int] = chart_fit.ANY_KEY,
) -> TrackResult:
    if track.bars is None:
        raise ValueError(f"{track.id}: the manifest gives no 'bars' for this track")
    grid = bar_grid(track, beats_dir, beat_backend, cache_dir)
    found, grid = form_start.locate(
        anchor_estimate(track, anchor_backend, cache_dir, anchor_dir, refresh),
        grid,
        reference,
        bars=track.bars,
        min_bars=form.CHORUS_MIN_BARS,
        shifts=shifts,
    )
    if found is None:
        raise ValueError(f"{track.id}: the recording is too short to hold the form")
    return score_performance(
        track,
        estimated,
        grid=grid,
        bars_per_chorus=track.bars,
        anchor=found,
        vocabularies=vocabularies,
        check_alignment=check_alignment,
        reference=reference,
    )


def key_search(mode: str) -> tuple[int, ...]:
    """The keys to search for a ``--transpose`` mode: ``auto`` all twelve, ``none`` the chart's."""
    if mode == "auto":
        return chart_fit.ANY_KEY
    if mode == "none":
        return chart_fit.AS_WRITTEN
    raise ValueError(
        f"unknown transpose mode {mode!r}, expected one of {list(form.TRANSPOSE_MODES)}"
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
    beats_dir: Path | None = None,
    beat_backend: str | None = None,
    anchor_backend: str | None = None,
    anchor_dir: Path | None = None,
    transpose: str = "auto",
) -> Report:
    """Score every usable track in ``dataset``.

    Exactly one source of estimates is required: ``backend`` runs a model, ``estimates_dir``
    reads ``.lab`` files somebody else produced. A track whose reference or estimate is
    missing is skipped and named, not scored as zero. A bar grid turns on per-chorus scoring,
    from a start found on an anchor estimate (``anchor_dir`` or ``anchor_backend``, see
    :func:`anchor_estimate`); ``transpose="none"`` pins the chart's key.
    """
    import subprocess

    from .. import beats, chords, lab, separate

    if (backend is None) == (estimates_dir is None):
        raise ValueError("pass exactly one of backend= or estimates_dir=")
    if beats_dir is not None and beat_backend is not None:
        raise ValueError("pass at most one of beats_dir= or beat_backend=")
    shifts = key_search(transpose)
    if beat_backend is not None:
        beats.sample_rate_for(beat_backend)
    # Checked here because the per-track catch below takes ValueError: an
    # unknown backend would otherwise skip every track instead of aborting.
    if backend is not None:
        chords.sample_rate_for(backend)
    per_chorus = beats_dir is not None or beat_backend is not None
    anchor_name = (
        _check_anchor(backend, estimates_dir, anchor_backend, anchor_dir) if per_chorus else None
    )
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
    fallbacks = 0

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
                estimated, separated = _estimate(track, backend, cache_dir, use_separation, refresh)
                if use_separation and not separated:
                    fallbacks += 1
        # demucs failing, or an undecodable file, on one track is a per-track
        # problem like a missing file. Specific types rather than RuntimeError,
        # which would also swallow NotImplementedError from an unwired backend
        # on every track. ImportError is left out for the same reason: a missing
        # model package fails every uncached track, and skipping them would
        # score a subset.
        except (
            OSError,
            ValueError,
            subprocess.CalledProcessError,
            separate.SeparationError,
            AudioDecodeError,
        ) as error:
            log.warning("%s: %s", track.id, error)
            skipped.append(track.id)
            continue

        try:
            if beats_dir is None and beat_backend is None:
                result = score_track(
                    track,
                    estimated,
                    vocabularies,
                    check_alignment=check_alignment,
                    reference=reference,
                )
            else:
                result = _score_form(
                    track,
                    estimated,
                    reference,
                    vocabularies,
                    check_alignment,
                    beats_dir,
                    beat_backend,
                    cache_dir,
                    anchor_backend,
                    anchor_dir,
                    refresh,
                    shifts,
                )
        # If the form cannot be found for this track, skip it and list it in the report.
        except (OSError, ValueError, AudioDecodeError) as error:
            log.warning("%s: %s", track.id, error)
            skipped.append(track.id)
            continue
        results.append(result)
        if on_track is not None:
            on_track(result)

    return Report(
        dataset=dataset.name,
        backend=backend or f"estimates:{estimates_dir}",
        vocabularies=tuple(vocabularies),
        results=tuple(results),
        skipped=tuple(skipped),
        separated=None if estimates_dir is not None else use_separation,
        fallbacks=fallbacks,
        anchor=anchor_name,
    )
