"""Beat and downbeat tracking, via pluggable backends.

Chord models emit boundaries wherever their frames fall; a musician reads
bars. This module supplies the bar lines.

Same registry contract as :mod:`songscribe.chords`: one function per backend
in :data:`BACKENDS`, its rate in :data:`BACKEND_SAMPLE_RATES`, no ``**kwargs``.
"""

from __future__ import annotations

import inspect
import logging
from bisect import bisect_right
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from math import isfinite, isnan, nan
from statistics import median

import numpy as np

from .types import BeatEvent

log = logging.getLogger(__name__)

DEFAULT_BACKEND = "madmom"

#: madmom's downbeat RNN was trained at 44.1 kHz, like its chord model.
BEAT_SAMPLE_RATE = 44_100

#: Frame rate of the RNN's activation function. The DBN has to be told, and
#: getting it wrong scales every beat time rather than failing.
MADMOM_FPS = 100

#: Metres the tracker is allowed to consider. The DBN takes a candidate list;
#: jazz standards are in 4/4 or 3/4, the same pair madmom's reference
#: DBNDownBeatTracker program uses.
METRE_CANDIDATES = (3, 4)

#: Assumed when a backend tracks beats without identifying the downbeat, and
#: when a beat timeline carries no position information at all.
DEFAULT_METRE = 4

#: How far one bar may differ from the median before it is called unreliable,
#: as a fraction. Catches gross faults only (a dropped bar is ~100% long);
#: good bars scatter up to 6.4%. EVALUATION.md has the measurement.
BAR_DRIFT_TOLERANCE = 0.10

#: A grid implying a tempo outside this range is not a tempo, it is a tracking
#: error: librosa output gone astray, or a corrupt or hand-edited ``.beats``
#: file. Not a half- or double-tempo lock from madmom -- its DBN is held to its
#: own 55-215 bpm, so an octave error there lands inside this range and needs
#: an external tempo reference to catch. Bounded in beats per minute rather
#: than in bar length so the metre is respected: at three beats to the bar,
#: 0.85 s is 212 bpm, not the leisurely tempo a bar-length bound would have
#: called it.
MIN_TEMPO_BPM = 30.0
MAX_TEMPO_BPM = 300.0


# --------------------------------------------------------------------------- #
# The grid
# --------------------------------------------------------------------------- #


# Lives here rather than in types.py, which holds event streams: a grid is
# derived from one.
@dataclass(frozen=True, slots=True)
class BeatGrid:
    """Bar lines in seconds from the start of the recording.

    Holds every downbeat rather than one bar length, because a real band
    drifts and a constant bar length accumulates error across a chorus: over
    32 bars, a 0.5% disagreement is most of a beat by the end.
    """

    downbeats: tuple[float, ...]
    beats_per_bar: int = DEFAULT_METRE

    def __post_init__(self) -> None:
        if self.beats_per_bar < 1:
            raise ValueError(f"beats_per_bar must be at least 1, got {self.beats_per_bar}")
        # Checked before the ordering test, which nan would pass silently: every
        # comparison against nan is False, so a nan downbeat would survive into
        # a grid whose bar lengths are all nan and whose health report is empty.
        if not all(isfinite(b) for b in self.downbeats):
            raise ValueError(f"downbeats must all be finite, got {self.downbeats}")
        if any(b <= a for a, b in zip(self.downbeats, self.downbeats[1:], strict=False)):
            raise ValueError("downbeats must be strictly ascending")

    @property
    def bars(self) -> int:
        """Complete bars. ``n`` downbeats delimit ``n - 1`` of them: the last
        opens a bar whose end the tracker never saw."""
        return max(len(self.downbeats) - 1, 0)

    @property
    def bar_length(self) -> float:
        """Median bar length, or ``nan`` with fewer than two downbeats.

        The median, not the mean: one dropped bar doubles a single interval,
        which would drag a mean but leaves a median alone.
        """
        lengths = self.bar_lengths
        return median(lengths) if lengths else nan

    @property
    def bar_lengths(self) -> tuple[float, ...]:
        """Length of each complete bar, in order."""
        return tuple(b - a for a, b in zip(self.downbeats, self.downbeats[1:], strict=False))

    @property
    def tempo(self) -> float:
        """Implied beats per minute, or ``nan`` without a measurable bar."""
        length = self.bar_length
        return nan if isnan(length) or length <= 0 else self.beats_per_bar * 60.0 / length

    @property
    def is_plausible(self) -> bool:
        """Whether the implied tempo is one a band could be playing."""
        tempo = self.tempo
        return not isnan(tempo) and MIN_TEMPO_BPM <= tempo <= MAX_TEMPO_BPM

    def bar_start(self, index: int) -> float:
        """Start of bar ``index``, counting from 0.

        Bounded rather than relying on tuple indexing, which would answer a
        negative index from the far end of the grid. The last downbeat is in
        range though it opens no complete bar: it is where the form stops.
        """
        if not 0 <= index < len(self.downbeats):
            raise IndexError(f"bar {index} outside the grid's {len(self.downbeats)} bar lines")
        return self.downbeats[index]

    def bar_span(self, index: int) -> tuple[float, float]:
        """``(start, end)`` of complete bar ``index``."""
        if not 0 <= index < self.bars:
            raise IndexError(f"bar {index} outside the grid's {self.bars} complete bars")
        return self.downbeats[index], self.downbeats[index + 1]

    def bar_index_at(self, time: float) -> int | None:
        """Which complete bar ``time`` falls in, or ``None`` if outside.

        Time before the first downbeat is outside on purpose: whatever plays
        there -- an intro, stage noise, a count-in -- is not in bar 1.
        """
        if not self.downbeats or time < self.downbeats[0]:
            return None
        # bisect on the tuple rather than np.searchsorted on a fresh array:
        # placing a whole chord timeline is one call per event, and this way
        # that costs no allocation.
        index = bisect_right(self.downbeats, time) - 1
        return index if index < self.bars else None

    def drift(self) -> tuple[float, ...]:
        """Each bar's deviation from the median, as a fraction of it."""
        length = self.bar_length
        if isnan(length) or length <= 0:
            return ()
        return tuple(abs(bar - length) / length for bar in self.bar_lengths)

    def unreliable_bars(self, tolerance: float = BAR_DRIFT_TOLERANCE) -> tuple[int, ...]:
        """Bars whose length is further than ``tolerance`` from the median.

        A gross-fault detector, not a "is this bar in the form" test -- see
        "Beat-grid drift" in EVALUATION.md for why the distinction matters.
        """
        return tuple(i for i, d in enumerate(self.drift()) if d > tolerance)


def grid(beats: Sequence[BeatEvent], beats_per_bar: int | None = None) -> BeatGrid:
    """Collapse a tracked beat timeline to its bar lines.

    ``beats_per_bar`` is inferred from the highest position seen when not
    given -- the tracker already decided the metre, and re-deriving it from
    beat spacing would only disagree with it.
    """
    positions = [b.position for b in beats]
    if beats_per_bar is None:
        beats_per_bar = max(positions) if positions else DEFAULT_METRE
        # An excerpt shorter than a bar infers its own length as the metre.
        if positions and beats_per_bar not in METRE_CANDIDATES:
            log.warning("inferred metre %d from %d beats", beats_per_bar, len(positions))
    elif positions and max(positions) != beats_per_bar:
        # The caller still wins, but every tempo figure scales with the metre:
        # 4/4 beats read as 3/4 report 90 bpm for 120, and nothing else says so.
        log.warning(
            "metre %d given, but the tracked positions reach %d", beats_per_bar, max(positions)
        )
    downbeats = tuple(sorted(b.time for b in beats if b.is_downbeat))
    # A timeline trimmed mid-bar carries no downbeat, and the resulting grid
    # places nothing at all -- worth a word, since it is indistinguishable
    # downstream from a recording the tracker simply found no bars in.
    if beats and not downbeats:
        log.warning("%d beats but no downbeat among them: the grid is empty", len(beats))
    return BeatGrid(downbeats=downbeats, beats_per_bar=beats_per_bar)


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


def _track_madmom(
    audio: np.ndarray,
    sample_rate: int,
    beats_per_bar: int | Sequence[int] = METRE_CANDIDATES,
) -> list[BeatEvent]:
    """RNN downbeat activations + DBN bar tracking, via madmom's weights.

    ``beats_per_bar`` is one metre or a list to choose between -- madmom's own
    name, and the librosa backend's, so ``track(..., beats_per_bar=3)`` means
    the same thing whichever backend runs.
    """
    # Validated before the import, so a wrong rate costs nothing and is
    # catchable in CI, where the weights are not installed.
    if sample_rate != BEAT_SAMPLE_RATE:
        raise ValueError(f"madmom backend expects {BEAT_SAMPLE_RATE} Hz, got {sample_rate}")

    from madmom.audio.signal import Signal
    from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor

    # Wrap in a madmom Signal for the same reason the chord backend does: the
    # processor chain reads the rate off the signal, and a bare ndarray would
    # inherit whatever default it was configured with.
    signal = Signal(audio, sample_rate=sample_rate, num_channels=1)
    activations = RNNDownBeatProcessor()(signal)
    # Passed as given: the DBN reads it with np.array(ndmin=1), so an int and a
    # list both work without converting here.
    tracker = DBNDownBeatTrackingProcessor(beats_per_bar=beats_per_bar, fps=MADMOM_FPS)
    # Rows are (time, position in bar), position 1-based -- verified against
    # madmom 0.17.dev0.
    return [
        BeatEvent(time=float(time), position=int(position))
        for time, position in tracker(activations)
    ]


def _track_librosa(
    audio: np.ndarray,
    sample_rate: int,
    beats_per_bar: int = DEFAULT_METRE,
    hop_length: int = 512,
) -> list[BeatEvent]:
    """Non-DL baseline: onset-envelope beat tracking, metre assumed.

    Kept so there is always a grid available with no weights on disk, the way
    the ``template`` chord backend keeps the pipeline runnable. It finds beats
    but **not** downbeats -- the bar phase here is assumed, not detected, so
    the first tracked beat is called bar 1. Never the default for that reason.
    """
    import librosa

    _, frames = librosa.beat.beat_track(y=audio, sr=sample_rate, hop_length=hop_length)
    times = librosa.frames_to_time(frames, sr=sample_rate, hop_length=hop_length)
    return [
        BeatEvent(time=float(time), position=index % beats_per_bar + 1)
        for index, time in enumerate(times)
    ]


BACKENDS: dict[str, Callable[..., list[BeatEvent]]] = {
    "madmom": _track_madmom,
    "librosa": _track_librosa,
}

#: Sample rate each backend wants its audio in.
BACKEND_SAMPLE_RATES = {
    "madmom": BEAT_SAMPLE_RATE,
    "librosa": 22_050,
}


def _check_backend(backend: str) -> None:
    """Reject an unknown backend by name.

    Never falls back to another tracker: a wrong grid is harder to notice than
    a missing one, because every number downstream still comes out looking
    like a measurement.
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown beat backend {backend!r}, expected one of {sorted(BACKENDS)}")


#: Parameter kinds a setting can be passed as. A backend's own positional-only
#: or ``*args`` parameters are unreachable through :func:`track`, which
#: forwards settings by keyword.
_KEYWORD_KINDS = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)


def backend_options(backend: str) -> set[str]:
    """The settings ``backend`` accepts, by name.

    Read off the backend function's own signature, so a backend declares its
    settings once by taking them as parameters. The first two are the audio and
    its sample rate, which :func:`track` passes positionally. An empty set
    means the backend takes no settings; a backend that accepts *anything* is a
    separate question, asked of :func:`accepts_any_option`.
    """
    _check_backend(backend)
    parameters = list(inspect.signature(BACKENDS[backend]).parameters.values())
    options = set()
    for parameter in parameters[2:]:
        if parameter.kind not in _KEYWORD_KINDS:
            continue
        # A setting without a default is not optional, so it cannot be reported
        # as one: `track` calls a backend with the audio and its rate alone, and
        # a caller who names nothing would get a TypeError from the call -- the
        # one error path this module promises, routed around. Rejecting the
        # registration points the backend's author at the missing default.
        if parameter.default is inspect.Parameter.empty:
            raise ValueError(
                f"beat backend {backend!r} declares setting {parameter.name!r} with no "
                "default; a backend must be callable with the audio and its rate alone"
            )
        options.add(parameter.name)
    return options


def accepts_any_option(backend: str) -> bool:
    """Whether ``backend`` keeps a catch-all, and so accepts every setting.

    Such a backend cannot be checked -- it really does take anything -- so this
    reports the exemption instead of folding it into :func:`backend_options`,
    where "anything" and "nothing" would be two indistinguishable empty
    answers. Nothing registered here is exempt, and a test holds that: a
    ``**kwargs`` on a backend is how the silently-ignored metre came back.
    """
    _check_backend(backend)
    return any(
        p.kind is inspect.Parameter.VAR_KEYWORD
        for p in inspect.signature(BACKENDS[backend]).parameters.values()
    )


def _check_options(backend: str, options: Iterable[str]) -> None:
    """Reject settings ``backend`` has no parameter for.

    The same rule as :func:`_check_backend`, one level down: a misspelled
    setting that is quietly dropped is indistinguishable from a tracker that
    ignored it, which is how a metre setting went unnoticed here. Raised in
    :func:`track` rather than in each backend so there is one error path.
    """
    if accepts_any_option(backend):
        return
    accepted = backend_options(backend)
    unknown = sorted(set(options) - accepted)
    if not unknown:
        return
    named = ", ".join(repr(name) for name in unknown)
    noun = "setting" if len(unknown) == 1 else "settings"
    detail = f"expected one of {sorted(accepted)}" if accepted else "which takes no settings"
    raise ValueError(f"unknown {noun} {named} for beat backend {backend!r}, {detail}")


def sample_rate_for(backend: str) -> int:
    """The sample rate ``backend`` needs its audio loaded at."""
    _check_backend(backend)
    return BACKEND_SAMPLE_RATES[backend]


def track(
    audio: np.ndarray,
    sample_rate: int,
    backend: str = DEFAULT_BACKEND,
    **kwargs,
) -> list[BeatEvent]:
    """Track beats and downbeats in ``audio`` (mono float32).

    Extra keyword arguments are the chosen backend's settings; one it does not
    accept raises rather than being dropped.
    """
    _check_backend(backend)
    _check_options(backend, kwargs)
    log.debug("beat backend=%s sample_rate=%d", backend, sample_rate)
    return BACKENDS[backend](audio, sample_rate, **kwargs)
