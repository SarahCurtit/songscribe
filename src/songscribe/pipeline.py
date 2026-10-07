"""End-to-end orchestration: audio file in, :class:`Song` out.

    audio ──┬─(separation)─> vocals ──> ASR ─────────> timed words ──┐
            │                                                        ├─> align ─> Song
            └─(separation)─> backing ──> chord model ─> chord events ┘

Three pretrained open-source models, one per branch plus separation. The
branches never talk to each other until :mod:`songscribe.align`, which is what
keeps each model independently swappable and testable.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from . import align, audio, chords, lyrics, separate
from .types import Song

log = logging.getLogger(__name__)


def transcribe(
    path: str | Path,
    chord_backend: str = chords.DEFAULT_BACKEND,
    model_size: str = lyrics.DEFAULT_MODEL,
    language: str | None = None,
    use_separation: bool = True,
    use_lyrics: bool = True,
    title: str | None = None,
) -> Song:
    """Transcribe the song at ``path`` into chords over lyrics.

    ``use_separation`` falls back to the full mix when demucs is unavailable
    rather than failing -- the result is worse, but still usable.

    ``use_lyrics=False`` gives a chords-only chart and loads no ASR model,
    which is the right mode for instrumentals and for testing the chord
    branch on its own.
    """
    path = Path(path)
    vocal_source: str | Path = path
    chord_source: str | Path = path
    tmp: tempfile.TemporaryDirectory | None = None

    # Separation runs even for a chords-only chart: the accompaniment stem is
    # the mix minus vocals, and singing smears the chroma the chord model
    # reads. Use --no-separate to trade that accuracy for speed -- that choice
    # stays the caller's, rather than being implied by --no-lyrics.
    if use_separation:
        if separate.available():
            log.info("separating stems with demucs (this is the slow part)")
            tmp = tempfile.TemporaryDirectory(prefix="songscribe-")
            vocal_source, chord_source = separate.isolate_vocals(path, tmp.name)
        else:
            log.warning("demucs not installed; running both models on the full mix")

    try:
        log.info("recognising chords (backend=%s)", chord_backend)
        samples, sample_rate = audio.load(chord_source, chords.sample_rate_for(chord_backend))
        chord_events = chords.strip_no_chord(
            chords.recognise(samples, sample_rate, backend=chord_backend)
        )

        if use_lyrics:
            log.info("transcribing lyrics (whisper-%s)", model_size)
            words = lyrics.transcribe(vocal_source, model_size=model_size, language=language)
        else:
            log.info("skipping lyrics; chords-only chart")
            words = []
    finally:
        if tmp is not None:
            tmp.cleanup()

    song = align.align(chord_events, words)
    song.title = title or path.stem.replace("_", " ").replace("-", " ")
    song.key = chords.estimate_key(chord_events)
    return song
