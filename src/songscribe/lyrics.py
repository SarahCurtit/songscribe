"""Lyric transcription from audio.

Wraps `faster-whisper <https://github.com/SYSTRAN/faster-whisper>`_ for ASR
with word-level timestamps. Timestamps are the whole point here -- a lyric
transcript without them cannot be aligned to chords.

Singing is harder for ASR than speech: long held vowels, backing vocals, and
words stretched across several beats. Running
:func:`songscribe.separate.isolate_vocals` first makes a large difference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .types import Word

DEFAULT_MODEL = "small"


def transcribe(
    audio: np.ndarray | str | Path,
    sample_rate: int | None = None,
    model_size: str = DEFAULT_MODEL,
    language: str | None = None,
    device: str = "auto",
) -> list[Word]:
    """Transcribe singing into timed words.

    ``audio`` may be a mono float32 array (then ``sample_rate`` is required,
    and must be 16 kHz) or a path, which faster-whisper decodes itself.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as error:
        raise RuntimeError(
            "faster-whisper not found. Install the extra: pip install 'songscribe[lyrics]' "
            "-- or pass --no-lyrics for a chords-only chart."
        ) from error

    if isinstance(audio, np.ndarray):
        if sample_rate is None:
            raise ValueError("sample_rate is required when passing an array")
        if sample_rate != 16_000:
            raise ValueError(f"expected 16 kHz audio, got {sample_rate} Hz")
        source: np.ndarray | str = audio
    else:
        source = str(audio)

    model = WhisperModel(model_size, device=device, compute_type="auto")
    segments, _info = model.transcribe(
        source,
        language=language,
        word_timestamps=True,
        # Singing has long gaps; VAD trimming tends to eat held notes.
        vad_filter=False,
    )

    words: list[Word] = []
    for segment in segments:
        for word in segment.words or ():
            text = word.word.strip()
            if text:
                words.append(Word(start=word.start, end=word.end, text=text))
    return words
