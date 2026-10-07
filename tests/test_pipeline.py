"""Pipeline wiring tests.

Every model is monkeypatched out: these check which stages get *invoked* and
what they are handed, not whether the models are any good. That keeps them
runnable in CI with no weights and no audio on disk.
"""

from __future__ import annotations

import numpy as np
import pytest

from songscribe import pipeline
from songscribe.types import ChordEvent, Word


@pytest.fixture
def stub_models(monkeypatch):
    """Replace every model call, recording what was invoked."""
    calls: dict[str, list] = {"chords": [], "lyrics": [], "separate": []}

    monkeypatch.setattr(
        pipeline.audio, "load", lambda src, sr: (np.zeros(sr, dtype=np.float32), sr)
    )

    def fake_recognise(samples, sample_rate, backend):
        calls["chords"].append(backend)
        return [ChordEvent(0.0, 2.0, "C"), ChordEvent(2.0, 4.0, "G")]

    def fake_transcribe(source, model_size, language):
        calls["lyrics"].append(model_size)
        return [Word(0.1, 0.5, "lantern"), Word(0.6, 1.0, "water")]

    def fake_isolate(path, out_dir):
        calls["separate"].append(str(path))
        return path, path

    monkeypatch.setattr(pipeline.chords, "recognise", fake_recognise)
    monkeypatch.setattr(pipeline.lyrics, "transcribe", fake_transcribe)
    monkeypatch.setattr(pipeline.separate, "isolate_vocals", fake_isolate)
    monkeypatch.setattr(pipeline.separate, "available", lambda: True)
    return calls


@pytest.fixture
def audio_file(tmp_path):
    path = tmp_path / "my_song.wav"
    path.write_bytes(b"")  # never read: audio.load is stubbed
    return path


class TestLyricsToggle:
    def test_no_lyrics_never_loads_the_asr_model(self, stub_models, audio_file):
        song = pipeline.transcribe(audio_file, use_lyrics=False)
        assert stub_models["lyrics"] == []
        assert stub_models["chords"] == ["madmom"]
        assert all(s.is_instrumental for s in song.sections)

    def test_no_lyrics_still_separates(self, stub_models, audio_file):
        # The accompaniment stem is the mix minus vocals, which is cleaner for
        # the chord model -- so chords-only runs still want separation. Only
        # --no-separate turns it off.
        pipeline.transcribe(audio_file, use_lyrics=False, use_separation=True)
        assert len(stub_models["separate"]) == 1

    def test_no_separate_is_the_only_thing_that_skips_demucs(self, stub_models, audio_file):
        pipeline.transcribe(audio_file, use_lyrics=False, use_separation=False)
        assert stub_models["separate"] == []

    def test_lyrics_on_by_default(self, stub_models, audio_file):
        song = pipeline.transcribe(audio_file)
        assert stub_models["lyrics"] == ["small"]
        assert any(not s.is_instrumental for s in song.sections)

    def test_lyrics_run_still_separates(self, stub_models, audio_file):
        pipeline.transcribe(audio_file, use_separation=True)
        assert len(stub_models["separate"]) == 1


class TestMetadata:
    def test_title_defaults_to_a_tidied_filename(self, stub_models, audio_file):
        song = pipeline.transcribe(audio_file, use_lyrics=False)
        assert song.title == "my song"

    def test_explicit_title_wins(self, stub_models, audio_file):
        song = pipeline.transcribe(audio_file, use_lyrics=False, title="Real Name")
        assert song.title == "Real Name"

    def test_key_is_estimated_from_the_chords(self, stub_models, audio_file):
        song = pipeline.transcribe(audio_file, use_lyrics=False)
        assert song.key in {"C", "G"}


class TestBackendSelection:
    def test_passes_the_requested_backend_through(self, stub_models, audio_file):
        pipeline.transcribe(audio_file, chord_backend="template", use_lyrics=False)
        assert stub_models["chords"] == ["template"]

    def test_unknown_backend_fails_before_any_model_runs(self, stub_models, audio_file):
        with pytest.raises(ValueError, match="unknown chord backend"):
            pipeline.transcribe(audio_file, chord_backend="nope", use_lyrics=False)
        assert stub_models["chords"] == []
