"""Separation wrapper tests.

demucs is never actually run here -- subprocess is stubbed. These pin *how* it
gets invoked and detected, which is where the bugs were: both were failures to
find a demucs that was installed all along.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from songscribe import separate


class TestAvailable:
    def test_detects_demucs_by_import_not_by_path(self, monkeypatch):
        # The console script is only on PATH when the venv is activated, so a
        # PATH probe reports "missing" for an absolute-path invocation and
        # separation gets skipped silently.
        monkeypatch.setattr(separate.importlib.util, "find_spec", lambda name: object())
        monkeypatch.delenv("PATH", raising=False)
        assert separate.available() is True

    def test_reports_missing_when_not_importable(self, monkeypatch):
        monkeypatch.setattr(separate.importlib.util, "find_spec", lambda name: None)
        assert separate.available() is False

    def test_looks_for_the_demucs_module(self, monkeypatch):
        asked = []
        monkeypatch.setattr(
            separate.importlib.util, "find_spec", lambda name: asked.append(name) or object()
        )
        separate.available()
        assert asked == ["demucs"]


class TestSeparate:
    def test_invokes_demucs_under_the_current_interpreter(self, monkeypatch, tmp_path):
        commands = []

        def fake_run(cmd, **kwargs):
            commands.append(cmd)
            # Produce the stems demucs would have written.
            stem_dir = tmp_path / separate.DEMUCS_MODEL / "song"
            stem_dir.mkdir(parents=True)
            (stem_dir / "vocals.wav").write_bytes(b"")
            (stem_dir / "no_vocals.wav").write_bytes(b"")
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(separate.importlib.util, "find_spec", lambda name: object())
        monkeypatch.setattr(separate.subprocess, "run", fake_run)

        stems = separate.separate(tmp_path / "song.wav", tmp_path)

        cmd = commands[0]
        # sys.executable -m demucs, never a bare "demucs" that needs PATH.
        assert cmd[:3] == [sys.executable, "-m", "demucs"]
        assert "--two-stems=vocals" in cmd
        assert set(stems) == {"vocals", "no_vocals"}

    def test_error_names_both_the_extra_and_the_escape_hatch(self, monkeypatch, tmp_path):
        monkeypatch.setattr(separate.importlib.util, "find_spec", lambda name: None)
        with pytest.raises(RuntimeError) as excinfo:
            separate.separate(tmp_path / "song.wav", tmp_path)
        message = str(excinfo.value)
        assert "songscribe[separate]" in message
        assert "--no-separate" in message

    def test_missing_stem_is_reported_with_what_was_found(self, monkeypatch, tmp_path):
        def fake_run(cmd, **kwargs):
            stem_dir = tmp_path / separate.DEMUCS_MODEL / "song"
            stem_dir.mkdir(parents=True)
            (stem_dir / "vocals.wav").write_bytes(b"")  # no_vocals missing
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(separate.importlib.util, "find_spec", lambda name: object())
        monkeypatch.setattr(separate.subprocess, "run", fake_run)

        with pytest.raises(RuntimeError, match="no_vocals"):
            separate.separate(tmp_path / "song.wav", tmp_path)

    def test_isolate_vocals_returns_both_stems_in_order(self, monkeypatch, tmp_path):
        def fake_run(cmd, **kwargs):
            stem_dir = tmp_path / separate.DEMUCS_MODEL / "song"
            stem_dir.mkdir(parents=True)
            (stem_dir / "vocals.wav").write_bytes(b"")
            (stem_dir / "no_vocals.wav").write_bytes(b"")
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(separate.importlib.util, "find_spec", lambda name: object())
        monkeypatch.setattr(separate.subprocess, "run", fake_run)

        vocals, accompaniment = separate.isolate_vocals(tmp_path / "song.wav", tmp_path)
        assert vocals.name == "vocals.wav"
        assert accompaniment.name == "no_vocals.wav"
