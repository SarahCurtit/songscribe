"""Source separation, so each downstream stage hears only what it needs.

The vocal stem goes to ASR; the accompaniment goes to chord recognition. Both
jobs get noticeably easier once the other instrument is out of the way.

Separation is slow and optional. Without it, ``songscribe`` runs both stages on
the full mix -- faster, and good enough for a sparse acoustic recording.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

DEMUCS_MODEL = "htdemucs"


def available() -> bool:
    """Whether demucs can be run from this interpreter.

    Deliberately an import check rather than ``shutil.which("demucs")``: the
    console script only lands on PATH when the virtualenv is *activated*, so a
    PATH probe reports "not installed" for anyone invoking songscribe by its
    absolute path -- and separation would then be skipped silently, which is
    the single biggest accuracy loss in the pipeline.
    """
    return importlib.util.find_spec("demucs") is not None


def separate(path: str | Path, out_dir: str | Path, model: str = DEMUCS_MODEL) -> dict[str, Path]:
    """Split ``path`` into stems with demucs; return ``{stem_name: wav_path}``.

    Uses ``--two-stems=vocals``, which is all this pipeline needs and roughly
    halves the work compared to a full four-stem split.
    """
    if not available():
        raise RuntimeError(
            "demucs not found. Install the extra: pip install 'songscribe[separate]' "
            "-- or pass --no-separate to work on the full mix."
        )

    path, out_dir = Path(path), Path(out_dir)
    # Invoked as a module under the *current* interpreter, so it resolves
    # without depending on PATH or on the venv being activated.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "demucs",
            "--two-stems=vocals",
            "-n",
            model,
            "-o",
            str(out_dir),
            str(path),
        ],
        check=True,
    )

    stem_dir = out_dir / model / path.stem
    stems = {wav.stem: wav for wav in stem_dir.glob("*.wav")}
    missing = {"vocals", "no_vocals"} - stems.keys()
    if missing:
        raise RuntimeError(
            f"demucs produced no {', '.join(sorted(missing))} stem in {stem_dir}; "
            f"found {sorted(stems) or 'nothing'}"
        )
    return stems


def isolate_vocals(path: str | Path, out_dir: str | Path) -> tuple[Path, Path]:
    """Convenience wrapper returning ``(vocals, accompaniment)`` paths."""
    stems = separate(path, out_dir)
    return stems["vocals"], stems["no_vocals"]
