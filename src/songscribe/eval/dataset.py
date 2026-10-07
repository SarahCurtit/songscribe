"""Evaluation manifests: which tracks to score, and where their files are.

A manifest holds track identifiers and *relative* paths, never audio and never
annotation text. That split is deliberate and is what makes an evaluation
publishable: the manifest, the metrics and the runner can all be committed and
shared, while the recordings stay on the machine that is licensed to hold
them. Someone else with their own copy of the same corpus points the roots at
it and reproduces the numbers.

Roots are resolved in this order, first hit winning:

1. an explicit argument (``--audio``/``--annotations`` on the command line),
2. the :data:`AUDIO_ROOT_ENV` / :data:`ANNOTATION_ROOT_ENV` environment
   variables,
3. the directory containing the manifest.

See ``EVALUATION.md`` for the corpora this is built for and the licence
position on each.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

#: Where the recordings live. Set this once in your shell rather than passing
#: ``--audio`` on every run.
AUDIO_ROOT_ENV = "SONGSCRIBE_EVAL_ROOT"

#: Where the ``.lab`` reference annotations live.
ANNOTATION_ROOT_ENV = "SONGSCRIBE_EVAL_ANNOTATIONS"

MANIFEST_VERSION = 1

#: Extensions tried when looking for the audio belonging to an annotation,
#: lossless first: a transcoded MP3 is fine for a sanity check but the chord
#: models read spectra, so prefer the better copy when both are present.
AUDIO_EXTENSIONS = (".flac", ".wav", ".m4a", ".mp3", ".ogg", ".aac", ".aiff", ".wma")


class ManifestError(ValueError):
    """A manifest was malformed or referred to files that are not there."""


@dataclass(frozen=True, slots=True)
class Track:
    """One evaluation track, with its paths already resolved."""

    id: str
    reference: Path
    audio: Path | None = None

    @property
    def is_runnable(self) -> bool:
        """Whether a model can actually be run on this track."""
        return self.audio is not None and self.audio.exists()


@dataclass(frozen=True, slots=True)
class Dataset:
    """A resolved manifest."""

    name: str
    tracks: tuple[Track, ...]
    notes: str = ""
    audio_root: Path | None = None
    annotation_root: Path | None = None

    def runnable(self) -> tuple[Track, ...]:
        return tuple(t for t in self.tracks if t.is_runnable)

    def missing_audio(self) -> tuple[Track, ...]:
        return tuple(t for t in self.tracks if not t.is_runnable)

    def missing_references(self) -> tuple[Track, ...]:
        return tuple(t for t in self.tracks if not t.reference.exists())


def resolve_root(explicit: str | Path | None, env_var: str, fallback: Path) -> Path:
    """Pick a root directory from argument, environment, then fallback."""
    if explicit is not None:
        return Path(explicit).expanduser()
    from_env = os.environ.get(env_var)
    if from_env:
        return Path(from_env).expanduser()
    return fallback


def build(
    name: str,
    entries: list[dict[str, str]],
    notes: str = "",
) -> dict:
    """Assemble a manifest document ready to be written as JSON."""
    return {
        "version": MANIFEST_VERSION,
        "dataset": name,
        "notes": notes,
        "audio_root_env": AUDIO_ROOT_ENV,
        "annotation_root_env": ANNOTATION_ROOT_ENV,
        "tracks": entries,
    }


def write_manifest(path: str | Path, manifest: dict) -> None:
    """Write a manifest as indented JSON, so diffs stay readable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def load_manifest(
    path: str | Path,
    audio_root: str | Path | None = None,
    annotation_root: str | Path | None = None,
) -> Dataset:
    """Read a manifest and resolve every path in it.

    Missing files are *not* an error here: a manifest covering 180 tracks is
    useful with 40 of them on disk, and the caller reports the shortfall
    rather than refusing to run. A structurally broken manifest does raise.
    """
    path = Path(path).expanduser()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ManifestError(f"{path}: not valid JSON: {error}") from error

    if not isinstance(document, dict):
        raise ManifestError(f"{path}: expected a JSON object at the top level")

    version = document.get("version", MANIFEST_VERSION)
    if version != MANIFEST_VERSION:
        raise ManifestError(
            f"{path}: manifest version {version}, this songscribe understands {MANIFEST_VERSION}"
        )

    raw_tracks = document.get("tracks")
    if not isinstance(raw_tracks, list):
        raise ManifestError(f"{path}: 'tracks' must be a list")

    audio_base = resolve_root(audio_root, AUDIO_ROOT_ENV, path.parent)
    annotation_base = resolve_root(annotation_root, ANNOTATION_ROOT_ENV, path.parent)

    tracks: list[Track] = []
    seen: set[str] = set()
    for index, entry in enumerate(raw_tracks):
        if not isinstance(entry, dict):
            raise ManifestError(f"{path}: track {index} is not an object")
        track_id = entry.get("id")
        reference = entry.get("reference")
        if not track_id or not reference:
            raise ManifestError(f"{path}: track {index} needs both 'id' and 'reference'")
        if track_id in seen:
            raise ManifestError(f"{path}: duplicate track id {track_id!r}")
        seen.add(track_id)

        audio = entry.get("audio")
        tracks.append(
            Track(
                id=track_id,
                reference=_resolve(annotation_base, reference),
                audio=_resolve(audio_base, audio) if audio else None,
            )
        )

    return Dataset(
        name=document.get("dataset") or path.stem,
        tracks=tuple(tracks),
        notes=document.get("notes", ""),
        audio_root=audio_base,
        annotation_root=annotation_base,
    )


def _resolve(root: Path, relative: str) -> Path:
    """Join a manifest path to its root, honouring an absolute override.

    An absolute path in a manifest is respected but is a sign the manifest is
    not portable -- ``songscribe-eval manifest`` never writes one.
    """
    candidate = Path(relative).expanduser()
    return candidate if candidate.is_absolute() else root / candidate
