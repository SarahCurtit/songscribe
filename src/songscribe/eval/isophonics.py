"""Build an evaluation manifest from an Isophonics annotation tree.

Isophonics (``isophonics.net/content/reference-annotations``) publishes
chord, key, beat and structure annotations for the Beatles, Queen, Carole King
and Zweieck. The chord annotations are the de facto benchmark for this task,
they are free to download, and crucially they ship *without* audio -- you
supply your own copy of the recordings. That is exactly the split this
project needs, and the reason Isophonics is the corpus wired up first.

The annotation tree looks roughly like::

    chordlab/The Beatles/01_-_Please_Please_Me/01_-_I_Saw_Her_Standing_There.lab

but the layout has varied between releases, and nobody's music library is
organised identically to it. So rather than hardcoding a structure, this
walks the tree for ``.lab`` files and pairs each with audio by trying, in
order: the same relative path, then the same filename under any album, then a
punctuation- and track-number-insensitive match on the title. Anything still
unpaired is reported rather than guessed at -- a manifest that quietly points
at the wrong recording would produce a plausible, wrong score.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .dataset import AUDIO_EXTENSIONS, build

#: Directory names inside an Isophonics download that hold chord annotations.
#: Both spellings have shipped; the key/beat/structure directories alongside
#: them are a different task and are skipped.
CHORD_DIRECTORIES = ("chordlab", "chords", "chord")

_TRACK_NUMBER = re.compile(r"^\d+")
_NOT_ALNUM = re.compile(r"[^a-z0-9]+")

#: Deliberately silent about provenance: this builder reads any
#: Isophonics-*shaped* tree, and stamping "isophonics.net" onto a manifest
#: built from something else would misattribute someone's annotations. Record
#: the real source in ``EVALUATION.md``, where its licence can sit beside it.
DEFAULT_NOTES = (
    "Chord reference annotations paired against a local audio library. Paths "
    "are relative, and no audio or annotation text is stored here -- only "
    "identifiers -- so this manifest carries no licensed material."
)


@dataclass(frozen=True, slots=True)
class BuildReport:
    """What :func:`build_manifest` found, so the caller can report coverage.

    ``total`` counts annotations, not pairings: a references-only manifest is
    a legitimate result with nothing paired, and deriving the total from the
    pairing counts would report it as empty.
    """

    manifest: dict
    total: int
    paired: int
    unpaired: tuple[str, ...]
    ambiguous: tuple[str, ...]


def fold_title(name: str) -> str:
    """Reduce a filename to something two libraries might agree on.

    ``"01_-_I_Saw_Her_Standing_There.lab"`` and
    ``"01 I Saw Her Standing There.flac"`` both fold to
    ``"isawherstandingthere"``. Leading track numbers go, as do case,
    punctuation and spacing -- the three things that differ between a
    hand-made annotation tree and a ripped library.
    """
    stem = Path(name).stem.lower()
    folded = _NOT_ALNUM.sub("", stem)
    # Strip the track number only after folding, so "01_-_Help" and "01 Help"
    # both lose exactly the same prefix.
    return _TRACK_NUMBER.sub("", folded) or folded


def find_annotations(annotation_root: Path) -> list[Path]:
    """Every chord ``.lab`` file under ``annotation_root``, sorted.

    If the tree has a recognised chord directory, only that subtree is read --
    Isophonics ships key and beat annotations in the same download, and those
    are ``.lab`` files too but describe a different thing.
    """
    if not annotation_root.is_dir():
        return []

    chord_roots = [
        child
        for child in sorted(annotation_root.iterdir())
        if child.is_dir() and child.name.lower() in CHORD_DIRECTORIES
    ]

    search = chord_roots or [annotation_root]
    found: list[Path] = []
    for root in search:
        found.extend(p for p in root.rglob("*.lab") if p.is_file())
    return sorted(found)


def index_audio(audio_root: Path) -> dict[str, list[Path]]:
    """Map folded titles to the audio files that carry them."""
    index: dict[str, list[Path]] = {}
    if not audio_root.is_dir():
        return index
    for path in sorted(audio_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            index.setdefault(fold_title(path.name), []).append(path)
    return index


def build_manifest(
    annotation_root: str | Path,
    audio_root: str | Path | None = None,
    name: str = "isophonics",
    notes: str = DEFAULT_NOTES,
) -> BuildReport:
    """Pair annotations with audio and assemble a manifest.

    ``audio_root`` may be omitted, which produces a manifest with references
    only -- enough to score pre-computed estimates, not to run a model.
    """
    annotation_root = Path(annotation_root).expanduser()
    if not annotation_root.is_dir():
        raise NotADirectoryError(f"no annotation directory at {annotation_root}")

    annotations = find_annotations(annotation_root)
    if not annotations:
        raise FileNotFoundError(f"no .lab files under {annotation_root}")

    audio_root = Path(audio_root).expanduser() if audio_root else None
    index = index_audio(audio_root) if audio_root else {}

    entries: list[dict[str, str]] = []
    unpaired: list[str] = []
    ambiguous: list[str] = []
    paired = 0

    for annotation in annotations:
        relative = annotation.relative_to(annotation_root)
        track_id = relative.with_suffix("").as_posix()
        entry = {"id": track_id, "reference": relative.as_posix()}

        if audio_root is not None:
            matches = _match_audio(relative, audio_root, index)
            if len(matches) == 1:
                entry["audio"] = matches[0].relative_to(audio_root).as_posix()
                paired += 1
            elif matches:
                ambiguous.append(track_id)
            else:
                unpaired.append(track_id)

        entries.append(entry)

    return BuildReport(
        manifest=build(name=name, entries=entries, notes=notes),
        total=len(annotations),
        paired=paired,
        unpaired=tuple(unpaired),
        ambiguous=tuple(ambiguous),
    )


def _match_audio(
    relative: Path,
    audio_root: Path,
    index: dict[str, list[Path]],
) -> list[Path]:
    """Candidate audio files for one annotation, best strategy first."""
    # 1. The library mirrors the annotation tree exactly.
    for extension in AUDIO_EXTENSIONS:
        exact = audio_root / relative.with_suffix(extension)
        if exact.is_file():
            return [exact]

    # 2. The album directory matches but the filename differs in punctuation.
    folded = fold_title(relative.name)
    candidates = index.get(folded, [])
    if len(candidates) > 1 and relative.parent != Path("."):
        album = fold_title(relative.parent.name)
        within_album = [p for p in candidates if any(fold_title(part) == album for part in p.parts)]
        if within_album:
            return within_album

    # 3. A unique title anywhere under the audio root. Several hits are left
    #    ambiguous: two albums with the same track title would otherwise be
    #    scored against each other's recording.
    return candidates
