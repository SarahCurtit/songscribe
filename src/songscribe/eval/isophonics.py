"""Build an evaluation manifest from an Isophonics annotation tree.

Walks the tree for ``.lab`` files and pairs each with audio by trying, in order: the same
relative path, then the same folded title within the matching album, then a unique folded
title anywhere (folding drops case, punctuation, and disc and track numbers). Unpaired files
are reported, never guessed at -- a wrong pairing would produce a plausible, wrong score.
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

#: The non-chord annotation directories an Isophonics archive ships beside the
#: chord one (``keylab``, ``seglab``, ``beat``), plus their obvious variants.
#: A chord directory only marks an archive when one of these is its sibling: an
#: ordinary album folder that happens to be called ``Chords`` has none.
OTHER_ANNOTATION_DIRECTORIES = (
    "keylab",
    "key",
    "keys",
    "seglab",
    "segments",
    "segmentation",
    "beat",
    "beats",
    "beatlab",
)

# Repeated so a disc number goes with the track number: the White Album's
# "CD1_-_01_-_..." and a rip's "Disc 1 - 01 - ..." or "1-01 ..." must all agree.
_TRACK_NUMBER = re.compile(r"^(?:(?:cd|disc|disk)?\d+)+")
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

#: Written instead of :data:`DEFAULT_NOTES` when nothing paired, so a
#: references-only manifest doesn't claim a pairing it never made. Decided by
#: what paired, not by whether an audio root was given: a wrong or empty
#: library yields no ``audio`` entries all the same.
REFERENCES_ONLY_NOTES = (
    "Chord reference annotations only, not paired with any audio. Paths are "
    "relative, and no annotation text is stored here -- only identifiers -- "
    "so this manifest carries no licensed material."
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
    ``"isawherstandingthere"``: leading disc and track numbers, case,
    punctuation and spacing all go, being what differs between the two.
    """
    stem = Path(name).stem.lower()
    folded = _NOT_ALNUM.sub("", stem)
    # Strip the track number only after folding, so "01_-_Help" and "01 Help"
    # both lose exactly the same prefix.
    return _TRACK_NUMBER.sub("", folded) or folded


def find_annotations(annotation_root: Path) -> list[Path]:
    """Every chord ``.lab`` file under ``annotation_root``, sorted.

    A directory holding both a chord directory and a key/segment/beat one is an
    Isophonics archive: below it, only the chord directory is read. Everything
    outside such an archive is read whole, so flat trees beside it survive.
    """
    if not annotation_root.is_dir():
        return []

    # Narrowed per archive, not tree-wide: one chord directory anywhere must not
    # silence unrelated trees beside it. Requiring a non-chord sibling as well
    # keeps an album folder named "Chords" from dropping the albums next to it.
    archives = {
        d for d in [annotation_root, *annotation_root.rglob("*")] if d.is_dir() and _is_archive(d)
    }

    found = []
    for path in sorted(annotation_root.rglob("*.lab")):
        parts = path.relative_to(annotation_root).parts
        # At every archive the file sits in, the step taken below it must be into
        # the chord directory; ancestors that aren't archives impose nothing.
        if path.is_file() and all(
            part.lower() in CHORD_DIRECTORIES
            for depth, part in enumerate(parts)
            if annotation_root.joinpath(*parts[:depth]) in archives
        ):
            found.append(path)
    return found


def _is_archive(directory: Path) -> bool:
    """True if ``directory`` holds a chord directory beside another annotation one."""
    names = {c.name.lower() for c in directory.iterdir() if c.is_dir()}
    return bool(names & set(CHORD_DIRECTORIES)) and bool(names & set(OTHER_ANNOTATION_DIRECTORIES))


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
    notes: str | None = None,
) -> BuildReport:
    """Pair annotations with audio and assemble a manifest.

    ``audio_root`` may be omitted, which produces a manifest with references
    only -- enough to score pre-computed estimates, not to run a model.
    ``notes`` defaults to the paired text only if at least one track paired.
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

    if notes is None:
        notes = DEFAULT_NOTES if paired else REFERENCES_ONLY_NOTES

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
    candidates = _best_copies(index.get(folded, []))
    if len(candidates) > 1 and relative.parent != Path("."):
        album = fold_title(relative.parent.name)
        # Directories below the root only: a title track's own filename folds to
        # the album name, and a directory above the root is shared by every
        # candidate, so either would let all of them through.
        within_album = [
            p
            for p in candidates
            if any(fold_title(part) == album for part in p.relative_to(audio_root).parent.parts)
        ]
        if within_album:
            return within_album

    # 3. A unique title anywhere under the audio root. Several hits are left
    #    ambiguous: two albums with the same track title would otherwise be
    #    scored against each other's recording.
    return candidates


def _best_copies(paths: list[Path]) -> list[Path]:
    """Keep one file per recording, the best-ranked by :data:`AUDIO_EXTENSIONS`.

    Only a shared directory and stem make two files the same recording: the same
    title in two directories may be two albums, and must stay ambiguous.
    """
    best: dict[tuple[Path, str], Path] = {}
    for path in paths:
        key = (path.parent, path.stem)
        held = best.get(key)
        if held is None or _rank(path) < _rank(held):
            best[key] = path
    return list(best.values())


def _rank(path: Path) -> int:
    return AUDIO_EXTENSIONS.index(path.suffix.lower())
