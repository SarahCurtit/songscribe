"""Build an evaluation manifest from an Isophonics annotation tree.

Walks the tree for ``.lab`` files and pairs each with audio by trying, in order: the same
relative path, the same folded title within the matching album, then a unique folded title
at the root or under a folder (the audio root's included) that is its artist or album, give
or take the other name and a few allowlisted words. Misses are reported, never guessed at.
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
_WORD = re.compile(r"[a-z0-9]+")
# Apostrophes join rather than split, as in _fold: "Day's" and "Days" must give
# the same word. Dots still split, which already makes "Sgt. Pepper" and "Sgt
# Pepper" agree; joining them would glue "Vol.1" and "Sgt.Pepper" into one word.
_APOSTROPHES = re.compile(r"['\u2018\u2019\u02bc]")

#: Words a rip adds to a folder name that say nothing about which album it is,
#: so "Help! [2009 Stereo Remaster]" and "1965 - Help! (Deluxe Edition)" both
#: reduce to the album's own words.
DECORATION_WORDS = frozenset(
    {
        "remaster",
        "remastered",
        "deluxe",
        "edition",
        "expanded",
        "anniversary",
        "reissue",
        "japan",
        "japanese",
        "import",
        "vinyl",
        "lp",
        "rip",
        "stereo",
        "mono",
        "flac",
        "mp3",
        "aac",
        "alac",
        "ogg",
        "wav",
        "cd",
        "disc",
        "disk",
    }
)

#: The only words a folder may add around an artist or album name it contains,
#: besides the annotation's other name ("Band - Album"). Anything else marks
#: someone else's record about the band -- "Band Covers", "Songs of Band",
#: "Band Tribute Band", "Album Demos" -- or simply a longer name ("Band Surname").
CONTAINED_NAME_EXTRAS = frozenset(
    {"the", "discography", "collection", "complete", "anthology", "box", "set"}
)

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
    pairing counts would report it as empty. ``unplaced``: a lone title found
    under no folder naming its artist or album -- a layout problem, not a miss.
    """

    manifest: dict
    total: int
    paired: int
    unpaired: tuple[str, ...]
    ambiguous: tuple[str, ...]
    unplaced: tuple[str, ...] = ()


def fold_title(name: str) -> str:
    """Reduce a filename to something two libraries might agree on.

    ``"01_-_I_Saw_Her_Standing_There.lab"`` and
    ``"01 I Saw Her Standing There.flac"`` both fold to
    ``"isawherstandingthere"``: leading disc and track numbers, case,
    punctuation and spacing all go, being what differs between the two.
    """
    return _fold(Path(name).stem)


def _fold_directory(name: str) -> str:
    """:func:`fold_title` for a directory name, which has no suffix to drop.

    ``.stem`` would cut ``"05_-_Mr._Placeholder's_Band"`` at its first dot, and
    every album sharing that prefix would then claim the same tracks.
    """
    return _fold(name)


def _fold(text: str) -> str:
    folded = _NOT_ALNUM.sub("", text.lower())
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
    # A mistyped audio root would otherwise index nothing and report every
    # track as unpaired, pointing at filenames rather than the path.
    if audio_root is not None and not audio_root.is_dir():
        raise NotADirectoryError(f"no audio directory at {audio_root}")
    index = index_audio(audio_root) if audio_root else {}

    entries: list[dict[str, str]] = []
    unpaired: list[str] = []
    ambiguous: list[str] = []
    unplaced: list[str] = []
    paired = 0

    for annotation in annotations:
        relative = annotation.relative_to(annotation_root)
        track_id = relative.with_suffix("").as_posix()
        entry = {"id": track_id, "reference": relative.as_posix()}

        if audio_root is not None:
            matches, misplaced = _match_audio(relative, audio_root, index)
            if len(matches) == 1:
                entry["audio"] = matches[0].relative_to(audio_root).as_posix()
                paired += 1
            elif matches:
                ambiguous.append(track_id)
            elif misplaced:
                unplaced.append(track_id)
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
        unplaced=tuple(unplaced),
    )


def _match_audio(
    relative: Path,
    audio_root: Path,
    index: dict[str, list[Path]],
) -> tuple[list[Path], bool]:
    """Candidate audio files for one annotation, best strategy first.

    The flag is true when a lone title was found but rejected for its folders.
    """
    # 1. The library mirrors the annotation tree exactly.
    for extension in AUDIO_EXTENSIONS:
        exact = audio_root / relative.with_suffix(extension)
        if exact.is_file():
            return [exact], False

    # 2. The album directory matches but the filename differs in punctuation.
    folded = fold_title(relative.name)
    candidates = _best_copies(index.get(folded, []))
    album = _fold_directory(relative.parent.name)
    album_words = _name_words(relative.parent.name)
    artist_words = [
        _name_words(part)
        for part in relative.parent.parent.parts
        if part.lower() not in CHORD_DIRECTORIES
    ]
    # A self-titled album names its artist, so in the library it would match the
    # artist folder above every album, compilations included. Checked on the
    # annotation side, whose artist/album layout is known; a library path can't
    # say which of its directories is the artist.
    self_titled = (album_words and album_words in artist_words) or any(
        _fold_directory(part) == album for part in relative.parent.parent.parts
    )
    if len(candidates) > 1 and relative.parent != Path(".") and not self_titled:
        # Directories below the root only, unlike step 3: a title track's filename
        # folds to the album name, and the root's own name (or any directory above
        # it) is shared by every candidate, so would let all of them through. Read
        # as words, as in step 3, so "Band - Album [FLAC]" beats "Greatest Hits".
        others = {w for words in artist_words for w in words}

        def names_album(part: str) -> bool:
            return _fold_directory(part) == album or _holds_name(
                _name_words(part), album_words, others
            )

        within_album = [
            p for p in candidates if any(map(names_album, p.relative_to(audio_root).parent.parts))
        ]
        if within_album:
            return within_album, False

    # 3. A unique title elsewhere. Several hits are left ambiguous: two albums
    #    with the same track title would otherwise be scored against each
    #    other's recording.
    if len(candidates) == 1 and not _plausible_home(candidates[0], relative, audio_root):
        # A lone hit is accepted only beside a folder named for the annotation's
        # artist or album, or loose at the root of a flat library: with one track
        # missing, a same-titled song by someone else would pair and score
        # silently. Reported apart from misses, since the fix is the layout.
        return [], True
    return candidates, False


def _plausible_home(candidate: Path, relative: Path, audio_root: Path) -> bool:
    """True if ``candidate`` sits at the root or under the annotation's artist or album.

    The audio root's own name counts: ``--audio ~/music/band`` names the artist.
    """
    library = candidate.relative_to(audio_root).parent.parts
    if not library:
        return True
    expected = [
        _name_words(part) for part in relative.parent.parts if part.lower() not in CHORD_DIRECTORIES
    ]
    expected = [words for words in expected if words]
    folders = [_name_words(part) for part in (audio_root.resolve().name, *library)]
    return any(_names_agree(folder, expected) for folder in folders if folder)


def _name_words(name: str) -> tuple[str, ...]:
    """The words of a folder name that identify an artist or album.

    Separate from :func:`_fold` so steps 1-2 keep their exact matching. Numbers
    (track, disc, year) and :data:`DECORATION_WORDS` go wherever they sit, and a
    leading "the", so "The Band - Album (2009) [FLAC]" reads as ``band album``.
    """
    text = _APOSTROPHES.sub("", name.lower())
    words = [w for w in _WORD.findall(text) if w not in DECORATION_WORDS]
    words = [w for w in words if not _TRACK_NUMBER.fullmatch(w)]
    return tuple(words[1:] if words[:1] == ["the"] else words)


def _names_agree(folder: tuple[str, ...], expected: list[tuple[str, ...]]) -> bool:
    """True if a library folder's words name one of the annotation's folders.

    Equal word lists agree. A folder holding one name contiguously agrees only if
    each word left over is in :data:`CONTAINED_NAME_EXTRAS` or the annotation's
    other name, so "Band - Album" agrees and "Band - Album (Karaoke)" doesn't. A
    folder is never matched by being *part of* a name: "Hearts" isn't an album.
    """
    if folder in expected:
        return True
    for index, words in enumerate(expected):
        # By position, not value: a self-titled album's name equals its artist's,
        # and "Band - Band" must still find the other one.
        others = {w for i, name in enumerate(expected) if i != index for w in name}
        if _holds_name(folder, words, others):
            return True
    return False


def _holds_name(folder: tuple[str, ...], name: tuple[str, ...], others: set[str]) -> bool:
    """True if ``folder`` is ``name``, or holds it contiguously among allowed words.

    Allowed: :data:`CONTAINED_NAME_EXTRAS` and ``others``, the annotation's other name.
    """
    if not name:
        return False
    for start in _runs(folder, name):
        leftover = folder[:start] + folder[start + len(name) :]
        if all(w in CONTAINED_NAME_EXTRAS or w in others for w in leftover):
            return True
    return False


def _runs(longer: tuple[str, ...], shorter: tuple[str, ...]) -> list[int]:
    """Every index at which ``shorter`` runs contiguously inside ``longer``."""
    size = len(shorter)
    return [i for i in range(len(longer) - size + 1) if longer[i : i + size] == shorter]


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
