"""Split a manifest into train and test sets, keeping every version of a song together.

Train includes validation: tune on it however you like, and score on test once. The
assignment is a hash of the song, not a draw from a random generator, so it does not depend
on which other tracks are in the manifest. A manifest built locally with audio paths gets the
same split as the committed references-only one.
"""

from __future__ import annotations

import hashlib
import re

#: Share of songs held out for testing. The hash makes the share approximate, not exact.
TEST_FRACTION = 0.2

#: Recordings whose file names differ but whose song is the same, keyed by the name the
#: parentheses-stripping rule leaves. Confirmed against the recordings' titles.
SAME_SONG = {
    "cotton_tail_fitzgerald": "cotton_tail",
    "dinah_fats_waller": "dinah",
    "dinah_red_nichols": "dinah",
    "new_east_st_louis": "east_st_louis",
}

#: Both separators ship inside one archive: Isophonics numbers the Beatles "05_-_Dig_It" and
#: Carole King "05 Beautiful". A number left in the key would tie the song to its position on
#: the album, so renumbering a file, or the same song appearing on a reissue at another track
#: number, would move it between the sets.
_TRACK_NUMBER = re.compile(r"^(cd\d+_-_)?\d+(_-_| )")
_PARENTHESES = re.compile(r"_?\([^)]*\)$")


def song_key(track_id: str) -> str:
    """The song a track id belongs to: recordings of one song share a key.

    A trailing parenthesis, a performer or "Reprise", is dropped, so
    ``labs/maple_leaf_rag(bechet)`` and ``labs/maple_leaf_rag(hyman)`` both give
    ``maple_leaf_rag``. The leading track number of an album file is dropped too.
    """
    stem = track_id.rsplit("/", 1)[-1].lower()
    stem = _PARENTHESES.sub("", _TRACK_NUMBER.sub("", stem)).rstrip("_")
    return SAME_SONG.get(stem, stem)


def position(key: str) -> float:
    """Where the song sits in the unit interval. Stable across runs and manifests.

    Exposed so a caller can say which fraction *would* hold a song out: on a small
    corpus every song can sit above the fraction, and "0 tracks" alone doesn't say
    whether that is the fraction or the corpus.
    """
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def is_test(key: str, test_fraction: float = TEST_FRACTION) -> bool:
    """Whether the song with this key is held out."""
    return position(key) < test_fraction


def split_manifest(manifest: dict, test_fraction: float = TEST_FRACTION) -> tuple[dict, dict]:
    """Return ``(train, test)`` manifests; each track goes to exactly one, order kept."""
    if not 0 < test_fraction < 1:
        raise ValueError(f"test_fraction must be between 0 and 1, got {test_fraction}")

    name = manifest.get("dataset", "corpus")
    train = {**manifest, "dataset": f"{name}-train", "tracks": []}
    test = {**manifest, "dataset": f"{name}-test", "tracks": []}
    for track in manifest["tracks"]:
        held_out = is_test(song_key(track["id"]), test_fraction)
        (test if held_out else train)["tracks"].append(track)
    return train, test
