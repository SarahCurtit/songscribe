"""Local, private word timelines for :func:`~songscribe.eval.metrics.placement_score`.

They hold lyric text, so they never enter the repo (constraint 5 in ``CLAUDE.md``):
files must end in ``.words.json``, which ``.gitignore`` blocks.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..types import Word

# No public corpus has timed words (Isophonics is chords only), so you produce the
# timeline from your own recording with your own pipeline. That is sound: placement_score
# holds one word timeline fixed for both sides, so word errors cancel out -- the timeline
# must be consistent, not correct.

#: Suffix the gitignore rule matches. Using it is what keeps a transcript out
#: of a commit by accident.
SUFFIX = ".words.json"

FORMAT_VERSION = 1


class ReferenceError(ValueError):
    """A word reference file was malformed."""


def write_words(path: str | Path, words: list[Word], source: str = "") -> Path:
    """Write a word timeline as JSON, returning the path written.

    ``source`` records where the timeline came from -- which recording, which
    model -- since a reference is only interpretable alongside that.
    """
    path = Path(path)
    if not path.name.endswith(SUFFIX):
        raise ReferenceError(f"a word reference must end in {SUFFIX}, got {path.name}")

    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "version": FORMAT_VERSION,
        "source": source,
        "words": [{"start": w.start, "end": w.end, "text": w.text} for w in words],
    }
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def read_words(path: str | Path) -> list[Word]:
    """Read a word timeline written by :func:`write_words`."""
    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ReferenceError(f"{path}: not valid JSON: {error}") from error

    if not isinstance(document, dict):
        raise ReferenceError(f"{path}: expected a JSON object at the top level")
    version = document.get("version", FORMAT_VERSION)
    if version != FORMAT_VERSION:
        raise ReferenceError(f"{path}: format version {version}, this one understands 1")

    raw = document.get("words")
    if not isinstance(raw, list):
        raise ReferenceError(f"{path}: 'words' must be a list")

    words: list[Word] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ReferenceError(f"{path}: word {index} is not an object")
        try:
            words.append(
                Word(start=float(entry["start"]), end=float(entry["end"]), text=str(entry["text"]))
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ReferenceError(f"{path}: word {index} needs start, end and text") from error

    # Sorted, because place_in_line walks words in order and a reference that
    # was hand-edited is easy to leave out of order.
    return sorted(words, key=lambda w: (w.start, w.end))
