"""Shared fixtures.

The lyrics in these fixtures are nonsense filler on purpose: the tests are
about timing and layout, and inventing the words keeps real songs out of the
repo.
"""

from __future__ import annotations

import pytest

from songscribe.types import ChordEvent, Word


def words_from(spec: list[tuple[float, float, str]]) -> list[Word]:
    return [Word(start, end, text) for start, end, text in spec]


@pytest.fixture
def two_line_words() -> list[Word]:
    """Two phrases separated by a 1.5s breath."""
    return words_from(
        [
            (0.0, 0.4, "lantern"),
            (0.5, 0.9, "over"),
            (1.0, 1.6, "water"),
            (3.1, 3.5, "counting"),
            (3.6, 4.0, "every"),
            (4.1, 4.8, "window"),
        ]
    )


@pytest.fixture
def simple_chords() -> list[ChordEvent]:
    """One chord per bar, changing mid-phrase in each line."""
    return [
        ChordEvent(0.0, 1.0, "C"),
        ChordEvent(1.0, 2.0, "G"),
        ChordEvent(3.0, 3.6, "Am"),
        ChordEvent(3.6, 5.0, "F"),
    ]
