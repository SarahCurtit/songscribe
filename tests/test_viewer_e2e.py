"""Grid viewer in a real browser: playback, follow, seeking, chord and staff lanes.

Runs headless Chromium through Playwright, and skips itself when Playwright or
its browser is not installed -- CI installs core deps only. Audio plays at
:data:`PLAYBACK_RATE`, Chromium's ceiling, so a half-minute grid plays through
in about two seconds. The track is silence and the song is invented.

Set up once with ``pip install playwright`` and ``playwright install chromium``.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from songscribe import chords, lab
from songscribe import melody as melody_stage
from songscribe.types import ChordEvent, NoteEvent
from songscribe.viewer import examples
from songscribe.viewer.server import Viewer
from tests.test_viewer import write_grid

sync_api = pytest.importorskip("playwright.sync_api")

pytestmark = pytest.mark.e2e

PLAYBACK_RATE = 16
TRACK_SECONDS = 32.0
BAR_SECONDS = 2.0
# Wider than the timeline at any test zoom, so follow has something to do.
VIEWPORT = {"width": 1000, "height": 800}
PROGRESSION = ["C:maj", "A:min", "F:maj7", "G:7"]
GRID_ID = "0/lantern-reel/3"
# An invented lead line, one whole note a bar: C5, D5, E5, F5.
MELODY_PITCHES = [72, 74, 76, 77]
# Bars the fake tracker gets wrong, and how: an octave up, then a semitone.
OCTAVE_BAR, WRONG_BAR = 1, 2
# A later bar carrying the same letter as OCTAVE_BAR, transcribed as written.
SAME_LETTER_BAR = OCTAVE_BAR + len(MELODY_PITCHES)
# An octave is seven staff steps, at four px a step (STAFF.step in app.js).
OCTAVE_PX = 28


def grid_payload() -> dict:
    bars = int(TRACK_SECONDS / BAR_SECONDS)
    return {
        "song": "Lantern Reel",
        "grid": 3,
        "events": [
            {
                "start": bar * BAR_SECONDS,
                "end": (bar + 1) * BAR_SECONDS,
                "chord": PROGRESSION[bar % len(PROGRESSION)],
            }
            for bar in range(bars)
        ],
        "annotation": {
            "bars": bars,
            "bar_seconds": BAR_SECONDS,
            "start_seconds": 0.0,
            "bpm_nominal": 120.0,
            "audio_seconds": TRACK_SECONDS,
            "sections": [{"bar": 1, "name": "A"}, {"bar": 9, "name": "B"}],
            "alternates": [{"bar": 2, "chord": "D:min7"}],
        },
    }


def written_melody() -> list[NoteEvent]:
    """The lead line the chart prints: one whole note a bar, right through."""
    return [
        NoteEvent(
            bar * BAR_SECONDS,
            (bar + 1) * BAR_SECONDS,
            MELODY_PITCHES[bar % len(MELODY_PITCHES)],
        )
        for bar in range(int(TRACK_SECONDS / BAR_SECONDS))
    ]


def fake_line(audio, sample_rate, **kwargs) -> list[NoteEvent]:
    """The written line, but one bar an octave out and one a semitone wrong."""
    notes = []
    for note in written_melody():
        shift = {OCTAVE_BAR: 12, WRONG_BAR: 1}.get(int(note.start / BAR_SECONDS), 0)
        notes.append(NoteEvent(note.start, note.end, note.pitch + shift))
    return notes


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as playwright:
        try:
            launched = playwright.chromium.launch(
                args=["--autoplay-policy=no-user-gesture-required"]
            )
        except sync_api.Error as error:
            pytest.skip(f"Chromium unavailable: {error.message.splitlines()[0]}")
        yield launched
        launched.close()


@pytest.fixture
def base_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "examples"
    write_grid(root / "lantern-reel", 3, grid_payload(), seconds=TRACK_SECONDS)
    lab.write_notes(root / "lantern-reel" / "grid3_melody.notes", written_melody())

    def fake_backend(audio, sample_rate, **kwargs) -> list[ChordEvent]:
        # Right on bar 1, plays bar 2's alternate (Dm7) as a triad, then C
        # to the half-way mark and a semitone off for the rest.
        return [
            ChordEvent(0.0, BAR_SECONDS, "C"),
            ChordEvent(BAR_SECONDS, 2 * BAR_SECONDS, "Dm"),
            ChordEvent(2 * BAR_SECONDS, TRACK_SECONDS / 2, "C"),
            ChordEvent(TRACK_SECONDS / 2, TRACK_SECONDS, "C#"),
        ]

    def stub_backend(audio, sample_rate, **kwargs):
        raise NotImplementedError("the stub backend is not implemented yet")

    def crashing_backend(audio, sample_rate, **kwargs):
        raise RuntimeError("weights file is corrupt")

    for name, backend in [
        ("fake", fake_backend),
        ("stub", stub_backend),
        ("crash", crashing_backend),
    ]:
        monkeypatch.setitem(chords.BACKENDS, name, backend)
        monkeypatch.setitem(chords.BACKEND_SAMPLE_RATES, name, 8_000)
    monkeypatch.setitem(melody_stage.BACKENDS, "fakeline", fake_line)
    monkeypatch.setitem(melody_stage.BACKEND_SAMPLE_RATES, "fakeline", 8_000)
    monkeypatch.setattr("songscribe.audio.load", lambda path, rate: (None, rate))

    viewer = Viewer(("127.0.0.1", 0), examples.discover([root]), tmp_path / "cache")
    threading.Thread(target=viewer.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{viewer.server_address[1]}"
    viewer.shutdown()
    viewer.server_close()


@pytest.fixture
def page(browser, base_url: str):
    context = browser.new_context(viewport=VIEWPORT)
    page = context.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("console", lambda m: m.type == "error" and errors.append(m.text))
    page.goto(base_url)
    page.wait_for_function("document.querySelectorAll('#svg rect.block').length > 0")
    page.wait_for_function("document.getElementById('audio').readyState >= 1")
    yield page
    context.close()
    assert not errors, f"browser errors: {errors}"


def zoom(page, px_per_second: int) -> None:
    page.eval_on_selector(
        "#zoom",
        "(el, v) => { el.value = v; el.dispatchEvent(new Event('input')); }",
        px_per_second,
    )


def play_fast(page) -> None:
    page.evaluate(f"(() => {{ const a = audio; a.playbackRate = {PLAYBACK_RATE}; a.play(); }})()")


def scroll_state(page) -> dict:
    return page.evaluate(
        """() => ({
            time: audio.currentTime,
            left: scroller.scrollLeft,
            visible: scroller.clientWidth,
            content: scroller.scrollWidth,
            playhead: Number(document.querySelector('.playhead').getAttribute('x1')),
            pageOverflow: document.documentElement.scrollWidth - window.innerWidth,
        })"""
    )


def test_page_draws_the_grid(page) -> None:
    assert page.locator("#grid-select option").all_inner_texts() == [
        "Lantern Reel — grid 3 · 120 bpm"
    ]
    blocks = page.locator("#svg rect.block:not(.alt)").count()
    assert blocks == int(TRACK_SECONDS / BAR_SECONDS)
    assert page.locator("#svg rect.block.alt").count() == 1
    # SVG text has no layout text, so inner_text() is empty; text_content() isn't.
    assert page.locator("#svg text.section-name").all_text_contents() == ["[A]", "[B]"]
    assert page.locator("#now .card .chord").first.inner_text() == "C"


def test_timeline_scrolls_inside_its_panel_not_the_page(page) -> None:
    zoom(page, 100)
    state = scroll_state(page)
    assert state["content"] > state["visible"], "timeline should overflow its scroller"
    assert state["pageOverflow"] <= 0, "the page itself must not scroll sideways"


def test_follow_keeps_the_playhead_in_view(page) -> None:
    zoom(page, 100)
    play_fast(page)
    page.wait_for_function(f"audio.currentTime > {TRACK_SECONDS * 0.6}")
    state = scroll_state(page)
    page.evaluate("audio.pause()")

    assert state["left"] > 0, "follow never scrolled the timeline"
    assert state["left"] <= state["playhead"] <= state["left"] + state["visible"]


def test_follow_off_leaves_the_scroll_alone(page) -> None:
    zoom(page, 100)
    page.uncheck("#follow")
    # Zooming keeps the view's centre, so the scroll need not start at 0.
    before = scroll_state(page)["left"]
    play_fast(page)
    page.wait_for_function(f"audio.currentTime > {TRACK_SECONDS * 0.6}")
    page.evaluate("audio.pause()")
    assert scroll_state(page)["left"] == before


def test_playback_moves_playhead_and_readout(page) -> None:
    zoom(page, 50)
    play_fast(page)
    page.wait_for_function(f"audio.currentTime > {BAR_SECONDS * 1.25}")
    page.evaluate("audio.pause()")
    page.wait_for_timeout(100)  # let one animation frame catch up
    state = scroll_state(page)
    assert state["playhead"] == pytest.approx(state["time"] * 50, abs=2)
    bar = int(state["time"] / BAR_SECONDS)
    expected = chords.normalise_label(PROGRESSION[bar % len(PROGRESSION)])
    assert page.locator("#now .card .chord").first.inner_text() == expected


def test_click_seeks(page) -> None:
    zoom(page, 50)
    box = page.locator("#svg").bounding_box()
    page.mouse.click(box["x"] + 10 * 50, box["y"] + 50)
    assert page.evaluate("audio.currentTime") == pytest.approx(10, abs=0.1)


def test_model_lane_runs_and_scores(page) -> None:
    page.check(".backend[data-backend='fake'] input")
    page.wait_for_function("document.querySelectorAll('#svg rect.agree').length > 0")
    levels = page.eval_on_selector_all(
        "#svg rect.agree", "els => els.map(e => e.getAttribute('class'))"
    )
    assert "agree exact" in levels
    assert "agree miss" in levels
    assert "models=fake" in page.evaluate("location.hash")
    score = page.locator("#lane-labels .score:not(.staff)").inner_text()
    assert score.endswith("%")


@pytest.mark.parametrize(
    ("backend", "shown"),
    [("stub", "not implemented yet"), ("crash", "failed: weights file is corrupt")],
)
def test_backend_failure_says_why(page, backend: str, shown: str) -> None:
    page.check(f".backend[data-backend='{backend}'] input")
    status = page.locator(f".backend[data-backend='{backend}'] .status")
    sync_api.expect(status).to_have_text(shown)
    assert page.locator("#svg rect.agree").count() == 0


def test_alternate_match_is_not_a_mistake(page) -> None:
    zoom(page, 50)
    page.check(".backend[data-backend='fake'] input")
    page.wait_for_function("document.querySelectorAll('#svg rect.agree').length > 0")
    strip = page.locator("#svg rect.agree").first.bounding_box()
    svg = page.locator("#svg").bounding_box()
    # Middle of bar 2, on the agreement strip.
    page.mouse.move(svg["x"] + 1.5 * BAR_SECONDS * 50, strip["y"] + strip["height"] / 2)
    tooltip = page.locator("#tooltip")
    assert tooltip.is_visible()
    assert tooltip.inner_text().startswith("same triad (alternate)")


# --------------------------------------------------------------------------- #
# The staff
# --------------------------------------------------------------------------- #


def note_classes(page) -> list[str]:
    return page.eval_on_selector_all("#svg g.note", "els => els.map(e => e.getAttribute('class'))")


def test_the_written_melody_is_drawn_before_any_model_runs(page) -> None:
    # The label carries the clef beside its name, so match the start of it.
    labels = page.locator("#lane-labels div").all_text_contents()
    assert any(text.startswith("Melody (written)") for text in labels)
    # One staff, five lines, and one head per written note.
    assert page.locator("#svg g.staff").count() == 1
    assert page.locator("#svg g.staff line.staffline").count() == 5
    assert page.locator("#svg g.note").count() == len(written_melody())
    # 120 bpm, so a 2s bar is four beats and each note is a whole note: hollow,
    # and with no flag.
    assert page.locator("#svg g.note ellipse.head.filled").count() == 0
    assert page.locator("#svg g.note path.flag").count() == 0


def test_melody_lane_runs_and_grades_every_note(page) -> None:
    page.check(".backend[data-backend='fakeline'] input")
    page.wait_for_function("document.querySelectorAll('#svg g.note.exact').length > 0")

    classes = note_classes(page)
    assert "note exact" in classes
    assert "note octave" in classes
    assert "note miss" in classes
    assert page.locator("#svg g.staff").count() == 2, "written and transcribed staves"
    assert "melody=fakeline" in page.evaluate("location.hash")


def test_an_octave_error_is_drawn_an_octave_higher(page) -> None:
    """The point of a staff: a wrong octave is visible as height, not a label."""
    page.check(".backend[data-backend='fakeline'] input")
    page.wait_for_function("document.querySelectorAll('#svg g.note.octave').length > 0")
    heads = page.eval_on_selector_all(
        "#svg g.staff:last-of-type g.note ellipse.head",
        "els => els.map(e => Number(e.getAttribute('cy')))",
    )
    # Both bars carry the same letter, and only OCTAVE_BAR was moved, so the
    # two heads are exactly an octave apart on one staff. y grows downwards,
    # so the higher note is the smaller number.
    assert heads[OCTAVE_BAR] < heads[SAME_LETTER_BAR]
    assert heads[SAME_LETTER_BAR] - heads[OCTAVE_BAR] == pytest.approx(OCTAVE_PX, abs=1)


def test_a_note_tooltip_names_the_pitch_and_the_value(page) -> None:
    zoom(page, 50)
    page.check(".backend[data-backend='fakeline'] input")
    page.wait_for_function("document.querySelectorAll('#svg g.note.exact').length > 0")
    head = page.locator("#svg g.staff:last-of-type g.note").first.bounding_box()
    page.mouse.move(head["x"] + head["width"] / 2, head["y"] + head["height"] / 2)
    tooltip = page.locator("#tooltip")
    assert tooltip.is_visible()
    text = tooltip.inner_text()
    assert text.startswith("C5")
    assert "whole" in text
    assert "exact note" in text


def test_the_melody_score_is_reported_beside_the_lane(page) -> None:
    page.check(".backend[data-backend='fakeline'] input")
    page.wait_for_function("document.querySelectorAll('#svg g.note.exact').length > 0")
    staff = page.locator("#lane-labels .score.staff").all_inner_texts()
    assert len(staff) == 2, f"written and transcribed staves, got {staff}"
    # 16 bars, one an octave out and one plain wrong: 14/16 exact, 15/16 by
    # pitch class, and one transcribed note per written one.
    assert "88%" in staff[-1]
    assert "94%" in staff[-1]
    assert "16/16 notes" in staff[-1]


def test_the_playhead_readout_names_the_sounding_note(page) -> None:
    page.check(".backend[data-backend='fakeline'] input")
    page.wait_for_function("document.querySelectorAll('#svg g.note.exact').length > 0")
    names = page.locator("#now .card .name").all_inner_texts()
    assert "Melody (written)" in names
    assert "fakeline" in names
    chords_shown = page.locator("#now .card .chord").all_inner_texts()
    # At time 0 the first written note is sounding, and the model agrees.
    assert chords_shown[names.index("Melody (written)")] == "C5"
    assert chords_shown[names.index("fakeline")] == "C5"
