"""Synthesise a demo track with known chords.

No audio ships with songscribe -- recordings are copyrighted and would bloat
the repo -- so there is nothing to run the pipeline against out of the box.
This generates something instead: additive-synthesis triads over a chord
progression you choose, plus the ground-truth annotation, so the chord branch
always has a target to be measured against.

What it cannot do is test the lyric branch. Synthesising singing is out of
scope, so the output is instrumental: Whisper will correctly find no words in
it. Use your own recording for that.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np

from . import lab
from .chords import PITCH_CLASSES
from .types import BeatEvent, ChordEvent

# Written with sharps, like PITCH_CLASSES; flats are accepted and mapped in.
FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}

# Semitone offsets from the root. Deliberately small: this is test material,
# not an instrument, and every quality here is one the pipeline can label.
INTERVALS = {
    "": (0, 4, 7),
    "m": (0, 3, 7),
    "7": (0, 4, 7, 10),
    "m7": (0, 3, 7, 10),
    "maj7": (0, 4, 7, 11),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
}

# A I-V-vi-IV turnaround: the most common progression in popular music, and
# squarely inside the default backend's major/minor vocabulary.
DEFAULT_PROGRESSION = ("C", "G", "Am", "F")

DEFAULT_BPM = 96.0
DEFAULT_BEATS_PER_BAR = 4
DEFAULT_REPEATS = 2
DEFAULT_SAMPLE_RATE = 44_100
ROOT_OCTAVE = 3

# Harmonic amplitudes. A handful of partials keeps the chroma from looking
# like a pure tone, which is unrealistically easy for a chord model.
HARMONICS = (1.0, 0.5, 0.33, 0.25)


def parse_chord(label: str) -> tuple[int, tuple[int, ...]]:
    """Split a chord label into its root pitch class and interval set.

    ``"Am"`` -> ``(9, (0, 3, 7))``. Raises :class:`ValueError` on anything
    outside :data:`INTERVALS`, rather than guessing at a quality.
    """
    label = label.strip()
    if not label:
        raise ValueError("empty chord label")

    # Longest root first, so "C#" wins over "C".
    for length in (2, 1):
        root, quality = label[:length], label[length:]
        root = FLAT_TO_SHARP.get(root, root)
        if root in PITCH_CLASSES and quality in INTERVALS:
            return PITCH_CLASSES.index(root), INTERVALS[quality]

    raise ValueError(
        f"cannot parse chord {label!r}; supported qualities: "
        f"{sorted(q or '(major)' for q in INTERVALS)}"
    )


def transpose_chord(label: str, semitones: int) -> str:
    """``label`` moved ``semitones`` up, spelled with sharps. Raises like ``parse_chord``."""
    pitch_class, _ = parse_chord(label)
    label = label.strip()
    quality = label[2:] if label[:2] in PITCH_CLASSES or label[:2] in FLAT_TO_SHARP else label[1:]
    return PITCH_CLASSES[(pitch_class + semitones) % 12] + quality


def chord_frequencies(
    label: str, octave: int = ROOT_OCTAVE, extra: Sequence[int] = ()
) -> list[float]:
    """Frequencies in Hz for the notes of ``label``, voiced from ``octave``.

    ``extra`` adds notes, as semitones above the root, to the chord's own.
    """
    pitch_class, intervals = parse_chord(label)
    midi_root = 12 * (octave + 1) + pitch_class
    return [440.0 * 2 ** ((midi_root + i - 69) / 12) for i in (*intervals, *extra)]


def _envelope(length: int, sample_rate: int) -> np.ndarray:
    """A plucked shape: fast attack, slow decay, short release.

    Without this every bar is a steady drone, and the chord boundaries a
    tracker has to find are far cleaner than in real music.
    """
    attack = min(int(0.01 * sample_rate), length)
    release = min(int(0.05 * sample_rate), length - attack)
    sustain = length - attack - release

    return np.concatenate(
        [
            np.linspace(0.0, 1.0, attack, endpoint=False),
            np.exp(-2.0 * np.linspace(0.0, 1.0, sustain, endpoint=False)),
            np.linspace(np.exp(-2.0), 0.0, release),
        ]
    )[:length]


def _voices(label: str, t: np.ndarray, extra: Sequence[int] = ()) -> np.ndarray:
    """Sum the sine tones of one chord at times ``t`` (in seconds), scaled to a peak of 1."""
    voices = sum(
        amplitude * np.sin(2 * np.pi * freq * harmonic * t)
        for freq in chord_frequencies(label, extra=extra)
        for harmonic, amplitude in enumerate(HARMONICS, start=1)
    )
    return voices / np.max(np.abs(voices))


def synthesise(
    progression: list[str] | tuple[str, ...] = DEFAULT_PROGRESSION,
    bpm: float = DEFAULT_BPM,
    beats_per_bar: int = DEFAULT_BEATS_PER_BAR,
    repeats: int = DEFAULT_REPEATS,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    noise: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, list[ChordEvent]]:
    """Render ``progression`` to mono float32 audio plus its chord timeline.

    One chord per bar. ``noise`` adds white noise at that amplitude, seeded so
    the output stays byte-identical across runs.
    """
    if not progression:
        raise ValueError("progression is empty")
    if bpm <= 0:
        raise ValueError(f"bpm must be positive, got {bpm}")
    if beats_per_bar < 1:
        raise ValueError(f"beats_per_bar must be at least 1, got {beats_per_bar}")
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}")
    if sample_rate < 1:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    if noise < 0:
        raise ValueError(f"noise must not be negative, got {noise}")

    bar_seconds = beats_per_bar * 60.0 / bpm
    samples_per_bar = round(bar_seconds * sample_rate)
    t = np.arange(samples_per_bar) / sample_rate
    envelope = _envelope(samples_per_bar, sample_rate)

    bars: list[np.ndarray] = []
    events: list[ChordEvent] = []

    for index, label in enumerate(list(progression) * repeats):
        bars.append(_voices(label, t) * envelope)
        start = index * samples_per_bar / sample_rate
        events.append(ChordEvent(start=start, end=start + bar_seconds, label=label))

    audio = np.concatenate(bars)
    if noise:
        audio = audio + np.random.default_rng(seed).normal(0.0, noise, audio.shape)

    # Leave headroom so writing to 16-bit PCM cannot clip.
    audio = audio / np.max(np.abs(audio)) * 0.89
    return audio.astype(np.float32), events


@dataclass(frozen=True, slots=True)
class Performance:
    """A synthetic recording of the chart played several times, with its ground truth.

    ``events`` are the chords of the whole recording, in seconds from the start of ``audio``.
    ``reference`` is one chorus of the chart from 0 at the requested tempo, like a lead sheet.
    ``downbeats`` are the starts of all bars, the first being bar 0, plus the end of the last.
    A pickup is in ``events`` and ``beat_events()``, not ``reference``; ``intro_bars`` come first.
    """

    audio: np.ndarray
    sample_rate: int
    events: list[ChordEvent]
    reference: list[ChordEvent]
    downbeats: tuple[float, ...]
    beats_per_bar: int
    bars_per_chorus: int
    pickup_beats: int = 0
    intro_bars: int = 0

    @property
    def head_start(self) -> float:
        """When the head's first bar begins, after any intro."""
        return self.downbeats[self.intro_bars]

    def beat_events(self, tracker_phase: int = 0) -> list[BeatEvent]:
        """The beats of the recording, in the format a beat tracker returns.

        ``tracker_phase`` moves every beat's position that many places on, as a tracker
        whose downbeats are off by that many beats would report them.
        """
        stub = self.pickup_beats if self.intro_bars == 0 else 0
        pickup = [
            BeatEvent(
                time=self.downbeats[0] * k / stub,
                position=self.beats_per_bar - stub + k + 1,
            )
            for k in range(stub)
        ]
        beats = [
            BeatEvent(time=a + (b - a) * k / self.beats_per_bar, position=k + 1)
            for a, b in pairwise(self.downbeats)
            for k in range(self.beats_per_bar)
        ]
        every = [*pickup, *beats, BeatEvent(time=self.downbeats[-1], position=1)]
        return [
            BeatEvent(e.time, (e.position - 1 + tracker_phase) % self.beats_per_bar + 1)
            for e in every
        ]


def synthesise_performance(
    progression: list[str] | tuple[str, ...] = DEFAULT_PROGRESSION,
    bpm: float = DEFAULT_BPM,
    beats_per_bar: int = DEFAULT_BEATS_PER_BAR,
    repeats: int = DEFAULT_REPEATS,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    noise: float = 0.0,
    seed: int = 0,
    pickup_beats: int = 0,
    tempo_drift: float = 0.0,
    substitutions: Mapping[int, Mapping[int, str]] | None = None,
    transpose: int = 0,
    intro: Sequence[str] = (),
    colour: Mapping[int, Sequence[int]] | None = None,
) -> Performance:
    """Synthesise ``repeats`` choruses of ``progression``, one chord per bar.

    ``pickup_beats`` of the chart's last chord precede the head: the last beats of the
    ``intro`` (one chord per bar) or, with none, a stub before bar 0. ``colour[repeat]`` adds
    notes (semitones above each root) to a whole chorus. ``tempo_drift`` is the relative
    tempo change from first to last bar. ``substitutions[repeat][bar]`` replaces one chord.
    The whole take, pickup included, is played ``transpose`` semitones above the chart.
    """
    if not progression:
        raise ValueError("progression is empty")
    if bpm <= 0 or beats_per_bar < 1 or repeats < 1 or sample_rate < 1 or noise < 0:
        raise ValueError("bpm, beats_per_bar, repeats, sample_rate must be positive; noise >= 0")
    if not 0 <= pickup_beats < beats_per_bar:
        raise ValueError(f"pickup_beats must be 0..{beats_per_bar - 1}, got {pickup_beats}")
    if tempo_drift <= -1:
        raise ValueError(f"tempo_drift must exceed -1, got {tempo_drift}")

    substitutions = substitutions or {}
    for repeat, changes in substitutions.items():
        if not 0 <= repeat < repeats or any(not 0 <= bar < len(progression) for bar in changes):
            raise ValueError(f"substitution outside the {repeats} x {len(progression)} take")

    rng = np.random.default_rng(seed)
    colour = colour or {}
    total = len(progression) * repeats
    pieces: list[np.ndarray] = []
    events: list[ChordEvent] = []
    beat_samples = 60.0 / bpm * sample_rate
    stub = round(pickup_beats * beat_samples) if not intro else 0
    cursor = 0
    last = transpose_chord(progression[-1], transpose)

    def play(label: str, length: int, extra: Sequence[int] = ()) -> None:
        nonlocal cursor
        t = np.arange(length) / sample_rate
        pieces.append(_voices(label, t, extra) * _envelope(length, sample_rate))
        events.append(ChordEvent(cursor / sample_rate, (cursor + length) / sample_rate, label))
        cursor += length

    if stub:
        play(last, stub)
    downbeats = [cursor / sample_rate]

    bar = round(beats_per_bar * beat_samples)
    for k, label in enumerate(intro):
        label = transpose_chord(label, transpose)
        if pickup_beats and k == len(intro) - 1:
            held = round((beats_per_bar - pickup_beats) * beat_samples)
            play(label, held)
            play(last, bar - held)
        else:
            play(label, bar)
        downbeats.append(cursor / sample_rate)

    for index in range(total):
        repeat, position = divmod(index, len(progression))
        label = substitutions.get(repeat, {}).get(position, progression[position])
        label = transpose_chord(label, transpose)
        ramp = tempo_drift * index / (total - 1) if total > 1 else 0.0
        length = round(beats_per_bar * 60.0 / (bpm * (1 + ramp)) * sample_rate)
        play(label, length, colour.get(repeat, ()))
        downbeats.append(cursor / sample_rate)

    audio = np.concatenate(pieces)
    if noise:
        audio = audio + rng.normal(0.0, noise, audio.shape)
    audio = audio / np.max(np.abs(audio)) * 0.89

    bar_seconds = beats_per_bar * 60.0 / bpm
    reference = [
        ChordEvent(i * bar_seconds, (i + 1) * bar_seconds, label)
        for i, label in enumerate(progression)
    ]
    return Performance(
        audio=audio.astype(np.float32),
        sample_rate=sample_rate,
        events=events,
        reference=reference,
        downbeats=tuple(downbeats),
        beats_per_bar=beats_per_bar,
        bars_per_chorus=len(progression),
        pickup_beats=pickup_beats,
        intro_bars=len(intro),
    )


def write_annotation(path: Path, events: list[ChordEvent]) -> None:
    """Write chord ground truth as a ``.lab`` file.

    Tab-separated ``start end label``, the Isophonics/MIREX convention -- so
    the demo doubles as a one-track evaluation target, not just a smoke test:
    ``songscribe.eval`` reads this file with no special casing.
    """
    # The format itself lives in songscribe.lab, which is also what the
    # evaluation reads reference annotations with: one implementation, so the
    # writer and the reader cannot drift apart.
    lab.write(path, events)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="songscribe-demo",
        description="Generate a synthetic demo track with known chords.",
        epilog="Instrumental by design: it exercises the chord branch, not the lyric branch.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("demo.wav"),
        help="WAV file to write (default: %(default)s)",
    )
    parser.add_argument(
        "-p",
        "--progression",
        nargs="+",
        default=list(DEFAULT_PROGRESSION),
        metavar="CHORD",
        help="chords, one per bar (default: %(default)s)",
    )
    parser.add_argument(
        "--bpm", type=float, default=DEFAULT_BPM, help="tempo (default: %(default)s)"
    )
    parser.add_argument(
        "--beats-per-bar", type=int, default=DEFAULT_BEATS_PER_BAR, help="(default: %(default)s)"
    )
    parser.add_argument(
        "-r",
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="times through the progression (default: %(default)s)",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE, help="(default: %(default)s)"
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="add white noise at this amplitude, e.g. 0.01, to make it less pristine",
    )
    parser.add_argument(
        "--pickup-beats",
        type=int,
        default=0,
        help="beats of the last chord played before bar 1, as a pickup (default: %(default)s)",
    )
    parser.add_argument(
        "--tempo-drift",
        type=float,
        default=0.0,
        help="relative tempo change between the first and last bar, e.g. 0.03 for 3%% faster",
    )
    parser.add_argument(
        "--vary",
        action="append",
        default=[],
        metavar="REPEAT:BAR=CHORD",
        help="replace the chord of one bar in one chorus (both counted from 0); repeatable",
    )
    parser.add_argument(
        "--transpose-take",
        type=int,
        default=0,
        metavar="SEMITONES",
        help="play the whole take this far above the chart, which stays as written",
    )
    parser.add_argument(
        "--intro",
        nargs="+",
        default=[],
        metavar="CHORD",
        help="chords, one per bar, played before the head; a pickup is the last beats of the last",
    )
    parser.add_argument(
        "--colour",
        action="append",
        default=[],
        metavar="REPEAT=SEMITONES",
        help="add notes, e.g. 1=10,14 (semitones above each root), to one whole chorus; repeatable",
    )
    parser.add_argument(
        "--tracker-phase",
        type=int,
        default=0,
        metavar="BEATS",
        help="write a .beats file whose downbeats are this many beats off, like a bad tracker",
    )
    parser.add_argument(
        "--no-annotation", action="store_true", help="skip writing the .lab ground truth"
    )
    return parser


def _parse_colour(texts: list[str]) -> dict[int, list[int]]:
    colour: dict[int, list[int]] = {}
    for text in texts:
        try:
            repeat, notes = text.split("=", 1)
            colour[int(repeat)] = [int(n) for n in notes.split(",")]
        except ValueError:
            raise ValueError(f"--colour wants REPEAT=SEMITONES, got {text!r}") from None
    return colour


def _parse_variation(texts: list[str]) -> dict[int, dict[int, str]]:
    substitutions: dict[int, dict[int, str]] = {}
    for text in texts:
        try:
            place, chord = text.split("=", 1)
            repeat, bar = place.split(":", 1)
            substitutions.setdefault(int(repeat), {})[int(bar)] = chord
        except ValueError:
            raise ValueError(f"--vary wants REPEAT:BAR=CHORD, got {text!r}") from None
    return substitutions


def _write_performance(args: argparse.Namespace) -> int:
    """Write the audio plus the ``.lab``, ``.reference.lab`` and ``.beats`` ground truth."""
    import soundfile as sf

    performance = synthesise_performance(
        progression=args.progression,
        bpm=args.bpm,
        beats_per_bar=args.beats_per_bar,
        repeats=args.repeats,
        sample_rate=args.sample_rate,
        noise=args.noise,
        pickup_beats=args.pickup_beats,
        tempo_drift=args.tempo_drift,
        substitutions=_parse_variation(args.vary),
        intro=args.intro,
        colour=_parse_colour(args.colour),
        transpose=args.transpose_take,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, performance.audio, args.sample_rate)
    print(f"wrote {args.output} ({len(performance.audio) / args.sample_rate:.1f}s)")
    if args.no_annotation:
        return 0
    stem = args.output.with_suffix("")
    for suffix, write, content in (
        (".lab", write_annotation, performance.events),
        (".reference.lab", write_annotation, performance.reference),
        (".beats", lab.write_beats, performance.beat_events(args.tracker_phase)),
    ):
        path = stem.with_name(stem.name + suffix)
        write(path, content)
        print(f"wrote {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        if (
            args.pickup_beats
            or args.tempo_drift
            or args.vary
            or args.transpose_take
            or args.intro
            or args.colour
            or args.tracker_phase
        ):
            return _write_performance(args)
        audio, events = synthesise(
            progression=args.progression,
            bpm=args.bpm,
            beats_per_bar=args.beats_per_bar,
            repeats=args.repeats,
            sample_rate=args.sample_rate,
            noise=args.noise,
        )
    except ValueError as error:
        print(f"songscribe-demo: {error}", file=sys.stderr)
        return 1

    import soundfile as sf

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, audio, args.sample_rate)
    print(f"wrote {args.output} ({len(audio) / args.sample_rate:.1f}s)")

    if not args.no_annotation:
        annotation = args.output.with_suffix(".lab")
        write_annotation(annotation, events)
        print(f"wrote {annotation} ({len(events)} chords)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
