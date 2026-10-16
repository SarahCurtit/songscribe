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
from collections.abc import Mapping
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

#: Standard deviation of the noise before bar 1, against a chord's peak of 1:
#: loud enough to be junk a tracker must see past, too flat to carry a chord.
LEAD_IN_LEVEL = 0.3


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


def chord_frequencies(label: str, octave: int = ROOT_OCTAVE) -> list[float]:
    """Frequencies in Hz for the notes of ``label``, voiced from ``octave``."""
    pitch_class, intervals = parse_chord(label)
    midi_root = 12 * (octave + 1) + pitch_class
    return [440.0 * 2 ** ((midi_root + i - 69) / 12) for i in intervals]


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


def _voices(label: str, t: np.ndarray) -> np.ndarray:
    """One chord's additive tones over times ``t``, scaled to a peak of 1."""
    voices = sum(
        amplitude * np.sin(2 * np.pi * freq * harmonic * t)
        for freq in chord_frequencies(label)
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
    """A multi-chorus take: its audio, true chords, and true bar lines.

    ``events`` is the whole take on the audio's clock; ``reference`` is the
    chart alone -- one chorus from 0 at the nominal tempo -- as a lead sheet
    would supply it. ``downbeats`` ends with the close of the last bar.
    """

    audio: np.ndarray
    sample_rate: int
    events: list[ChordEvent]
    reference: list[ChordEvent]
    downbeats: tuple[float, ...]
    beats_per_bar: int
    bars_per_chorus: int

    def beat_events(self) -> list[BeatEvent]:
        """The true beats, as a tracker would hand them over."""
        beats = [
            BeatEvent(time=a + (b - a) * k / self.beats_per_bar, position=k + 1)
            for a, b in pairwise(self.downbeats)
            for k in range(self.beats_per_bar)
        ]
        return [*beats, BeatEvent(time=self.downbeats[-1], position=1)]


def synthesise_performance(
    progression: list[str] | tuple[str, ...] = DEFAULT_PROGRESSION,
    bpm: float = DEFAULT_BPM,
    beats_per_bar: int = DEFAULT_BEATS_PER_BAR,
    repeats: int = DEFAULT_REPEATS,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    noise: float = 0.0,
    seed: int = 0,
    lead_in_seconds: float = 0.0,
    tempo_drift: float = 0.0,
    substitutions: Mapping[int, Mapping[int, str]] | None = None,
) -> Performance:
    """Render ``repeats`` choruses the way a band plays them, not a loop.

    ``lead_in_seconds`` of noise precede bar 1 (no chord there, no bar line);
    ``tempo_drift`` is the fractional tempo change from first bar to last;
    ``substitutions[repeat][bar]`` swaps one bar's chord in one chorus.
    """
    if not progression:
        raise ValueError("progression is empty")
    if bpm <= 0 or beats_per_bar < 1 or repeats < 1 or sample_rate < 1 or noise < 0:
        raise ValueError("bpm, beats_per_bar, repeats, sample_rate must be positive; noise >= 0")
    if lead_in_seconds < 0:
        raise ValueError(f"lead_in_seconds must not be negative, got {lead_in_seconds}")
    if tempo_drift <= -1:
        raise ValueError(f"tempo_drift must exceed -1, got {tempo_drift}")

    substitutions = substitutions or {}
    for repeat, changes in substitutions.items():
        if not 0 <= repeat < repeats or any(not 0 <= bar < len(progression) for bar in changes):
            raise ValueError(f"substitution outside the {repeats} x {len(progression)} take")

    rng = np.random.default_rng(seed)
    total = len(progression) * repeats
    lead = round(lead_in_seconds * sample_rate)
    pieces: list[np.ndarray] = [rng.normal(0.0, LEAD_IN_LEVEL, lead)]
    events: list[ChordEvent] = []
    downbeats = [lead / sample_rate]
    cursor = lead

    for index in range(total):
        repeat, bar = divmod(index, len(progression))
        label = substitutions.get(repeat, {}).get(bar, progression[bar])
        ramp = tempo_drift * index / (total - 1) if total > 1 else 0.0
        length = round(beats_per_bar * 60.0 / (bpm * (1 + ramp)) * sample_rate)
        t = np.arange(length) / sample_rate
        pieces.append(_voices(label, t) * _envelope(length, sample_rate))
        events.append(ChordEvent(cursor / sample_rate, (cursor + length) / sample_rate, label))
        cursor += length
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
        "--lead-in-seconds",
        type=float,
        default=0.0,
        help="noise before bar 1, like a count-in or stage noise (default: %(default)s)",
    )
    parser.add_argument(
        "--tempo-drift",
        type=float,
        default=0.0,
        help="fractional tempo change from first bar to last, e.g. 0.03",
    )
    parser.add_argument(
        "--vary",
        action="append",
        default=[],
        metavar="REPEAT:BAR=CHORD",
        help="swap one bar's chord in one chorus, counting from 0; repeatable",
    )
    parser.add_argument(
        "--no-annotation", action="store_true", help="skip writing the .lab ground truth"
    )
    return parser


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
    """The multi-chorus take: audio, full timeline, chart, and true bar lines."""
    import soundfile as sf

    performance = synthesise_performance(
        progression=args.progression,
        bpm=args.bpm,
        beats_per_bar=args.beats_per_bar,
        repeats=args.repeats,
        sample_rate=args.sample_rate,
        noise=args.noise,
        lead_in_seconds=args.lead_in_seconds,
        tempo_drift=args.tempo_drift,
        substitutions=_parse_variation(args.vary),
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
        (".beats", lab.write_beats, performance.beat_events()),
    ):
        path = stem.with_name(stem.name + suffix)
        write(path, content)
        print(f"wrote {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        if args.lead_in_seconds or args.tempo_drift or args.vary:
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
