"""Command line entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__
from .chords import BACKENDS, DEFAULT_BACKEND
from .lyrics import DEFAULT_MODEL
from .render import RENDERERS, render


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="songscribe",
        description="Transcribe chords and lyrics from an audio file.",
    )
    parser.add_argument("audio", type=Path, help="audio file (mp3, wav, flac, m4a, ...)")
    parser.add_argument("-o", "--output", type=Path, help="write to this file instead of stdout")
    parser.add_argument(
        "-f",
        "--format",
        choices=RENDERERS,
        default="text",
        help="output format (default: %(default)s)",
    )
    parser.add_argument(
        "-c",
        "--chord-backend",
        choices=sorted(BACKENDS),
        default=DEFAULT_BACKEND,
        help="chord recognition model; 'template' is the non-DL baseline (default: %(default)s)",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        help="whisper model size for lyrics (default: %(default)s)",
    )
    parser.add_argument(
        "-l", "--language", help="lyric language as an ISO code; autodetected if omitted"
    )
    parser.add_argument("--title", help="song title for the output header")
    parser.add_argument(
        "--no-separate",
        action="store_true",
        help="skip stem separation and work on the full mix (much faster, less accurate)",
    )
    parser.add_argument(
        "--no-lyrics",
        action="store_true",
        help="chords only; loads no ASR model, the right mode for instrumentals",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="suppress progress logging")
    parser.add_argument("--version", action="version", version=f"songscribe {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s",
        stream=sys.stderr,
    )

    if not args.audio.exists():
        print(f"songscribe: no such file: {args.audio}", file=sys.stderr)
        return 1

    # Imported late so --help and --version stay instant: librosa, torch and
    # the model backends are all slow to import.
    from .pipeline import transcribe

    try:
        song = transcribe(
            args.audio,
            chord_backend=args.chord_backend,
            model_size=args.model,
            language=args.language,
            use_separation=not args.no_separate,
            use_lyrics=not args.no_lyrics,
            title=args.title,
        )
    except NotImplementedError as error:
        print(f"songscribe: {error}", file=sys.stderr)
        return 2
    except (RuntimeError, OSError, ValueError, ImportError) as error:
        print(f"songscribe: {error}", file=sys.stderr)
        return 1

    output = render(song, args.format)
    if args.output:
        args.output.write_text(output, encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
