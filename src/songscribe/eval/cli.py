"""``songscribe-eval`` -- build an evaluation manifest, or score a backend."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from math import isnan
from pathlib import Path

from .. import __version__
from ..chords import BACKENDS, DEFAULT_BACKEND
from .dataset import (
    ANNOTATION_ROOT_ENV,
    AUDIO_ROOT_ENV,
    ManifestError,
    load_manifest,
    write_manifest,
)
from .metrics import DEFAULT_VOCABULARIES, VOCABULARIES

#: Width of the leading column in the per-track table. Long enough for an
#: album/track path, short enough to leave room for four scores.
_ID_WIDTH = 52


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="songscribe-eval",
        description="Evaluate a chord backend against reference annotations.",
        epilog=(
            "Manifests hold relative paths only: no audio and no annotations, "
            f"so they are safe to commit. Point {AUDIO_ROOT_ENV} at your "
            f"recordings and {ANNOTATION_ROOT_ENV} at your .lab files. "
            "See EVALUATION.md."
        ),
    )
    parser.add_argument("--version", action="version", version=f"songscribe {__version__}")
    parser.add_argument("-q", "--quiet", action="store_true", help="suppress progress logging")

    # Accepting -q on either side of the subcommand needs a shared parent, and
    # SUPPRESS on it: a plain store_true would default to False in the
    # subparser and overwrite a -q given before the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        default=argparse.SUPPRESS,
        help="suppress progress logging",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser(
        "manifest",
        parents=[common],
        help="build a manifest from an annotation tree",
        description=(
            "Walk an Isophonics-style annotation tree, pair each .lab file with "
            "your copy of the recording, and write a manifest."
        ),
    )
    manifest.add_argument(
        "annotations", type=Path, help="directory holding the reference .lab files"
    )
    manifest.add_argument("-o", "--output", type=Path, required=True, help="manifest file to write")
    manifest.add_argument(
        "-a",
        "--audio",
        type=Path,
        help="directory holding your recordings; omit for a references-only manifest",
    )
    manifest.add_argument(
        "-n", "--name", default="isophonics", help="dataset name (default: %(default)s)"
    )
    manifest.add_argument(
        "--list-unpaired",
        action="store_true",
        help="print every track with no matching audio, not just the count",
    )

    score = subparsers.add_parser(
        "score",
        parents=[common],
        help="score a backend or a directory of estimates",
        description=(
            "Score chord estimates against a manifest's reference annotations. "
            "Estimates come from a backend (-c) or from .lab files (--estimates)."
        ),
    )
    score.add_argument("manifest", type=Path, help="manifest written by 'songscribe-eval manifest'")
    source = score.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "-c",
        "--chord-backend",
        choices=sorted(BACKENDS),
        help=f"run this backend (the baseline to beat is 'template'; default model is "
        f"'{DEFAULT_BACKEND}')",
    )
    source.add_argument(
        "--estimates", type=Path, help="score pre-computed .lab estimates in this directory"
    )
    score.add_argument("-a", "--audio", type=Path, help=f"audio root (overrides {AUDIO_ROOT_ENV})")
    score.add_argument(
        "--annotations", type=Path, help=f"annotation root (overrides {ANNOTATION_ROOT_ENV})"
    )
    score.add_argument(
        "-v",
        "--vocabulary",
        action="append",
        choices=sorted(VOCABULARIES),
        metavar="VOCAB",
        help=(
            "comparison vocabulary, repeatable (default: "
            f"{' '.join(DEFAULT_VOCABULARIES)}); choose from {', '.join(sorted(VOCABULARIES))}"
        ),
    )
    score.add_argument(
        "--cache",
        type=Path,
        help="cache estimates here, so re-scoring does not re-run the model",
    )
    score.add_argument("--refresh", action="store_true", help="recompute estimates even if cached")
    score.add_argument(
        "--separate",
        action="store_true",
        help="run demucs first (much slower, and the single biggest accuracy factor)",
    )
    score.add_argument("--limit", type=int, help="score only the first N tracks")
    score.add_argument("--per-track", action="store_true", help="print a row per track")
    score.add_argument("--json", type=Path, help="also write the full results as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Configure our own logger only, for the reason songscribe.cli gives: the
    # root logger would put every third-party library at INFO too.
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    package_log = logging.getLogger("songscribe")
    package_log.setLevel(logging.WARNING if args.quiet else logging.INFO)
    package_log.handlers.clear()
    package_log.addHandler(handler)
    package_log.propagate = False

    if args.command == "manifest":
        return _manifest(args)
    return _score(args)


def _manifest(args: argparse.Namespace) -> int:
    from .isophonics import build_manifest

    try:
        report = build_manifest(args.annotations, args.audio, name=args.name)
    except (OSError, ValueError) as error:
        print(f"songscribe-eval: {error}", file=sys.stderr)
        return 1

    write_manifest(args.output, report.manifest)
    print(f"wrote {args.output}: {report.total} annotations")

    if args.audio is None:
        print("no audio root given; manifest has references only (scoring estimates still works)")
        return 0

    print(f"  paired with audio: {report.paired}")
    for label, tracks in (("no audio found", report.unpaired), ("ambiguous", report.ambiguous)):
        if not tracks:
            continue
        print(f"  {label}: {len(tracks)}")
        shown = tracks if args.list_unpaired else tracks[:5]
        for track_id in shown:
            print(f"    {track_id}")
        if len(tracks) > len(shown):
            print(f"    ... and {len(tracks) - len(shown)} more (--list-unpaired to see all)")
    return 0


def _score(args: argparse.Namespace) -> int:
    from .runner import evaluate

    vocabularies = tuple(args.vocabulary) if args.vocabulary else DEFAULT_VOCABULARIES

    try:
        dataset = load_manifest(
            args.manifest, audio_root=args.audio, annotation_root=args.annotations
        )
    except (ManifestError, OSError) as error:
        print(f"songscribe-eval: {error}", file=sys.stderr)
        return 1

    # These three only steer a model run. Silently ignoring them would leave
    # someone believing they had just scored a separated-stem run.
    if args.estimates is not None:
        ignored = [
            flag
            for flag, given in (
                ("--cache", args.cache),
                ("--separate", args.separate),
                ("--refresh", args.refresh),
            )
            if given
        ]
        if ignored:
            print(
                f"songscribe-eval: {', '.join(ignored)} ignored when scoring --estimates",
                file=sys.stderr,
            )

    missing_audio = dataset.missing_audio()
    if args.estimates is None and missing_audio:
        print(
            f"songscribe-eval: {len(missing_audio)} of {len(dataset.tracks)} tracks have no "
            f"audio under {dataset.audio_root}; scoring the rest",
            file=sys.stderr,
        )
    if not dataset.tracks:
        print("songscribe-eval: manifest lists no tracks", file=sys.stderr)
        return 1

    printer = _per_track_printer(vocabularies) if args.per_track else None
    try:
        report = evaluate(
            dataset,
            backend=args.chord_backend,
            estimates_dir=args.estimates,
            vocabularies=vocabularies,
            cache_dir=args.cache,
            use_separation=args.separate,
            refresh=args.refresh,
            limit=args.limit,
            on_track=printer,
        )
    except (OSError, ValueError, ImportError) as error:
        print(f"songscribe-eval: {error}", file=sys.stderr)
        return 1
    except NotImplementedError as error:
        print(f"songscribe-eval: {error}", file=sys.stderr)
        return 2

    if not report.results:
        print("songscribe-eval: nothing could be scored", file=sys.stderr)
        return 1

    sys.stdout.write(format_report(report))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report_to_dict(report), indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json}", file=sys.stderr)
    return 0


def _per_track_printer(vocabularies: tuple[str, ...]):
    """A callback printing one row per track as the run proceeds.

    Printed live rather than collected, because a full corpus run takes long
    enough that silence is indistinguishable from a hang.
    """
    header_written = False

    def write(result) -> None:
        nonlocal header_written
        if not header_written:
            columns = "".join(f"{name:>10}" for name in vocabularies)
            print(f"{'track':<{_ID_WIDTH}}{columns}{'seg':>10}")
            header_written = True
        scores = "".join(f"{_percent(result.scores[v].wcsr):>10}" for v in vocabularies)
        boundary = _percent(result.segmentation.score)
        print(f"{_truncate(result.track_id):<{_ID_WIDTH}}{scores}{boundary:>10}")

    return write


def format_report(report) -> str:
    """Render the corpus-level summary table."""
    lines = [
        "",
        f"dataset   {report.dataset}",
        f"backend   {report.backend}",
        f"tracks    {len(report.results)} scored"
        + (f", {len(report.skipped)} skipped" if report.skipped else ""),
        f"duration  {report.total_duration / 60:.1f} min of annotated audio",
        "",
        f"{'vocabulary':<12}{'mean':>10}{'weighted':>10}   {'definition'}",
    ]
    for vocabulary in report.vocabularies:
        lines.append(
            f"{vocabulary:<12}{_percent(report.mean(vocabulary)):>10}"
            f"{_percent(report.weighted(vocabulary)):>10}   {VOCABULARIES[vocabulary]}"
        )
    lines += [
        "",
        f"{'segmentation':<12}{_percent(report.mean_segmentation()):>10}"
        f"{'':>10}   boundary agreement, labels ignored",
        "",
        "mean = per-track average; weighted = by duration across the corpus.",
        "",
    ]
    return "\n".join(lines)


def report_to_dict(report) -> dict:
    """The full results, including per-track rows, as JSON-ready data."""
    return {
        "dataset": report.dataset,
        "backend": report.backend,
        "vocabularies": list(report.vocabularies),
        "total_duration": report.total_duration,
        "skipped": list(report.skipped),
        "summary": {
            vocabulary: {
                "mean": _jsonable(report.mean(vocabulary)),
                "weighted": _jsonable(report.weighted(vocabulary)),
            }
            for vocabulary in report.vocabularies
        },
        "segmentation": {"mean": _jsonable(report.mean_segmentation())},
        "tracks": [
            {
                "id": result.track_id,
                "duration": result.duration,
                "reference_chords": result.reference_chords,
                "estimate_chords": result.estimate_chords,
                "segmentation": {
                    "over": _jsonable(result.segmentation.over),
                    "under": _jsonable(result.segmentation.under),
                    "score": _jsonable(result.segmentation.score),
                },
                "scores": {
                    vocabulary: {
                        "wcsr": _jsonable(score.wcsr),
                        "correct": score.correct,
                        "compared": score.compared,
                        "ignored": score.ignored,
                    }
                    for vocabulary, score in result.scores.items()
                },
            }
            for result in report.results
        ],
    }


def _jsonable(value: float) -> float | None:
    """JSON has no NaN; an unscoreable track is a null, not a zero."""
    return None if isnan(value) else value


def _percent(value: float) -> str:
    return "--" if isnan(value) else f"{100 * value:.1f}%"


def _truncate(text: str) -> str:
    """Keep the tail of a long track id: the title is more use than the album."""
    return text if len(text) <= _ID_WIDTH - 1 else "..." + text[-(_ID_WIDTH - 4) :]


if __name__ == "__main__":
    raise SystemExit(main())
