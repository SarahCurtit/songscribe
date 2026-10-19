"""``songscribe-eval`` -- build an evaluation manifest, or score a backend."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from math import isnan
from pathlib import Path

from .. import __version__
from ..beats import BACKENDS as BEAT_BACKENDS
from ..chords import BACKENDS, DEFAULT_BACKEND
from .dataset import (
    ANNOTATION_ROOT_ENV,
    AUDIO_ROOT_ENV,
    ManifestError,
    load_manifest,
    write_manifest,
)
from .form import TRANSPOSE_MODES
from .metrics import (
    ALIGNMENT_WARN_GAIN,
    ALIGNMENT_WARN_OFFSET,
    DEFAULT_VOCABULARIES,
    VOCABULARIES,
)
from .survey import TOP_LABELS

#: Width of the leading column in the per-track table. Long enough for an
#: album/track path, short enough to leave room for four scores.
_ID_WIDTH = 52

#: Printed under the unplaced count: those tracks *were* found, so renaming the
#: files -- the fix for a miss -- would not help; the folders are what failed.
UNPLACED_HINT = "point --audio at the artist folder or rename album folders"

#: Printed under the ambiguous count: the title is in several folders and none
#: names the album -- typically a compilation beside a folder that doesn't. A
#: self-titled album can't be told from its artist folder, so only moving the
#: other copies out of --audio settles it.
AMBIGUOUS_HINT = "name the album folder after its album, or move compilations out of --audio"


def _positive_int(text: str) -> int:
    """An argparse type for counts that must be at least 1.

    A plain ``int`` lets ``--limit -1`` through, and slicing with it silently
    drops the last track instead of limiting anything.
    """
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="songscribe-eval",
        description="Evaluate a chord backend against reference annotations.",
        epilog=(
            "Manifests hold relative paths only: no audio and no annotations, "
            f"so they are safe to commit. Point {AUDIO_ROOT_ENV} and {ANNOTATION_ROOT_ENV} "
            "at the directories passed to 'songscribe-eval manifest' as --audio and as "
            "its annotation argument (for Isophonics, the one containing chordlab/). "
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
        "-n",
        "--name",
        help="dataset name for report headers (default: the annotation directory's name)",
    )
    manifest.add_argument(
        "--list-unpaired",
        action="store_true",
        help="print every track that did not pair, not just the first few",
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
    score.add_argument(
        "-a",
        "--audio",
        type=Path,
        help=f"directory passed to 'manifest --audio' (overrides {AUDIO_ROOT_ENV})",
    )
    score.add_argument(
        "--annotations",
        type=Path,
        help=f"annotation directory passed to 'manifest' (overrides {ANNOTATION_ROOT_ENV})",
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
    score.add_argument("--limit", type=_positive_int, help="score only the first N tracks")
    score.add_argument("--per-track", action="store_true", help="print a row per track")
    grid_source = score.add_mutually_exclusive_group()
    grid_source.add_argument(
        "--beats",
        type=Path,
        help=(
            "directory of <track>.beats files with the recordings' bar lines. Scores each "
            "chorus separately (tracks need 'bars' in the manifest, and audio)"
        ),
    )
    grid_source.add_argument(
        "--beat-backend",
        choices=sorted(BEAT_BACKENDS),
        help="find the bar lines with this beat tracker instead (results cached under --cache)",
    )
    score.add_argument(
        "--per-chorus",
        action="store_true",
        help="print a row per chorus; needs --beats or --beat-backend",
    )
    score.add_argument(
        "--transpose",
        choices=TRANSPOSE_MODES,
        default="auto",
        help=(
            "keys a take may be played in: auto searches all twelve for the whole take and "
            "scores against the chart moved to it; none insists on the chart's own "
            "(default: %(default)s)"
        ),
    )
    score.add_argument("--json", type=Path, help="also write the full results as JSON")
    score.add_argument(
        "--no-alignment-check",
        action="store_true",
        help="skip the time-offset sweep that flags recordings off the reference timeline",
    )

    survey = subparsers.add_parser(
        "survey",
        parents=[common],
        help="summarise a corpus's annotations; needs no audio and no model",
        description=(
            "Tally the chord labels in a manifest's reference annotations: how much "
            "annotated time there is, whether every label parses, and how much of "
            "that time each comparison vocabulary can actually score. Worth running "
            "on a new corpus before committing hours of CPU to it."
        ),
    )
    survey.add_argument("manifest", type=Path, help="manifest to survey")
    survey.add_argument(
        "--annotations",
        type=Path,
        help=f"annotation directory passed to 'manifest' (overrides {ANNOTATION_ROOT_ENV})",
    )
    survey.add_argument(
        "-v",
        "--vocabulary",
        action="append",
        choices=sorted(VOCABULARIES),
        metavar="VOCAB",
        help="vocabulary to report scoreable time for, repeatable",
    )
    survey.add_argument(
        "--labels",
        type=int,
        default=TOP_LABELS,
        help="how many labels to list by time (default: %(default)s; 0 for none)",
    )
    survey.add_argument("--json", type=Path, help="also write the survey as JSON")

    form = subparsers.add_parser(
        "form",
        parents=[common],
        help="print the bar grid, anchor and choruses found in one recording",
        description=(
            "Find the bar where the chart starts in a recording and split the recording "
            "into choruses. Prints the result without scoring anything. Exits 2 if the "
            "start bar is not clear."
        ),
    )
    form.add_argument("reference", type=Path, help="the chart as a .lab file, one chorus")
    form.add_argument(
        "--bars", type=_positive_int, required=True, help="number of bars in the chart"
    )
    form.add_argument("--audio", type=Path, help="the recording, compared with the chart by chroma")
    form.add_argument(
        "--estimate",
        type=Path,
        help="a chord .lab file to use instead of --audio. Only for inspecting a form: "
        "the start found is the one that suits the estimate",
    )
    form.add_argument(
        "--transpose",
        choices=TRANSPOSE_MODES,
        default="auto",
        help=(
            "keys a take may be played in: auto searches all twelve for the whole take and "
            "scores against the chart moved to it; none insists on the chart's own "
            "(default: %(default)s)"
        ),
    )
    form_grid = form.add_mutually_exclusive_group(required=True)
    form_grid.add_argument("--beats", type=Path, help="a .beats file for the recording")
    form_grid.add_argument(
        "--beat-backend", choices=sorted(BEAT_BACKENDS), help="find the bar lines in --audio"
    )
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
    if args.command == "survey":
        return _survey(args)
    if args.command == "form":
        return _form(args)
    return _score(args)


def _manifest(args: argparse.Namespace) -> int:
    from .isophonics import build_manifest

    try:
        # Naming the corpus after the directory it came from, rather than
        # defaulting to "isophonics": this builder reads any Isophonics-shaped
        # tree, and a report headed "isophonics" for something else is a lie
        # that ends up pasted into a comparison table. Inside the try because
        # resolve() can raise on a symlink loop.
        name = args.name or args.annotations.expanduser().resolve().name or "corpus"
        report = build_manifest(args.annotations, args.audio, name=name)
    except (OSError, ValueError) as error:
        print(f"songscribe-eval: {error}", file=sys.stderr)
        return 1

    write_manifest(args.output, report.manifest)
    print(f"wrote {args.output}: {report.total} annotations")

    if args.audio is None:
        print("no audio root given; manifest has references only (scoring estimates still works)")
        return 0

    print(f"  paired with audio: {report.paired}")
    groups = (
        ("no audio found", report.unpaired, None),
        ("ambiguous", report.ambiguous, AMBIGUOUS_HINT),
        ("title found but no folder names the artist or album", report.unplaced, UNPLACED_HINT),
    )
    for label, tracks, hint in groups:
        if not tracks:
            continue
        print(f"  {label}: {len(tracks)}")
        if hint:
            print(f"    ({hint})")
        shown = tracks if args.list_unpaired else tracks[:5]
        for track_id in shown:
            print(f"    {track_id}")
        if len(tracks) > len(shown):
            print(f"    ... and {len(tracks) - len(shown)} more (--list-unpaired to see all)")
    return 0


def _survey(args: argparse.Namespace) -> int:
    from .survey import survey

    vocabularies = tuple(args.vocabulary) if args.vocabulary else DEFAULT_VOCABULARIES

    try:
        dataset = load_manifest(args.manifest, annotation_root=args.annotations)
    except (ManifestError, OSError) as error:
        print(f"songscribe-eval: {error}", file=sys.stderr)
        return 1

    result = survey(dataset, vocabularies)
    if not result.tracks:
        print(
            f"songscribe-eval: no readable annotations under {dataset.annotation_root}",
            file=sys.stderr,
        )
        return 1

    sys.stdout.write(format_survey(result, label_limit=args.labels))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(survey_to_dict(result), indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json}", file=sys.stderr)
    # An unparseable label is a gap in our own vocabulary, not bad data, so it
    # is worth a non-zero exit: this is the shape of thing CI should catch.
    return 2 if result.unparseable else 0


def _form(args: argparse.Namespace) -> int:
    from .. import beats, lab
    from . import chroma, form, runner
    from .dataset import Track
    from .form_start import locate
    from .metrics import extent

    if (args.audio is None) == (args.estimate is None):
        print("songscribe-eval: pass exactly one of --audio or --estimate", file=sys.stderr)
        return 1
    if args.beat_backend is not None and args.audio is None:
        print("songscribe-eval: --beat-backend tracks beats in --audio", file=sys.stderr)
        return 1

    track = Track(
        id=args.reference.stem, reference=args.reference, audio=args.audio, bars=args.bars
    )
    try:
        reference = lab.read(args.reference)
        if args.beats is not None:
            grid = beats.grid(lab.read_beats(args.beats))
        else:
            grid = runner.bar_grid(track, None, args.beat_backend, None)
        evidence = (
            runner.chroma_of(track)
            if args.audio is not None
            else chroma.chroma_from_events(lab.read(args.estimate))
        )
        shifts = runner.key_search(args.transpose)
        found, grid = locate(
            evidence,
            grid,
            chroma.bar_templates(reference, args.bars),
            min_bars=form.CHORUS_MIN_BARS,
            shifts=shifts,
        )
        blocks = ()
        if found is not None and not found.refusal():
            start, end = extent(reference)
            blocks = form.in_key(
                form.choruses(grid, args.bars, found.bar_index, end - start), found.shift
            )
    except (OSError, ValueError, runner.AudioDecodeError) as error:
        print(f"songscribe-eval: {error}", file=sys.stderr)
        return 1

    sys.stdout.write(format_form(grid, found, blocks, from_estimate=args.estimate is not None))
    return 0 if blocks else 2


def format_form(grid, anchor, blocks, from_estimate: bool = False) -> str:
    """Format the output of ``form``: the bar grid, the anchor, and a table of choruses."""
    from . import form

    tempo = f"{grid.tempo:.1f}" if not isnan(grid.tempo) else "--"
    length = f"{grid.bar_length:.3f} s" if not isnan(grid.bar_length) else "--"
    lines = [
        "",
        f"bar grid  {grid.bars} bars, median {length} ({tempo} bpm in {grid.beats_per_bar} beats)",
        f"          {len(grid.unreliable_bars())} bars far from the median length",
    ]
    if anchor is None:
        lines += ["anchor    none: the recording is too short for the chart", ""]
        return "\n".join(lines)
    refusal = anchor.refusal()
    lines.append(
        f"anchor    bar {anchor.bar_index} at {anchor.start:.2f} s   "
        f"agreement {anchor.mean_agreement:.2f}  runner-up {anchor.runner_up:.2f}  "
        f"margin {anchor.margin:.2f}  key {_key(anchor.shift)}"
    )
    lines += _start_lines(anchor)
    if from_estimate:
        lines.append("          anchored on the estimate, not on audio: do not use it for scoring")
    if refusal:
        lines += [
            f"          NOT CLEAR: {refusal}",
            "          choruses are not listed because the start is not reliable",
            "",
        ]
        return "\n".join(lines)
    lines += [
        "",
        f"{'#':>3} {'kind':<10}{'first bar':>10}{'bars':>6}{'start':>10}{'end':>10}"
        f"{'scale':>8}{'key':>6}",
    ]
    for chorus in blocks:
        scale = "--" if isnan(chorus.scale) else f"{chorus.scale:.3f}"
        key = _key(chorus.shift) if chorus.is_scored else "--"
        lines.append(
            f"{chorus.index:>3} {chorus.kind:<10}{chorus.first_bar:>10}{chorus.bars:>6}"
            f"{chorus.start:>9.2f}s{chorus.end:>9.2f}s{scale:>8}{key:>6}"
        )
    unscored = form.unscored_seconds(blocks)
    lines += ["", f"{unscored:.1f} s not scored (intro, tail, unaligned blocks)", ""]
    return "\n".join(lines)


def _start_lines(anchor) -> list[str]:
    """How the start was found: the evidence, the intro and pickup, the ending, the check."""
    if anchor.evidence == "recurrence":
        lines = [
            f"          found by recurrence (strength {anchor.recurrence_ratio:.1f}x the best "
            "other lag)",
            f"          intro {anchor.intro_bars} bars, pickup {anchor.pickup_beats} beats, "
            f"about {anchor.ending_bars} bars after the last recurring pass",
        ]
        if anchor.phase_shift:
            lines.append(
                f"          bar lines moved {anchor.phase_shift:+d} beats from the tracker's"
            )
        if anchor.chart_agrees is not None:
            verdict = "agrees" if anchor.chart_agrees else "DISAGREES"
            lines.append(f"          chart match alone {verdict}")
        return lines
    why = (
        "nothing recurs"
        if isnan(anchor.recurrence_ratio)
        else f"recurrence {anchor.recurrence_ratio:.1f}x the best other lag is not enough"
    )
    return [f"          found by chart match ({why}); intro and pickup not measured"]


def format_survey(result, label_limit: int = TOP_LABELS) -> str:
    """Render a corpus survey as a text report."""
    lines = [
        "",
        f"dataset   {result.dataset}",
        f"tracks    {result.tracks} readable"
        + (f", {len(result.unreadable)} unreadable" if result.unreadable else ""),
        f"annotated {result.duration / 3600:.2f} h in {result.segments} segments",
        f"labels    {len(result.labels)} distinct",
        "",
    ]

    for track_id, error in result.unreadable:
        lines.append(f"  unreadable: {track_id}: {error}")
    if result.unreadable:
        lines.append("")

    if result.unparseable:
        lines.append("UNPARSEABLE LABELS -- time no vocabulary can score:")
        for stats in result.unparseable:
            lines.append(
                f"  {stats.label!r:28} {stats.segments:6d} segments  {stats.duration / 60:7.1f} min"
            )
        lines.append("")
    else:
        lines.append("every label parses")
        lines.append("")

    lines.append(f"{'vocabulary':<12}{'scoreable':>11}{'excluded':>11}   definition")
    for vocabulary, fraction in result.scoreable.items():
        excluded = (1.0 - fraction) * result.duration / 60
        lines.append(
            f"{vocabulary:<12}{_percent(fraction):>11}{excluded:>9.1f} m   "
            f"{VOCABULARIES[vocabulary]}"
        )
    lines.append("")
    lines.append("scoreable = reference time the vocabulary can express, so the most")
    lines.append("a backend could possibly score there. Excluded time is not a penalty.")
    lines.append("")

    if label_limit > 0 and result.labels:
        lines.append(f"most common labels by time (top {label_limit}):")
        for stats in result.labels[:label_limit]:
            flag = "" if stats.parses else "  <-- UNPARSEABLE"
            # A corpus can total zero duration -- an annotation of nothing but
            # zero-length segments is accepted by the parser, since published
            # sets contain them -- so the share is guarded rather than assumed.
            share = f"{stats.duration / result.duration:5.1%}" if result.duration else "    --"
            lines.append(f"  {stats.label!r:20} {stats.duration / 60:7.1f} min  {share}{flag}")
        lines.append("")

    return "\n".join(lines)


def survey_to_dict(result) -> dict:
    """The survey as JSON-ready data, every label included."""
    return {
        "dataset": result.dataset,
        "tracks": result.tracks,
        "segments": result.segments,
        "duration": result.duration,
        "unreadable": [{"id": i, "error": e} for i, e in result.unreadable],
        "scoreable": dict(result.scoreable),
        "labels": [
            {
                "label": s.label,
                "segments": s.segments,
                "duration": s.duration,
                "parses": s.parses,
            }
            for s in result.labels
        ],
    }


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

    if args.per_chorus and args.beats is None and args.beat_backend is None:
        print("songscribe-eval: --per-chorus needs --beats or --beat-backend", file=sys.stderr)
        return 1

    printer = None
    if args.per_track or args.per_chorus:
        printer = _per_track_printer(vocabularies, args.per_chorus)
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
            check_alignment=not args.no_alignment_check,
            beats_dir=args.beats,
            beat_backend=args.beat_backend,
            transpose=args.transpose,
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


def _per_track_printer(vocabularies: tuple[str, ...], per_chorus: bool = False):
    """Return a callback that prints one row per track, or per chorus if ``per_chorus``.

    Rows are printed as each track finishes, because a full corpus run is
    slow enough that no output looks like a hang.
    """
    header_written = False

    def row(label: str, scores, segmentation, bar: float | None, key: str = "") -> None:
        nonlocal header_written
        if not header_written:
            columns = "".join(f"{name:>10}" for name in vocabularies)
            print(
                f"{'track':<{_ID_WIDTH}}{columns}{'seg':>10}"
                + (f"{'bar':>10}{'key':>6}" if per_chorus else "")
            )
            header_written = True
        cells = "".join(f"{_percent(scores[v].wcsr):>10}" for v in vocabularies)
        line = f"{label:<{_ID_WIDTH}}{cells}{_percent(segmentation.score):>10}"
        if per_chorus:
            line += f"{'' if bar is None else _percent(bar):>10}{key:>6}"
        print(line)

    def write(result) -> None:
        if not (per_chorus and result.choruses):
            row(_truncate(result.track_id), result.scores, result.segmentation, None)
            return
        for part in result.choruses:
            chorus = part.chorus
            name = f"{_truncate(result.track_id)[: _ID_WIDTH - 22]} #{chorus.index} {chorus.kind}"
            if chorus.kind == "partial":
                name += f" ({chorus.bars} bars)"
            row(name, part.scores, part.segmentation, part.bar, _key(chorus.shift))

    return write


def format_report(report) -> str:
    """Render the corpus-level summary table."""
    lines = [
        "",
        f"dataset   {report.dataset}",
        f"backend   {report.backend}",
        f"input     {_input_variant(report)}",
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
        f"{'chart':<12}{_percent(report.mean_chart()):>10}"
        f"{'':>10}   chord order as printed, timing ignored",
        f"{'grid':<12}{_percent(report.mean_grid()):>10}"
        f"{'':>10}   order + harmonic rhythm in bars, tempo-free"
        + (
            f" ({report.graded_grids()}/{len(report.results)} tracks)"
            if report.graded_grids() != len(report.results)
            else ""
        ),
        f"{'segmentation':<12}{_percent(report.mean_segmentation()):>10}"
        f"{'':>10}   boundary agreement, labels ignored",
    ]
    if report.chorus_count():
        lines.append(
            f"{'bar':<12}{_percent(report.mean_bar()):>10}"
            f"{'':>10}   bars with the right chord (independent of tempo and offset)"
        )
    lines += [
        "",
        "mean = per-track average; weighted = by duration across the corpus.",
        "chart and grid are per-track and unweighted: a chart is a chart at any length.",
        "",
    ]
    if report.chorus_count():
        lines += [
            f"{report.chorus_count()} choruses in {len(report.results)} tracks, scored separately.",
            "The track rows combine the full choruses. Intro, tail and unaligned blocks are",
            f"not scored ({sum(r.unscored_seconds for r in report.results):.0f} s in total).",
            "",
        ]
    if report.graded_grids() < len(report.results):
        lines += [
            "grid is skipped where the reference's own chords do not fall on a regular",
            "harmonic grid, which no estimate of the bar length can fix. Supplying real",
            "downbeats (roadmap) is what would make it available everywhere.",
            "",
        ]
    lines += _alignment_warning(report)
    return "\n".join(lines)


def _input_variant(report) -> str:
    """What the backend heard: separated and full-mix scores are not comparable."""
    if report.separated is None:
        return "not applicable (pre-computed estimates)"
    if not report.separated:
        return "full mix"
    if not report.fallbacks:
        return "separated stems"
    return (
        f"separated stems ({report.fallbacks} of {len(report.results)} fell back to the full mix)"
    )


def _alignment_warning(report) -> list[str]:
    """Name any track that scores markedly better at a time shift.

    Printed after the scores rather than mixed into them, because the scores
    stand as measured: a shift is never folded in. The point is to stop a
    misaligned recording being read as a model result, which is a mistake the
    numbers alone look perfectly reasonable for.
    """
    suspicious = [
        result
        for result in report.results
        if result.alignment is not None
        and result.alignment.is_suspicious(ALIGNMENT_WARN_GAIN, ALIGNMENT_WARN_OFFSET)
    ]
    if not suspicious:
        return []

    lines = [
        f"WARNING: {len(suspicious)} of {len(report.results)} tracks score markedly better",
        "at a time shift, which means the recording is not on the reference's timeline.",
        "The scores above are as measured and are too low for these tracks:",
        "",
        f"  {'track':<{_ID_WIDTH - 2}}{'shift':>8}{'measured':>10}{'shifted':>9}",
    ]
    for result in sorted(suspicious, key=lambda r: -r.alignment.gain):
        alignment = result.alignment
        lines.append(
            f"  {_truncate(result.track_id):<{_ID_WIDTH - 2}}{alignment.offset:+7.2f}s"
            f"{_percent(alignment.measured):>10}{_percent(alignment.shifted):>9}"
        )
    lines += [
        "",
        "Fix the audio -- trim the lead-in, or get the pressing the annotation was",
        "made from. Do not treat the shifted column as the result: choosing the",
        "offset that scores best is fitting to the reference.",
        "",
    ]
    return lines


def report_to_dict(report) -> dict:
    """The full results, including per-track rows, as JSON-ready data."""
    return {
        "dataset": report.dataset,
        "backend": report.backend,
        # null for pre-computed estimates, whose input is not known here.
        "separated": report.separated,
        "separation_fallbacks": report.fallbacks,
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
        "chart": {"mean": _jsonable(report.mean_chart())},
        "grid": {
            "mean": _jsonable(report.mean_grid()),
            "graded_tracks": report.graded_grids(),
        },
        "segmentation": {"mean": _jsonable(report.mean_segmentation())},
        "bar": {"mean": _jsonable(report.mean_bar()), "choruses": report.chorus_count()},
        "tracks": [
            {
                "id": result.track_id,
                "duration": result.duration,
                "reference_chords": result.reference_chords,
                "estimate_chords": result.estimate_chords,
                "chart": {
                    "similarity": _jsonable(result.chart),
                    "reference_chords": result.reference_chart_chords,
                    "estimate_chords": result.estimate_chart_chords,
                },
                "grid": {
                    "similarity": _jsonable(result.grid),
                    "unit": None if result.harmonic is None else _jsonable(result.harmonic.unit),
                    "residual": None
                    if result.harmonic is None
                    else _jsonable(result.harmonic.residual),
                    "reliable": result.harmonic is not None and result.harmonic.is_reliable,
                },
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
                "alignment": None
                if result.alignment is None
                else {
                    "best_offset": result.alignment.offset,
                    "shifted": _jsonable(result.alignment.shifted),
                    "measured": _jsonable(result.alignment.measured),
                    "gain": _jsonable(result.alignment.gain),
                    "suspicious": result.alignment.is_suspicious(
                        ALIGNMENT_WARN_GAIN, ALIGNMENT_WARN_OFFSET
                    ),
                },
                "bar": _jsonable(result.bar),
                "anchor": None if result.anchor is None else _anchor_to_dict(result.anchor),
                "unscored_seconds": result.unscored_seconds,
                "choruses": [_chorus_to_dict(part) for part in result.choruses],
            }
            for result in report.results
        ],
    }


def _anchor_to_dict(anchor) -> dict:
    return {
        "bar_index": anchor.bar_index,
        "start": anchor.start,
        "agreement": _jsonable(anchor.mean_agreement),
        "runner_up": _jsonable(anchor.runner_up),
        "margin": _jsonable(anchor.margin),
        "shift": anchor.shift,
        "keys_searched": anchor.keys,
        "evidence": anchor.evidence,
        "intro_bars": anchor.intro_bars,
        "pickup_beats": anchor.pickup_beats,
        "phase_shift": anchor.phase_shift,
        "ending_bars": anchor.ending_bars,
        "recurrence_ratio": _jsonable(anchor.recurrence_ratio),
        "chart_agrees": anchor.chart_agrees,
        "key_margin": _jsonable(anchor.key_margin),
    }


def _chorus_to_dict(part) -> dict:
    chorus = part.chorus
    return {
        "index": chorus.index,
        "kind": chorus.kind,
        "start": chorus.start,
        "end": chorus.end,
        "first_bar": chorus.first_bar,
        "bars": chorus.bars,
        "scale": _jsonable(chorus.scale),
        "shift": chorus.shift,
        "bar": _jsonable(part.bar),
        "chart": _jsonable(part.chart),
        "grid": _jsonable(part.grid),
        "segmentation": _jsonable(part.segmentation.score),
        "scores": {name: _jsonable(score.wcsr) for name, score in part.scores.items()},
    }


def _jsonable(value: float) -> float | None:
    """JSON has no NaN; an unscoreable track is a null, not a zero."""
    return None if isnan(value) else value


def _key(shift: int) -> str:
    """A key relative to the chart's: ``+0`` as written, ``+2`` a tone up, ``-3`` down."""
    return f"{shift:+d}"


def _percent(value: float) -> str:
    """Render a score as a percentage, or ``--`` if it could not be computed.

    Two decimals, which is what MIREX tables carry -- and enough that float
    summation noise (~1e-15, from an estimate read back off disk at 6dp
    instead of held in memory) cannot flip a rendered digit and make two
    identical runs look like they disagree.
    """
    return "--" if isnan(value) else f"{100 * value:.2f}%"


def _truncate(text: str) -> str:
    """Keep the tail of a long track id: the title is more use than the album."""
    return text if len(text) <= _ID_WIDTH - 1 else "..." + text[-(_ID_WIDTH - 4) :]


if __name__ == "__main__":
    raise SystemExit(main())
