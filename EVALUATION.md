# Evaluating chord recognition

`songscribe-eval` scores a chord-recognition backend against reference annotations, following
MIREX conventions. It reads annotations and your own recordings from local paths. The
repository contains the code and manifests (track ids and relative paths), no audio and no
annotation text.

## Install

```bash
pip install -e '.[dev]'    # all backends, pytest, ruff
```

The `template` backend needs no model weights. `madmom` and `btc` need the `chords` extra.

## Metrics

| Metric | Asks | Robust to |
|---|---|---|
| WCSR | is each second labelled right | nothing; this is the metric in published results |
| `chart` | is the printed chord order right | offset and tempo |
| `grid` | is the order right, and the harmonic rhythm in bars | offset and tempo |
| `segmentation` | are the chord boundaries in the right places | nothing; labels are ignored |
| `bar` | does each bar carry the right chord | offset, tempo and length |
| `placement` | does each chord sit over the right word | offset and tempo |

`bar` applies to recordings of a repeated chart (see
[Scoring a chart played several times](#scoring-a-chart-played-several-times)) and
`placement` to songs (see [Lyric-chord placement](#lyric-chord-placement)).

### WCSR and vocabularies

Weighted Chord Symbol Recall is the fraction of annotated time labelled correctly. It is
computed under a comparison vocabulary:

| Vocabulary | Counts a segment as correct when |
|---|---|
| `root` | the root matches; quality is ignored |
| `majmin` | the major or minor triad matches; sevenths reduce to their triad |
| `triads` | as `majmin`, plus dim, aug, sus2, sus4 |
| `sevenths` | the seventh also matches: maj, min, 7, maj7, min7 |
| `mirex` | the two chords share at least 3 pitch classes |

The default is `root majmin sevenths mirex`; choose others with `-v`, repeatable. The gap
between vocabularies shows where errors come from. A backend that only emits triads scores
well on `majmin` and near zero on `sevenths` against seventh-chord references, while `mirex`
stays high because a triad shares three pitch classes with its seventh.

Two aggregates are printed:

- **mean**: the unweighted mean of per-track WCSR; each song counts once.
- **weighted**: total correct duration over total compared duration; long songs count more.

### chart, grid, segmentation

WCSR is sensitive to timing: a transcription that is uniformly late loses many points even
though the progression is correct. It also does not penalise a backend that changes chord far
too often, because most seconds still carry a plausible label. These three metrics cover
those cases.

- **`chart`** projects both sides to `majmin`, drops `N`, merges repeats and flicker shorter
  than `align.MIN_CHORD_DURATION` (the same collapse `align.dedupe` applies when printing), and
  compares the chord sequences by normalised edit distance. A substitution is a wrong chord,
  an insertion a chord that was invented, a deletion a missed one.
- **`grid`** compares `(label, bars)` pairs. The harmonic unit (one bar) is estimated from the
  reference chord durations alone and applied to both sides, so it tells one chord per bar from
  two. It reports `nan` for a track whose chords do not fall on a recoverable grid, and `score`
  prints how many tracks were gradeable. Treat it as indicative. `grid_sequence(..., unit=)`
  takes a known bar length and skips the estimate.
- **`segmentation`** measures boundary placement and ignores labels. High `segmentation` with
  low WCSR means the boundaries are right and the labels are wrong; both low together points
  to a timing problem in the recording (see [Checking the alignment](#checking-the-alignment)).

### What gets excluded

In WCSR, reference time that the vocabulary cannot express is removed from both numerator and
denominator, and `survey` reports it as excluded. This covers `X` (a chord the annotator could
not name), labels that fail to parse, and qualities outside the vocabulary (`sus4` is excluded
from `majmin` and scored under `triads`). `mirex` also excludes reference chords with fewer
than 3 pitch classes, such as power chords.

`chart`, `grid` and `placement` keep a reference `X` as a token that no backend emits, because
removing it would merge its neighbours. It lowers those scores on that track.

An estimate that is unparseable or outside the vocabulary counts as wrong. A track whose
reference or estimate is missing is skipped and named, not scored as zero.

### Differences from `mir_eval`

The metrics are reimplemented without a `mir_eval` dependency. Results are close to its
values but not identical, so do not mix the two in one table.

- A reference chord is reduced to its triad by reading the third and fifth from its interval
  set, so `C:maj(9)` becomes `C:maj` and is scored. `mir_eval` excludes it. WCSR is therefore
  slightly lower here.
- Inversions are ignored in every vocabulary.
- Intervals compare modulo the octave: a 9th is a 2nd, an 11th a 4th.

## Quick start with the demo track

`songscribe-demo` writes a synthetic instrumental track with known chords, a ground-truth
`.lab` and a `.beats` file:

```bash
songscribe-demo -o demo.wav
```

Run `songscribe-demo --help` for the options (progression, tempo, repeats, intro, key, noise).

## Isophonics

[Isophonics](http://isophonics.net/content/reference-annotations) provides chord
annotations for The Beatles, Queen, Carole King and Zweieck, without audio. It is the
corpus the harness is wired up for.

**1. Annotations.** Download the reference annotations and extract them into
`data/isophonics/` (gitignored) so that `chordlab/` sits directly under it. Only
`chordlab/` is read; `keylab/`, `seglab/`, `beat/` and `all/` are skipped.

**2. Recordings.** Supply your own copies. The Beatles annotations are timed against the
original 1987 CD issues (catalogue numbers are listed on the Isophonics page), not the
remasters or mono editions. A different master can start at another offset or run at another
speed, which shifts every boundary. See [Checking the alignment](#checking-the-alignment).

**3. Set the roots.** From the repository root:

```bash
export SONGSCRIBE_EVAL_ROOT=~/music/beatles                 # your recordings
export SONGSCRIBE_EVAL_ANNOTATIONS="$PWD/data/isophonics"   # the directory containing chordlab/
```

The annotation root is the directory containing `chordlab/`, not `chordlab/` itself.
`--audio` and `--annotations` override the variables per command.

**4. Build a manifest.**

```bash
songscribe-eval manifest data/isophonics --audio ~/music/beatles --output eval/isophonics.json
```

Each reference `.lab` is paired with a recording by mirrored relative path, then by a folder
named after the album, then by a title match that ignores punctuation, track numbers, years and
words like "Remastered" or "FLAC". A title match is accepted only when the file is in the audio
root or in a folder named for the artist or album. Tracks that cannot be paired uniquely are
reported, not guessed: no file with that title, a title whose folders name neither artist nor
album, or a title found under several albums. `--list-unpaired` lists every one.

The manifest holds paths that follow your library, so keep it uncommitted. The committed
`eval/isophonics-beatles.json` is references-only (no `audio` paths); it works for `survey` and
`score --estimates`.

**5. Survey.** Needs no audio or model:

```bash
songscribe-eval survey eval/isophonics-beatles.json
```

It reports annotated time, distinct labels, whether every label parses, and for each
vocabulary how much reference time is scoreable. That figure is the upper bound for any
backend, and the denominator to quote beside a WCSR. `survey` exits non-zero if a label does
not parse, which indicates a gap in `eval/harte.py`.

**6. Score.**

```bash
songscribe-eval score eval/isophonics.json -c madmom --cache .eval-cache --per-track
```

A full pass is long: `madmom` runs at about 2x realtime on one CPU, so 8 hours of audio takes
about 4 hours; `template` is about 180x. `--cache` stores each estimate as a `.lab`, so an
interrupted run resumes and re-scoring is instant. Try `--limit 5` first.

```bash
# Re-score cached estimates under other vocabularies (no model).
songscribe-eval score eval/isophonics.json -c madmom --cache .eval-cache -v triads -v sevenths

# Score estimates produced by another tool.
songscribe-eval score eval/isophonics.json --estimates ~/estimates/btc

# Smoke test on a few tracks.
songscribe-eval score eval/isophonics.json -c template --limit 5
```

`--separate` runs demucs first. It is much slower and improves accuracy most; results with and
without it are not comparable, so report which was used. `--json FILE` also writes the full
results.

### Checking the alignment

Scores are meaningful only if the recording is on the reference's timeline. After scoring,
`score` shifts each estimate over a range of offsets and warns when a track scores at least 5
points higher at a shift of 0.15 s or more:

```
WARNING: 1 of 1 tracks score markedly better
at a time shift, which means the recording is not on the reference's timeline.
```

The warning means the recording needs fixing, not that the shifted score is the result;
picking the best offset fits the estimate to the reference. A negative shift means the
recording starts late: trim it.

```bash
ffmpeg -ss 1.70 -i in.m4a -ac 1 -ar 44100 out.wav
```

A positive shift means it starts early: pad silence instead (here in milliseconds).

```bash
ffmpeg -i in.m4a -af "adelay=1700|1700" -ac 1 -ar 44100 out.wav
```

Then replace the original in the audio library with `out.wav` (same folder and stem),
rebuild the manifest with `manifest --audio`, and re-run with a fresh `--cache`, since cached
estimates are keyed by backend, input variant and track, not by the audio. The sweep takes a
few seconds per hundred tracks; `--no-alignment-check` skips it.

## Scoring a chart played several times

A recording with an intro, a chart played several times and an ending does not line up with a
one-chorus reference. To score one, add `"bars"` (the chart's bar count) to the track in the
manifest and give the bar lines with `--beats DIR` or `--beat-backend`:

```json
{"version": 1, "dataset": "synthetic-take", "tracks": [
  {"id": "take", "reference": "take.reference.lab", "audio": "take.wav", "bars": 8}
]}
```

Saved as `take.json`, this is a complete manifest. A synthetic example, with the demo's ground-truth chords standing in for the anchoring
estimate:

```bash
songscribe-demo -o take.wav -p C Am Dm G Em A F Bb -r 3 --intro C G
songscribe-eval form take.reference.lab --bars 8 --estimate take.lab --beats take.beats
songscribe-eval score take.json -c template --anchor-estimates . --beats . --per-chorus
```

`form` prints the detected start and the choruses without scoring, and exits 2 if the start
is not clear. `score --beats DIR` reads one `<track>.beats` file per track; `songscribe-demo`
writes `take.beats` beside the audio. On a real recording, `--audio FILE --beat-backend madmom`
replaces `--estimate` and `--beats`.

- The start is found on a chord estimate of the recording: `--anchor-backend` (default
  `madmom`), or `.lab` files via `--anchor-estimates` (`--estimate` for `form`). Use a different
  backend from the one being scored; `score` warns when they match.
- `--transpose auto` (default) finds the key the take is played in and scores against the
  chart transposed to it; `--transpose none` keeps the chart as written. One key per take.
- Blocks are labelled `intro`, `head`, `chorus`, `partial`, `tail` or `unaligned`. Seconds not
  scored are reported.
- Each chorus is mapped onto the reference timeline, so WCSR, `chart`, `grid` and
  `segmentation` apply per chorus. `--per-chorus` prints a row for each.
- `bar` is the fraction of bars with the right chord, each judged by the chord that sounds
  longest in it. Use it as the main score for a recording. It is reported for every vocabulary
  with a class per chord (`root`, `majmin`, `triads`, `sevenths`; not `mirex`): `bar` is the
  `majmin` one, `--per-chorus` adds a `bar-<vocabulary>` column for each, and `--json` has
  `bar_by_vocabulary`.
- The track row combines all full choruses; a partial last chorus is reported but excluded.

How the start is found is described in [`src/songscribe/eval/README.md`](src/songscribe/eval/README.md).

## Lyric-chord placement

For a song, `metrics.placement_score` measures whether each chord sits over the right word.
It builds lines with `align.group_words`, places chords with `align.place_in_line`, and
compares `(line index, word index, label)` sequences. It works in word indices, so it is
independent of offset and tempo and needs no lyric text.

```python
placement_score(reference_chords, estimate_chords, words)
placement_score(reference_chords, estimate_chords, words, compare_labels=False)
```

The first call checks chord and word; with `compare_labels=False` only the word, which
separates a mistimed chord from a misnamed one. Both chord lists are placed against the same
`words` timeline. The timeline need only be consistent between the two sides: the output of
your own ASR run on the recording works.

`songscribe.eval.reference` stores a timeline as JSON and accepts only names ending in
`.words.json`, which `.gitignore` excludes:

```python
from songscribe import lyrics
from songscribe.eval import reference, metrics

words = lyrics.transcribe("my_track.wav", model_size="base", language="en")
reference.write_words("data/placement/my_track.words.json", words, source="whisper-base")

metrics.placement_score(reference_chords, estimate_chords, words)
```

`placement` is not part of `songscribe-eval`; it needs a recording with both time-aligned
words and chords. A self-recorded track is the simplest source.
