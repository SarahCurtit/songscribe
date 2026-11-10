# Evaluating chord recognition

`songscribe-eval` scores a chord-recognition backend against reference annotations, following
MIREX conventions. MIREX, the Music Information Retrieval Evaluation eXchange, is the yearly
community benchmark whose chord-estimation task fixed the standard metric (the fraction of
time labelled correctly, below) and the `mirex` comparison, two chords sharing three pitch
classes. It reads annotations and your own recordings from local paths. The repository
contains the code and manifests (track ids and relative paths), no audio and no annotation
text.

## Install

```bash
pip install -e '.[dev]'    # all backends, pytest, ruff
```

The `template` backend needs no model weights. `madmom` needs the `chords` extra and `btc` the `btc` extra.

## Metrics

| Metric | Asks | Robust to |
|---|---|---|
| WCSR | is each second labelled right | nothing; this is the metric in published results |
| `chart` | is the printed chord order right | offset and tempo |
| `grid` | is the order right, and the harmonic rhythm in bars | offset and tempo |
| `segmentation` | are the chord boundaries in the right places | nothing; labels are ignored |
| `bar` | does each bar carry the right chord | offset, tempo and length |
| `theory` | how close is a wrong chord, musically | nothing; it grades the same seconds WCSR does |
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
| `substitutions` | the two chords stand in a substitution a player would accept |

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

### theory: how close a wrong chord is

Every vocabulary above compares two chords for equality after reducing them to a class. That
asks whether the symbol is right, not how close a wrong one is, and on a jazz chart the two come
apart because substitution is correct practice:

- a tritone substitution (`G:7` for `Db:7`) and a ii written for its V (`D:min7` for `G:7`) are
  misses under `majmin`, `sevenths` **and** `mirex` — nothing scores them as near;
- where the annotation names a rootless dominant such as `B:hdim7`, `majmin` and `sevenths`
  cannot express it and that time leaves their denominator, while `mirex` counts it correct
  against `G:7` — the two share B, D and F;
- the four vocabularies disagree so sharply on the rest that "how close was it" takes reading
  all four.

`theory` names the relation between the two chords instead and gives it partial credit, as one
number over the same seconds WCSR uses:

| Relation | Credit | Example |
|---|---|---|
| `same` | 1.00 | `C:maj` / `C:maj`, and `C:dim7` / `Eb:dim7` — one symmetric chord, two names |
| `extension` | 0.95 | `C:7` / `C:13`, and a triad model's `C:7` / `C:maj` |
| `sus_for_dominant` | 0.90 | `C:sus4(b7)` / `C:7` |
| `guide_tones` | 0.90 | `B:hdim7` / `Bb:maj7` — same third and seventh, same roles |
| `tritone_sub` | 0.80 | `G:7` / `Db:7` |
| `dim_for_dominant` | 0.80 | `B:hdim7` / `G:7` — the dominant without its root |
| `ii_V_merge` | 0.70 | `D:min7` / `G:7` |
| `relative` | 0.70 | `C:maj6` / `A:min7` |
| `sus_for_triad` | 0.70 | `C:sus4` / `C:maj` |
| `seventh_swap` | 0.60 | `C:maj7` / `C:7` |
| `parallel` | 0.40 | `C:maj` / `C:min` |
| `none` | 0.00 | anything else |

It is reported two ways, and **neither changes WCSR**. Both grade the same seconds with the
same table; they differ in what a grade is worth:

| | `substitutions` | `theory` |
|---|---|---|
| Scores a segment | 1 if credit ≥ `metrics.SUBSTITUTION_MIN_CREDIT` (0.75), else 0 | its credit, 0 to 1 |
| Answers | would a player accept it? | how close was it? |
| `relative`, `ii_V_merge`, `sus_for_triad`, `seventh_swap`, `parallel` | 0 | 0.4 to 0.7 |
| `extension` (0.95) | 1 | 0.95 |

So `theory` is usually higher: those five near misses count for nothing in `substitutions`
and for part of their credit in `theory`. It can be lower on a run dominated by `extension`.
A large gap means the errors are mostly near misses; a small one means they are mostly `none`.

`score` also prints the share of compared time each relation accounted for, and `--json`
records it per track. That breakdown is the thing to read first: **the credits are musical
judgement, not measured frequencies.** Nothing here has yet measured how often two independent
lead sheets of one tune disagree in each of these ways, so a run whose credit leans on
`relative` or `ii_V_merge` is evidence those two are too generous rather than evidence the model
is doing well. Read `theory` as indicative, the way `grid` is.

`substitutions` and `mirex` need the reference and the estimate together to decide, so a single
chord has no label in them (`metrics.project` raises for both).
`chart`, `grid` and `bar` are built on projected label strings and leave both out; asking for
`-v substitutions` adds a WCSR column and no bar row.

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

`theory` and `substitutions` exclude exactly what WCSR does — `X` and unparseable reference
labels — and nothing else: every relation is defined on pitch content, so no chord is outside
their vocabulary. A track whose reference is entirely `X` has no `theory` score and is absent
from its mean, not counted as zero.

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

One archive holds several artists, and `--include` takes one of them as its own corpus. That
is how the committed per-artist manifests were built, and how to rebuild one:

```bash
songscribe-eval manifest data/isophonics --include chordlab/Zweieck \
  --name isophonics-zweieck --output eval/isophonics-zweieck.json
```

`--include` keeps the annotations at or under a path relative to the annotation directory, and
is repeatable. It matches whole directory names.

Keeping the artists apart matters because Isophonics rates its own chord annotations per
artist: The Beatles "use with confidence", Zweieck "moderate confidence", Carole King "have
not been carefully checked, use with care". One average over all three would hide that, and a
poor Carole King score is as likely to be the reference as the model. The chord references
cover three of the four artists — Michael Jackson's set is structural segmentation only, with
no chords at all, so it never reaches a manifest.

| Manifest | Tracks | Annotated | Confidence upstream |
|---|---|---|---|
| `eval/isophonics-beatles.json` | 180 | 8.15 h | use with confidence |
| `eval/isophonics-zweieck.json` | 18 | 0.98 h | moderate confidence |
| `eval/isophonics-carole-king.json` | 7 | 0.44 h | not carefully checked |

Carole King is Tapestry's first seven tracks (01–07).

`eval/backing-track.json` lists every song of a jazz backing-track collection, one track per
song, with tempo and key variants as separate tracks. No chord references exist for them yet,
so each `reference` is a placeholder: the path, `labs/<id>.lab`, where that song's annotation
will go. Write a `.lab` at that path and the track counts, with no change to the manifest.
The loader accepts a missing reference and `survey` lists it as unreadable; until the first
`.lab` is written, `survey` has nothing to read and stops with "no readable annotations".

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

## JAAH

The [Jazz Audio-Aligned Harmony](https://github.com/MTG/JAAH) dataset annotates 113 jazz
recordings (1917–1989, 6.9 hours) with chords that include sevenths, so it is the corpus for
telling a `sevenths` score apart from a `majmin` one. Like Isophonics it ships no audio.

**1. Annotations.** Extract the repository's `labs.zip` into `data/jaah/` (gitignored) so that
`labs/` sits directly under it. The repository has no licence file; keep the annotations
local, as with Isophonics.

**2. Recordings.** Each annotation is timed against one specific recording, which the
repository's JSON identifies by MusicBrainz id and `duration`. A different release or master
can start at another offset or run at another speed; see
[Checking the alignment](#checking-the-alignment).

**3. Survey.** The committed `eval/jaah.json` is references-only:

```bash
export SONGSCRIBE_EVAL_ANNOTATIONS="$PWD/data/jaah"   # the directory containing labs/
songscribe-eval survey eval/jaah.json
```

It reports that 93.5% of the annotated time is expressible in `majmin` and 93.3% in `sevenths`;
the rest is chords neither vocabulary has a class for. The most common labels are dominant
sevenths (`Bb:7`, `F:7`, `Eb:7`).

**4. Score.** Pair your recordings by naming each after its lab (`giant_steps.flac`) or by
mirroring the `labs/` tree, then build a manifest and score as for Isophonics:

```bash
songscribe-eval manifest data/jaah --audio ~/music/jaah --name jaah --output eval/jaah-audio.json
songscribe-eval score eval/jaah-audio.json -c btc -v majmin -v sevenths --cache .eval-cache
```

`-c madmom` on the same manifest is the comparison: its `sevenths` score is capped by its
triad vocabulary.

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

## Train and test sets

`eval/jaah.json`, `eval/isophonics-beatles.json`, `eval/isophonics-zweieck.json` and
`eval/backing-track.json` each have a train and a test manifest next to them, about 20% of the
songs held out. Train includes validation: tune on it, and score on test once you are done
tuning. The first three are hash splits (below); the backing tracks are
[split by band](#backing-tracks-are-split-by-band).

```bash
songscribe-eval split eval/jaah.json     # writes eval/jaah-train.json and eval/jaah-test.json
```

Recordings of one song always go to the same set. A trailing parenthesis is ignored, so
`autumn_leaves(e_concert)` and `autumn_leaves(g_concert)` stay together, and a leading track
number is ignored in either spelling Isophonics ships (`05_-_Dig_It`, `05 Beautiful`), so a
song is not tied to its position on the album. The set comes from a hash of the song name, so
a manifest built locally splits like the committed one. If file names hide that two recordings
are one song, add them to `SAME_SONG` in `src/songscribe/eval/split.py` and run `split` again,
before tuning.

**Carole King is held out whole**, as `eval/isophonics-carole-king-test.json` with no train
manifest: score it, never tune on it. Seven songs cannot carry a hash split — the lowest of
them hashes at 0.39, so no sensible fraction holds one out. That it is also the set Isophonics
has not checked carefully is a second reason to keep it off the tuning side rather than
average it in. That test manifest is **maintained by hand**: it is every track of the corpus,
and `split` does not produce it, so edit it alongside `eval/isophonics-carole-king.json` when
the corpus changes.

A hash split is approximate by design, and on a small corpus it can land every song on one
side. `split` refuses that rather than writing a 0-track manifest, and says where the nearest
song hashes:

```
songscribe-eval: test would be empty -- 7 songs is too few for a 0.2 hash split;
  the lowest song hashes at 0.39. Nothing written; score the corpus whole
  rather than holding out none of it.
```

Neither manifest is written and the exit is 1, so a corpus held out whole keeps its test
manifest. Raising `--test-fraction` until something lands in test is choosing a test set by
the number it gives, which is the thing a held-out set exists to prevent.

### Backing tracks are split by band

Every backing track is played by one trio (piano, bass, drums), and the same few trios play
many songs. A hash split by song puts the same trio on both sides, so a model tuned on train has
already heard the test set's players, and a test score overstates how it does on a new band.
`eval/backing-track-train.json` and `eval/backing-track-test.json` are split by band instead:

- **no trio plays on both sides**, and every version of a song stays on one side (`song_key`);
- **some musicians play only in test**: 13 of 33, against 14 only in train; the 6 on both
  sides never play in the same trio on both;
- **test's mix of styles** (swing, ballad, latin, …) is within 0.10 of the collection's, as total
  variation distance. Without that limit the most separable bands are the style specialists, and
  test came out 31% latin against 5% in train, which would mix "new players" with "new styles";
- test holds 40 of the 169 tracks (24%), 8 of the 21 bands.

The line-ups are not public, so the split is made in the private data_extraction repository
(`labels/band_split.py`, which tries every grouping of bands and keeps the best) and only its
result, track ids, is committed here. The manifests carry `"split": "by band: …"`, and
`songscribe-eval split` refuses a manifest that declares its split rather than replacing it with a
hash split. When the collection changes, re-run `band_split.py` there, not `split` here.

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
