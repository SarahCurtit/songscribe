# Evaluating the chord branch

How to put a number on "did that change help". Chord recognition is the one branch with a
real benchmark tradition behind it, so this follows MIREX convention rather than inventing a
metric.

The rule that shapes everything here: **the evaluation code is public, the corpus is not.**
Manifests hold track identifiers and relative paths — never audio, never annotation text —
so the harness, the metrics and the manifests can all be committed while the recordings stay
on the machine that is licensed to hold them. Someone else with their own copy of the same
corpus points two environment variables at it and reproduces the numbers.

```
src/songscribe/eval/
  harte.py       chord labels -> pitch content            (pure Python)
  metrics.py     WCSR, vocabularies, segmentation         (pure Python)
  dataset.py     manifests and path resolution            (pure Python)
  isophonics.py  manifest builder for an Isophonics tree
  runner.py      runs a backend, caches estimates, aggregates
  cli.py         songscribe-eval
```

Only `runner.py` touches audio or models. That is what lets the scoring rules be tested
exhaustively in CI with no weights and no recordings on disk — the same reasoning that keeps
`align.py` and `render.py` dependency-free.

## Quick start: Isophonics

[Isophonics](http://isophonics.net/content/reference-annotations) is the corpus wired up
first, for three reasons: its chord annotations are the de facto benchmark for this task,
they are a free download, and they ship **without audio** — you supply your own copy of the
recordings. That last point is the whole reason it fits this project.

**1. Get the annotations.** Download "Reference Annotations: The Beatles" (and Queen, Carole
King, Zweieck if you want them) and extract into `data/isophonics/` — `data/` is gitignored
wholesale — so that `chordlab/` sits directly under it. The archive carries several formats
side by side:

```
chordlab/   .lab chord annotations  <- the only thing this harness reads
keylab/     key annotations
seglab/     structural segmentation
beat/       beat annotations
all/        the same data as Turtle RDF (.ttl)
```

Pass the directory *containing* `chordlab/` (here `data/isophonics`, not
`data/isophonics/chordlab`) to `songscribe-eval manifest` and the builder
finds it: `keylab/` and `seglab/` are `.lab` files too, but they annotate a different
thing, so they're skipped by name rather than by glob.

**2. Supply the audio.** Isophonics distributes **no audio, from any page** — the download
is annotations only, by design, and that is what makes it usable here. The recordings are
your own copies. See [What you may and may not use](#what-you-may-and-may-not-use) below
before you source them. Everything up to and including `survey` works without audio; only
`score -c <backend>` needs it.

**Match the pressing.** The annotations are timed against specific CD issues, which
`reference-annotations.html` lists by catalogue number — for the Beatles, the original
1987 CD issues (`CDP 7 46435 2` *Please Please Me* through `CDP 7 46447 2` *Let It Be*),
not the 2009 remasters and not the mono editions. A different master can run at a different
length or start at a different offset, which shifts every boundary in the file.

How much does that cost? Measured by scoring one Beatles reference against a shifted copy
of itself — a pure timing error, no model involved:

| Offset | `majmin` | `segmentation` |
|---|---|---|
| 0.05 s | 98.4% | 98.2% |
| 0.10 s | 96.8% | 96.5% |
| 0.25 s | 91.9% | 91.2% |
| 0.50 s | 83.8% | 82.4% |
| 1.00 s | 67.6% | 74.2% |

So the budget is tight: **a tenth of a second costs ~3 points, half a second costs ~16.**
A constant speed difference is worse than an offset, because it accumulates — a master
running 1% fast drifts 1.8 s over a three-minute track and scores 71%. CD-to-CD remaster
differences are usually well inside the top row; a vinyl or tape transfer is not, and
should be assumed unusable.

The diagnostic is the *relationship* between the two columns, not either alone. A timing
error moves `segmentation` **in step with** `majmin`, because the boundaries themselves are
wrong. A genuine model weakness puts wrong labels on correct boundaries, so `segmentation`
stays high while `majmin` falls. If a whole album drops with both columns together, suspect
the rip before the backend.

**`score` now checks this for you.** After scoring, it sweeps a time offset over each
estimate — pure timeline arithmetic, no model time — and warns when a track scores markedly
better shifted:

```
WARNING: 1 of 1 tracks score markedly better
at a time shift, which means the recording is not on the reference's timeline.
The scores above are as measured and are too low for these tracks:

  track                                                shift  measured  shifted
  ...lab/The Beatles/07_-_Revolver/02_-_Eleanor_Rigby  -1.70s    50.09%   79.51%
```

That is a real example: an m4a with ~1.7 s of encoder lead-in scored 50.09%, and 79.51%
once trimmed — the harness had been measuring the lead-in. Both backends peaked at the same
offset, which is the tell that it is the recording and not the model, since two unrelated
backends do not share a bias.

**The shift is never folded into a score, and you shouldn't either.** Choosing the offset
that scores best is fitting to the reference: do that and a misaligned corpus quietly
becomes a flattering one. Treat the number as an instruction to fix the audio. The shift is
what the estimate had to move to fit, so a negative one means the recording starts late:
trim its magnitude with `ffmpeg -ss 1.70 -i in.m4a -ac 1 -ar 44100 out.wav` (never
`-ss -1.70`, which trims nothing). A positive one means it starts early, which `-ss` cannot
fix; pad that much silence instead, delaying every channel before `-ac 1` downmixes them:
`ffmpeg -i in.m4a -af "adelay=<ms>|<ms>" -ac 1 -ar 44100 out.wav` (a mono source ignores
the second delay, and this form, unlike `all=1`, works on FFmpeg 4.2). The manifest still
points at `in.m4a`, so move that out of the audio library, put `out.wav` in its folder
with the same stem, and rebuild the manifest with `manifest --audio`. Keeping both loses:
a copy in another folder fails the album match, and one beside a FLAC original ranks
below it. Then re-run with a fresh `--cache`, since cached estimates are keyed by backend,
input variant and track, not by the audio they came from.

The sweep costs about 10 s across 180 tracks, nothing against 4 hours of inference but
noticeable on a cached re-score; `--no-alignment-check` skips it.

The thresholds (a 5-point gain at ≥0.15 s) are set so a model's ordinary boundary slack
doesn't trip it: on the trimmed file above, the same check stays silent.

**Confidence varies by annotation and artist**, and Isophonics says so. The Beatles *chord*
annotations are the top tier — "checked several times by Christopher Harte and the MIR
community, use with confidence" — which is exactly the branch this harness measures. Queen
and Zweieck chords are "moderate confidence"; Carole King chords are "not carefully
checked"; keys generally are "use with care". So quote Beatles chord numbers plainly, and
hedge anything derived from the other three.

**3. Point the harness at both.** Once, in your shell:

```bash
# run from the repo root; $PWD makes the path absolute, so it works from any directory
export SONGSCRIBE_EVAL_ROOT=~/music/beatles                 # your recordings
export SONGSCRIBE_EVAL_ANNOTATIONS="$PWD/data/isophonics"   # the directory containing chordlab/
```

The annotation root must be the directory *containing* `chordlab/`, because manifest
`reference` paths — the committed one's included — start at `chordlab/`. Point it at
`chordlab/` itself and `survey` reports no readable annotations.

**4. Build a manifest.**

```bash
songscribe-eval manifest data/isophonics \
    --audio ~/music/beatles \
    --output eval/isophonics.json
```

It walks the tree for chord `.lab` files and pairs each with a recording, trying the
mirrored relative path, then a folder named exactly after the album, then a punctuation- and
track-number-insensitive title match. That last step takes a lone match only when it sits
directly in the audio root or under a folder that names the annotation's artist or album, so
a same-titled song in someone else's folders is not picked up when your copy is missing. The
`--audio` directory's own name counts (`~/music/beatles` names the artist), and years, disc
and track numbers, apostrophes, a leading "The" and words like *Remastered*, *Deluxe
Edition*, *Vinyl Rip* or *FLAC* are ignored, so `1965 - Help! (Remastered)/`,
`The Beatles - Help! (2009) [FLAC]/` and `A Hard Days Night/` qualify. Numbers in the
annotation's own album name (after its track or disc prefix) are not ignored: the folder
must carry them too, so `Live 1/` never stands in for *Live 2*, nor `Live in Paris 1970/`
for *Live in Paris 1971*, while a folder's extra numbers of any kind are tolerated. A folder
that holds the name among other words qualifies only when those other words are the annotation's other
name (artist beside album) or one of *the*, *discography*, *collection*, *complete*,
*anthology*, *box*, *set* — so `Beatles, The/` and `Beatles Discography/CD 05/` count, while
`Beatles Covers/`, `Songs of The Beatles/`, `Help Yourself Sessions/`, `Abbey Road Demos/`
and `The Beatles - Help! (Karaoke)/` do not, and neither does `Queen Latifah/` for Queen. A folder that is only *part* of a name (`Lonely Hearts/`) never counts. The check
reads folder names, not recordings: a folder named exactly like the album — someone else's
`Let It Be/` — or an `--audio` root named for the artist still vouches for whatever is in
it. Anything it cannot pair uniquely it *reports* rather than guessing at — a manifest
quietly pointing at the wrong recording would produce a plausible, wrong score.

The report splits what did not pair. **No audio found** means no file carries the title:
fix those filenames. **Title found but no folder names the artist or album** means the file
is there but its folders don't say whose it is: point `--audio` at the artist's folder, or
rename the album folders, rather than the files. **Ambiguous** means the title is in several
folders and none of them is named exactly after the annotation's album. The word rule above
does not apply here: only a folder named exactly like the album — case, punctuation and a
leading year or track number aside, so `Help!/` or `1965 - Help/` — picks that copy over a
`Greatest Hits/` one, while `The Beatles - Help!/` or `Help! [FLAC]/` leaves the title
ambiguous. A looser reading kept choosing the wrong disc or volume silently. Rename the album
folder, or move compilations out of `--audio`; a self-titled album (*The Beatles*) can't be
told from its artist folder, so for it only the latter works.
`--list-unpaired` lists every track in each group.

This manifest is yours alone: its audio paths follow your library's filenames, so it is
not committed. The committed `eval/isophonics-beatles.json` is the Beatles chordlab set
**references-only** — 180 tracks, no `audio` paths. It is enough for `survey` and for
`score --estimates`, which need only the annotation root; `score -c` on it finds no
recordings and scores nothing. To run a backend, build your own as above.

**5. Survey the corpus.** Needs no audio and no model, so do it before the long run:

```bash
songscribe-eval survey eval/isophonics-beatles.json
```

```
dataset   isophonics-beatles
tracks    180 readable
annotated 8.15 h in 14621 segments
labels    407 distinct

every label parses

vocabulary    scoreable   excluded   definition
root            100.00%      0.0 m   root only; ignores quality entirely
majmin           96.43%     17.4 m   major and minor triads; sevenths reduce to their triad
sevenths         96.27%     18.2 m   maj, min, 7, maj7, min7
mirex           100.00%      0.0 m   at least 3 pitch classes in common
```

**"every label parses" is the line that matters.** All 407 distinct labels across those
14,621 segments — including `E:7/3`, `A:min/b3`, `C#:maj7(*b5)` and `D:min7(2,*b3,4)` —
resolve to pitch content. An unparseable label would be a gap in `harte.py` or in
`chords.HARTE_QUALITIES`, not bad data, and the time it accounted for would be time *no*
vocabulary could score. `survey` exits non-zero when any label fails to parse, so it works
as a corpus health check.

`scoreable` is the ceiling, not a score: it is the reference time a vocabulary can express,
so the most any backend could possibly get right there. On this corpus `majmin` excludes
17 minutes out of 489 — the suspensions, diminished and augmented chords — which is the
honest denominator to quote alongside a WCSR figure.

The `mirex` row above predates a fix: `mirex` now also excludes reference chords with fewer
than 3 pitch classes (power chords like `A:(1,5)`, bare roots like `E:(1)`), which no
estimate could ever match. Any such labels in the corpus will pull it below 100% once
re-measured.

**6. Score a backend.**

```bash
songscribe-eval score eval/isophonics.json -c madmom --cache .eval-cache --per-track
```

**Budget most of a working day for a full corpus.** The Beatles chordlab set is 180 tracks
and **8.15 hours of annotated audio**. Measured on one CPU, `madmom` runs at roughly
**2× realtime**, so a pass over it is **about 4 hours**; `--separate` adds demucs on top of
that. The `template` baseline is ~180× realtime — about 3 minutes for the same corpus —
which is why it's the right thing to shake the plumbing out with.

`--cache` matters at that scale: it writes each estimate as a `.lab`, so an interrupted run
resumes instead of restarting, and re-scoring under different vocabularies is instant. The
scoring rules get iterated on far more often than the model does. Use `--limit 5` to confirm
the manifest and paths are right *before* committing to the long run.

```bash
# Re-score cached estimates under different vocabularies: no model, seconds.
songscribe-eval score eval/isophonics.json -c madmom --cache .eval-cache -v triads -v sevenths

# Score estimates some other tool produced.
songscribe-eval score eval/isophonics.json --estimates ~/estimates/btc

# Sanity-check the plumbing on a handful of tracks first.
songscribe-eval score eval/isophonics.json -c template --limit 5
```

Add `--separate` to run demucs first. It is much slower, and it is the single largest
accuracy factor in the pipeline — so a number produced without it is a lower bound, and
numbers with and without are not comparable. Say which you used.

## Reading the numbers

Weighted Chord Symbol Recall is the fraction of *annotated time* labelled correctly.
Duration-weighted, not segment-counted: a model should not win by getting a hundred passing
chords right and the tonic wrong.

Two corpus aggregates are printed because they answer different questions — MIREX reports
both:

- **mean** — unweighted mean of per-track WCSR. Every song counts once.
- **weighted** — total correct duration over total compared duration. Every second counts
  once, so long songs pull harder.

Several vocabularies are printed because **the gap between them is diagnostic**, and it is
the first thing to check before blaming a model:

| Vocabulary | Asks |
|---|---|
| `root` | is the root right, ignoring quality entirely |
| `majmin` | is the triad right; sevenths reduce to their triad |
| `triads` | `majmin` plus dim, aug, sus2, sus4 |
| `sevenths` | is the seventh right too: maj, min, 7, maj7, min7 |
| `mirex` | do the two chords share at least 3 pitch classes |

On a synthetic `Cmaj7 Am7 Dm7 G7` demo, both current backends score ~92–99% on `majmin` and
**0.0% on `sevenths`**, while `mirex` stays ~99%. Nothing is broken: neither backend can
*emit* a seventh, so every segment is a triad guess against a seventh reference. `mirex`
stays high because a triad shares three pitch classes with its parent seventh. This is
exactly the "chord vocabulary" entry under *where accuracy is actually lost* in
[CLAUDE.md](CLAUDE.md) — it looks like a model error and isn't. A backend with a larger
vocabulary (BTC) is what moves `sevenths`; more smoothing will not.

A **segmentation** score is printed alongside, which ignores labels entirely and measures
boundary placement only. It catches a model that is harmonically right but smears every
change across a bar — a different failure from naming the wrong chord, and one WCSR alone
hides.

### WCSR is not the same question as "is the chart playable"

WCSR is onset-sensitive by construction: it asks whether *each second* carries the right
label. That is the right question for comparing against published numbers, and the wrong
question for a chart. A transcription that is uniformly 1.7 s late annoys nobody — the
musician reads the same progression — yet WCSR drops ~32 points for it.

So a **chart** score is reported too: project both sides into `majmin` (always, whatever
`-v` asks for, so charts compare across runs), drop `N`,
collapse them exactly as `align.dedupe` does (merge repeats, drop flicker below
`align.MIN_CHORD_DURATION`), and compare the resulting chord *sequences* by normalised
edit distance. Timing is absent by construction, so a substitution is a wrong
chord, an insertion one the model invented, a deletion one it missed — the three things a
musician notices. Because it is built from `align.dedupe`, it scores the chart `render.py`
would actually print, not the raw frame timeline. Unlike WCSR it keeps a reference `X` as a
token no backend emits, so an `X` that prints costs a backend something, depending on what
it put there and on the chords around it, and can tip a close comparison; see
[What gets excluded](#what-gets-excluded).

Measured on Eleanor Rigby, before and after trimming a 1.7 s lead-in:

| | misaligned | trimmed |
|---|---|---|
| `majmin` | 50.09% | 79.51% |
| `chart` | **67.44%** | **67.44%** |

Identical to the decimal. That is the invariance working: the lead-in is invisible to the
chart score, exactly as it is invisible to a musician.

**But the chart score is not WCSR with the timing taken out — it is often harsher**, and
that is its real value. On the same track:

| Backend | `majmin` | `chart` | chart chords (ref: 34) |
|---|---|---|---|
| madmom | 79.51% | 67.44% | 43 |
| template | 58.02% | **28.57%** | **112** |

The `template` baseline prints 112 chord changes where there are 34. Every second carries a
defensible label, so WCSR says 58% — but the chart is unplayable, and the chart score says
28.57%. WCSR flatters over-segmentation; the chart score does not.

Note what that does to the comparison. On WCSR, madmom beats the baseline by 21.5 points.
On the metric that matches the product goal, it beats it by **38.9**. Measuring the wrong
thing was understating the model.

### `grid`: order *and* harmonic rhythm, in bars

`chart` ignores duration entirely, so it cannot tell one chord per bar from two. That
distinction *is* the harmonic rhythm, and it is most of what makes an instrumental chart
readable — so `grid` reports it: chord durations quantised to a harmonic unit and compared
as `(label, bars)` pairs.

The unit is estimated from the chord timeline alone, no beat tracking required: anchored on
the median chord duration — harmonic rhythm means most chords last one unit — then refined
to whichever nearby value the durations are closest to whole multiples of. Because the unit
is measured rather than assumed, it **scales with the tempo**, which is what makes the grid
tempo-free:

| Tempo | `majmin` | `chart` | `grid` units |
|---|---|---|---|
| 0.5× – 2.0× | 46.89% – 79.51% | 67.44% throughout | identical throughout |

Two design choices worth knowing:

**Both sides are measured against the *reference's* unit**, as WCSR takes its time base from
the reference. Letting each side estimate its own would normalise away the error the metric
exists to catch: a model that systematically merges chords has a longer typical chord, so a
per-side unit would rescale it back to "one chord per bar" and call it correct.

**Ties resolve to the coarser grid.** When every chord lasts 2 s, a 2 s unit (one per bar)
and a 1 s unit (two per bar) both fit perfectly; the musician's reading is the former.

### Where the grid stops working, and what would fix it

On regular material the unit estimate is exact — residual ~0% — and the grid catches what
`chart` cannot: a transcription with the right chord order but a two-bar chord written as
one scores 100% on `chart` and under 50% on `grid`.

On **irregular real material it does not work**, and `grid` reports `nan` rather than a
number. The Isophonics Eleanor Rigby reference fits its best unit at a 10.2% residual — its
chords simply do not fall on a grid recoverable from durations alone — so no score is
emitted for it. `score` prints how many tracks were gradeable.

Be sceptical of that gate. The unit is a free continuous parameter over a ±50% window, so on
a short timeline almost any durations fit *something* to within a few percent; the residual
only separates regular from irregular once there are enough chords for a wrong unit to
accumulate error. `GRID_MAX_RESIDUAL` is calibrated on one real track, and
`tests/test_eval_grid.py::TestTheReliabilityGateIsWeak` pins three plainly irregular
fixtures that pass it anyway. **Treat `grid` as indicative, not decisive.**

The exact path is to stop estimating: `grid_sequence(..., unit=...)` takes a known bar
length and does no search at all. Supplying real downbeats — the beat-tracking roadmap item
in `README.md` — is what turns `grid` from a heuristic into a measurement, and it is the
strongest argument for doing that work next.

### Beat-grid drift: what `BAR_DRIFT_TOLERANCE` does and does not catch

`BeatGrid.unreliable_bars()` flags any tracked bar more than `BAR_DRIFT_TOLERANCE` (10%)
away from the median bar length. It is meant to catch a *gross* grid fault — a dropped or
added bar, or a stretch tracked at half tempo — and deliberately nothing finer.

Measured on one real 236 s take: across the steady body of the performance, individual
tracked bars still scatter up to 6.4% around the median. That figure is dominated by the
tracker placing each beat, not by the band's tempo moving — the median itself drifts under
1% over the same span — so anything tighter fires on a perfectly good grid. A dropped bar,
by contrast, is about 100% long, so there is a wide margin.

Deciding *which* bars belong to the form is not this statistic's job. On the same take the
out-of-tempo intro ran 4–12% long against the body tempo, a range that straddles the
threshold: its first bars are flagged and its last are not, having already accelerated into
tempo. Form membership needs the form's known bar count and an anchor, not a per-bar length
test.

### A chart played several times: `form`, `bar`, and `--beats`

A real recording is an intro, the chart several times through, and an ending, none of it on
the reference's clock. Give a manifest track `"bars"` (the chart's bar count) and score it
with `--beats DIR` (or `--beat-backend`); `songscribe-eval form` shows what was detected
first. The pieces, and what each refuses to do:

- **Start** (`eval/recurrence.py`, `eval/chart_fit.py`, `eval/form_start.py`): a chord
  estimate of the recording finds where the form starts; the chart does not. The estimate is
  one symbol per beat (root and third, so `C`, `C7` and `Cmaj7` agree and flicker or an
  anticipated change is outvoted within the beat), and each bar is compared with the bar a form
  later (the chart gives only the length in bars; it is not searched for, since self-similarity
  alone returns half of it). The stretch where that holds begins at the head, or at the pickup
  into it, so the bars before it are the intro and the beats just before the head that also
  recur are the pickup. Where the chords change says which beat the bar lines fall on, so a
  tracker whose downbeats are a few beats off is moved. A run counts only if it beats every
  other lag as a null and spans three quarters of a form (`MIN_RUN_FORMS`); the tolerance for
  choruses that differ is the step limit of the run. The chart's labels are then compared with
  the estimate from the start, beat by beat, in the key asked for (the chart transposed, never
  the estimate). That fraction is read against the best that bar-shuffled copies of the chart
  reach over the same starts and keys (`Anchor.chance`): the chart has to beat it by
  `ANCHOR_MIN_EXCESS`, so a vamp or a loop, where the shuffles agree as well, is refused.
  The chart chooses between starts within a form of the opening (an intro that echoes the
  tune; a head that differs from the solos) and, if nothing recurs (one chorus), finds the
  start alone. `form` says which evidence found the start. The thresholds come from synthetic
  takes (shuffled, unrelated, one-chord and looped controls under three levels of perturbed
  estimates), never from a real recording.
  The estimate is a model's output, so the start can be the one that suits it: anchor on a
  different backend from the one scored (`--anchor-backend`, default `madmom`; `score` warns
  when it is the same one) or on pre-computed `.lab` files (`--anchor-estimates`, `form
  --estimate`).
- **Key** (`--transpose auto`, the default): a tune is often played in another key, so the
  chart is also tried in each of the twelve keys (its labels transposed, never the estimate).
  The shuffled-chart chance level is taken over the same starts and keys, so twelve keys raise
  the bar a chart has to clear instead of lowering it, with no separate gate. A start found by
  recurrence is already fixed, so only the key is chosen: the chart must clear the chance level
  and the best key must lead the next by `ANCHOR_MIN_MARGIN`. One key is assumed for the whole
  take: the anchor's shift applies to every chorus, which is scored against the chart
  transposed by it; the estimate is never altered. `form` and `--per-chorus` show it as
  `key` (`+2` a tone up), the JSON as `shift`. A chart is refused when no key gives a clear
  fit. `--transpose none` insists on the written key and refuses a transposed take as before. A
  song that changes key partway is not supported.
- **Choruses** (`eval/form.py`): contiguous blocks of the chart's bar count from the anchor.
  `intro`, `head` (first and last full chorus), `chorus`, `partial` (at least four bars),
  `tail` (shorter: an ending), `unaligned`. A bar far from its own block's median length
  loses the count, so that block and every later one is `unaligned`; so is a chorus at
  another tempo than the rest. Unscored time is reported, never folded in.
- **Mapping**: each chorus's estimate is moved onto the reference's clock by one affine map
  (start, scale), so WCSR, `chart`, `grid` and `segmentation` apply unchanged per chorus.
  The offset sweep per chorus is one bar wide, not three seconds.
- **`bar`**: the fraction of bars holding the right chord, read against the recording's own
  bar lines, so tempo, offset and length cannot move it. Lead with it on a real take.
- **Aggregation**: the track row sums the full choruses. It is never the best chorus, and a
  partial last pass is reported but left out.

### `placement`: lyric-chord synchronisation, for songs

For a song rather than an instrumental, the thing that matters is whether each chord sits
over the right *syllable*. `metrics.placement_score` measures it, reusing `align.py`'s own
`group_words` and `place_in_line` so it scores the chart that gets printed.

It compares sequences of `(line index, word index, label)` — so it works in **word indices,
never seconds**, and needs no lyric text at all. That has two happy consequences: it is
immune to offset and tempo exactly as `chart` is, and it can be developed and tested with
invented placeholder words, keeping copyrighted lyrics out of the repo entirely.

```python
placement_score(reference_chords, estimate_chords, words)  # right chord, right syllable
placement_score(
    reference_chords, estimate_chords, words, compare_labels=False
)  # right syllable at all
```

The pair separates a *mistimed* chord from a *misnamed* one, which are different bugs in
different branches.

**Both sides are placed against one word timeline — the reference's.** Scoring each against
its own ASR output would confound two independent failures; the branches are kept
independent everywhere else for that reason, and `placement_score` takes a single `words`
argument so the confusion is not expressible.

#### The word timeline does not have to be a gold transcript

`placement_score` holds **one** word timeline fixed across both sides, so word errors apply
equally to reference and estimate and cancel out. The timeline has to be *consistent*, not
*correct*. That is what makes this measurable with no external data at all: run your own ASR
on your own recording, and the comparison still isolates chord placement.

`songscribe.eval.reference` stores such a timeline as JSON. It refuses any filename not
ending in `.words.json`, because that suffix is what `.gitignore` matches — these files hold
lyric text and must stay local.

```python
from songscribe import lyrics
from songscribe.eval import reference, metrics

words = lyrics.transcribe("data/audio-aligned/my_track.wav", model_size="base", language="en")
reference.write_words("data/placement/my_track.words.json", words, source="whisper-base")

metrics.placement_score(reference_chords, estimate_chords, words)
```

Measured that way on Eleanor Rigby — Isophonics chords as reference, madmom as estimate,
195 Whisper words over 23 lines:

| Metric | Score |
|---|---|
| `majmin` WCSR | 79.51% |
| `chart` | 67.44% |
| `placement`, right syllable at all | 73.21% |
| `placement`, right chord **and** right syllable | **57.14%** |

The last row is the lead-sheet number: 57% of chords are the right chord over the right
syllable, **22 points below the WCSR headline**. The 73% → 57% gap is chords landing in the
right place but misnamed; `placement` below `chart` means some correctly-ordered chords land
on the wrong word. The reference places 45 chords where the estimate places 56 — the
over-segmentation again.

Caveat worth keeping attached to that number: those words are uncorrected Whisper output on
an unseparated mix, so "syllable" means "Whisper's idea of a word boundary". The *comparison*
is sound, since both sides share it, but hand-correcting the timeline is what would make the
absolute figure trustworthy.

It is **not wired into `songscribe-eval`**, because no *corpus* here carries lyrics:
Isophonics is chords only, and a manifest path would have nothing to read. For a shareable
reference rather than a local one, the options are, best first:

- **A self-recorded track.** The chords played and the words sung are both known, so the
  reference is exact and free of licensing questions entirely.
- **MTG Jamendo Lyrics.** CC-licensed audio with word-level aligned lyrics — small, but
  genuinely redistributable. It has no chord annotations, so you would annotate those
  yourself; on CC audio that is legitimate and the result is even committable.
- Commercial tab and sheet-music sites are **not** an option, for the reasons in
  [What you may and may not use](#what-you-may-and-may-not-use) plus the lyrics themselves
  being copyrighted — see hard constraint 5 in `CLAUDE.md`.

### A placement bug this metric found

Writing the metric surfaced a real `align.py` behaviour. When a chord's span spills past a
line break, `place_in_line` places it at the *new* line's start; its "two changes on one
syllable, keep the first" rule then discards the chord that genuinely belongs there. The
line is printed with the previous chord and no sign of the right one.

A 0.2 s shift is enough to trigger it, and `placement` reports it rather than smoothing it
over — see `tests/test_eval_placement.py::...::test_a_chord_spilling_over_a_line_break_displaces_the_next_one`.
Whether the fix is to clip chords at line boundaries or to prefer the chord that *starts*
within the line is an `align-render` decision, not an evaluation one, so the behaviour is
pinned by a test rather than changed here.

### What gets excluded

In WCSR, reference time the comparison vocabulary cannot express leaves the score entirely,
numerator and denominator both, and `survey` counts it as excluded rather than scoreable.
That covers `X` (the annotator heard a chord but could not name it), labels this project
cannot parse, and qualities outside the vocabulary — a `sus4` is excluded from `majmin` but
scored under `triads`. Penalising a model for disagreeing with an admitted non-answer would
measure the annotation, not the model.

`chart`, `grid` and `placement` do **not** exclude a reference `X`: they keep it as a token
no backend emits (an unparseable reference label is kept as its raw text the same way),
because dropping it would merge its neighbours and change the sequence. So an `X` that
prints in the chart costs a backend something, and how much depends on what the backend
put there and on the chords around it. It keeps `chart` and `grid` below 100% on that
track — `C X G A:min F` against `C N G Am F` is 100% WCSR and 80% `chart` — and it can tip
a close comparison: against reference `C X G`, `C D G E` scores 50% `chart` and `C G` 67%,
though `C D G E` would score 75% were the `X` read as its `D`. Read small differences
between backends on tracks with an `X` with that in mind.

An unparseable or out-of-vocabulary *estimate*, by contrast, is simply wrong. A model is not
excused by emitting something unreadable.

A track whose reference or estimate is missing is **skipped and named**, never scored as
zero: a missing measurement is absent, not wrong, and averaging it in as zero would
understate the model.

### Differences from `mir_eval`

The metrics are reimplemented here rather than taking a dependency, so that CI can score
with no extra packages. Numbers should be close to `mir_eval`'s but are not guaranteed
identical, so don't mix them in one table:

- **Projection is more inclusive.** A reference chord is reduced to its triad by reading the
  third and fifth off the interval set, so `C:maj(9)` projects to `C:maj` and is scored.
  `mir_eval`'s bitmap test excludes it. More hard segments stay in the denominator, which
  pushes reported WCSR slightly *down* — the conservative direction.
- **Inversions are ignored** in every vocabulary, because no backend here emits them;
  `normalise_label` drops the bass note on the way out. `Chord.bass` is parsed and kept, so
  an inversion-sensitive vocabulary can be added without reparsing, but there is nothing to
  measure yet.
- **Comparison is modulo the octave.** A 9th is a 2nd, an 11th a 4th. No vocabulary in use
  distinguishes voicings.

## What you may and may not use

*Not legal advice — this is the practical position, recorded here so it doesn't get
re-litigated every time a tempting shortcut appears. Same purpose as the "Rejected" table in
[MODELS.md](MODELS.md).*

The thing that matters for a dataset is **lawful access to the copy you analyse**. The
research-friendly doctrines — US fair use for non-expressive analysis (the Google Books /
HathiTrust line), the EU TDM exceptions — are genuinely favourable for *what you do with*
audio once you hold it legitimately. They all presuppose you obtained it legitimately.
"Private use" and "I only published the script" do not substitute for that: they shrink the
practical risk, they don't make the access lawful.

### Usable

| Source | Position |
|---|---|
| Recordings you own | A rip of your own CD or a DRM-free purchased download, analysed locally. The lawful-access case, and what Isophonics assumes. |
| Isophonics annotations | Free download for research, audio never included. Their pages state **no explicit licence**, so treat redistribution as not granted: use them locally, cite Harte's work, and keep them out of the repo. `data/` is gitignored for this reason. |
| McGill Billboard annotations | Published for research, audio not included. Also distributes precomputed chroma features. |
| RWC Popular Music | Licensed to researchers for a modest fee **with** the audio. The cleanest end-to-end option. |
| CC0 / CC-BY audio (FMA, MTG-Jamendo) | Redistributable, but no chord ground truth — sanity checks, not accuracy numbers. |
| `songscribe-demo` | Synthetic, known-correct, ships no audio. The CI-runnable target. |

### Not usable

| Source | Why not |
|---|---|
| **Ultimate Guitar tabs** | Their ToS forbids automated extraction, so scraping is a contract breach whatever you do with the result. The transcriptions are user-submitted derivative works that UG licenses and pays publishers for. A bare chord *sequence* is probably too thin to protect; a time-aligned full transcription is not. And crowd tabs are frequently in the wrong key, wrong inversion, or simplified — unverifiable ground truth makes a bad gold standard even setting the licence aside. |
| **Audio ripped from streaming services** | Extracting from Spotify/Apple Music means circumventing DRM, which is a *separate* violation from infringement under DMCA §1201 and the EU Art. 6 equivalents — fair use and research exceptions do not reach the act of circumvention. YouTube is DRM-free in practice but its ToS prohibits downloading, so that route is contract breach instead. |
| **Sheet-music and chord-chart sites** (boiteachansons, Musicnotes, Chordify and the like) | Commercial services that license transcriptions from publishers and forbid extraction in their terms — the Ultimate Guitar analysis, plus the lyrics themselves being copyrighted, which hard constraint 5 in `CLAUDE.md` rules out independently. There is also no technical upside: a chord chart carries **no timestamps**, so it cannot feed `placement_score` (which needs word onsets) without word timings from audio anyway. Its only unique contribution would be the licensed transcription. |
| **"Royalty-free" stock libraries** | Licensed, not free. Their terms forbid redistributing the audio. |

A corollary worth stating because it is easy to miss: if the *public* eval script contained a
scraper or a stream-ripper, the script would be the problem regardless of how private the
dataset was. `songscribe-eval` therefore only ever reads local paths plus a manifest, and
acquiring the corpus is deliberately left outside the tool.

## Other corpora worth adding

Recorded as possibilities, in rough order of usefulness. All follow the same
annotations-without-audio split, so each needs only a manifest builder beside
`isophonics.py` — `dataset.py` and `metrics.py` are corpus-agnostic.

- **McGill Billboard** — ~890 tracks sampled from the Billboard charts, high-quality Harte
  annotations with a large vocabulary. The most valuable addition: it is far more
  stylistically varied than Isophonics, which is heavily 1960s British pop, and it is what
  large-vocabulary models are usually trained and reported on. It also distributes
  **precomputed NNLS and tuned chroma features**, which sidesteps audio entirely — though
  not for madmom or BTC, which both want waveforms. Scoring against it is the check that a
  gain on Isophonics isn't overfitting.
- **RWC Popular Music** — 100 tracks, licensed to researchers *with* the audio for a modest
  fee. The only option here that arrives complete, which makes it the best choice if
  assembling a library is the bottleneck. AIST distributes chord annotations separately.
- **JAAH** — 113 jazz tracks with Harte annotations. Dense extended harmony, so it is the
  natural target once a backend can emit sevenths; today it would mostly re-measure the
  vocabulary ceiling.
- **Schubert Winterreise** — classical lieder, multiple performances of the same scores.
  Useful for checking the chord branch hasn't quietly become a pop-progression prior.
- **Chordify / HookTheory** — large, but check the terms carefully before touching either;
  both are commercial products built on user submissions, which is the same shape of problem
  as Ultimate Guitar.

Adding one means writing a manifest builder that emits `{id, reference, audio}` entries
with relative paths — see `isophonics.build_manifest` — and a row in the tables above
recording where it came from and under what terms.
