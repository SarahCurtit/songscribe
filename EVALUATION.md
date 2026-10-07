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
King, Zweieck if you want them). Unzip somewhere outside the repo — `*.lab` is gitignored,
but keep them out anyway.

**2. Supply the audio.** Your own copies of the recordings, in a directory of their own.
See [What you may and may not use](#what-you-may-and-may-not-use) below before you source
these.

**3. Point the harness at both.** Once, in your shell:

```bash
export SONGSCRIBE_EVAL_ROOT=~/music/beatles              # your recordings
export SONGSCRIBE_EVAL_ANNOTATIONS=~/corpora/isophonics  # the .lab files
```

**4. Build a manifest.**

```bash
songscribe-eval manifest ~/corpora/isophonics \
    --audio ~/music/beatles \
    --output eval/isophonics.json
```

It walks the tree for chord `.lab` files and pairs each with a recording, trying the
mirrored relative path, then the album directory, then a punctuation- and
track-number-insensitive title match. Anything it cannot pair uniquely it *reports* rather
than guessing at — a manifest quietly pointing at the wrong recording would produce a
plausible, wrong score. Use `--list-unpaired` to see every miss and fix your filenames.

**5. Score a backend.**

```bash
songscribe-eval score eval/isophonics.json -c madmom --cache .eval-cache --per-track
```

**Budget overnight for a full corpus.** Measured on one CPU, `madmom` runs at roughly
**2× realtime**, so the ~180-track Beatles set (about 12 hours of audio) is **5–6 hours**;
`--separate` adds demucs on top of that. The `template` baseline is ~180× realtime, so the
same corpus takes about 4 minutes — which is why it's the right thing to shake the plumbing
out with.

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

### What gets excluded

Reference time the comparison vocabulary cannot express leaves the score entirely,
numerator and denominator both. That covers `X` (the annotator heard a chord but could not
name it), labels this project cannot parse, and qualities outside the vocabulary — a `sus4`
is excluded from `majmin` but scored under `triads`. Penalising a model for disagreeing with
an admitted non-answer would measure the annotation, not the model.

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
| Isophonics annotations | Free download for research. Check the current terms before redistributing them; this repo never commits them. |
| McGill Billboard annotations | Published for research, audio not included. Also distributes precomputed chroma features. |
| RWC Popular Music | Licensed to researchers for a modest fee **with** the audio. The cleanest end-to-end option. |
| CC0 / CC-BY audio (FMA, MTG-Jamendo) | Redistributable, but no chord ground truth — sanity checks, not accuracy numbers. |
| `songscribe-demo` | Synthetic, known-correct, ships no audio. The CI-runnable target. |

### Not usable

| Source | Why not |
|---|---|
| **Ultimate Guitar tabs** | Their ToS forbids automated extraction, so scraping is a contract breach whatever you do with the result. The transcriptions are user-submitted derivative works that UG licenses and pays publishers for. A bare chord *sequence* is probably too thin to protect; a time-aligned full transcription is not. And crowd tabs are frequently in the wrong key, wrong inversion, or simplified — unverifiable ground truth makes a bad gold standard even setting the licence aside. |
| **Audio ripped from streaming services** | Extracting from Spotify/Apple Music means circumventing DRM, which is a *separate* violation from infringement under DMCA §1201 and the EU Art. 6 equivalents — fair use and research exceptions do not reach the act of circumvention. YouTube is DRM-free in practice but its ToS prohibits downloading, so that route is contract breach instead. |
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
