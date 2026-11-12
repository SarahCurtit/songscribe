# songscribe — notes for Claude

Transcribe chords and lyrics from an audio file into a lead sheet. Deep learning end to end,
**open-source models only**, everything local.

## Hard constraints

These are the project's identity, not preferences. Don't trade them away for accuracy.

1. **Open-source models with openly published weights only.** No hosted inference APIs, no
   API keys, no gated or request-only checkpoints. Before adding any model, check it against
   the bar in `MODELS.md` and add a row to the inventory table there in the same change.
2. **PyTorch, or no framework at all. Never TensorFlow or Keras.** Reject a TF model however
   good its numbers are — one DL runtime is enough for a project this size. Framework-free
   numpy inference is welcome (the default chord backend is exactly that), and ONNX Runtime
   is fine for `.onnx` weights. `MODELS.md` has the reasoning and a list of candidates
   already rejected on this basis, so they don't get re-litigated.
3. **Runs locally on CPU.** GPU may be an optimisation, never a requirement. If a change
   makes CPU-only use impossible, it's the wrong change.
4. **No audio, lyrics, or weights in the repo.** `.gitignore` blocks them; don't work around
   it. Tests use synthetic data or hand-written event fixtures, never song files.
5. **Never commit transcribed lyrics of real songs.** Lyrics are copyrighted. Examples in
   docs and tests use invented placeholder words — keep it that way.

## Layout

```
src/songscribe/
  types.py      ChordEvent, BeatEvent, Word, LyricLine, Placement, Section, Song — all frozen
  audio.py      loading/resampling; caller asks for the rate its model wants
  separate.py   demucs wrapper (subprocess), optional
  chords.py     pluggable chord backends behind recognise()
  beats.py      pluggable beat/downbeat backends behind track(); BeatGrid
  tempo.py      BeatGrid vs a known tempo: octave-error check          (pure Python)
  registry.py   name/setting checks shared by chords and beats       (pure Python)
  lyrics.py     faster-whisper wrapper, word-level timestamps
  align.py      chord timeline + word timeline -> printable sections   (pure Python)
  render.py     Song -> text or ChordPro                               (pure Python)
  pipeline.py   orchestration of the above
  cli.py        argparse entry point
  demo.py       synthesises a demo track + .lab ground truth  (pure Python)
  lab.py        .lab chord annotation read/write               (pure Python)
  eval/         chord accuracy measurement
    harte.py      chord labels -> pitch content                (pure Python)
    timeline.py   fill gaps, merge two chord timelines         (pure Python)
    metrics.py    WCSR, vocabularies, segmentation, bar score  (pure Python)
    substitution.py  graded chord relations: how close a wrong chord is  (pure Python)
    labels.py     chord labels -> one rough symbol per beat, and how well two agree (numpy)
    chart_fit.py  the chart's labels against an estimate from a start, with its chance level (numpy)
    recurrence.py the stretch of an estimate that repeats one form later: head, intro, pickup (numpy)
    form_start.py recurrence first, chart as check and fallback -> the form's start (numpy)
    form.py       splits a recording into choruses, on the chart's timeline (pure Python)
    README.md     how labels, recurrence, chart_fit, form_start and form fit together
    dataset.py    manifests and path resolution                (pure Python)
    split.py      train/test manifests, versions of a song kept together (pure Python)
    isophonics.py manifest builder for an Isophonics tree
    survey.py     corpus label/coverage report, no audio or model needed
    reference.py  local word timelines for placement scoring (gitignored data)
    viz.py        shaping for plots — no plotting dependency      (pure Python)
    runner.py     runs a backend, caches estimates, aggregates
    cli.py        songscribe-eval entry point
notebooks/      annotations.ipynb — look at annotations and estimates
tests/          unit tests for the pure-Python stages
MODELS.md       model inventory, licences, and the bar for adding one
EVALUATION.md   how to measure a backend
```

## Test material

There is no audio in this repo and there must never be. `songscribe-demo` (`demo.py`)
generates a synthetic track with known chords plus a `.lab` annotation, which is how the
chord branch gets measured without shipping a recording. It is **instrumental on purpose** —
don't try to make it sing; that is what a real recording is for.

For a real accuracy number, `songscribe-eval` scores a backend against reference
annotations; `EVALUATION.md` explains how. The rule the harness is built around: manifests
hold track ids and relative paths only, so the evaluation code is publishable while the
corpus stays local. Never commit annotations, estimates, or a manifest with absolute paths
in it.

Real audio goes in the gitignored `audio/` directory and is never committed, whatever its
licence. For anything claiming an accuracy number, prefer a recording where the chords and
words are known in advance.

Corpus decisions (settled; don't re-open them):

- **Usable:** recordings you own, analysed locally; Isophonics and McGill Billboard
  annotations (audio not included, keep them out of the repo); RWC Popular Music (licensed
  to researchers with the audio); CC0/CC-BY audio (FMA, MTG-Jamendo) for sanity checks only,
  since it has no chord ground truth; `songscribe-demo`.
- **Not usable:** Ultimate Guitar and other chord/tab/sheet-music sites (terms forbid
  extraction, crowd tabs are unverified, lyrics are copyrighted); audio ripped from
  streaming services (circumvents DRM; YouTube downloads breach its terms); "royalty-free"
  stock libraries (licensed, not free, redistribution forbidden).
- The eval tool only reads local paths and manifests. It never contains a scraper or ripper.

## Architecture rules

- **Everything is a timeline in seconds from start.** Every stage consumes and produces
  timed events on that one clock. This is what keeps models swappable — don't introduce
  frame indices or sample offsets into cross-stage interfaces.
- **The two branches stay independent.** Chord recognition and lyric transcription must not
  know about each other; they meet only in `align.align()`. Resist "the chord model could
  use the vocal onsets" shortcuts — they couple the branches and break testability.
- **Models go behind a registry.** A chord backend is one function returning
  `list[ChordEvent]`, registered in `chords.BACKENDS` with its sample rate in
  `BACKEND_SAMPLE_RATES`. Nothing else in the pipeline learns its name. Every
  setting it accepts is a named parameter *with a default*, after the audio and
  its rate — `recognise()` and `track()` reject a setting the chosen backend
  doesn't declare. A `**kwargs` on a backend would turn that check off, so a
  registered backend must not have one; a test enforces it.
- **One chord vocabulary downstream.** Models emit Harte notation (`C:maj`, `A:min`);
  everything after the backend expects compact shorthand (`C`, `Am`). Backends call
  `chords.normalise_label()` on the way out. An unrecognised quality passes through
  untouched — never guess at it, or a wider vocabulary gets silently mislabelled.
- **Keep `align.py` and `render.py` model-free and dependency-free.** They're the only stages
  that can be tested exhaustively and fast; that's worth protecting.
- **Heavy imports stay inside functions.** `librosa`, `torch`, `madmom` and whisper are all
  slow to import — `songscribe --help` must stay instant. Don't move them to module scope.
- **Unknown backend or format raises.** Never silently fall back to a worse model; a typo in
  a config shouldn't quietly degrade output. The one sanctioned fallback is separation,
  which warns and continues on the full mix.
- The `template` chord backend is **not** deep learning. It exists as the baseline models
  must beat and so CI can run with no weights on disk. Never make it the default.

## Commands

```bash
pip install -e '.[dev]'    # all models + pytest + ruff; compiles madmom, ~30s
pytest                     # fast, no weights, no audio
pytest -k align            # one stage
ruff check . && ruff format --check .
songscribe song.mp3 -c template --no-separate   # quickest end-to-end smoke test

# the manifests and the corpora they name both live in the private data repo
export DATA=~/perso/songscribe-data-extraction
export SONGSCRIBE_EVAL_ANNOTATIONS="$DATA/data/isophonics"   # the directory holding chordlab/
export SONGSCRIBE_EVAL_ROOT="$DATA/data"                     # one root, every corpus
songscribe-eval survey "$DATA"/eval/isophonics-beatles.json  # no audio, no model
songscribe-eval score "$DATA"/eval/isophonics-beatles-audio-test.json -c madmom --per-track
```

A recording of the chart played several times (a manifest track with `"bars"`) is scored per
chorus with `score --beats DIR --per-chorus`; `songscribe-eval form` prints the detected
start and choruses without scoring. The key is searched by default (`--transpose none` keeps
the chart as written). See `EVALUATION.md`.

`survey` is the corpus health check: it reports annotated time and whether every label
parses, and **exits non-zero if any label doesn't** — an unparseable label is a gap in
`eval/harte.py`, not bad data. Run it on a new corpus before anything else. The one exception
is a label that isn't valid notation: Carole King's `E:4` is a bare degree where Harte
parenthesises a degree list, so it stays unparsed on purpose and `survey` exits 2 on that
corpus for good. Don't "fix" it — `EVALUATION.md` says why, and a test pins it.

`notebooks/annotations.ipynb` draws the same data: chord timelines raw vs collapsed-to-chart,
harmonic rhythm, root profile, corpus label coverage, reference-against-estimate with an
agreement strip, and the alignment offset curve. The shaping lives in `eval/viz.py`, which has
**no plotting dependency** so it stays testable on the core deps CI installs; matplotlib and
jupyterlab are in the `dev` extra. Commit the notebook with outputs stripped.

`score` sweeps a time offset per track and **warns when a recording isn't on the reference's
timeline** — an encoder lead-in or the wrong master makes the harness measure the mismatch
instead of the model. A real case: an m4a with ~1.7s of lead-in scored 50% and 80% trimmed.
Never fold the suggested shift into a score; that is fitting to the reference. Fix the audio,
**replace** the original with it (same folder and stem), rebuild with `manifest --audio`,
and re-run with a **fresh `--cache`**, since cache keys are backend, input variant and track,
not audio.

`--cache` writes each estimate as a `.lab`, so re-scoring under a different vocabulary is
instant and a long run is resumable — which matters because `madmom` runs at ~2x realtime,
making the 8-hour Beatles set ~4 hours (`template` is ~180x, so ~3 minutes). Shake the
paths out with `-c template --limit 5` first. See `EVALUATION.md`.

**No manifest is committed here.** A manifest names a corpus's own files, so it lives with the
corpus: on this machine they are all in the private data_extraction repo's `eval/` —
`isophonics-beatles`, `-zweieck`, `-carole-king`, `jaah` and `backing-track`, references-only
as downloaded, each with an `-audio` pairing beside it that adds the audio path of every track
a recording was found for — all but `-zweieck`, whose record nobody there has.
`score -c` runs straight off those pairings; the references-only
ones serve `survey` and `score --estimates`, and `score -c` on them scores nothing. Isophonics
is one manifest per artist, built with `manifest --include "chordlab/<artist>"`, because it
rates its annotations per artist and one average would hide that. Carole King is held out
whole (`-test`, no `-train`): 7 songs cannot carry a hash split, and `split` refuses rather
than writing an empty side. The backing tracks are **split by band**, not by hash — no trio on
both sides, some musicians test-only, styles balanced — from line-ups that are not public;
those manifests say `"split"`, and `split` refuses them (`EVALUATION.md`). Michael Jackson's
set has no chords at all, only segments, so it never reaches a manifest.

The Isophonics annotations live in that repo's `data/isophonics/` (extract the archive there,
so `chordlab/` sits directly under it), and their `reference` paths start at `chordlab/`, so
`SONGSCRIBE_EVAL_ANNOTATIONS` — or `--annotations` — must be the directory *containing*
`chordlab/`; unset, it falls back to the manifest's own directory and finds nothing.
`SONGSCRIBE_EVAL_ROOT` is that repo's `data/` for every corpus, each audio path starting with
its corpus directory. Both roots have to point back at the trees a manifest was built from, or
`score` finds nothing.

`[chords]` pulls madmom from a **pinned git commit**, not PyPI — the 2018 release can't
import on Python 3.10+. Don't "simplify" it back to `madmom>=0.16`; that breaks the default
backend. `MODELS.md` records why, and the dead ends, so they don't get retried.

CI runs `ruff check`, `ruff format --check`, and `pytest` on 3.10–3.12, installing only the
core deps — so **tests must not require model weights or network access**.

## Conventions

- Python ≥3.10, `from __future__ import annotations`, `X | None` not `Optional[X]`.
- Frozen slotted dataclasses for event types; they're passed around a lot and must not be
  mutated mid-pipeline.
- Line length 100. Ruff with `E,F,I,UP,B,SIM,RUF`.
- Comments explain *why* a threshold or choice exists (`MIN_CHORD_DURATION` is about model
  flicker, not about taste). Don't narrate what the code already says.
- Tuned constants are module-level and named, never inline magic numbers.

## Where accuracy is actually lost

Useful priors when debugging bad output, roughly in order of impact:

1. **Separation skipped** — both models on the full mix is the biggest single quality drop.
2. **Chord vocabulary** — the default backend only knows major/minor triads, so sevenths get
   flattened to their nearest triad. This looks like a model error but isn't. The gap
   between `songscribe-eval`'s `majmin` and `sevenths` scores is exactly this effect, which
   is why both are reported: a high `majmin` next to a near-zero `sevenths` means the
   vocabulary, not the model.
3. **Whisper timestamp drift on held notes** — pushes chord labels a word too far right.
   Forced alignment is the planned fix, not more smoothing in `align.py`.
4. **Line grouping** — `group_words` splits on pauses, which mis-segments very legato or
   very staccato singing. Tune `max_gap`/`max_words` before suspecting the ASR.
5. **Over-segmentation** — a model that flickers between plausible chords keeps a
   respectable WCSR (most seconds carry a defensible label) while printing a chart with
   several times too many changes, which is what makes a chart unplayable. Measured: on
   Eleanor Rigby the `template` backend prints 112 chord changes where the reference has 34,
   scoring 58% WCSR but 29% on `songscribe-eval`'s `chart` metric. **WCSR flatters this
   failure; `chart` and `segmentation` are what expose it.**

Which metric to believe depends on the question, and `songscribe-eval` reports five because
they disagree sharply (see `EVALUATION.md`):

| Metric | Asks | Robust to |
|---|---|---|
| WCSR | is each second labelled right | nothing — this is the published-numbers metric |
| `chart` | is the printed chord order right | offset and tempo |
| `grid` | …and the harmonic rhythm, in bars | offset and tempo |
| `segmentation` | are the boundaries in the right places | nothing — labels ignored |
| `theory` | how close is a wrong chord, musically | nothing — same seconds as WCSR |

`theory` (`eval/substitution.py`) grades the *relation* between two chords — extension,
tritone sub, rootless dominant, relative, parallel — instead of comparing symbols. A tritone
sub and a ii written for its V miss every vocabulary including `mirex`, and a rootless
dominant is outside `majmin` and `sevenths`, so that time is dropped rather than scored. It
never touches WCSR: the partial credit lives in its own `GradedScore`, and the `substitutions`
vocabulary is the conservative pass/fail over the same table. **Its credits are musical
judgement, not measured frequencies** — read the per-relation breakdown `score` prints before
believing the single number, and don't re-fit the credits until several lead sheets per tune
exist.

`grid` is the one that distinguishes one chord per bar from two, which is most of what makes
an instrumental chart readable. Its harmonic unit is **estimated from chord durations**, so
it returns `nan` on material whose chords don't fall on a recoverable grid, and its
reliability gate is calibrated on one track — treat it as indicative. `grid_sequence` takes
a known `unit=`, which is exact; supplying real downbeats (roadmap) is what would make it a
measurement rather than a heuristic.

For songs, `metrics.placement_score` measures lyric-chord synchronisation — whether each
chord sits over the right syllable. It compares `(line, word index, label)` sequences via
`align.py`'s own placement logic, so it works in **word indices, never seconds**: immune to
offset and tempo, and testable with invented placeholder words, which is how copyrighted
lyrics stay out of the repo. `compare_labels=False` asks the weaker question (right syllable
at all), separating a mistimed chord from a misnamed one. Both sides are placed against the
*reference's* word timeline, so it measures the chord branch without ASR error leaking in.

It is **not** wired into `songscribe-eval`: no corpus here has lyrics. It needs time-aligned
words *and* chords for one recording — a self-recorded track, or CC audio such as MTG
Jamendo Lyrics with chords annotated by hand.

**Known `align.py` bug it surfaced:** a chord spilling past a line break is placed at the new
line's start, and `place_in_line`'s "keep the first change on a syllable" rule then discards
the chord that belongs there. 0.2s of drift triggers it. Pinned by a test in
`tests/test_eval_placement.py`; the fix is an `align-render` call, not an evaluation one.

## Sub-agents

Specialised agent definitions live in `.claude/agents/`. Use them for work that falls
squarely in one stage — they carry that stage's context so you don't have to rebuild it:

| Agent | Use for |
|---|---|
| `chord-model` | chord backends, DSP features, vocabularies, decoding |
| `lyrics-asr` | Whisper, timestamps, forced alignment, languages |
| `align-render` | line grouping, chord placement, output formats |
| `eval-harness` | metrics, reference datasets, benchmarking backends |
| `model-scout` | finding and licence-vetting candidate open-source models |
