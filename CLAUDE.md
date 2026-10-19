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
    metrics.py    WCSR, vocabularies, segmentation, bar score  (pure Python)
    labels.py     chord labels -> one rough symbol per beat, and how well two agree (numpy)
    chart_fit.py  the chart's labels against an estimate from a start, with its chance level (numpy)
    recurrence.py the stretch of an estimate that repeats one form later: head, intro, pickup (numpy)
    form_start.py recurrence first, chart as check and fallback -> the form's start (numpy)
    form.py       splits a recording into choruses, on the chart's timeline (pure Python)
    README.md     how labels, recurrence, chart_fit, form_start and form fit together
    dataset.py    manifests and path resolution                (pure Python)
    isophonics.py manifest builder for an Isophonics tree
    survey.py     corpus label/coverage report, no audio or model needed
    reference.py  local word timelines for placement scoring (gitignored data)
    viz.py        shaping for plots — no plotting dependency      (pure Python)
    runner.py     runs a backend, caches estimates, aggregates
    cli.py        songscribe-eval entry point
eval/           committed manifests: track ids + relative paths, no content
notebooks/      annotations.ipynb — look at annotations and estimates
tests/          unit tests for the pure-Python stages
MODELS.md       model inventory, licences, and the bar for adding one
EVALUATION.md   how to measure a backend, and which corpora are usable
```

## Test material

There is no audio in this repo and there must never be. `songscribe-demo` (`demo.py`)
generates a synthetic track with known chords plus a `.lab` annotation, which is how the
chord branch gets measured without shipping a recording. It is **instrumental on purpose** —
don't try to make it sing; that is what a real recording is for.

For a real accuracy number, `songscribe-eval` scores a backend against reference
annotations. **`EVALUATION.md` records which corpora are usable and which are not, and
why** — Ultimate Guitar and ripped streaming audio are both settled no, so don't re-open
them. The rule the harness is built around: manifests hold track ids and relative paths
only, so the evaluation code is publishable while the corpus stays local. Never commit
annotations, estimates, or a manifest with absolute paths in it.

Real audio goes in the gitignored `audio/` directory and is never committed, whatever its
licence. For anything claiming an accuracy number, prefer a recording where the chords and
words are known in advance; CC0 sources are fine for sanity checks but come with no ground
truth. "Royalty-free" stock libraries are **not** usable here — they're licensed, not free,
and their terms forbid redistributing the audio.

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

# run from the repo root; $PWD makes the path absolute, so it works from any directory
export SONGSCRIBE_EVAL_ANNOTATIONS="$PWD/data/isophonics"    # the directory holding chordlab/
export SONGSCRIBE_EVAL_ROOT=~/music                          # your recordings
songscribe-eval survey eval/isophonics-beatles.json          # no audio, no model
songscribe-eval manifest data/isophonics -a ~/music -o eval/isophonics.json
songscribe-eval score eval/isophonics.json -c madmom --cache .eval-cache --per-track
```

A recording of the chart played several times (manifest track with `"bars"`) is scored per
chorus with `score --beats DIR --per-chorus`. Its form starts where a chord estimate of the
recording repeats one form later: from `--anchor-backend` (default madmom, run on the full
mix and cached under `--cache`, so a later scoring run reuses it) or from `--anchor-estimates
DIR`. Anchoring on the backend being scored warns, since the start found can be the one that
flatters it. `songscribe-eval form` prints the grid, anchor and choruses, and exits 2 if the
start is not clear. The key is searched by default: one key is assumed for the whole take and
every chorus is scored against the chart transposed to it (`--transpose none` turns this
off). See `EVALUATION.md`.

`survey` is the corpus health check: it reports annotated time and whether every label
parses, and **exits non-zero if any label doesn't** — an unparseable label is a gap in
`eval/harte.py`, not bad data. Run it on a new corpus before anything else.

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

The Beatles annotations live in the gitignored `data/isophonics/` (extract the Isophonics
archive there, so `chordlab/` sits directly under it); `eval/isophonics-beatles.json` is the
committed manifest over them. Its `reference` paths start at `chordlab/`, so the annotation
root — `SONGSCRIBE_EVAL_ANNOTATIONS`, as above, or `--annotations` — must be the directory
*containing* `chordlab/`; unset, it falls back to `eval/` and finds nothing. The manifest is
**references-only** — no `audio` paths —
so it serves `survey` and `score --estimates` (both need that root), and `score -c` on it
scores nothing. A `-c` run needs a manifest built locally with `manifest -a`, as above; its
audio paths follow your own library's filenames, so it stays uncommitted. Its paths are
relative to the two directories it was built from, so `score` needs both roots pointing
back at them — the two exports above, or `-a` and `--annotations` — or it finds nothing.

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

Which metric to believe depends on the question, and `songscribe-eval` reports four because
they disagree sharply (see `EVALUATION.md`):

| Metric | Asks | Robust to |
|---|---|---|
| WCSR | is each second labelled right | nothing — this is the published-numbers metric |
| `chart` | is the printed chord order right | offset and tempo |
| `grid` | …and the harmonic rhythm, in bars | offset and tempo |
| `segmentation` | are the boundaries in the right places | nothing — labels ignored |

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
