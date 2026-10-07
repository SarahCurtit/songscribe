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
  types.py      ChordEvent, Word, LyricLine, Placement, Section, Song — all frozen
  audio.py      loading/resampling; caller asks for the rate its model wants
  separate.py   demucs wrapper (subprocess), optional
  chords.py     pluggable chord backends behind recognise()
  lyrics.py     faster-whisper wrapper, word-level timestamps
  align.py      chord timeline + word timeline -> printable sections   (pure Python)
  render.py     Song -> text or ChordPro                               (pure Python)
  pipeline.py   orchestration of the above
  cli.py        argparse entry point
  demo.py       synthesises a demo track + .lab ground truth  (pure Python)
  lab.py        .lab chord annotation read/write               (pure Python)
  eval/         chord accuracy measurement
    harte.py      chord labels -> pitch content                (pure Python)
    metrics.py    WCSR, vocabularies, segmentation             (pure Python)
    dataset.py    manifests and path resolution                (pure Python)
    isophonics.py manifest builder for an Isophonics tree
    runner.py     runs a backend, caches estimates, aggregates
    cli.py        songscribe-eval entry point
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
  `BACKEND_SAMPLE_RATES`. Nothing else in the pipeline learns its name.
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

songscribe-eval manifest ~/corpora/isophonics -a ~/music -o eval/isophonics.json
songscribe-eval score eval/isophonics.json -c madmom --cache .eval-cache --per-track
```

`--cache` writes each estimate as a `.lab`, so re-scoring under a different vocabulary is
instant and a long run is resumable. The scoring rules get iterated on far more often than
the model does. See `EVALUATION.md`.

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
