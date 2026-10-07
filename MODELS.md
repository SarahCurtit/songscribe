# Models

songscribe is deep-learning based and uses **open-source models with openly published
weights only**. No hosted inference APIs, no proprietary or weights-unavailable checkpoints,
nothing that needs an account or an API key. Everything runs locally.

## The bar for adding a model

A model may be added only if all of these hold:

1. **Open-source code** under an OSI-approved licence (or equivalent).
2. **Openly downloadable weights** — a public URL, HuggingFace Hub, or bundled with the
   package. No gated or request-only checkpoints.
3. **Licence permits our use** and is recorded in the table below.
4. **Runs locally on CPU**, even if slowly. GPU may be an optimisation, never a requirement.
5. **Pinned and reproducible** — a specific version or commit, not "latest".

If a candidate fails any of these, it does not go in, however good the benchmark numbers are.

## Current inventory

| Stage | Model | Code licence | Weights | Notes |
|---|---|---|---|---|
| Separation | [Demucs v4](https://github.com/adefossez/demucs) (`htdemucs`) | MIT | Auto-downloaded on first use | Hybrid transformer; `--two-stems=vocals` is all we need |
| Chords (default) | [madmom](https://github.com/CPJKU/madmom) CNN + CRF | BSD-2 (+ academic clause) | Bundled with the package | Korzeniowski & Widmer; major/minor vocabulary |
| Chords (planned) | [BTC](https://github.com/jayg996/BTC-ISMIR19) | MIT | Public checkpoint in the repo | Large vocabulary incl. sevenths/inversions; not wired up yet |
| Lyrics | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (Whisper) | MIT (code), MIT (weights) | HuggingFace Hub, auto-downloaded | CTranslate2 reimplementation; word-level timestamps |

The `template` chord backend (chroma + triad templates) is **not** a model — it is the
non-DL baseline that any model we add has to beat, and the only backend that runs in CI
with no weights on disk.

## Licence note on madmom

madmom is BSD-2-Clause with an added clause asking for citation in academic work. That is
fine for an MIT-licensed project, but it is why madmom sits behind the `[chords]` extra
rather than in the core dependencies — installing songscribe shouldn't silently pull in a
differently-licensed dependency.

## Candidates under consideration

- **Chord recognition** — [Chord-Former](https://github.com/)-style CRNN/transformer models;
  anything trained with a large vocabulary on McGill Billboard + Isophonics.
- **Beat/downbeat tracking** — [BeatNet](https://github.com/mjhydri/BeatNet) (MIT) or
  madmom's DBN tracker, so chords snap to bars rather than raw frames.
- **Lyric alignment** — [WhisperX](https://github.com/m-bain/whisperX) (BSD-4) for
  phoneme-level forced alignment, which would fix the held-note timestamp drift.
- **Singing-specific ASR** — Whisper fine-tunes on singing voice, if one appears with open
  weights. Generic Whisper is the weakest link in the pipeline.
- **Structure/section detection** — self-similarity on top of an open embedding model, to
  label verse/chorus.

## Weights and caching

Model weights are downloaded on first use into each library's own cache
(`~/.cache/torch/hub`, `~/.cache/huggingface`). They are never committed here —
`.gitignore` blocks `models/`, `*.pt`, `*.ckpt`, and `*.onnx` so a stray checkpoint can't be
added by accident.
