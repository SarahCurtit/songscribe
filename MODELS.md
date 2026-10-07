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
4. **PyTorch, or no framework at all.** See below — this one rules out otherwise strong
   candidates.
5. **Runs locally on CPU**, even if slowly. GPU may be an optimisation, never a requirement.
6. **Pinned and reproducible** — a specific version or commit, not "latest".

If a candidate fails any of these, it does not go in, however good the benchmark numbers are.

## Framework: PyTorch

**This project is PyTorch-based. Do not add a TensorFlow or Keras dependency.** A model that
needs TF is rejected regardless of its accuracy — one deep-learning runtime is enough weight
for a project this size, and mixing two means two sets of version pins, two CPU thread pools,
and roughly a gigabyte of wheels for no user-visible gain.

Framework-free models (plain numpy/scipy inference) are also fine, and are in some ways
preferable — they add no runtime at all. The default chord backend is one of these.

ONNX Runtime is acceptable for a model shipped as `.onnx` weights, since it brings no
training framework with it.

## Current inventory

| Stage | Model | Framework | Code licence | Weights | Verified | Notes |
|---|---|---|---|---|---|---|
| Separation | [Demucs v4](https://github.com/adefossez/demucs) (`htdemucs`) | PyTorch | MIT | Auto-downloaded on first use | not yet | Hybrid transformer; `--two-stems=vocals` is all we need |
| Chords (default) | [madmom](https://github.com/CPJKU/madmom) CNN + CRF | none (numpy) | BSD-2 (+ academic clause) | Bundled with the package | **yes** — 0.17.dev0, Py3.11 | Korzeniowski & Widmer; major/minor vocabulary |
| Chords (planned) | [BTC](https://github.com/jayg996/BTC-ISMIR19) | PyTorch | MIT | Public checkpoint in the repo | n/a | Large vocabulary incl. sevenths/inversions; not wired up yet |
| Lyrics | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (Whisper) | CTranslate2 | MIT (code), MIT (weights) | HuggingFace Hub, auto-downloaded | not yet | Word-level timestamps; see the note below |

Two entries need a word of explanation against the PyTorch policy:

- **madmom** carries no framework at all. Its CNN and CRF inference is hand-rolled over
  numpy/scipy (`madmom/ml/nn`, plus a Cython HMM); its only dependencies are `numpy`,
  `scipy`, and `mido`. It satisfies the policy by adding no runtime, not by being PyTorch.
- **faster-whisper** runs on CTranslate2, a standalone C++ inference engine — neither
  TensorFlow nor PyTorch. It is here because it is markedly faster on CPU than
  `openai-whisper`, which matters for a 4-minute track. If strict single-runtime purity is
  wanted later, `openai-whisper` or HuggingFace `transformers` Whisper are the PyTorch
  equivalents, at a real speed cost. **Open decision**, not an oversight.

## Installing madmom (the awkward one)

**Install it from git, not PyPI.** The last PyPI release (0.16.1, 2018) cannot import on any
Python this project supports — it does `from collections import MutableSequence`, which 3.10
removed. Upstream `main` fixed that and has proper PEP 518 build metadata, so a plain install
works:

```bash
pip install 'songscribe[chords]'   # resolves to the pinned git commit
```

Verified on this machine, Python 3.11.9 / numpy 2.4.6: builds from source with build
isolation, imports cleanly, and needs no `setuptools` pin (upstream dropped `pkg_resources`).
It compiles Cython on install, so budget ~30s.

Dead ends, recorded so nobody retries them:

- `pip install madmom` (0.16.1) — fails at build, because `setup.py` imports Cython without
  declaring it. Forcing past that with `--no-build-isolation` then fails at import on
  `MutableSequence`, and before that on `pkg_resources` with `setuptools>=81`.

### Label vocabulary

madmom emits **Harte notation** (`C:maj`, `A:min`, `N`), not the compact shorthand a chart
wants (`C`, `Am`). `chords.normalise_label()` converts at the backend boundary, so every
backend hands the same vocabulary downstream. Any new backend must normalise on the way out —
an unrecognised quality passes through untouched rather than being guessed at.

The `template` chord backend (chroma + triad templates) is **not** a model — it is the
non-DL baseline that any model we add has to beat, and the only backend that runs in CI
with no weights on disk.

## Licence note on madmom

madmom is BSD-2-Clause with an added clause asking for citation in academic work. That is
fine for an MIT-licensed project, but it is why madmom sits behind the `[chords]` extra
rather than in the core dependencies — installing songscribe shouldn't silently pull in a
differently-licensed dependency.

## Candidates under consideration

All PyTorch or framework-free, per the policy above.

- **Chord recognition** — [BTC](https://github.com/jayg996/BTC-ISMIR19) (PyTorch, MIT) is the
  chosen upgrade and is already stubbed in `chords.py`; it needs vendoring, since there is no
  pip package. Any other candidate must be a PyTorch CRNN/transformer trained with a large
  vocabulary on McGill Billboard + Isophonics.
- **Beat/downbeat tracking** — [BeatNet](https://github.com/mjhydri/BeatNet) (PyTorch, MIT),
  or madmom's DBN tracker (framework-free), so chords snap to bars rather than raw frames.
- **Lyric alignment** — [WhisperX](https://github.com/m-bain/whisperX) (PyTorch/torchaudio,
  BSD-4) for phoneme-level forced alignment, which would fix the held-note timestamp drift.
  Check the BSD-4 advertising clause against our MIT licence before adopting.
- **Singing-specific ASR** — a Whisper fine-tune on singing voice, if one appears with open
  weights. Generic Whisper is the weakest link in the pipeline.
- **Structure/section detection** — self-similarity on top of an open PyTorch embedding
  model, to label verse/chorus.

## Rejected

Recorded so they don't get re-litigated. All of these are otherwise reasonable — they fail
on framework or on packaging, not on quality.

| Candidate | Why rejected |
|---|---|
| [crema](https://github.com/bmcfee/crema) | **TensorFlow/Keras.** Genuinely tempting otherwise: large-vocabulary chord model (McFee & Bello, ISMIR'17), MIT, pip-installable with bundled weights. Fails the framework policy, and its pre-Keras-3 vintage would likely break on current TF anyway. |
| [autochord](https://github.com/cjbayron/autochord) | **TensorFlow** (tflite). Also needs the NNLS-Chroma Vamp plugin installed system-wide. |
| [basic-pitch](https://github.com/spotify/basic-pitch) | **TensorFlow.** Also note transcription, not chords — it would need a chord-labelling layer on top. An ONNX build would clear the framework bar if one is maintained. |
| [omnizart](https://github.com/Music-and-Culture-Technology-Lab/omnizart) | **TensorFlow**, with old pinned deps. |
| [Chordino](https://github.com/ohollo/chord-extractor) | Not deep learning (NNLS-Chroma, C++ Vamp plugin), and awkward to install. The `template` backend already fills the "simple baseline" slot. |
| madmom 0.16.1 from PyPI | Cannot import on Python 3.10+. See the install section above — use the git pin. |

## Weights and caching

Model weights are downloaded on first use into each library's own cache
(`~/.cache/torch/hub`, `~/.cache/huggingface`). They are never committed here —
`.gitignore` blocks `models/`, `*.pt`, `*.ckpt`, and `*.onnx` so a stray checkpoint can't be
added by accident.
