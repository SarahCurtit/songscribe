# songscribe

Transcribe **chords and lyrics** from an audio file into a lead sheet you can actually play
from. Deep learning end to end, using **open-source models only** — everything runs locally,
no API keys, no hosted inference.

Point it at a song, get back chord labels sitting above the words where the changes happen:

```
$ songscribe your-song.mp3
```

```
Your Song
=========
key: G

G                 D
lantern over water, counting
Em            C
every window lit tonight
```

Or as [ChordPro](https://www.chordpro.org/), which opens in most setlist and songbook apps:

```
$ songscribe your-song.mp3 --format chordpro --output song.cho
```

```
{title: Your Song}
{key: G}

[G]lantern over [D]water, counting
[Em]every window [C]lit tonight
```

*Illustrative output shape — the words are invented placeholders, and you supply the audio.
No recordings ship with songscribe. For something runnable right now, see
[Try it with no audio](#try-it-with-no-audio).*

## Install

```bash
pip install 'songscribe[all]'      # everything: chords, lyrics, separation
pip install 'songscribe[chords]'   # chord recognition only
pip install 'songscribe[lyrics]'   # lyric transcription only
```

Each extra pulls one pretrained open-source model — see [MODELS.md](MODELS.md) for the full
inventory and licences. Weights download themselves on first run. `ffmpeg` needs to be on
your PATH for anything other than WAV input.

> **Note:** `[chords]` installs madmom from a pinned git commit, because its last PyPI
> release (2018) can't import on Python 3.10+. That makes the extra ineligible for PyPI
> upload, so for now install songscribe from a source checkout. It compiles Cython on
> install — budget ~30s. Details and the dead ends are in [MODELS.md](MODELS.md).

## Usage

```bash
songscribe song.mp3                        # text lead sheet to stdout
songscribe song.mp3 -f chordpro -o out.cho # ChordPro to a file
songscribe song.mp3 --no-separate          # skip demucs: much faster, less accurate
songscribe song.mp3 --no-lyrics            # chords only, loads no ASR model
songscribe song.mp3 -c template            # non-DL baseline, no weights needed
songscribe song.mp3 -m medium -l en        # bigger ASR model, force language
songscribe --help
```

As a library, each stage is independently usable:

```python
from songscribe import audio, chords, lyrics, align, render

samples, sr = audio.load("song.mp3", sample_rate=44_100)
chord_events = chords.recognise(samples, sr, backend="madmom")
words = lyrics.transcribe("song.mp3")
print(render.render(align.align(chord_events, words), "chordpro"))
```

## Try it with no audio

No recordings ship with songscribe — they're copyrighted and would bloat the repo. Generate
one instead:

```bash
pip install -e '.[chords]'                      # chord model; no ASR needed
songscribe-demo --output demo.wav               # 20s of I-V-vi-IV, plus demo.lab truth
songscribe demo.wav --no-lyrics --no-separate   # chords-only chart
```

```
demo
====
key: C

C G Am F C G Am F
```

`--no-lyrics` matters here: the demo is instrumental, and without it songscribe would load
an ASR model to find words that aren't there — and fail if the `[lyrics]` extra isn't
installed.

`songscribe-demo` writes a synthetic track by additive synthesis, alongside a `.lab`
annotation in the Isophonics/MIREX format `mir_eval` reads — so it's an evaluation target,
not just a smoke test. You know the right answer, which is the whole point.

```bash
songscribe-demo -p C Am F G --bpm 120 -r 4   # your own progression
songscribe-demo --noise 0.01                 # less pristine, harder to track
songscribe-demo --help
```

On the default track the madmom backend scores **8/8 chords, 98% frame accuracy** (the 2% is
the final bar fading under its envelope, correctly read as no-chord).

**It's instrumental by design.** Synthesising singing is out of scope, so this exercises the
chord branch only — Whisper will correctly find no words in it. For the lyric branch you need
real singing: your own recording is ideal, since you know both the chords you played and the
words you sang. Otherwise use CC0 audio, and keep it in the gitignored `audio/` directory.

## Measuring it

`songscribe-eval` scores a chord backend against reference annotations, using the MIREX
measure — Weighted Chord Symbol Recall, the fraction of annotated *time* labelled correctly.

```bash
songscribe-eval manifest ~/corpora/isophonics --audio ~/music -o eval/isophonics.json
songscribe-eval score eval/isophonics.json -c madmom --cache .eval-cache --per-track
```

```
vocabulary        mean  weighted   definition
root             92.3%     92.3%   root only; ignores quality entirely
majmin           92.3%     92.3%   major and minor triads; sevenths reduce to their triad
sevenths          0.0%      0.0%   maj, min, 7, maj7, min7
mirex            99.7%     99.7%   at least 3 pitch classes in common
```

Several vocabularies, because the gap between them is the diagnostic. A high `majmin` beside
a near-zero `sevenths` is the vocabulary ceiling, not a broken model — the backend cannot
emit a seventh at all, so it guesses the triad every time, and `mirex` stays high because a
triad shares three pitch classes with its parent seventh.

Manifests hold track identifiers and relative paths — **no audio, no annotation text** — so
the harness is shareable while the corpus stays yours. [EVALUATION.md](EVALUATION.md) covers
setup, how the metrics work, and which corpora can and cannot legitimately be used.

## How it works

```
audio ──┬─(demucs)─> vocals ──> whisper ──────> timed words ──┐
        │                                                     ├─> align ─> render
        └─(demucs)─> backing ──> CNN+CRF ────> chord events ──┘
```

Three models and one piece of logic, all talking in seconds-from-start so every stage stays
swappable:

1. **Separation** (`separate.py`) — Demucs v4 splits the mix so each downstream model hears
   only what it needs. Slow, optional, and the single biggest quality win.
2. **Chord recognition** (`chords.py`) — pluggable backends behind one function. Default is
   madmom's CNN chord features with CRF decoding; BTC (transformer, larger vocabulary) is
   the planned upgrade; `template` is a chroma baseline kept only for comparison and CI.
3. **Lyric transcription** (`lyrics.py`) — faster-whisper with word-level timestamps. The
   timestamps are the point; a transcript without them can't be aligned to chords.
4. **Alignment and rendering** (`align.py`, `render.py`) — pure Python, no model. Words are
   grouped into lines on breath-length pauses, each chord is pinned to the syllable being
   sung when it changes, and gaps over 2s become instrumental breaks.

Adding a chord model means writing one function returning `list[ChordEvent]` and registering
it in `chords.BACKENDS`. Nothing else in the pipeline needs to know.

## What to expect

It is honest about being v0. Realistically today:

- **Good** on sparse recordings with clear diatonic triads and intelligible vocals.
- **Shaky** on dense mixes, key changes, and anything where backing vocals fight the lead.
- **Limited vocabulary** — the default backend knows major and minor triads, so sevenths and
  extensions get rounded to their nearest triad. That is what the BTC backend is for.
- **Timestamps drift** on long held notes, which pushes chord labels a word too far right.
  Forced alignment (WhisperX) is the planned fix.

Treat the output as a first draft to correct, not a finished chart.

## Roadmap

- [ ] Wire up the BTC backend (sevenths, inversions, no-chord)
- [ ] Beat and downbeat tracking, so chords snap to bars instead of raw frames
- [ ] Phoneme-level forced alignment to fix held-note drift
- [x] Evaluation harness against reference annotations (Isophonics; McGill Billboard next)
- [ ] Section detection (verse/chorus) from repetition structure
- [ ] Forced alignment of known lyrics, for when you have the words but not the timings
- [ ] Capo and transposition options
- [ ] MusicXML / MIDI export

## Contributing

Issues and PRs welcome.

```bash
git clone git@github.com:SarahCurtit/songscribe.git
cd songscribe
pip install -e '.[dev]'
pytest
ruff check .
```

Any new model has to clear the bar in [MODELS.md](MODELS.md): open code, open weights,
runs locally on CPU, licence recorded.

The unit tests deliberately avoid audio fixtures and model weights — the pure-Python stages
(alignment, rendering, chord templates) are tested directly so the suite stays fast and the
repo stays small.

## A note on copyright

songscribe analyses audio you already have and produces your own transcription of it. Chord
progressions aren't copyrightable, but **lyrics are** — so transcripts of commercial songs
are for your personal use. Don't commit song audio or generated lyric transcripts to this
repo; `.gitignore` is set up to help you not do that by accident.

The same applies to evaluation corpora, where the constraint is lawful access to the copy
you analyse — your own rips, or a corpus licensed to you. `EVALUATION.md` has the detail,
including why tabs scraped from tab sites and audio ripped from streaming services are not
usable here.

## License

MIT — see [LICENSE](LICENSE).
