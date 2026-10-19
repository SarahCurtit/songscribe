# Stage 2 by example: anchor the form, segment the choruses, score each

Worked runs of `songscribe-demo` and `songscribe-eval`, output captured verbatim on
2026-10-19. They need no audio files and no weights: the audio is synthesised, and the
`template` chord backend and the supplied bar lines are the only machinery. Background and
the rules behind each piece are in [EVALUATION.md](../EVALUATION.md#a-chart-played-several-times-form-bar-and---beats).

**These are synthetic takes.** The anchor thresholds (`ANCHOR_MIN_AGREEMENT`,
`ANCHOR_MIN_MARGIN`) have not been calibrated on a real recording, and the pristine bar
lines here are the true ones, which a tracker will not give. Numbers from a real recording
belong in a local, uncommitted run.

## 0. The take

Four times through `C G Am F`, 3.5 s of noise before bar 1, a 3% tempo drift over the take,
and two choruses that stray from the chart (chorus 1 plays `Dm` in bar 3, chorus 2 plays `E`
in bar 1).

```bash
songscribe-demo -o take.wav -r 4 --lead-in-seconds 3.5 --tempo-drift 0.03 \
    --vary 1:3=Dm --vary 2:1=E --noise 0.01
```

```
wrote take.wav (42.9s)
wrote take.lab              # what was really played, on the recording's clock
wrote take.reference.lab    # the chart: one chorus from 0 s
wrote take.beats            # the true bar lines
```

The manifest (`manifest.json`, next to the files) names the chart and its length in bars:

```json
{"version": 1, "dataset": "synthetic-take",
 "tracks": [{"id": "take", "reference": "take.reference.lab", "audio": "take.wav", "bars": 4}]}
```

## 1. `form`: what was found, before any score

```bash
songscribe-eval form take.reference.lab --bars 4 --audio take.wav --beats take.beats
```

```
bar grid  16 bars, median 2.463 s (97.4 bpm in 4 beats)
          0 bars far from the median length
anchor    bar 0 at 3.50 s   agreement 0.89  runner-up 0.03  margin 0.86

  # kind       first bar  bars     start       end   scale
  0 intro              0     0     0.00s     3.50s      --
  1 head               0     4     3.50s    13.47s   0.997
  2 chorus             4     4    13.47s    23.36s   0.989
  3 chorus             8     4    23.36s    33.17s   0.981
  4 head              12     4    33.17s    42.91s   0.974

3.5 s in blocks no score covers (intro, ending, unaligned)
```

The form starts at 3.50 s, exactly where the lead-in ends, with a wide margin over the
runner-up phase. `scale` falls from 0.997 to 0.974 as the take slows. The last chorus is
labelled `head` because the first and last full choruses are the head by default (head in,
head out), even though it is the fourth pass.

## 2. Refusal: the start is not clear

The same recording against a chart in the wrong key (`Eb Bb Cm Ab`):

```bash
songscribe-demo -o wrong.wav -p Eb Bb Cm Ab -r 1 --lead-in-seconds 3.5   # only for wrong.reference.lab
songscribe-eval form wrong.reference.lab --bars 4 --audio take.wav --beats take.beats
echo $?
```

```
bar grid  16 bars, median 2.463 s (97.4 bpm in 4 beats)
          0 bars far from the median length
anchor    bar 6 at 18.43 s   agreement 0.30  runner-up 0.11  margin 0.18
          REFUSED: agreement 0.30 under 0.50
          no choruses are drawn from a guess
2
```

Exit status 2, with the gate that failed. `score --beats` skips such a track and names it
rather than scoring against a guess.

## 3. `score --per-chorus`

```bash
songscribe-eval score manifest.json -c template --beats . --per-chorus -q
```

```
track                                                     root    majmin  sevenths     mirex       seg       bar
take #1 head                                            98.75%    98.75%    98.75%    98.75%    98.75%   100.00%
take #2 chorus                                          74.41%    74.41%    74.41%    74.41%    98.83%    75.00%
take #3 chorus                                          73.99%    73.99%    73.99%    73.99%    98.94%    75.00%
take #4 head                                            98.85%    98.85%    98.85%    98.85%    98.85%   100.00%

chart           87.50%             chord order as printed, timing ignored
grid            87.50%             order + harmonic rhythm in bars, tempo-free
segmentation    98.84%             boundary agreement, labels ignored
bar             87.50%             right chord in the right bar, tempo- and offset-free

4 choruses in 1 tracks, each scored on
its own; the track rows aggregate the full choruses. Intro, ending and unaligned
choruses are left out (4 s).
```

(Abridged: the header and the `mean`/`weighted` vocabulary table are omitted.) The estimate
is nearly perfect for every chorus (`seg` about 98.8% throughout); the two choruses at 75%
`bar` are the two where the performer left the chart, one bar in four. The per-chorus rows
separate "the model erred" from "the player varied", which one whole-track number cannot.
The 3.5 s intro is reported as excluded, not counted as error.

### An ending the tracker lost

Cutting the bar lines after bar 14 (a tracker that gave up early) leaves a two-bar remainder:

```bash
songscribe-eval score manifest.json -c template --beats short/ --per-chorus -q
```

```
take #1 head                                            98.75%    98.75%    98.75%    98.75%    98.75%   100.00%
take #2 chorus                                          74.41%    74.41%    74.41%    74.41%    98.83%    75.00%
take #3 head                                            73.99%    73.99%    73.99%    73.99%    98.94%    75.00%

3 choruses in 1 tracks, each scored on
its own; the track rows aggregate the full choruses. Intro, ending and unaligned
choruses are left out (8 s).
```

The two-bar remainder is a `tail` and is left out (8 s unscored: intro plus ending). Note
that chorus 3 is now labelled `head`, since it is the last full chorus the grid contains:
labels follow the detected form, not the performer's intent.

## 4. Before and after

The same take, `template` backend, scored with no form knowledge (no `--beats`):

```bash
songscribe-eval score manifest.json -c template -q
```

```
root             2.79%     2.79%   root only; ignores quality entirely
majmin           1.86%     1.86%   major and minor triads; sevenths reduce to their triad
chart           18.18%             chord order as printed, timing ignored
grid            18.18%             order + harmonic rhythm in bars, tempo-free
segmentation    52.24%             boundary agreement, labels ignored

WARNING: 1 of 1 tracks score markedly better
at a time shift, which means the recording is not on the reference's timeline.

  take                                                -3.00s     1.86%   83.00%
```

|                    | whole track, no anchor | per chorus, anchored (section 3) |
|--------------------|-----------------------:|---------------------------------:|
| `majmin`           | 1.86%                  | 86.50%                           |
| `chart`            | 18.18%                 | 87.50%                           |
| `segmentation`     | 52.24%                 | 98.84%                           |

Without the anchor, the lead-in, the three extra choruses and the drift all count as model
error, and the harness can only suggest a shift (it warns, correctly, not to adopt it).
With it, the same estimate scores 86.5% `majmin`, and the remaining gap is the performer's
variations, visible per chorus.

## Viewer

There is no viewer in this repository or on any pushed branch, so there are no screenshots
here. Showing a multi-chorus take would need the viewer to draw the detected chorus blocks
(kind, first bar, scale) on its bar grid and the anchor's refusal state; see the plan's
"What the viewer needs for a real multi-chorus track".
