# Finding the form in a recording

A recording is usually an intro, the head, choruses that repeat its changes, and an ending.
To score it against a one-chorus chart, the harness finds the bar where the first chorus
starts and cuts the recording into choruses. These modules do it; none needs model weights.

```
  chord estimate + beat grid            chart (one chorus) + its bar count
            |                                      |
   labels: one symbol per beat            chart_fit: chart as symbols
            |                                      |
   recurrence: stretch that repeats               |
   one form later                                 |
            |                                      |
            +------------> form_start.locate <-----+
                           recurrence found: chart_fit picks the bar of the form
                           none found:       chart_fit alone
                                  |
                           Anchor + the grid it indexes
                                  |
                      form: intro / head / chorus / partial / tail
                                  |
                      runner.score_performance: metrics per chorus
```

## Modules

`labels` reduces a chord label to a symbol (root, minor or not third), makes one symbol per
beat from a chord timeline, and scores two symbol sequences: 1 for equal, 0.5 for the same
root with the other third, 0 otherwise.

`recurrence` takes the symbols, the beat grid and the form length in bars. It compares each
bar with the bar one form later and returns a `Recurrence`: the stretch that matches (where
the head or its pickup opens, where the recurring music ends) and how it compares with the
other lags, which serve as the null. `is_found` says whether it counts. `bar_line_phase`
checks whether the chords change on the tracker's downbeat.

`chart_fit` cuts the chart into equal bars and beats and compares it with the estimate from
each candidate start, in each key asked for. It also returns `chance`, the best fit of
bar-shuffled charts over the same starts and keys.

`form_start.locate` combines them and returns an `Anchor` and the grid its `bar_index`
refers to. `Anchor.refusal()` gives the reason a start is unreliable, or `None`.

`form.choruses` cuts the grid into blocks of the form length from the anchor bar. Blocks with
irregular bars are `unaligned`; a short last block is `partial` or `tail`. Scored blocks are
rescaled onto the chart's timeline with `project_estimate`.

## Order and fallback

1. The estimate and the beat grid become one symbol per beat.
2. `recurrence.measure` looks for a stretch that repeats one form later.
3. If it finds one, its opening is the start. The chart can move it to another bar of the
   form if that fits clearly better, and bar lines move by whole beats if the chords change
   on another beat than the tracker's downbeat.
4. If nothing recurs, the start is the downbeat where the chart alone fits best. It must
   exceed `chance` and lead the other positions in the form, or the anchor is refused. Intro
   and pickup are not measured.

## What the chart contributes

- The form length: its bar count is the lag recurrence is measured at.
- A check: a start only counts if the chart fits there better than a shuffled chart does.
- The key: the searched shift of the chart that fits best at the start.
- The head's bar, when the recurrence opens on another bar of the form.

## The anchoring estimate

The start is found on the full-mix estimate of `--anchor-backend` (default
`chords.DEFAULT_BACKEND`) or on `.lab` files from `--anchor-estimates`. A backend's estimate is
cached under `--cache`, so scoring the same backend later reuses it. Use a different backend
from the one being scored; `score` warns when they match. `songscribe-eval form` prints the
grid, anchor and choruses for one recording without scoring.
