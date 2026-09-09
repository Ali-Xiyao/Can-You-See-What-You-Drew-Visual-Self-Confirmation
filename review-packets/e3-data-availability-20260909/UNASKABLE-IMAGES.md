# The images that support no question are exactly the failures

New file, 2026-09-09. `NOTES.md`, `availability.py`, `output.txt` and
`CORRECTION-checkpoint-count.md` are left exactly as they were.

Found while dry-running `scripts/v4_e3_selection_observe.py` against the main
run, before spending any GPU time on it.

## What was found

The selection pass drops an image that supports no question, which is
`scripts/v4_run_pipeline.py`'s `stage_observe` rule and not a new one. On the
main run's first three checkpoints that is 8 to 14 of 256 candidates per
arm-checkpoint, and **every one of them is externally incorrect**:

```
naive     step  0: askable 242 p=0.302 | unaskable 14 p=0.000
naive     step  8: askable 245 p=0.310 | unaskable 11 p=0.000
naive     step 16: askable 243 p=0.309 | unaskable 13 p=0.000
rfo_gold  step  0: askable 242 p=0.302 | unaskable 14 p=0.000
rfo_gold  step  8: askable 246 p=0.305 | unaskable 10 p=0.000
rfo_gold  step 16: askable 247 p=0.300 | unaskable  8 p=0.000
```

`p = 0.000` is not a small number, it is zero out of 70 across the six
arm-checkpoints. It is also not a coincidence: `build_counting` returns None
when none of the requested categories was detected at all, and the other three
builders need a detected object too. "None of the requested categories is in
the picture" is also the definition of a miss. So the exclusion is
**deterministically confounded with the outcome** — it can only ever remove
candidates that were going to be wrong.

## Why it matters, and how much

Endpoint 3's y axis is `pick(pool, "blind") − pick(pool, "prompted")`, where
`pick` is the external correctness of the highest-scoring candidate. Removing
all-wrong candidates makes the pool easier: fewer wrong answers to be tempted
by, and a higher ceiling for both conditions.

Two things keep it from being a threat to the endpoint, and one thing keeps it
from being nothing:

1. **It cancels between the conditions by construction.** The question set is
   built from the image and the detections, not from the condition, so blind
   and prompted see the same pool at every checkpoint. Endpoint 3's y is a
   difference *within* one pool.
2. **It is small.** Pool sizes over those six arm-checkpoints:

   | arm | step | pool sizes | prompts kept | ceiling kept | ceiling all |
   |---|---|---|---|---|---|
   | naive | 0 | 1×2, 2×2, 3×4, 4×56 | 62/64 | 0.645 | 0.625 |
   | naive | 8 | 1×1, 2×1, 3×6, 4×56 | 63/64 | 0.667 | 0.656 |
   | naive | 16 | 1×3, 3×4, 4×57 | 61/64 | 0.705 | 0.672 |
   | rfo_gold | 0 | 1×2, 2×2, 3×4, 4×56 | 62/64 | 0.645 | 0.625 |
   | rfo_gold | 8 | 1×1, 2×1, 3×5, 4×57 | 63/64 | 0.667 | 0.656 |
   | rfo_gold | 16 | 2×1, 3×7, 4×56 | 64/64 | 0.672 | 0.672 |

   At most 3 prompts of 64 leave, and the ceiling moves by at most +0.033.
3. **The two arms are not affected equally.** At step 16 naive loses 3 prompts
   and `rfo_gold` loses 0, because the arm that draws better pictures has fewer
   catastrophic ones to exclude. That asymmetry is small here and there is no
   reason to assume it stays small over 88 steps.

## What was done about it

Nothing to the protocol. The exclusion is already what
`review-packets/bsv-selection-20260908/bsv_selection.py` did on the corpus
runs — the +0.190 and +0.034 selection gains deviation 13.2 pins the 口径 to
were computed with it — so removing it now would be a new 口径, not a fix.

What was added is a measurement, at zero GPU cost.
`scripts/v4_e3_selection_observe.py` now records four counts per
arm-checkpoint in its manifest:

```
prompts                  every prompt with at least one adjudicated candidate
pools_kept               those with >= MIN_CANDIDATES askable candidates
pools_kept_with_correct  of those, how many contain a correct candidate
prompts_with_correct     over every adjudicated candidate, excluded ones too
```

so the ceiling with and against the exclusion can be read off all 120
arm-checkpoints rather than off the three I looked at by hand. The last of the
four is deliberately counted over the candidates that left: that the excluded
ones are always wrong is a finding about this run, and a counter that assumed
it would report a cost of zero by construction. A test pins that
(`test_the_ceiling_is_counted_over_the_candidates_that_left_as_well`), and a
second pins the writer's definition of "kept" to the reader's `MIN_CANDIDATES`.

## What the paper has to say

That the selection task is scored on the candidates a question could be asked
about, that those exclude 3–5% of drawings and that all of the excluded ones
had failed, and the measured size of the resulting ceiling shift. Not that it
is negligible — the numbers above are three checkpoints of 120, and the run
will produce the rest.

## Reproducing

```
envs/core/python.exe scripts/v4_e3_selection_observe.py \
    --run runs/v4/decoupling-main-20260908 \
    --config configs/v4_decoupling_main_20260908.yaml \
    --arms naive rfo_gold --dry-run
```

The `unaskable` column is the count; the per-checkpoint breakdown above came
from a throwaway script over the same two files the pass reads
(`evaluations/<arm>/step-NNNNN/{manifest,verified}.jsonl`), and is what the
manifest counts now record without one.
