# The registered D* rule has never had a pair to test

Registered 2026-09-10, while `decoupling-main-20260908` was at round 5 of 11.
The frozen `decoupling_report.json` written at 05:04:40 is kept as it is. This
packet is a new measurement of the same artifacts, not a replacement.

## What the frozen report says

Every window, both arms, all four looks:

    "n_paired_specs": 0
    "n_start_complete": 0
    "n_end_complete": 0
    "external": {"delta": null, "ci_low": null, "ci_high": null}
    "ci_status": "insufficient_paired_specs"
    "candidate": false

`first_candidate_step: None` reads as a negative result. It is not one. The
primary registered analysis has never been evaluated, because it has never had
an input.

## Why

`read_outcome` keeps a spec only when every image in it carries both an
`s_select` score and a verdict (`scripts/v4_decoupling_report.py:165`). The
writer does not produce that shape and never did. At every checkpoint of both
arms:

    images = 256   specs = 64   images per spec = 4 (uniform)
    verified.jsonl  = 256 rows   every image adjudicated
    s_select.jsonl  =  64 rows   exactly one scored image per spec
    scored-images-per-spec distribution = {1: 64}
    fully-scored specs = 0

All 64 `s_select` paths are present in the manifest, so this is not a path
mismatch. The wiring is correct and the granularity is not: `s_select` scores
the selected candidate, one per spec; `verified` adjudicates all four. The
reader was written against the shape it expected rather than the shape the
writer emits, which is the same defect family as sections 3.4 and 3.8.

Nothing downstream can recover from it. `complete_specs` is empty, so
`paired_changes` returns `n_paired_specs = 0`, `external.delta` is `null`, and
`candidate` is `false` by short-circuit. The remaining six rounds would not have
changed this, and the five replicates would have inherited it.

## The repair, and what it does not change

One predicate. A spec counts when at least one of its images is scored and all
of its images are adjudicated.

    s_select = mean over the images that are scored   (was: mean over all four)
    external = mean over all images                   (unchanged)

`reanalyze_dstar_join.py` imports the frozen module and drives it. Every
estimator, threshold, bootstrap count and seed is the frozen module's own;
`read_outcome` is called for real first, so its duplicate-image and spec-id
mismatch checks still run, and only `complete_specs` is rebuilt afterwards.
Nothing in the run directory is written.

Validated against a number the run had already published: per-spec `s_select`
averaged over the repaired complete set lands within 0.01 of the
`checkpoint_metrics.csv` column at all twelve checkpoints, the residual being
that the CSV averages over all 64 specs and the repair over the 44-49 complete
ones.

## What the rule does once it can run

    bootstrap_rule_supported = candidate
                               and internal.ci_low > 1e-12
                               and external.ci_high <= 0.02
    candidate                = internal.delta >= 0.02 and external.delta <= 0.02

| arm | end | n | int delta | int ci_low | ext delta | ext ci_high | cand | sup |
|---|---|---|---|---|---|---|---|---|
| naive | 16 | 48 | +0.0208 | -0.0243 | +0.0156 | +0.0417 | yes | no |
| naive | 24 | 45 | +0.0056 | -0.0407 | +0.0056 | +0.0333 | no | no |
| naive | 32 | 49 | +0.0170 | -0.0102 | +0.0051 | +0.0306 | no | no |
| naive | 40 | 44 | +0.0398 | +5.05e-18 | -0.0057 | +0.0114 | yes | no |
| rfo_gold | 16 | 45 | +0.0222 | -0.0241 | +0.0056 | +0.0389 | yes | no |
| rfo_gold | 24 | 45 | +0.0000 | -0.0444 | -0.0111 | +0.0222 | no | no |
| rfo_gold | 32 | 46 | +0.0399 | +0.0127 | +0.0109 | +0.0380 | yes | no |
| rfo_gold | 40 | 47 | +0.0496 | +0.0142 | +0.0160 | +0.0585 | yes | no |

`first_candidate_step` becomes 16 for both arms.
`first_bootstrap_rule_supported_step` stays `None` for both.

Three things in that table are worth saying out loud.

The two arms fail on opposite halves. `naive` clears the external condition at
step 40 and fails the internal one; `rfo_gold` clears the internal condition at
steps 32 and 40 and fails the external one. Each half has been satisfied at
least once, so the rule is not unattainable at this scale -- it is a precision
problem, not a structural one.

`naive` at step 40 fails by the narrowest margin the arithmetic allows: its
2.5th bootstrap percentile is 5.05e-18, which is exactly zero plus float
residue, against a threshold of 1e-12. A 95% interval whose lower bound is zero
does not exclude zero, so `not supported` is the correct call and is not to be
argued away. It is worth recording only because `s_select` deltas over 44 specs
are discrete, so exactly-zero is a mass point rather than a coincidence.

The pairs lost are the adjudication gap: 14 to 23 of 256 images per checkpoint
carry no verdict, which costs 15 to 20 of the 64 specs. Whether closing that gap
would change any verdict is a question for the registered procedure over the
five replicates, not for this packet.

## What this packet does not license

One seed. Exploratory, multiple looks unadjusted, as the frozen report's own
limitations already state. The repaired numbers are a descriptive trajectory and
not a significance claim, and the choice of repaired granularity is a
pre-registration question that this packet raises rather than settles.

## Files

- `reanalyze_dstar_join.py` -- the harness; run it with no arguments to print
  both tables, or with a path to dump both full reports as JSON
- `dstar_reanalysis.json` -- frozen and repaired reports side by side
- `reanalysis.log` -- the run that produced them
