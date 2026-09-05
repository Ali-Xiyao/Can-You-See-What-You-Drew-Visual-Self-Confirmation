# Dynamic decoupling pilot, frozen before training

The user authorized implementation, both local RTX 3090 cards, and sustained
execution after reviewing the measurement defects and time estimate. This new
run is `runs/v4/decoupling-pilot-20260906`. Existing runs and their failed
registered verdicts remain unchanged. This is a descriptive pilot, not a new
confirmation of the old selector gate and not a promise that either event exists.

## Fixed experiment

- Ordinary Show-o2 1.5B, revision `07ec16589d4fc5422a74dddbbc4b2cd11e551039`,
  the audited rank-16 LoRA targets, and seed 20260906.
- Naive versus RFO-Gold, paired prompts, latent seeds and update budgets, with
  independent model/optimizer/RNG state. The frozen Qwen2-VL observer is used
  only for the gradient probe, never for training selection.
- The two main v4 corpora provide 228 unique specs. Deterministically reserve
  64 for outcomes and 32 for probes; the remaining 132 are training only.
  Understanding replay must also exclude outcome/probe specs and images.
- Ten rounds, 12 fresh training prompts per round, K=8 candidates per arm,
  eight optimizer updates per arm per round, microbatch 1, accumulation 8,
  learning rate 1e-4, 25% understanding replay restricted to training specs.
  The 120 training prompt slots do not repeat specs. No extension or threshold
  change is made in response to a favorable-looking checkpoint.
- Save and evaluate the base plus every completed round. Use the same 64
  outcome prompts and latent seeds across arms and checkpoints. Score both
  cycle log probability and the actual prompted atomic selection score on
  each outcome image. Atomic scoring uses all questions as its denominator;
  missing/error/abstaining answers receive zero and have separate coverage.
- External image correctness uses the existing dual-detector/crop ladder.
  A known generation failure remains a failure; an unresolved verdict remains
  explicitly unavailable, with its denominator and score bounds reported.
  An agreed image-level verdict is not a complete factual object inventory.
- Freeze at most 16 balanced probe specs from the reserved probe split and
  the existing openct2 bank, at most one pool per spec, selected deterministically.
  Report the actual number, requiring at least four. Keep images, questions,
  independent RFO answers and image verdicts fixed. Recompute the current
  model's Naive answers and all selected-example gradients at each checkpoint.
  This measures gradient evolution on a fixed held-out bank, not the changing
  distribution of newly generated probe images.
- Preserve the definition cos(mean paired gradients) for GDA-free and its
  Gold reference. Save Gram matrices, per-prompt statistics and paired
  bootstrap results; do not retain full multi-gigabyte gradients per checkpoint.

## Interpretation fixed before observing trajectories

The primary pilot output is an interpretable trajectory and a continuation or
stopping rationale. Check finite losses/gradients and actual parameter changes
before interpreting any curve. Ordinary convergence and absence of events are
valid outcomes.

For each arm, use the most recent three available checkpoints and paired specs
at the first and last checkpoint. A descriptive behavioral candidate requires
selection score change >=0.02 and external correctness change <=0.02. Report
spec-bootstrap 95% intervals for both changes and, separately, whether the
internal lower bound exceeds zero and external upper bound is <=0.02. These
are exploratory repeated looks without confirmatory error control. Availability
time is the last checkpoint in the window, not a retrospectively chosen knot.

A gradient candidate requires GDA-free to decline by >=0.01 from the frozen
base on two consecutive measured checkpoints. A pointwise flag alone is not
interval-supported evidence. Any interval-supported flag must use an actual
paired bootstrap difference on the identical bank/replicate indices, not
subtract marginal CI endpoints. Alarm availability is the second qualifying
checkpoint. Record early, simultaneous, late, missed and no-event cases
separately; no event or no alarm never silently becomes a zero lead.

Even a positive pilot does not establish a generalizable early-warning method.
That requires separately frozen replication and added predictive value beyond
the external trajectory's own history. The existing static selector results and
the defective old oracle report do not establish that claim.

## Execution boundaries

Keep raw answers, per-image metrics, selected candidates, model/checkpoint hashes,
configuration, source fingerprints, stage logs and exit status. Start with a
detector loading canary, then run the frozen schedule one round at a time with
evaluation and probes between rounds. A successful round must contain actual
finite parameter updates and at least two paired training prompts.

Pause for diagnosis on a failed stage, invalid/missing checkpoint, failed
invariant, fewer than two paired prompts, insufficient disk (8 GiB), or 60 hours
of pilot wall time. Preserve evidence; never recycle old result directories or
invent labels to finish a stage. Technical repairs require provenance and
revalidation; changing the scientific protocol requires a new run version.

The assistant may continue the authorized project after reviewing this pilot,
but completing this pilot alone does not mean that two research signals have
been found. Neither old failed gates nor missing measurements are relabeled.
