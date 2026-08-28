# Visual Self-Confirmation v2.3 Amendment: RFO-Gold Mechanism Gate

## Status and scope

This amendment defines a new local diagnostic experiment dated 2026-08-29. It does not revise,
replace or reinterpret the frozen v2.2 readiness decisions. Existing v2.1/v2.2 manifests, reports,
thresholds and run directories are immutable inputs. v2.3 is non-formal until a later registration
explicitly promotes a design after the local mechanism gate passes.

No A800 run and no new model download are authorized by this amendment. The trainable backbone is
the already materialized `showlab/show-o2-1.5B-HQ` revision
`d3a220ec55feaacbdfcb053847edee14edd4e69a`. GPU0 hosts the trainable model; GPU1 hosts the frozen
step-0 observer. The GPUs are independent workers and are never treated as pooled memory.

## Primary question

Can a perfect candidate selector make the registered correction-style SFT update outperform Naive
training under an identical K=4 candidate pool and update budget?

The four trajectories are:

1. Base: frozen step-0 checkpoint.
2. Naive: candidate chosen by the current arm's RGB observation.
3. RFO-Self: candidate chosen by the frozen step-0 Show-o2 RGB-only observer.
4. RFO-Gold: candidate chosen by the deterministic generated-image verifier.

RFO-Gold is a mechanism positive control. If it does not produce a repeatable advantage over Naive,
the next action is to revise the loss, weighting or gradient probe—not to scale compute or replace
the observer.

## Visual vocabulary

The internal geometry enum remains `square` so frozen schemas and verifier logic do not change. v2.3
visible prompts and questions use `box` for that geometry. A box is a four-sided filled region whose
axis-aligned bounding-box aspect ratio is between 0.5 and 2.0 inclusive; rotated quadrilaterals are
allowed. This matches the annotation policy the reviewer stated they had already applied when
rectangle-like outputs were counted as square.

The primary mechanism families are `existence`, `color` and `spatial`. Count and binding are retained
only as a separately reported hard tier. Absolute size is excluded from v2.3 mechanism decisions.
Training cannot start until a hash-bound human/verifier calibration artifact reaches at least 95%
agreement overall and per primary family under the box vocabulary.

## Data isolation

All v2.3 data are derived into `data/selfsight-v2.3`; all runs go under `runs/v2.3-rfo-gold`.
Original scene signatures and RGB hashes are retained, while scene IDs, prompts and questions are
versioned. Train, gradient-probe and outcome sources come from the corresponding frozen v1 splits,
so their pre-existing zero-overlap guarantee remains intact and is re-audited after transformation.

Registered local sizes are:

- training search pool: 432 prompts (144 per primary family);
- fixed gradient probe: 96 prompts (32 per primary family);
- checkpoint outcome set: 90 prompts (30 per primary family);
- hard-tier diagnostic: 60 prompts (30 count, 30 binding), not used for the primary decision.

## Candidate-pool and RNG contract

Each seed/round schedules 48 unique prompts and four candidate seeds per prompt. One shared candidate
pool is generated from the frozen Base checkpoint and is presented unchanged to all three selectors.
A pool is informative only when the RFO-Gold candidate scores span at least 1.0, which for the single
primary atom means that the pool contains at least one verifier-correct and one verifier-incorrect
candidate. The first 16 informative pools in deterministic schedule order are used. A round fails
closed if fewer than 16 pools qualify.

All trainable arms use exactly the same accepted prompt IDs, K, candidate RGBs, optimizer steps,
microbatch order, diffusion/flow latent seeds and LoRA-dropout RNG seeds. Any selector abstention
removes the prompt from every arm before the fixed count is taken. Tie-breaking is deterministic by
sampling seed and candidate ID.

## Gradient survival gate

Before three-seed training, run 96 fixed probe prompts in three independent candidate-seed repeats.
Each repeat reports:

- identical-selection repeat cosine and per-block cosines;
- Naive/RFO-Self and Naive/RFO-Gold cosine and norm ratio;
- selection agreement and Gold selected-score advantage;
- split-half noise diagnostics;
- accepted informative-pool count and family counts.

The gate passes only when all repeats have identical cosine at least 0.999, at least two of three
repeats have Naive/Gold cosine at most 0.995, Gold improves selected verifier score over Naive by at
least 0.10 in at least two repeats, and at least 64 common informative probes exist in every repeat.
Thresholds are locked before looking at v2.3 gradient outputs.

## Local short curves

Use seeds `20260829`, `20260830` and `20260831`. Each seed runs three rounds, 16 accepted informative
prompts per round, eight optimizer steps per round, BF16, effective batch eight and the existing
LoRA/AdamW configuration. Checkpoint evaluation reports external correctness, internal score,
verifier coverage, public-view consistency and SCFR with its raw denominator. No confidence interval
or significance claim is made from three small diagnostic seeds.

## Decision tree

- RFO-Gold does not consistently exceed Naive: mechanism gate red; redesign correction loss,
  sample weighting or gradient probe and stop scale-up.
- RFO-Gold improves but RFO-Self does not: mechanism is viable; diagnose atomic questions,
  confidence, abstention and self-observer selection.
- Gold and Self improve but generated measurability remains unstable: only then consider a backbone
  revision, subject to a new download authorization.
- A800 remains not-tested until a separately authorized formal design follows a green local gate.
