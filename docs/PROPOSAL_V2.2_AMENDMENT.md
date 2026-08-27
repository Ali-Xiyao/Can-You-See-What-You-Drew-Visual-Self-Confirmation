# Visual Self-Confirmation Proposal v2.2 Amendment

Status: normative amendment to the frozen v2.1 proposal  
Approved: 2026-08-28  
Frozen predecessor: Git tag `v2.1-showo-gate-red`

## Scope

This amendment does not reinterpret or overwrite the v2.1 Show-o result. Show-o v1 remains a
negative control that failed the preregistered generated-domain and observation-readiness gates.
The amendment changes the applicability claim, backbone selection rule, controlled benchmark, and
prerequisite gate for all new work.

## Revised scientific question

> For unified multimodal models that cross a joint generate-and-observe readiness threshold, when
> does self-consistency stop tracking publicly verifiable visual correctness?

The claim is conditional. A unified model below the readiness threshold is outside the domain of
the phenomenon experiment; its failure neither confirms nor refutes the later self-confirmation
dynamics.

## Backbone selection

The trainable backbone is selected by Gate -2 rather than named in advance. The registered ladder is:

1. `showlab/show-o2-1.5B`, audited at its official 432x432 setting;
2. `showlab/show-o2-1.5B-HQ`, audited only after a recorded failure of candidate 1 and used at
   512x512;
3. `showlab/show-o2-7B`, considered only after both 1.5B checkpoints fail.

Fallback checkpoints are not downloaded or audited speculatively. Each transition is justified by
the previous candidate's immutable decision. Show-o v1 is retained only as the frozen negative
control.

## Gate -2: Joint Generate–Observe Readiness

### Gate -2A — Unified functionality

The same checkpoint must:

- perform text-to-image generation;
- answer atomic questions from a hard-rendered RGB image;
- permit a frozen step-0 self-observer;
- accept LoRA on an audited generation/shared-transformer module set;
- complete generation-loss and understanding-replay backward passes;
- save and resume adapter, optimizer, scheduler, RNG, and full configuration state.

The six-sample inference canary runs before the larger audits. The LoRA/backward/resume canary runs
only after the observation and generated-domain audits leave a plausible path to four eligible
families, but it is required for a final Gate -2 pass.

### Gate -2B — Reference-image observation

The audit uses 120 balanced program-rendered images, 20 per main family, and reports open answer,
forced choice, order reversal, yes-bias, repeatability, and abstention.

| Requirement | Threshold |
|---|---:|
| Open accuracy for each retained family | >=80% |
| Retained families | >=4 |
| Absolute yes-bias | <=10 percentage points |
| Repeated-answer agreement | >=90% |
| Abstention rate | <=20% |

### Gate -2C — Generated-image measurability

The audit uses 60 family-specific minimal prompts, 10 per main family. It first generates K=1,
then generates K=4 for families retained by the reference audit. The deterministic verifier is a
diagnostic instrument whose answered cases are checked by a blinded human audit.

| Requirement | Threshold |
|---|---:|
| Blind-manual precision of answered verifier cases | >=95% |
| Overall primary-answer coverage | >=80% |
| Coverage for each retained family | >=70% |
| Oracle@K=4 has at least one correct candidate | >=70% |
| Coverage swing across fixed seeds | <=10 percentage points |

Abstention is preferable to forced guessing. The coverage relaxation from the v2.1 95% rule cannot
be used to re-open the v2.1 result: that result had 55% overall coverage, 0% binding coverage, and
0% exact-scene success.

### Gate -2D — Joint eligible families

For a family `f`:

```text
JointEligible(f) =
    SelfObservationAccuracy(f) >= 0.80
    AND GeneratedCoverage(f) >= 0.70
    AND VerifierPrecision(f) >= 0.95
    AND OracleAt4(f) >= 0.70
```

At least four main families must be jointly eligible before E1 or E2. The selected set is frozen in
the decision and bounds every subsequent conclusion.

## Controlled benchmark v2.2

Main families use family-specific minimal prompts on a plain white background:

- Existence: one target object, with a matched non-target scene for negative questions.
- Color: one shape with one named color; no size or position constraint.
- Absolute size: one explicitly large or small object.
- Spatial: two distinct objects with left/right or above/below only.
- Count: exactly N identical, separated objects.
- Binding: two separated objects with distinct shape-color bindings.

`larger_than` is removed from the main spatial family. Relative size is an independent appendix
family and is never mixed with absolute size or main spatial results.

All v2.2 data live under a new `selfsight-v2.2` namespace. No v1 manifest, rendered RGB, decision,
or figure may be overwritten or recalculated.

## Model and observer roles

- Trainable `Show-o2_t`: candidate generation and self-training.
- Frozen `Show-o2_t0`: RFO-Self selection, fixed for the entire experiment.
- Frozen Qwen2-VL-2B: public observer, `g_rfo` detector, and capability upper bound.
- Deterministic verifier: gold/public-pixel evaluation with explicit abstention.

Qwen2-VL capability matching is no longer a prerequisite for RFO training or GDA detection. It is
required only when making a causal/mechanistic statement that compares its perception with the
unified model, and that comparison is conditioned on the frozen family/item set.

## Readiness execution order

1. A1: six-sample load/generate/hard-render/re-observe canary and memory profile.
2. A2: 120-image program-reference audit.
3. A3: 60 minimal prompts at K=1, then K=4 for retained families, contact sheet, and blind audit.
4. A4: LoRA generation loss, understanding replay, optimizer step, corruption/restore, and resume.
5. Finalize Gate -2 with model/source/dependency revisions and SHA-256 of every input report.
6. Run E1, Gate -1b, and local paired E2 only after a green decision with at least four families.
7. Migrate to A800 only after the local short run is stable and interpretable.

## Conditional warm-up

If reference observation passes but controlled generation fails, try the HQ checkpoint, verify the
minimal benchmark, and only then permit a non-self-feedback generation-readiness warm-up. If
generation passes but observation fails, an image-to-atom VQA warm-up with general understanding
replay may create a new common `t0`. Both arms and the frozen self-observer must begin from that same
`t0`, while raw-pretrained versus adapted readiness is reported separately.

## Decision integrity

`readiness/decision.json` must bind:

- backbone ID and immutable model revision;
- official source revision and dependency revisions;
- native resolution and candidate rank;
- SHA-256 of reference, generated, blind-human, and LoRA reports;
- retained observation families and final joint-eligible families;
- every threshold and check result;
- final pass/fail plus the registered fallback action.

E1/E2 read the selected backbone and eligible families from this file. They may not accept a manual
backbone override.

## Official implementation references

- [Show-o official repository](https://github.com/showlab/Show-o)
- [Show-o2 1.5B checkpoint](https://huggingface.co/showlab/show-o2-1.5B)
- [Show-o2 1.5B-HQ checkpoint](https://huggingface.co/showlab/show-o2-1.5B-HQ)

