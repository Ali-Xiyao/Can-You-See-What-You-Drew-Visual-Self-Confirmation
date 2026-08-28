# Task Plan: Visual Self-Confirmation v2.3 RFO-Gold Mechanism Gate

## Goal

Preserve v2.2 as a frozen conditional-negative result, then determine locally whether perfect
candidate feedback can produce a measurable training advantage. v2.3 first tests RFO-Gold as a
mechanism positive control on the existing Show-o2-1.5B-HQ backbone. No A800 work or new model
download is allowed while the mechanism gate remains unresolved.

## Authoritative Inputs

- `Can You See What You Drew Visual Self-Confirmation.md` (frozen proposal v2.1)
- `docs/PROPOSAL_V2.2_AMENDMENT.md` (normative v2.2 amendment, to be implemented here)
- User-approved model/backbone revision dated 2026-08-28
- User-approved v2.3 RFO-Gold mechanism-first design dated 2026-08-29
- Frozen v2.1 Git tag: `v2.1-showo-gate-red`
- Frozen v2.2 implementation commit: `509e774631fadcb1acb8a9820327d697e312dc32`
- Active branch target: `experiment/v2.3-rfo-gold`

## Current Phase

v2.3 is fail-closed at Phase 12. The first gradient repeat made the 64/96 informative-pool floor
mathematically unreachable after 42 screened prompts, so repeats 2/3 and all three-seed training are
not tested. A future candidate-bank probe requires a separately named amendment; A800 remains N/T.

## Registered Backbone Ladder

1. `showlab/show-o2-1.5B` at its locked revision, native 432x432.
2. `showlab/show-o2-1.5B-HQ` only if candidate 1 fails, native 512x512 for this project.
3. `showlab/show-o2-7B` only if both 1.5B candidates fail and local feasibility justifies A800 use.
4. Show-o v1 remains a frozen negative control; it is never silently promoted back to the mainline.

The public frozen observer for RFO/GDA is the already audited Qwen2-VL-2B. Observer ability
matching is required only for claims about the self-confirmation mechanism, and is evaluated per
eligible family/item rather than as a single six-family macro constraint.

## Gate -2: Joint Generate–Observe Readiness

Gate -2 is family-conditioned. The six main families are existence, count, color, absolute size,
horizontal/vertical spatial relation, and two-object attribute binding. Relative size
(`larger_than`) is an independent appendix family and is not mixed into main spatial.

- A. Unified functionality: the same checkpoint must do text-to-image and image-to-text atomic QA,
  support a frozen step-0 observer, accept generation-branch LoRA, and resume adapter/optimizer
  state. The six-sample inference canary runs first; the backward/resume part runs only when B/C are
  promising, but it remains mandatory for a final pass.
- B. Reference observation: 120 balanced program renders. Retained-family open accuracy must be
  >=80%; at least four families must pass; yes-bias <=10pt, repeat agreement >=90%, and abstention
  <=20% are global requirements.
- C. Generated measurability: 60 minimal prompts, K=1 first and then K=4 on retained families.
  Blind-manual verifier precision >=95%, overall primary-answer coverage >=80%, each retained-family
  coverage >=70%, Oracle@K=4 >=70%, and fixed-seed coverage swing <=10pt.
- D. Joint eligibility: a family must meet self-observation accuracy >=80%, generated coverage
  >=70%, verifier precision >=95%, and Oracle@4 >=70%. At least four families in the intersection
  are required before E1/E2.

Every reported conclusion must name its frozen eligible-family set. Fewer than four eligible
families means the scientific claim is unsupported for that backbone and stops phenomenon work.

## Phases

### Phase 0 — Preserve v2.1

- [x] Verify the main worktree is clean at commit `5e5543853aaf0d6bf8428e9c9e30e049b01d6a9d`.
- [x] Create and push annotated tag `v2.1-showo-gate-red`.
- [x] Create branch `experiment/v2.2-joint-readiness`.
- [x] Add an in-repository v2.1 evidence index with hashes and immutable external run locations.
- **Status:** completed

### Phase 1 — Lock v2.2 design and model assets

- [x] Add the normative v2.2 proposal amendment and decision schema.
- [x] Resolve official Show-o2 source commit and the three checkpoint revisions.
- [x] Expand the pinned sparse source checkout to `show-o2/` without deleting v1 paths.
- [x] Lock all first-candidate dependencies, including Wan2.1 VAE, SigLIP, and Qwen2.5 tokenizer.
- [x] Add a gate-ordered download plan; do not materialize HQ/7B before their fallback condition.
- **Status:** completed

### Phase 2 — Versioned data and minimal prompts

- [x] Create a `selfsight-v2.2` data namespace; never overwrite v1 manifests or RGBs.
- [x] Remove `larger_than` from main spatial and register relative size as appendix-only.
- [x] Add family-specific minimal generation prompts with exact scene graphs and fixed IDs/seeds.
- [x] Materialize six-sample A1, balanced 120-reference A2, and 60-prompt A3 manifests.
- [x] Add zero-overlap, answer-normalization, hash, and v1-immutability tests.
- [ ] Materialize the optional appendix relative-size split only after the main readiness route.
- **Status:** completed_main_pending_appendix

### Phase 3 — Backbone abstraction

- [x] Add `backbones/base.py`, `backbones/showo_v1.py`, and `backbones/showo2.py`.
- [x] Preserve the proven old `ShowoAdapter` path and provide a v1 negative-control wrapper.
- [x] Expose generate, observe, image-target encoding, LoRA target discovery, gradient, and resource
  reporting through a common contract.
- [x] Keep the Qwen observer in its isolated JSONL service and add a locked official config.
- **Status:** completed

### Phase 4 — Readiness runners and decisions

- [x] Implement `run_backbone_readiness.py` for A1/A2 and evidence hashing.
- [x] Implement `audit_generated_precision.py` with blind packet export/import and verifier diagnostics.
- [x] Implement `finalize_joint_readiness.py` with fail-closed family intersection and SHA binding.
- [x] Implement `render_readiness_matrix.py` with publication-safe color/vector/grayscale exports.
- [x] Make E1 automatically read the v2.2 decision and reject ineligible families/backbones.
- [x] Bind Gate -1b to the same decision, public-observer audit, eligible families, and A4-selected
  Show-o2 LoRA module hash.
- [x] Bind the paired local E2 loop and checkpoint evaluator to the same decision, eligible
  families, frozen step-0 Show-o2, objective-specific training batches, and checkpoint contract.
- [x] Bind the formal A800 E2 orchestrator, migration canary, seed configs, and eligible-family data
  preflight to the same v2.2 contracts.
- **Status:** implementation_complete_runtime_gated

### Phase 5 — First Show-o2 candidate on local 3090s

- [x] Create an isolated native-Windows Show-o2 environment on H: and capture an exact lock.
- [x] Download only base 1.5B and its locked dependencies to H: with size/hash verification.
- [x] Run A1 (six samples) at official 432x432; log GPU assignment, peak memory, time, outputs, and hashes.
- [x] Run A2 and collision-safe A3-r2 under the locked rank-1 model, manifest, and seeds; retain r1
  only as an invalid diagnostic. r2 completed 210 unique IDs/paths and reproduced r1's automatic
  metrics exactly.
- [x] Implement a fail-closed upstream-stop decision for an automatic A3 failure; require
  collision-free candidate artifacts and record blind-human/A4 evidence as skipped rather than
  manufacturing missing measurements.
- [x] Apply the preregistered stop rule: automatic A3 was red, so blind-human precision and A4 are
  explicitly N/T rather than fabricated failures.
- **Status:** completed_rank1_red

### Phase 6 — Conditional fallback and local phenomenon work

- [ ] If base generation fails but reference observation passes, allow the registered short generation
  warm-up; otherwise move directly to HQ.
- [x] Download/audit HQ only after a recorded base decision; 7B only after both 1.5B decisions. The
  immutable base/HQ red decisions authorize the exact 7B revision, whose full locked model group,
  A1, and A2 are now complete; 7B A3 is running on physical GPU1.
- [x] Enforce the candidate ladder inside the downloader: fallback downloads require the immediately
  prior red Gate -2 decision, exact rank/model/fallback identity, and valid evidence SHA-256 records.
- [x] Evaluate the E1/Gate -1b/E2 authorization condition. It is not met because rank-3 Gate -2C
  is red; these phenomenon experiments are preregistered N/T rather than pending work.
- [x] Because the final candidate failed automatic Gate -2C, freeze the conditional negative result
  and stop before blind human review, A4, E1, Gate -1b, or self-training.
- **Status:** completed_conditional_negative
- [x] Validate the completed human CSV: 49/49 rows, one reviewer, valid parseability fields, SHA-256
  `70fe561e0f25f6bf94e6a138d930a86ed9712969dd86c618155e926905dcb68e`.
- [x] Accept arbitrary nonnegative numeric human counts and bind the exact review CSV SHA into the
  scored human report.
- [x] Add and test a fail-closed stop-after-human finalizer: measured human precision red, A4 N/T,
  no LoRA/backward run, adjacent 7B fallback only.
- [x] Score/freeze the HQ human report and Gate -2 decision, recursively revalidating A3 rows/RGBs
  and the rank-1 predecessor.
- [x] Render the HQ red readiness matrix with measured precision failures and A4 marked N/T.
- [x] Complete Show-o2 7B A3 after its green A1/A2 evidence. A3 failed automatic coverage/stability,
  so freeze the stop-before-human/A4 decision with the exact HQ predecessor and no next fallback.
- [x] Run final full verification; the complete test suite and Ruff pass after the frozen decision
  and three-candidate figure QA.
- **Current blocker:** none. The registered local candidate ladder is exhausted and no registered
  phenomenon experiment is authorized. The separate
  user-authorized engineering diagnostic is complete under Phase 8.

### Phase 6a — Project-root path normalization

- [x] Inventory every `H:\selfsight-*` root and every absolute path reference in source, config,
  scripts, documentation, registries, reports, checkpoints, and review artifacts.
- [x] Keep model weights only at `H:\selfsight-models`; move environments, data, runs, cache,
  temporary files, and review packets below the current project root.
- [x] Replace hard-coded non-model H-drive roots with project-root-derived paths and add fail-closed
  path-policy validation.
- [x] Rewrite movable runtime metadata without changing scientific payloads; preserve original
  content hashes wherever paths are not part of the evidence object.
- [x] Revalidate Gate evidence chains, tests, Ruff, environment executables, and the human-review
  ZIP after relocation.
- [x] Scan H: and the repository for residual non-model `selfsight-*` paths; confirm no physical
  source copies remain outside the project and retain only the five validated evidence junctions.
- [x] Commit and push the complete migration on `experiment/v2.2-joint-readiness`.
- **Status:** completed

### Phase 7 — A800 migration and formal experiments

- [x] Evaluate A800/formal authorization. The local candidate ladder has no green Gate -2 and no
  eligible family set, so the migration canary and formal E2 are N/T under the registered design.
- [ ] Re-materialize exact locked revisions on Linux/A800 only if a future, separately registered
  backbone/design revision reopens the Gate; this is outside the completed v2.2 ladder.
- **Status:** not_authorized_for_v2.2

### Phase 8 — User-authorized post-Gate exploratory continuation

- [x] Inventory the existing-only model/evidence/runtime surface and choose the smallest diagnostic
  backbone/family set without changing any frozen Gate report or downloading weights. Use
  Show-o2-1.5B-HQ with `existence/color/spatial` only.
- [x] Add a fail-loud exploratory authorization artifact and runner mode whose outputs are physically
  separated from `runs/readiness`; every report must state that it is non-preregistered/non-formal.
- [x] Run the Show-o2 LoRA A4 backward/resume canary first. Stop and diagnose on NaN, OOM, invalid
  target modules, or checkpoint mismatch before launching E1/E2.
- [x] Run E1 and RFO isolation checks using RGB/question-only observer traffic, then run Gate -1b
  gradient diagnostics on the same exploratory family set.
- [x] If all mechanism checks are stable, run the one-seed paired local Base/Naive/RFO-Self pilot
  with identical prompt IDs, candidate pools, update counts, and checkpoint evaluation.
- [x] Produce exploratory figures/reports with explicit provenance and retain the frozen v2.2
  conditional-negative conclusion unchanged.
- [x] Do not run formal three-seed/A800 work unless separately authorized after local diagnostics.
- **Status:** completed_local_exploratory_formal_not_authorized

### Phase 9 — v2.3 protocol and isolation

- [x] Push `experiment/v2.3-rfo-gold`, created from frozen implementation commit `509e774`.
- [x] Add a normative v2.3 amendment and a hash-bound local authorization that cannot modify or
  reinterpret v2.2 evidence.
- [x] Register existing-only model assets, three local seeds, K=4, fixed diffusion/dropout RNG and
  the four trajectories Base/Naive/RFO-Self/RFO-Gold.
- [x] Lock mechanism-first stop rules before any new training run.
- **Status:** completed

### Phase 10 — v2.3 visual semantics and calibration

- [x] Replace ambiguous `square` wording in the v2.3 main path with `box`/`quadrilateral`; record an
  explicit aspect-ratio policy and keep v2.1/v2.2 manifests byte-for-byte unchanged.
- [x] Keep `existence/color/spatial` as the main mechanism families; place count/binding in a hard
  diagnostic tier and exclude absolute size from the mechanism gate.
- [x] Materialize versioned v2.3 train/probe/outcome manifests with zero overlap and exact hashes.
- [x] Produce a blind human/verifier calibration packet and require >=95% agreement under the new
  vocabulary before training. If human labels are not yet available, stop at this gate.
- **Status:** completed

### Phase 11 — RFO-Gold and informative candidate selection

- [x] Add `rfo_gold` as the third selection/gradient arm without changing the Naive or RFO-Self
  objectives; do not instantiate train checkpoints after a red gradient gate.
- [x] Generate K=4 paired candidates with fixed seeds and select only pools whose gold-verifier
  best/worst score gap clears a preregistered informativeness threshold.
- [x] Keep prompt IDs and candidate seeds identical across Naive/RFO-Self/RFO-Gold; abstain
  symmetrically when a pool is not informative.
- [x] Add packet resume, candidate-pool identity, selector isolation and blind-wire evidence audits;
  checkpoint/resume remains N/T because the survival gate forbids training initialization.
- **Status:** completed_selector_training_not_authorized

### Phase 12 — v2.3 gradient survival gate

- [x] Expand the fixed gradient probe to 96 items; stop repeat 1 at the fail-closed mathematical
  boundary (9 informative after 42 screened, optimistic maximum 63 < 64), so repeats 2/3 are N/T.
- [x] Compare identical, Naive, RFO-Self and RFO-Gold gradients on the nine available diagnostic
  pools, including
  cosine, norm ratio, per-block statistics and split-half noise intervals.
- [x] Apply identical cosine >=0.999 and stable Gold/noise ordering requirements. Identical is 1.0,
  but sample supply fails and the Gold cosine remains inside the wide split-half interval; stop and
  diagnose the candidate probe before any training.
- **Status:** completed_red_informative_pool_supply

### Phase 13 — three-seed local short curves

- [ ] Run three small local seeds only if semantics, RFO-Gold selection and gradient gates pass.
- [ ] Produce Base/Naive/RFO-Self/RFO-Gold checkpoint curves with external correctness, internal
  score, verifier coverage, public-view consistency and SCFR with explicit denominators.
- [ ] Decide: Gold red -> redesign loss/probe; Gold green and Self red -> improve self-observation;
  Gold/Self green but measurability red -> consider backbone only after a new authorization.
- [x] Keep A800 migration N/T until the v2.3 local mechanism gate is green.
- **Status:** not_tested_gradient_gate_red

## Hard Stops

- Do not download any additional model or model weight without fresh, explicit user approval. If a
  future step appears to require one, pause first and report the exact model/revision, expected disk
  use, purpose, and no-download alternatives; planning or inspecting an existing lock is read-only.
- Never rewrite or re-decide the v2.1 Gate -1 evidence.
- Never download the whole ladder speculatively; each fallback requires a hashed predecessor decision.
- Never interpret reference-image understanding as generated-image self-confirmation without A3.
- Never run LoRA/self-training if A-C do not leave a plausible route to four joint families.
- Never use Qwen2-VL training or selection outputs as evidence that the unified backbone can see its
  own drawing; it remains a frozen external observer/detector.
- Never generalize beyond the Gate -2 eligible families.
- Post-Gate exploratory outputs must never be written into a frozen readiness directory, used to
  overwrite a decision, or described as preregistered/formal evidence.
- v2.3 must not write inside any v2.2 output directory. Its data and runs live under versioned
  `selfsight-v2.3` and `runs/v2.3-rfo-gold` roots.
- Do not start v2.3 training before the new vocabulary's blind human/verifier agreement reaches 95%.
- Do not scale a failed RFO-Gold mechanism gate; a red Gold control routes to loss/probe diagnosis.

## Fixed Local Resource Policy

- GPU0 and GPU1 are independent 24GB workers, not a pooled 48GB device. A complete Show-o2 stage may
  run on either card via an explicit logical-to-physical CUDA mapping; never split one model across
  the non-NVLink pair.
- Default concurrency remains GPU0 for trainable Show-o2 work and GPU1 for a frozen observer. When
  stages are dependency-serial, either idle card may instead run the next complete stage; the 7B A3
  uses `CUDA_VISIBLE_DEVICES=1` so locked logical `cuda:0` maps reproducibly to physical GPU1.
- Models remain at `H:\selfsight-models`; every other environment, cache, data, temporary, run,
  and review artifact stays below this repository root.
- Native Windows is attempted first. A Windows-incompatible official kernel may trigger a documented
  WSL2 fallback, but not an unrecorded implementation change.

## Verification Checklist

- [x] Unit/integration suite and Ruff pass after the 7B sharded-loader/low-RAM fixes.
- [x] Every decision binds model/source/dependency revisions and all input SHA-256 values.
- [x] Checkpoint code restores adapter, optimizer, scheduler, RNG, and full config; exploratory GPU A4 passed.
- [x] Observer subprocess receives only RGB path/bytes plus atomic question.
- [x] Windows/A800 canary is explicitly N/T because the registered A800 route is not authorized.
- [x] Branch is committed and pushed; main remains the frozen v2.1 baseline.

## Errors

See the append-only error log in `progress.md`.
