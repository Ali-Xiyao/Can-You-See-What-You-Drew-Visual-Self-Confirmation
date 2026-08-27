# Task Plan: Visual Self-Confirmation v2.2 Joint Readiness

## Goal

Determine whether an openly reproducible unified backbone can both draw and read the controlled
visual concepts required by Visual Self-Confirmation. Only a backbone that passes the registered
Joint Generate–Observe Readiness gate may enter E1/E2. Local work targets one engineering seed on
two independent RTX 3090 cards; formal three-seed experiments remain single-A800-80GB work.

## Authoritative Inputs

- `Can You See What You Drew Visual Self-Confirmation.md` (frozen proposal v2.1)
- `docs/PROPOSAL_V2.2_AMENDMENT.md` (normative v2.2 amendment, to be implemented here)
- User-approved model/backbone revision dated 2026-08-28
- Frozen v2.1 Git tag: `v2.1-showo-gate-red`
- Active branch: `experiment/v2.2-joint-readiness`

## Current Phase

Phase 5: first local Show-o2 candidate. The v2.1 red result is immutable/tagged, the v2.2 code and
data path are implemented and pushed, the H-drive Windows environment and candidate-1 assets are
locked, and A1 is green. The next operation is the 120-image A2 reference-observation audit.

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
- [ ] Download/audit HQ only after a recorded base decision; 7B only after both 1.5B decisions. The
  immutable base red decision, authorized HQ download, HQ A1, and HQ A2 are complete; HQ A3 is next.
- [x] Enforce the candidate ladder inside the downloader: fallback downloads require the immediately
  prior red Gate -2 decision, exact rank/model/fallback identity, and valid evidence SHA-256 records.
- [ ] If Gate -2 is green with at least four families, run family-restricted E1, Gate -1b, and paired local E2.
- [ ] If fewer than four families are eligible, finalize the conditional negative result and stop before self-training.
- **Status:** in_progress

### Phase 7 — A800 migration and formal experiments

- [ ] Re-materialize exact locked revisions on Linux/A800 and run the 32-prompt migration canary.
- [ ] Require >=95% answer/verifier-label agreement and <=1pt metric deviation.
- [ ] Run three-seed E2, then later gates in proposal order, only for the frozen eligible family set.
- **Status:** pending

## Hard Stops

- Never rewrite or re-decide the v2.1 Gate -1 evidence.
- Never download the whole ladder speculatively; each fallback requires a hashed predecessor decision.
- Never interpret reference-image understanding as generated-image self-confirmation without A3.
- Never run LoRA/self-training if A-C do not leave a plausible route to four joint families.
- Never use Qwen2-VL training or selection outputs as evidence that the unified backbone can see its
  own drawing; it remains a frozen external observer/detector.
- Never generalize beyond the Gate -2 eligible families.

## Fixed Local Resource Policy

- GPU0: Show-o2 generation, gradient, and LoRA work.
- GPU1: frozen Qwen observer, verifier support, and parallel audits.
- Environments, caches, models, data, temporary files, and runs stay under short H-drive roots.
- Native Windows is attempted first. A Windows-incompatible official kernel may trigger a documented
  WSL2 fallback, but not an unrecorded implementation change.

## Verification Checklist

- [x] Unit/integration suite and Ruff pass (60 tests after the A1 Windows/runtime fixes).
- [x] Every decision binds model/source/dependency revisions and all input SHA-256 values.
- [x] Checkpoint code restores adapter, optimizer, scheduler, RNG, and full config; GPU A4 still pending.
- [ ] Observer subprocess receives only RGB path/bytes plus atomic question.
- [ ] Windows/A800 canary comparison is deterministic within registered tolerances.
- [x] Branch is committed and pushed; main remains the frozen v2.1 baseline.

## Errors

See the append-only error log in `progress.md`.
