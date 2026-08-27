# Task Plan: Visual Self-Confirmation Implementation

## Goal
Implement a reproducible SelfSight research codebase, execute the local dual-RTX-3090 gate sequence,
continue into E1/Gate -1b/miniature E2 only when its preregistered prerequisites are green, and leave
a decision-complete single-A800 configuration for eventual formal three-seed runs.

## Authoritative Inputs
- `Can You See What You Drew Visual Self-Confirmation.md` (proposal v2.1)
- User-approved implementation plan in the current task
- Gate order and stop rules below override convenience or scale-up pressure

## Current Phase
Delivery complete at the preregistered hard stop. All locked local-stage snapshots, canaries, and
diagnostic audits are complete;
the generated-RGB coverage sub-gate and finalized Gate -1 are both red. Qwen and Janus confirm that
capable observers exist but are respectively 25pt and 22.5pt above Show-o, not capability-matched.

## Gate Order and Stop Rules
1. Gate 0: engineering canary, hard-render pipeline, deterministic verifier, generated-RGB
   answer coverage >=95%, and LoRA step/resume.
2. Gate -1: observer capability floor, debiased questions, and capability-matched observer.
3. Gate 1: pixel-override mechanism evidence. Failure does not block E2.
4. Gate -1b: gradient instrument noise floor and detector/trainer separation.
5. Local miniature E2: one seed only; engineering/effect-shape evidence, never a paper claim.
6. A800 migration canary before formal E2.
7. Formal E2 Gate 2/2b before E3/E4/E5.

Hard stops:
- Do not run phenomenon claims before Gate -1 passes.
- Do not start self-training while generated RGB answer coverage is below 95%; program-reference
  accuracy and generated-RGB parseability are separate requirements.
- Do not report GDA before Gate -1b passes.
- Do not scale to formal E2 if local checkpoints cannot resume or candidate pools are not paired.
- Do not run E3/E4/E5 if the prerequisite proposal Gate is red; follow the registered fallback narrative.

## Phases

### Phase 0: Repository and host setup
- [x] Initialize Git and non-destructive ignore rules.
- [x] Create source/config/test/script structure.
- [x] Route caches, environments, temporary files, data, and runs to short H-drive paths.
- [x] Implement host/GPU/software manifest and model revision lock schema.
- **Status:** complete

### Phase 1: Core data and verifier
- [x] Implement typed scene graph and atom schemas.
- [x] Implement deterministic prompt generation and split isolation.
- [x] Implement reference renderer, question generation, answer normalization, and pixel verifier.
- [x] Implement train/Tier-A/Tier-B manifest generation at planned scales.
- [x] Validate verifier accuracy and parse coverage on all deterministic fixtures (3,200/3,200).
- [x] Validate all 400 registered Tier-B interventions and category counts.
- [x] Materialize and audit the fixed 600-image Tier-D subset (300 Tier A + 150 complete Tier-B pairs).
- [ ] Complete the preregistered manual stratified agreement audit (programmatic gate is green; human agreement is not yet claimed).
- **Status:** programmatic_complete_manual_audit_pending

### Phase 2: Interfaces and RFO isolation
- [x] Implement ModelAdapter and JSONL ObserverService contracts.
- [x] Implement CandidateManifest with hashes and provenance.
- [x] Enforce hard RGB write/reload and context-redaction checks.
- [x] Implement observation conditions, RFO selection, paired candidate budgets, and metrics.
- [x] Run subprocess and adversarial no-leak tests.
- **Status:** complete

### Phase 3: LoRA and gradient instrumentation
- [x] Implement Show-o PEFT integration with fixed target modules.
- [x] Implement adapter-only checkpoints including optimizer/scheduler/RNG/config state.
- [x] Implement g_naive/g_rfo/g_gold collection, cosine, norm ratio, per-block summaries, and noise floor.
- [x] Unit-test identical-selection, accumulated gradients, split-half controls, and adapter-only resume.
- **Status:** complete

### Phase 4: Analysis and publication figure pipeline
- [x] Implement D*/D_g exploratory breakpoint analysis and bootstrap helpers.
- [x] Implement Figure 1 from tidy checkpoint metrics.
- [x] Implement PNG/PDF/SVG plus grayscale export and programmatic layout QA.
- [x] Render the full mock-pilot preview and complete color/grayscale visual review.
- **Status:** complete

### Phase 5: Model environments and downloads
- [x] Create the native-Windows Show-o/core environment under H: and verify two-GPU CUDA access.
- [x] Create and verify the isolated observer environment under H:, including exact package/CUDA locks.
- [x] Download and verify the pinned Show-o/MAGVIT/CLIP/Phi core snapshots on H:.
- [x] Download and register the minimal SmolVLM PyTorch snapshot while excluding unrelated ONNX/TF/Flax exports.
- [x] Run the Show-o generation/MMU/RGB reload/LoRA-step/checkpoint-resume canary.
- [x] Run the SmolVLM load/RGB/repeatability canary.
- [x] Download, lock, and canary InternVL2-1B.
- [x] Download, lock, canary, and capability-audit Qwen2-VL-2B.
- [x] Download, lock, and repeatability-canary the Qwen2.5-VL-7B upper bound on one 3090.
- [x] Download, canary, and capability-audit pure-discrete Show-o.
- [x] Download and canary Janus-Pro in its Torch>=2.6 audit-only environment.
- [x] Complete the Janus-Pro 120-image capability audit and six-backbone diagnostic figure.
- **Status:** complete

### Phase 6: Gates and local one-seed pilot
- [ ] Resolve Gate 0 generated-RGB coverage: strict palette 50–66.7% on the fixed 12-image
  canary; contour CV v2 reached 83.3% on those fixed RGBs but only 55% on the decisive balanced
  60-image dev canary. All are below the preregistered 95% threshold; threshold search is frozen.
- [x] Finalize Gate -1 decision: Show-o failed with 2/6 families; Qwen passed 5/6 but differs by
  25pt rather than <=3pt; the hashed decision is red and invokes the capability-floor fallback.
- [ ] Run E1/Tier-B pixel counterfactual audit.
- [ ] Run Gate -1b gradient noise-floor audit.
- [ ] Run paired Naive/RFO miniature E2 and render Figure 1 draft.
- **Status:** stopped_before_e1_and_e2_as_preregistered

### Phase 7: A800 handoff package
- [x] Create Linux/A800 environment lock, H-independent path bootstrap, and migration canary commands.
- [x] Create formal three-seed E2 configs, resumable runner, aggregation, and Gate 2/2b decisions.
- [x] Complete the fail-closed operator runbook and hash-verified Windows-to-Linux manifest rebaser.
- [ ] Verify the handoff on the actual A800 host.
- **Status:** implemented_pending_host_verification

### Phase 8: Verification and delivery
- [x] Run unit/integration/static tests (33 passed; ruff and compileall pass).
- [x] Verify original and rebased manifests, RGB hashes, Tier-B/Tier-D integrity, and no split leakage.
- [x] Review evidence stamps, frozen input hashes, model registries, and unresolved Gate status.
- [x] Deliver code, commands, outputs, and explicit remaining compute/design work.
- **Status:** complete_with_preregistered_hard_stop

## Fixed Decisions
| Decision | Rationale |
|---|---|
| Train `showlab/show-o-w-clip-vit-512x512` only | Strongest defense against the observer-incompetence confound |
| Native Windows first, PyTorch SDPA | User preference; avoids Linux-only optional extensions |
| GPU0 train/generate, GPU1 observe/evaluate | Two 24GB cards have no active NVLink; independent services avoid memory-pooling assumptions |
| Local one-seed miniature loop | Validates implementation and signal measurability without overstating evidence |
| Formal jobs target one A800 80GB | Lowest common server denominator; seeds/arms can be parallelized later without redefining experiments |
| All large paths under H: | C: has insufficient free space and user explicitly selected H: |
| No RL/QLoRA/full pretraining data | Proposal scope is rejection-sampling SFT with BF16 LoRA |

## Errors Encountered
| Error | Attempt | Resolution |
|---|---:|---|
| PowerShell parser rejected piping directly after two sequential `foreach` blocks | 1 | Record results in a task-specific array before formatting; do not repeat the pipeline form |
| Hugging Face Xet transport stalled with zero-byte progress and many CLOSE_WAIT sockets | 1 | Disable Xet locally and use resumable, checksum-verified aria2 range downloads for large locked LFS files |
| Family-blocked manifests made every former `first N` pilot subset badly unbalanced | 1 | Replace truncation with deterministic ID/seed-based stratified sampling everywhere |
| Balanced mock corruption asserted that metadata-derived and visible-answer atoms behave identically | 1 | Preserve existence/count/spatial metadata atoms; require color/size/binding visible answers to change; add all-family regression tests |
| SmolVLM repository inventory included about 6GB of unused ONNX exports | 1 | Stop the transfer, add default non-PyTorch export filters, and verify the required inventory is about 1.02GB |
| InternVL2-1B lock contained a non-existent SHA | 1 | Refuse fallback to `main`, verify the official Hub and `git ls-remote` agree on `0d75ccd166b1d0b79446ae6c5d1a4a667f1e6187`, then correct the lock |
| Show-o geometric generations remained ambiguous or compositionally wrong | 1 | Add train-only prompt-style and 12/60-image generated-domain canaries; keep E1/E2 blocked until deterministic answer coverage reaches 95% or a registered fallback is chosen |
| README used obsolete CLI names and option syntax | 1 | Replace it with commands generated from the implemented parser before handoff |
| Windows Hub cache required symlink privilege for small snapshot files | 1 | Materialize ordinary files directly into the locked snapshot on Windows; keep aria2 size/SHA checks for large LFS objects |
| Transformers refused Janus `.bin` weights under Torch 2.5.1 | 1 | Create a minimal H-drive Janus environment with Torch 2.6.0+cu118; leave the proven Qwen/Intern environment unchanged |
| Conda clone tried to fetch a missing VC runtime from an unavailable mirror | 1 | Preserve the partial clone under `H:\selfsight-tmp`, create a clean venv, and install the exact minimal Janus dependency set |
| Pure-discrete Show-o BF16 attention bias mismatched the official FP32 generation loop | 1 | Keep the inference-only discrete backend on the official FP32 path; main Show-o training remains BF16 |
| First Qwen2.5 canary shell omitted `SELFSIGHT_MODEL_ROOT` | 1 | Preserve the failed directory; source `set_h_env.ps1` and rerun in a new `-r2` directory. The service failed before model loading and never fell back to C: |

## Notes
- Every formal claim requires command, config digest, log, artifact/checkpoint path, metrics, and Gate decision in `progress.md` or `findings.md`.
- Re-read this file before any model download, training scale increase, or A800 migration.
