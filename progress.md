# Progress Log

## Session: 2026-08-27

### Phase 0: Repository and host setup
- **Status:** complete
- **Started:** 2026-08-27
- Actions taken:
  - Read the complete proposal and user-approved implementation plan.
  - Inspected host GPU, memory, storage, Python, Git LFS, and WSL state.
  - Researched official model repositories, weight sizes, training dependencies, and benchmark sources.
  - Locked native Windows, H-drive storage, one-seed local pilot, single-A800 formal target, and CLIP-ViT Show-o main checkpoint.
  - Created persistent planning/evidence files.
  - Confirmed there are no additional repository instructions or pre-existing code files.
  - Rechecked H-drive free space (147GB); deferred bulk model snapshots until cleanup while continuing code implementation.
  - Initialized an empty Git repository.
  - Captured immutable repository revisions using `git ls-remote` after the REST batching approach failed to yield output.
  - Created `H:\selfsight-envs\core` from the host's working CUDA environment, disabled user-site leakage, and verified Torch CUDA access to both RTX 3090s.
  - Synchronized locked Show-o and Janus source revisions to H: and added the implicit `microsoft/phi-1_5` tokenizer/config dependency to the model lock.
- Files created/modified:
  - `task_plan.md`
  - `findings.md`
  - `progress.md`
  - Phase 0 configuration/bootstrap/model-lock files
  - `src/selfsight/` core schema, data, observer, RFO, training, and analysis modules
  - immutable model/repository downloader scripts

### Phase 1–4 implementation
- **Status:** complete
- Implemented deterministic global split isolation: 2,400 train + 200 fixed probe + 600 outcome, with 3,200 unique scene signatures.
- Implemented exact Tier-B composition: 100 deletion/count, 100 color, 50 left/right, 50 size-relation, and 100 binding-swap pairs.
- Implemented an RGB pixel verifier and detected a real counterfactual construction defect during audit (31/400 pairs initially failed pixel-level flipping); corrected same-x left/right pairs and size-label semantics.
- Implemented a blind observer JSONL contract. Expected answers and original prompts are stripped before serialization and forbidden keys are rejected recursively.
- Implemented paired selection, H-drive candidate manifests, PEFT LoRA restrictions, adapter-only resume checkpoints, LoRA gradient vector/per-block analysis, noise intervals, segmented D*/D_g estimates, formal paired bootstrap helpers, and Gate 2/2b decision functions.
- Implemented a three-panel, shared-x Figure 1 design using colorblind-safe colors plus line/marker redundancy; no dual y-axis is used.
- Implemented real Show-o generation, MMU observation, T2I LoRA loss, 25% frozen-tower understanding replay, accumulated LoRA gradients, and exact adapter-only checkpoint restoration.
- Implemented Gate -1 observer audits, E1, Gate -1b, the paired/resumable local training loop, checkpoint evaluation, migration canaries, and formal three-seed E2 aggregation with Gate 2/2b stop decisions.
- Rebuilt the complete registered dataset after adding explicit Tier-B intervention intent to every pair.
- Completed the full 10-round mock pipeline: 2,560 candidates/decisions, both arms resumed exactly at round 10/step 250, and all figure formats were exported with non-scientific evidence stamps.
- Visually reviewed the color and grayscale Figure 1 previews; line/marker redundancy, panel alignment, legends, and clipping are acceptable, and automated layout QA passed.
- Exported a blinded 120-image manual reference audit packet, balanced 20 per question family, with separate digest-protected answer key and an automated 98% pass rule.
- Created and validated the isolated H-drive observer environment (Torch 2.5.1+cu121, Transformers 4.57.6, locked Janus checkout); user-site is disabled and both RTX 3090s are visible.
- Captured exact core and observer package/CUDA inventories under `H:\selfsight-envs\locks`.
- Replaced all local `first N` manifest truncation with deterministic balanced sampling after
  detecting that the family-blocked manifests had produced single-family pilot subsets.
- Re-ran the full 10-round mock pipeline with balanced train/probe/outcome subsets at
  `H:\selfsight-runs\mock-pilot-balanced-v2`; all 2,560 decisions, resume checks, and figure
  exports completed.

### Phase 5–6 real-model and Gate evidence
- **Status:** in_progress_with_gate_0_red
- Downloaded the complete locked core group to H: and independently verified the large LFS
  lengths and SHA-256 values.
- Ran the real Show-o canary at `H:\selfsight-runs\canaries\showo-20260827T1905-r2`:
  25-step generation, RGB reload/MMU, 14,155,776-parameter LoRA attach, T2I+MMU backward,
  AdamW update, deliberate adapter corruption/exact restore, and checkpoint resume all passed;
  peak allocated GPU memory was 5,428,942,336 bytes. The model answered the canary MMU question
  incorrectly (`yes` instead of `no`), so observation competence remains a measured Gate.
- Visually audited four train-external prompt styles. All produced object fusion, extra geometry,
  misplaced objects, or dropped attributes; none made the exact-palette verifier reliable.
- Added a train-only balanced generated-domain canary. On 12 fixed images, the 25-step strict
  verifier had 50% answer coverage/16.7% accuracy; the official README 50-step setting had
  66.7%/25%. Neither approached the 95% parseability threshold.
- Added a separate deterministic contour verifier for approximate generated geometry while
  retaining the exact verifier for program references. On the same fixed 50-step RGBs, its
  current development version reached 83.3% answer coverage and 41.7% correctness. This is an
  engineering diagnostic only, has not passed a generated-image human audit, and does not clear
  Gate 0.
- Completed the decisive balanced 60-image train-only canary at
  `H:\selfsight-runs\canaries\generated-domain-cv2-dev60-20260827`: coverage 55%, correctness
  23.3%, intended-scene recovery 6.7%, and exact-scene recovery 0%. Binding coverage was 0% and
  spatial coverage 30%. Visual review confirmed genuine multi-instance ambiguity and composition
  errors, so verifier-threshold development is frozen rather than optimized toward the Gate.
- Filtered ONNX/OpenVINO/CoreML/GGUF/TF/Flax exports from model acquisition after SmolVLM's Hub
  inventory revealed about 6GB of irrelevant exports. Required SmolVLM inventory is 1.02GB and
  is registered at its locked SHA.
- Ran the SmolVLM 6-image repeatability canary: 100% repeat consistency, 83.3% accuracy.
- Ran the balanced 120-image/360-question SmolVLM Gate -1 audit: macro open accuracy 83.3%,
  no material yes-bias (3.75pt), but only color/existence/binding reached 80%; capability Gate
  failed because only 3/6 families passed. Forced-choice order agreement was 41.7%.
- Corrected a malformed InternVL2-1B lock only after the official Hub and `git ls-remote`
  independently returned the same full SHA. The locked snapshot is complete on H:.
- Quarantined third-party model stdout from the observer JSONL channel after InternVL remote code
  emitted an informational line. The rerun completed with 100% repeat consistency; the client now
  fails explicitly on any future non-JSON protocol output.
- Ran the balanced 120-image/360-question InternVL Gate -1 audit: macro open accuracy 70.8%,
  color/existence/binding were the only families at or above 80%, and absolute yes-bias was 30pt.
  Both capability and bias conditions failed. Subsequent Show-o, Qwen, pure-discrete Show-o, and
  Janus audits are recorded below.
- Added fail-closed prerequisite validation to E1, Gate -1b, local E2, and the formal E2
  orchestrator. Gate identity, condition consistency, generated-domain sample basis, metric basis,
  and the non-lowered 95% threshold are checked before outputs or models are created.
- Ran the balanced 120-image/360-question Show-o Gate -1 audit: macro open accuracy 65.0%; only
  color (100%) and binding (95%) reached 80%, while spatial/existence/size/count were
  60/70/40/25%. Bias control passed at 5pt, but the required four-family capability floor failed.
  Gate -1 therefore cannot turn green even if Qwen is a strong detector.
- Completed the locked Qwen2-VL-2B snapshot, 6-image repeatability canary, and balanced
  120-image/360-question audit. Repeat consistency was 100%; the full audit reached 90.0% macro,
  5/6 families at >=80%, and 6.25pt yes-bias. It is capable but 25pt above Show-o, so it violates
  the <=3pt capability-matching requirement.
- Finalized `H:\selfsight-runs\gate-minus-1\decision.json` with source paths and SHA256 hashes.
  The reference and Show-o bias conditions pass; Show-o capability, matched-observer availability,
  and both downstream matched-observer conditions fail. The registered next action is to stop E1,
  Gate -1b, and E2 and use the capability-floor analysis fallback.
- Implemented and visually audited the Figure 2 fallback at
  `H:\selfsight-runs\gate-minus-1\figure2-local`: colorblind-safe heatmap, numeric labels,
  threshold outlines, separate bias panel, PNG/PDF/SVG/grayscale exports, tidy CSV, and QA JSON.
- Materialized the registered Tier-D mechanism subset: 300 Tier-A images (50 per family) plus
  150 complete Tier-B pairs (30 per intervention category, 300 images). The selection digest is
  `a77263dd407f865cb705a1b4600e1160804d395ff4977935d4332a103a0f4f32`; all 600 RGBs and
  atomic answers pass at `H:\selfsight-runs\audits\tier-d-audit.json`.
- Added a non-destructive dataset-manifest rebaser for A800 migration. A complete Windows rebase
  dry run left the source manifests unchanged and revalidated 3,200/3,200 references, 400/400
  Tier-B pairs, and 600/600 Tier-D images from the rebased view.
- Bound every E1/Gate-1b/evaluation/formal detector invocation to the exact selected model,
  revision, and capability-audit SHA-256 from Gate -1. A real red-Gate CLI test again exited before
  creating output.
- Fixed Windows Hub small-file materialization to avoid symlink privilege requirements while
  retaining resumable length/SHA-verified aria2 for large LFS weights.
- Downloaded and registered Janus-Pro-1B and pure-discrete Show-o at their locked revisions.
  The pure-discrete 6-image canary passes with 100% repeat consistency; its 120-image audit reaches
  80.0% macro but only color/binding (2/6 families) clear 80%, so its capability Gate fails.
- Created `H:\selfsight-envs\janus` with Torch 2.6.0+cu118 after current Transformers correctly
  refused unsafe `.bin` loading under Torch 2.5.1. The Janus r3 canary passes with 100% repeat
  consistency and 83.3% canary accuracy. Its balanced 120-image audit reaches 87.5% macro,
  5/6 passing families, 7.5pt yes-bias, and 15.8% abstention; it is 22.5pt above Show-o and therefore
  diagnostic rather than matched.
- Rendered and visually reviewed the six-backbone Figure 2 expansion at
  `H:\selfsight-runs\gate-minus-1\figure2-all-backbones-local`; PNG/PDF/SVG/grayscale, tidy CSV,
  and QA JSON all agree, and the artifact explicitly records that the Gate -1 decision is frozen.
- Materialized all 16,595,981,281 registered bytes of the locked Qwen2.5-VL-7B upper-bound model.
  Its `-r2` six-family canary loads on GPU1 in 20.46 seconds with 16,636,220,928 peak allocated
  bytes, reaches 6/6 first-answer accuracy and 100% repeat consistency over 12 requests. It remains
  outside the frozen Gate decision and no full post-hoc audit is run.

## Test Results
| Test | Input | Expected | Actual | Status |
|---|---|---|---|---|
| Session catchup | project path | No stale planning context | No report/output | pass |
| Split generator smoke | registered scales | 2400/200/600 and no duplicate signature | 2400/200/600; 3200 unique | pass |
| Reference verifier smoke | first 60 outcome scenes | exact answers | 60/60 exact | pass |
| Tier-B composition | outcome scenes | 400 with registered category counts | 400; 100/100/50/50/100 | pass |
| Tier-B first pixel audit | all 400 pairs | both images parse and answer flips | 369/400 | fail_fixed |
| Reference verifier full audit | 3,200 scenes, six families | accuracy/coverage >=98% | 3,200/3,200; both 100% | pass |
| Split isolation full audit | train/probe/outcome | zero prompt/template/signature overlap | all pairwise overlap counts 0 | pass |
| Tier-B corrected full audit | 400 registered pairs | 400 parse and answer flips | 400/400 across 100/100/50/50/100 | pass |
| Python test suite | unit/integration tests | all pass | 33 passed | pass |
| Python compileall | src/scripts/tests | no syntax errors | pass | pass |
| Mock paired pipeline | 10 rounds x 2 arms | paired candidates, resume, metrics, figures | 2,560 decisions; both arms round 10/step 250 | pass |
| Figure visual QA | color + grayscale preview | readable, aligned, redundant encoding, no clipping | inspected; automated QA passed | pass |
| Manual audit tooling | blinded export/score test | separate key, exact IDs, >=98% rule | 120-item real packet + unit test | pass_pending_human_labels |
| Observer environment isolation | H-drive observer Python | no user-site; CUDA and imports pass | prefix/modules on H:, 2 GPUs | pass |
| Real Show-o canary | generate/MMU/LoRA/update/restore/resume | all engineering operations stable | all operations pass; MMU answer wrong | engineering_pass_capability_pending |
| Balanced mock rerun | balanced six-family local subsets | 10 rounds, paired decisions, resume, figures | 2,560 decisions; complete | pass_non_scientific |
| Generated-domain strict canary | 12 balanced train-only prompts | answer coverage >=95% | 50% at 25 steps; 66.7% at 50 steps | fail |
| Generated contour verifier dev | fixed 12 generated RGBs | answer coverage >=95% plus later human agreement | 83.3% coverage; human audit not run | fail |
| Generated-domain decisive dev | 60 balanced train-only prompts | answer coverage >=95% | 55% coverage; 23.3% correctness | fail_hard_stop |
| SmolVLM repeatability canary | six balanced reference images x2 | repeat agreement >=90% | 100%; accuracy 83.3% | pass |
| SmolVLM Gate -1 local | 120 balanced references | >=4 families at >=80%; bias <=10pt | 3 families; bias 3.75pt | fail |
| InternVL repeatability canary | six balanced reference images x2 | repeat agreement >=90% | 100%; accuracy 50% | pass_engineering_only |
| InternVL Gate -1 local | 120 balanced references | >=4 families at >=80%; bias <=10pt | 3 families; bias 30pt | fail |
| Show-o Gate -1 local | 120 balanced references | >=4 families at >=80%; bias <=10pt | 2 families; bias 5pt | fail_hard_stop |
| Qwen2-VL repeatability canary | six balanced reference images x2 | repeat agreement >=90% | 100%; accuracy 83.3% | pass |
| Qwen2-VL Gate -1 local | 120 balanced references | >=4 families at >=80%; bias <=10pt; within 3pt of Show-o | 5 families; bias 6.25pt; macro delta 25pt | pass_capability_fail_matching |
| Gate -1 final decision | reference + Show-o + three heterogeneous audits | every registered condition green | 3/6 conditions red; evidence hashed | fail_hard_stop |
| Figure 2 fallback QA | color/grayscale + vector/raster exports | no clipping; redundant threshold encoding | visual and programmatic QA pass | pass |
| Tier-D registered subset | 300 Tier A + 150 complete Tier-B pairs | 600 unique, balanced, RGB/atom exact | 600/600; all composition checks true | pass |
| Manifest rebase dry run | five copied manifests | path-only rewrite; every RGB unchanged | source/rebased hashes identical on Windows; all audits pass | pass |
| Detector identity binding | red Gate + arbitrary detector args | fail before output/model load | exit nonzero; output absent | pass |
| Pure-discrete Show-o canary | six balanced references x2 | repeat agreement >=90% | 100%; accuracy 66.7% | pass |
| Pure-discrete Show-o audit | 120 balanced references | >=4 families at >=80%; bias <=10pt | 2 families; macro 80%; bias 0pt | fail_diagnostic |
| Janus-Pro repeatability canary | six balanced references x2 | repeat agreement >=90% | 100%; accuracy 83.3% | pass |
| Janus-Pro diagnostic audit | 120 balanced references | >=4 families at >=80%; bias <=10pt; within 3pt of Show-o | 5 families; macro 87.5%; bias 7.5pt; macro delta 22.5pt | pass_capability_fail_matching |
| Six-backbone Figure 2 QA | six audit reports; color/grayscale/vector/raster | no clipping; redundant threshold encoding; frozen-decision label | visual and programmatic QA pass | pass |
| Qwen2.5-VL-7B upper-bound canary | six balanced references x2 on GPU1 | load within 24GB; repeat agreement >=90% | peak 16,636,220,928 bytes; 100% agreement; 6/6 accuracy | pass_diagnostic |

## Error Log
| Timestamp | Error | Attempt | Resolution |
|---|---|---:|---|
| 2026-08-27 | PowerShell `empty pipe element` while querying revision locks | 1 | Next attempt stores both loops in `$revisionRows` and formats after collection |
| 2026-08-27 | Tier-B pixel audit found 31 non-flipping relations | 1 | Exclude equal-x sources for left/right and compare detected categorical sizes rather than shape areas; full rerun pending |
| 2026-08-27 | Windows rejected `fsync` on a read-only hard-render handle (`Errno 9`) | 1 | Reopen the completed temporary PNG as `r+b` before `fsync`; rerun full suite |
| 2026-08-27 | Filtered Show-o clone remained silent during checkout and was interrupted, leaving an index-only worktree | 1 | Preserve it as `.incomplete`; switch repository sync to cone-mode sparse checkout of research-relevant code paths with explicit progress |
| 2026-08-27 | Full 3,200-image audit was 99.53%; 15 spatial labels differed because triangle mass centroids sit below layout centers | 1 | Define layout relations using detected bounding-box centers; rerun the complete audit rather than accepting a merely above-threshold result |
| 2026-08-27 | PyTorch 2.2.1 Windows wheel transferred at ~85KB/s (estimated >8 hours) | 1 | Stop the non-resumable pip attempt; add an explicit local-Conda-clone bootstrap path using the host's working Torch 2.5.1+cu121 environment, while keeping exact 2.2.1 on A800 and recording the local deviation |
| 2026-08-27 | Successful Conda clone was followed by a missing `Scripts/python.exe` error | 1 | Support both Conda root `python.exe` and venv `Scripts/python.exe`; resume the completed H-drive clone without copying again |
| 2026-08-27 | Hugging Face Xet download stayed at zero bytes with proxy/CLOSE_WAIT connections | 1 | Preserve cache, disable Xet, and switch large locked LFS files to resumable aria2 with expected-size and SHA-256 verification |
| 2026-08-27 | Audit CLI rejected README-style `--data-root/--report` flags | 1 | Use the implemented positional root plus `--output` interface and correct the README |
| 2026-08-27 | Former local subsets used the first rows of family-blocked manifests | 1 | Implement and test deterministic stratified sampling for train, probe, outcome, canary, E1, observer, and migration paths |
| 2026-08-27 | First balanced mock rerun failed on color/size/binding corruption semantics | 1 | Separate metadata-derived from visible-answer invariants; all six-family corruption tests pass and balanced v2 completes |
| 2026-08-27 | Show-o import failed on missing `jaxtyping`/`typeguard` | 1 | Add exact official versions to the training environment and recapture the environment lock |
| 2026-08-27 | SmolVLM plan resolved 6.9GB because ONNX exports were included | 1 | Interrupt safely, default-ignore non-PyTorch exports, re-plan to 1.02GB, and finish the required snapshot |
| 2026-08-27 | InternVL2-1B revision returned Hub 404 | 1 | Refuse mutable fallback; correct the transcribed SHA only after two official-resolution methods agree |
| 2026-08-27 | InternVL remote code printed generation text to the JSONL stdout channel | 1 | Redirect third-party load/inference stdout to stderr, make protocol violations explicit, and rerun in a new immutable canary directory |
| 2026-08-27 | Hub small-file completion failed with Windows symlink privilege error | 1 | Use `local_dir` ordinary-file materialization for Windows snapshots; rerun both locked audit models to READY |
| 2026-08-27 | Janus `.bin` load was blocked under Torch 2.5.1 by the CVE-2025-32434 safety floor | 1 | Build a separate minimal Torch 2.6.0+cu118 Janus environment and capture its exact lock |
| 2026-08-27 | Partial Conda clone required an unavailable VC runtime mirror | 1 | Move the failed clone recoverably to `H:\selfsight-tmp\failed-janus-clone-20260827T2052`; use a clean H-drive venv |
| 2026-08-27 | Pure-discrete Show-o returned a query/attention-bias dtype mismatch in BF16 | 1 | Restore the official inference-only FP32 path; r2 canary and the full audit completed |
| 2026-08-27 | First Qwen2.5 canary shell omitted `SELFSIGHT_MODEL_ROOT` | 1 | Keep the failed output, source `set_h_env.ps1`, and rerun in an immutable `-r2` directory; the service failed before loading and did not write to C: |
| 2026-08-28 | A1 peak-memory reset rejected an uninitialized `torch.device` on Windows | 1 | Normalize the index, initialize the selected CUDA context, test the real API, and rerun as r2 |
| 2026-08-28 | A1 treated the official `WanVAE` wrapper as an `nn.Module` | 2 | Freeze/audit its internal `.model`, add residual-meta checks, and rerun as r3 |

## 5-Question Reboot Check
| Question | Answer |
|---|---|
| Where am I? | Phase 8: hard-red capability fallback and all local diagnostic audits are complete; delivery verification remains |
| Where am I going? | Lock the code/evidence handoff; any attempt to clear the red Gates requires a user-approved material design revision |
| What's the goal? | A reproducible local one-seed research loop plus formal A800 package |
| What have I learned? | See `findings.md` |
| What have I done? | Implemented and tested the full code path; completed core downloads, balanced mock closure, generated-domain diagnostics, the six-backbone ability ladder, and fail-closed real experiment entry points |

## 2026-08-28 — v2.2 Joint Readiness revision

- The user approved a material backbone/design revision after the frozen Show-o v1 Gate -1 red
  result. The new scientific claim is conditional on a unified model passing joint generation and
  observation readiness for a registered subset of families.
- Pushed annotated tag `v2.1-showo-gate-red` at
  `5e5543853aaf0d6bf8428e9c9e30e049b01d6a9d` and created local branch
  `experiment/v2.2-joint-readiness`.
- Verified official Show-o2 source commit
  `45a5a2de01d1ebd10cd5864d29310a76476cdf23` and current immutable Hub revisions for base 1.5B,
  1.5B-HQ, and 7B. Confirmed base uses the official 432x432 path while HQ supplies the 512x512
  configuration.
- Replaced the active task plan with v2.2 phases and fail-closed Gate -2 sequencing. The v2.1 plan
  remains recoverable exactly from the pushed tag and Git history.
- Tooling note: `apply_patch` rejected a single patch containing delete-and-recreate operations for
  `task_plan.md`; splitting it into two atomic patches completed successfully without losing the
  tagged v2.1 version.
- Source note: the first sparse-checkout command outlived the command window and retained its Git
  locks while downloading. The live process was verified and allowed to finish; `show-o2/` is now
  present at the same locked HEAD, and no lock file was removed.
- API note: direct anonymous PowerShell calls to the public Hugging Face JSON API returned an
  authentication error in this host's proxy path. The installed `huggingface_hub` client resolved
  the same public repositories and immutable file metadata successfully without printing or
  modifying credentials.
- Added the v2.2 proposal amendment, frozen v2.1 evidence index, exact readiness thresholds, locked
  Qwen observer config, first Show-o2 candidate config, and gate-ordered model groups. Inspecting the
  immutable Hub config (which overrides the demo YAML at load time) confirmed the official path uses
  `load_from_showo=false`; the four-snapshot plan therefore retains Qwen weights and totals
  12,775,937,051 bytes.
- Implemented the isolated family-minimal v2.2 benchmark and build script. Targeted tests pass 5/5;
  Ruff found one local import-order issue, which was corrected with `apply_patch` before continuing.
- Added the unified-backbone protocol, a backwards-compatible Show-o v1 negative-control wrapper,
  and a lazy local-only Show-o2 adapter implementing official 432 generation, deterministic RGB
  observation, latent targets, audited explicit-target LoRA attachment, generation/replay loss,
  gradients, adapter-only checkpointing, and resource reporting. Package-origin collisions fail
  closed so v1/v2 cannot contaminate one process.
- Added the Gate -2 finalizer and prerequisite validator. Thirteen focused data/backbone/decision
  tests pass, including evidence-tamper and gate-ordered fallback cases; targeted Ruff is clean.
- Implemented A1/A2 runtime audits and pure summary tests, including repeated open answers,
  forced-choice order reversal, abstention, yes-bias, native-resolution RGB evidence, peak GPU
  memory, and the actual shared-transformer module tree needed before any LoRA target is selected.
- Implemented A3 as a precision-first three-command workflow: generated K=1 coverage for all 60
  prompts, K=4 only for A2-retained families, equal-denominator fixed-seed stability, a blinded
  all-answered-K1 human packet, and fail-closed scoring with per-family precision. The audit packet
  excludes prompt, scene graph, intended answer, verifier answer, and generation seed.
- Implemented an A1-hash-bound LoRA target selection artifact and A4 backward/resume canary. A4
  cannot start until A1, A2, A3 generated measurability, and the completed blind precision audit are
  green; it checks generation and replay losses, LoRA-only trainability, optimizer step, intentional
  corruption, exact adapter restoration, resumed optimizer step, and adapter-only checkpoints.
- Added exact base/HQ/7B candidate profiles and gate-ordered language dependencies. The generic
  adapter now derives latent geometry, sequence/token counts, inference steps, language-base path,
  and dependency locks from the selected candidate rather than assuming the 432px 1.5B profile.
  Nineteen focused v2.2 tests pass and targeted Ruff is clean.
- Materialized the complete isolated benchmark at `H:\selfsight-data\selfsight-v2.2`: 6 canary,
  120 reference, and 60 generated-domain rows. Registry hashes are
  `675a999418fe13949ab6d8d51df1af93732c7351852873d329ce5f8b20024141` (canary),
  `2fe81a6e3d83fac2bb1c04b9158046e32b4f83b5f2885ecab317fddc868b819a` (reference), and
  `0059d62a168394adfefd7882f8e22419b86734d41e7595c09da2e9c27d3ab78a` (generated).
- Full repository Ruff and all 52 tests pass. Committed the first v2.2 implementation as
  `8fb06b0` and pushed branch `experiment/v2.2-joint-readiness` to `origin`.
- Implemented `render_readiness_matrix.py` under the SciPilot scientific-figure workflow. The
  first render exposed an overlapping family label and rasterized SVG; the second uses pure vector
  rectangles, shifted labels, exact annotated cells, colorblind-safe blue/orange plus hatch/text,
  and a 600-DPI grayscale preview. Programmatic layout QA, direct color/grayscale review, PNG strict
  compliance, and SVG compliance pass; the PDF checker emits only a conservative Type-0 font
  embedding warning and no Type-3/FAIL.
- Materialized `H:\selfsight-envs\showo2` by cloning the working H-drive CUDA environment, then
  installed the minimal v2.2 dependency extra and captured
  `H:\selfsight-envs\locks\windows-showo2.json`. An independent canary confirms Python 3.10.19,
  Torch 2.5.1+cu121, both RTX 3090 cards, Transformers 4.47.0, Diffusers 0.31.0, PEFT 0.11.1,
  Timm 1.0.12, Torchdiffeq 0.2.5, OpenCV 4.10.0, and the locked Show-o2 source origin.
- The cloned environment inherited inconsistent pip code/metadata and initially raised
  `ImportError: cannot import name BuildDependencyInstallError`. Reinstalled the exact working
  core-environment pip into the clone and added the same self-repair check to the bootstrap script;
  the rerun completed without recreating the environment.
- A full-suite rerun exposed a Windows-only Matplotlib test-order failure when Tk was selected in a
  headless process. All three publication-figure modules now select `Agg` before importing pyplot;
  Ruff and all 53 tests pass again, followed by a fresh Show-o2 CUDA import canary.
- Downloaded only the registered rank-1 group (12,775,937,051 expected bytes): Show-o2-1.5B, the
  single Wan2.1 VAE file, SigLIP SO400M, and Qwen2.5-1.5B. Independent local SHA-256 checks match
  every generated registry entry; HQ and 7B remain absent. H: had 62.35GB free after completion.
- A1 attempts r1/r2 failed before producing evidence: r1 exposed a Torch 2.5/Windows peak-memory
  API requirement for an initialized integer-indexed CUDA context; r2 exposed that official
  `WanVAE` is a wrapper rather than an `nn.Module`. Added a shared CUDA initialization helper,
  updated all affected canaries, froze/audited `WanVAE.model`, and added a fail-closed meta-tensor
  materialization audit.
- A1-r3 passed all four engineering checks for six native 432x432 samples in 84.32 seconds. Peak
  allocated GPU memory was 8,107,403,776 bytes; the loaded checkpoint reports 3,063,740,640 total
  and zero trainable parameters. Report SHA-256 is
  `a815b4d53c2c3f4dd12c01dde70e447ace3579501a0b098e1467bb0566f4d6ba`; rows and LoRA-tree
  hashes are `2bb81e7d7b6d8551485212b160f4b30e4d751afa7164285a42b61fce5bf5c8e7` and
  `4eccd45f86dbf04a91e4a1577901f93badf3442ac360bf05d92c65b5511f7ab2`.
- All six A1 reference and generated-image atomic answers repeated exactly and matched the expected
  canary answer. Direct RGB review nevertheless found a possibly over-counted/cropped count image
  and touching spatial/binding shapes; these are not promoted to correctness evidence and remain
  for A3 verifier plus blind-human adjudication.
- Full Ruff and all 60 tests pass after the CUDA-context, WanVAE, and materialization-audit fixes.
- A2-r1 passed in 90.53 seconds over all 120 balanced program renders. Open accuracy is 100% for
  existence/color/spatial/binding, 90% for count, and 55% for size, so the registered retained set
  for A3 is `{existence, count, color, spatial, binding}`. Macro accuracy is 90.83%, repeat
  agreement 100%, abstention 0%, and absolute yes-bias 0 points. Report SHA-256 is
  `c21d19c4623ea18b30bcb2306c8d8c8f1c3fe21c1024707cb885f56ea29a92e5`; rows SHA-256 is
  `7dd7654e5663a8376f11b1a4ba7d8c0a37f40fed6a85aee8646cca9f9746dcfe`.
- A2's 11 errors are structured rather than diffuse: two count-4 images were answered as 3, and
  nine large objects were answered as small. Size is therefore excluded before A3 K=4 and cannot
  re-enter the main eligible-family set through generated-domain performance.
- Reworked E1 against the public `ModelAdapter.observe_atoms` contract and added a fail-closed
  v2.2 entry path. It validates every Gate -2 evidence hash, exact Show-o2/Qwen identities, the
  frozen public-observer per-family floor, and filters Tier B to the decision's eligible families.
- Added v2.2 Gate -1b wiring without changing the frozen v2.1 route. The runner validates the A4
  report's exact target-config hash and selection digest before constructing an audited Show-o2
  adapter. Gradient batches now preserve each backbone's actual objective: flow-matching
  `Showo2GenerationBatch` for Show-o2 and masked discrete `ShowoSFTBatch` for Show-o v1.
- Added non-finite/out-of-range observer-evidence rejection and adapter-contract tests. Full Ruff
  and all 77 tests pass; the only warnings are Pillow's already-known future PDF palette-mode
  deprecation notices from publication-figure tests.
- Added explicit local/A800 Show-o2 experiment profiles while preserving the frozen Show-o v1
  profiles. They lock 432x432/50-step Show-o2 generation, audited-target-only LoRA, GPU0/GPU1 local
  separation, and single-GPU A800 placement.
- Reworked the paired local E2 loop for the v2.2 backbone contract. Its blind JSONL subprocess now
  has an explicit `showo2` backend that loads an independent frozen step-0 checkpoint on GPU1. The
  trainable arm dispatches flow-matching generation and atomic-QA replay batches, filters the prompt
  pool to Gate -2 families, and refuses checkpoint resume unless the base config, Gate hash,
  backbone hash, and exact LoRA module list all match.
- Reworked checkpoint evaluation to reconstruct Show-o2 with the same audited targets, validate the
  completed run's model identity, filter outcome/probe manifests before deterministic sampling, and
  load adapters with the saved joint training-contract digest. The GDA evaluator now passes the
  adapter explicitly to the backbone-specific gradient-batch dispatcher; this fixes a runtime-only
  omission that static imports and the prior v1-only suite could not exercise.
- A3-r1 completed 210 prompt-question rows in 2693.77 seconds but is not admissible Gate evidence.
  Its automatic result was red (70% overall coverage; retained-family spatial coverage 40%;
  Oracle@4 84% overall but 60% for count/spatial; 16pt fixed-seed swing). The report SHA-256 is
  `a8aa9f1bd3ba65368804ad8efa6756c385b3349721ca1074dd1fb09cb96cc660`; rows SHA-256 is
  `07e3cf63eceaf8b87f16e90f9aae32800ea7bc16206fa4742a428154cdcc0807`.
- Post-run artifact audit found only 186 unique candidate IDs/image paths for 210 rows. The A3
  manifest intentionally contains some same-drawing-prompt/different-question cases, while the
  adapter's filename identity used only prompt hash plus seed. Twenty-four paths were therefore
  overwritten even though deterministic RGB bytes were reproducible. A3 now namespaces checkpoint
  IDs by scene ID, writes to an output-specific image root, and rejects any row/ID/path cardinality
  mismatch before writing a report. Collision-safe r2 is running under the unchanged manifest,
  fixed seeds, model revision, and verifier.
- Completed the v2.2 A800 implementation path. The bootstrap creates a dedicated Show-o2
  environment; migration canaries use the green Gate/backbone and eligible probe families; the
  comparison hashes both hosts' rows/summaries and requires full backbone identity equality.
- The formal orchestrator now validates three seed configs differ only in seed/profile, rechecks all
  Gate -2/public-observer/A4/migration hashes and identities, enforces unique eligible sample
  capacity before creating outputs, runs Show-o2 Gate -1b/training/evaluation sequentially, and
  aggregates only the three registered seeds.
- Added a decision-bound eligible E2 data builder. It redistributes exactly 2400/200/600 cases over
  the selected families, preserves train/probe/outcome template separation, excludes every A1/A2/A3
  signature, renders deterministic replay references, and writes a Gate-hashed registry. The
  portability layer now rebases this three-manifest dataset without requiring unrelated Tier B/D.
- Added a preregistered upstream-stop finalizer for automatic A3 failures. It revalidates the exact
  A1/A2/A3 identity and runtime locks, recomputes the automatic A2/A3 checks, requires one unique
  candidate ID and image path per A3 row, refuses to run when the automatic A3 checks are green,
  and emits an immutable red decision with blind-human/A4 explicitly skipped. Seven focused joint
  readiness tests pass.
- Full repository verification after fallback-download integration passes: Ruff clean and 84/84
  tests green. The only
  warnings are the already-known future Pillow PDF palette-mode deprecations in figure-export tests.
- Extended the publication readiness matrix for upstream-stop decisions. Registered but unmeasured
  Gate/family cells now use gray dotted `N/T`, while measured failures remain orange hatched
  `FAIL/NO`; the CSV carries an explicit `measured` field and the QA profile counts six untested
  precision cells. The focused renderer tests, vector/grayscale exports, and manual PNG review pass.
- Closed the fallback-download authorization gap. `readiness_fallback_hq` and
  `readiness_fallback_7b` now require `--predecessor-decision` for actual downloads (plans remain
  read-only), reject rank/model ladder skips, green or inconsistent decisions, incomplete/tampered
  evidence, and bind the predecessor decision hash into every downloaded model registry. Direct
  `--model-id` selection cannot bypass the group check.
