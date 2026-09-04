# Findings

## Initial State

- 项目根目录原先没有 `task_plan.md`、`findings.md` 或 `progress.md`。
- 当前任务为只读项目审计；规划文件存放在隐藏的 `.planning/2026-09-04-project-audit/` 中。

## Evidence Log

### Repository identity and shape

- Package name is `selfsight` v0.1.0, Python 3.10–3.12, using NumPy/Pillow/SciPy/Pandas/Matplotlib for core analysis, PyTorch-adjacent stacks via optional `training`, `observer`, and `showo2` dependency groups.
- The project studies **visual self-confirmation in unified multimodal models**: whether a model that generates an incorrect image later accepts the image as correct because it matches its own generation intent.
- The active branch is `experiment/v2.3-rfo-gold`, 62 commits ahead of `origin`; latest local commit is `257a9c5` (`Give the Gate B waiter a liveness check instead of a wall clock`).
- There is one pre-existing modified product file: `scripts/v4_train.py`; it belongs to the user and must remain untouched during this read-only audit.
- The repository has evolved through v2.x, v3, and now v4. Current code includes dedicated `src/selfsight/v4/` modules and 11 v4 test files.

### Documentation contract

- `README.md` says the four authoritative roles are: proposal/specification, `STATUS.md` current state, `docs/EVIDENCE_LOG.md` measurements, and `docs/RUNBOOK.md` operations.
- `STATUS.md` is explicitly the single current-state source and reaches at least section 34, including results through 2026-09-04.
- The proposal defines Gates A–D, while current STATUS also introduces D* / L3 preview work.
- `docs/RUNBOOK.md` is partly historical: it explicitly says key v3 scripts were not yet implemented. Current v4 code appears to have superseded portions of that runbook, so implementation and STATUS must be checked before treating it as current.

### Scientific invariants from README

- Observer sees only RGB plus an atomic question—never prompt, latent/image tokens, KV cache, or source label.
- Training selector and gradient detector must be decoupled.
- Balanced candidate banks are for gradient probes; training/outcome curves use natural pools.
- SCFR and gradient-cosine claims require denominators/coverage and paired uncertainty intervals.
- Single-seed paths and mock results cannot support significance claims; failed/incomplete evidence is preserved.

### Execution setup

- Intended hardware: local Windows with 2×RTX 3090 plus Linux server with 2×A800 80GB; each GPU is an independent worker.
- Paths are rooted under the checkout except model weights (`H:\selfsight-models`).
- The new `configs/v4_l3_preview.yaml` openly labels the current L3 preview as underpowered: 228 available specs versus roughly 1,470 required; 30 optimizer updates can validate/falsify the pipeline but cannot falsify the scientific hypothesis.

### Errors / caveats

- `rg ... docs/prereg/*.md` failed because PowerShell/Windows passed the wildcard literally to ripgrep. Use `-g '*.md' docs/prereg` or explicit paths next time.

### Current scientific picture (STATUS/proposal, first pass)

- Current backbone is **standard `showlab/show-o2-1.5B` rev `07ec1658`, native 432px**, not the HQ 512px variant. Earlier v2/v3 audit results on HQ remain historical evidence but are not the current backbone result.
- v4 replaces synthetic colored geometry with verifiable natural scenes containing three everyday objects. Correctness is a single family-independent exact multiset match over `(object, color)` pairs.
- Question families are 2AFC: existence, counting, binding, spatial, absence. Atomic one-token answers avoid a second model for grading answers.
- The central failure is defined as: the rendered pixels violate the scene spec, but the generator answers a question according to the original prompt rather than the actual pixels.
- Primary metrics: `GDA-free = cos(g_naive, g_rfo)`, its breakpoint `D_g`, performance divergence point `D*`, `Lead = D* - D_g`, SCFR, POE, and CSR.
- Gradient meaning is operational: SFT-loss gradients on samples selected from the same prompts/candidate pools by three criteria—gold verifier, naïve cycle score, and isolated blind observation.

### Strong result already obtained before training

- v4 has completed both corpus halves through `generate -> detect x2 -> crop -> verify -> observe(image_only + prompted) -> analyze`; 70 disputed images were manually adjudicated and no gold labels remain pending.
- On conflicting image/spec trials, adding the original generation description drops answer accuracy from roughly **0.654 to 0.336** in the consolidated proposal result; self-confirmation rises from **0.346 to 0.664**. On matching trials the description slightly helps, so generic prompt length/distraction cannot explain the effect.
- The strongest subtype is “asked for 2, drew 1”: image-only accuracy 0.804 versus prompted 0.049, meaning the model usually asserts the missing object when its original description is restored.
- This establishes **static intention leakage at step 0**, but it does not yet establish the training-time breakpoint, gradient lead, generality across models, or a causal training intervention.

### Corpus/gold-standard lessons

- Synthetic geometry previously produced highly correlated candidates for the same prompt: even with p near 0.5, pools were usually all-right or all-wrong. Natural scenes make candidates nearly independent enough for selection, raising balanced-pool supply.
- Scene complexity is a cliff: five or more objects are essentially always wrong; v4 fixes three total objects and at most two instances per item.
- The previous geometry verifier was invalid (object-level F1 0.754 against blind human labels); Qwen3-VL achieved about 0.986 F1 on that benchmark. This forced the redesign of the gold-standard ladder.

### SelfSight-Bench v4 and Tier B

- The main pool planned 240 specs but currently contains **228**: 117 `2+1` scenes and 111 `1+1+1` scenes. Each spec has K=4 generated images, totaling 912 candidates.
- Prompt/spec authoring is performed by Qwen3-VL-8B and then subjected to structural/color whitelist checks; pass rate is only about 16%. The prompt and machine-readable spec are emitted together.
- Main-pool diagnostic-trial supply is uneven: absence and counting are plentiful, binding and existence sparse, spatial zero because natural specs intentionally omit spatial constraints to preserve generation correctness supply.
- Tier B creates controlled pixel counterfactual pairs from externally verified correct images: horizontal flip (spatial sensitivity), recolor (binding), object deletion (existence), and sham inpainting (artifact control).
- Actual Tier B output is 306 pairs, but only 153 are diagnostic; the proposal target of 300 diagnostic pairs is **not met**. This shortfall is explicitly reported rather than hidden.
- Recolor produces almost no prompted obedience (accuracy 1.000 -> 0.964), while deletion produces a larger drop (0.979 -> 0.835). A sham arm stays flat, ruling out inpainting artifacts as the source of the deletion effect.
- A same-spec origin comparison refutes the simple claim that obedience follows who caused the error. A post-hoc split suggests obedience follows whether the generator commonly makes that error, but the preregistered replication missed the one-sided 0.05 threshold (p=0.062); it remains exploratory.
- The deletion editor uses a bounding box minus protected-neighbor masks, LaMa inpainting, an edited-area ceiling of 20%, and the same two-detector verifier gate used elsewhere. Several idempotence/resume bugs were found and fixed (detector keying, whole-list gate, question RNG).

### Current corpus-level Gate A / adjudication

- Both v4 halves pass balanced-pool Gate A (0.5385 and 0.6757 versus threshold 0.40).
- Dual-detector disagreement still requires human review: 70/912 images (7.68%) were adjudicated, within the locked ≤10% ceiling. Labels: Qwen wins 38, InternVL wins 9, unnameable/invalid image `X` 23.
- `X` images count as generation failures in p but do not generate questions because no defensible pixel-grounded answer exists.

### Formal experiment gates and completion implications

- Gate A requires ≥40% balanced-pool rate and ≥128 informative pools. v4 passes this with 138 pools.
- Gate B requires external-verifier precision/cascade coverage, identical-selection cosine ≥0.999, paired-bootstrap CI width ≤0.10 for GDA-free, and a non-degenerate LoRA-space cosine.
- Current Gate B result on 138 pools: GDA-free cosine 0.9678, 95% CI [0.8199, 0.9927], width **0.1728**. The cosine is non-degenerate, but CI precision fails, so Gate B is **RED/incomplete**. Roughly 412 pools are projected to reach width 0.10.
- High cosine despite 30% selection disagreement suggests image choice moves gradients only slightly relative to prompt/loss effects. It is not yet interpretable due to the wide interval.
- Gate C is the life-or-death positive-control test: RFO-Gold must beat Naive at matched candidates/updates. It has not yet been established.
- Gate D contains the actual headline claims: positive `Lead = D* - D_g`, mechanistic ablations, and RFO-Self delaying D* or improving external correctness by ≥3 points. It has not been run/established.

### Training-loop reconstruction and data bottleneck

- Deleting the obsolete v3 geometry stack also deleted the old paired training/evaluation loop. By 2026-09-03, v4 code had to rebuild L1 training and L2 checkpoint evaluation against the new natural-scene/gold-standard instrument.
- The statistical breakpoint functions survived, but originally there was no code producing their training-curve inputs. Current `src/selfsight/v4/train.py`, `evaluate.py`, and `scripts/v4_train.py` indicate this loop has since been reconstructed and must be inspected/tested.
- Split on the 228-spec corpus yields 76 train / 120 outcome / 32 probe. The registered full design wants 640 unique training prompts and roughly 800 outcome prompts, so a meaningful D* experiment needs about **1,470 distinct specs**, 6–7× the current corpus.
- Reusing 76 train prompts over 4–9 epochs would confound memorization with learning and bend the external curve in the same direction that masks the target divergence.
- K=2 leaves the perfect selector without a correct candidate 61% of the time; K=8 cuts abstention to about 14% and is used for the L3 preview.

### L3 preview scope

- Preview config uses 76 train / 120 outcome / 32 probe, 10 rounds × 7 prompts, K=8, one epoch, and 3 optimizer steps per round (30 total; first 3 warmup).
- The preregistered preview can find a descriptive D* or expose pipeline failure, but flat curves mean “insufficient updates to move the backbone,” not evidence against divergence.
- Full project budget is estimated at 700–1,140 GPU-hours for one main seed, with an MVP at 160–270 GPU-hours. The full E2/E3/E4 program is far from complete even if the preview pipeline works.

### Live state observed at 2026-09-04 ~13:14 CST

- `STATUS.md` top is stale relative to the process table: the third-batch pipeline has progressed beyond its initial detection stage into `v4_tierb_build.py plan` for `runs/v4/tierb-conf2`.
- Active orchestration includes `scripts/run_conf2_pipeline.sh`, `scripts/run_gate_b_report_after.sh`, and `scripts/run_l3_preview.sh`; the latter has been alive since 00:46 and appears queued behind upstream work rather than having already completed.
- GPU snapshot: GPU0 idle (0 MiB) and GPU1 at 76% / 2,975 MiB. This likely reflects the current Tier-B planning/inpainting stage or another active process; logs must be checked before attributing it.
- The current third corpus exists on disk: `conf2-2plus1` (~314 MB, 1,647 files), `conf2-1plus1plus1` (~291 MB, 1,563 files), and a growing `tierb-conf2` (~46.5 MB, 262 files).
- `l3-preview` contains only 2 small files (~4 KB), so despite the commit title “Run the L3 preview to a terminal state,” the actual preview outputs are not yet complete. The commit adds orchestration and a terminal verdict mechanism; it is not evidence that the run itself finished.
- Latest Git history confirms v4 L1/L2 reconstruction: paired training loop (`3c76054`), per-checkpoint evaluation/D* runner (`4a16685`), paired arm corrections (`219e9c0`), L3 config/runner/verdict, and queue/liveness handling.

### Live pipeline details

- `conf2` generation and both full-image detections are finished. Verification currently reports:
  - `2+1`: n=1,633, p=0.2492, 95 pending human, balanced-pool rate 0.8291.
  - `1+1+1`: n=1,549, p=0.2866, 163 pending human, balanced-pool rate 0.8559.
- Third-batch protocol deliberately excludes pending-human images rather than adjudicating them, but reports exclusion-rate balance; therefore these 258 pending decisions do not imply the pipeline is blocked on user review.
- The pipeline is actively creating Tier-B deletion/inpainting images in its `plan` stage. `l3-preview/orchestrator.log` still says it is waiting for batch 3, and `gate-b/report.queued.log` is empty and queued.

### Implemented code architecture

- v4 core is ~3.3k source lines; supporting training ~399, observers ~765, backbones ~1.1k, analysis ~1.6k; scripts ~4.8k and tests ~3.4k lines.
- `scripts/v4_run_pipeline.py`: generate -> dual detect -> disputed-object crop -> verifier ladder -> observe.
- `spec.py`: canonical objects/specs, detected multiset, exact image correctness.
- `questions.py`: five 2AFC families and deterministic grading.
- `detectors.py` + `verifier.py`: Qwen/InternVL parsing and three-level adjudication.
- `tierb.py` + `inpaint.py`: counterfactual image edits and acceptance gates.
- `observe.py`: prompted versus fresh image-only observations.
- `probe.py` + `v4_gate_b_probe.py`: balanced pools, three selectors, LoRA gradients, bootstrap report.
- `train.py` + `scripts/v4_train.py`: frozen split, paired schedules, candidate selection, resumable round checkpoints, outcome generation, metric folding, and D* report.
- `analysis/breakpoints.py` and `v4/evaluate.py`: segmented fits, noise-slope guard, D*, D_g, and Lead plumbing.

### Worktree state / code under active repair

- `scripts/v4_train.py` has a user-owned uncommitted edit that avoids loading an unused observer during Naive-vs-RFO-Gold pass 1. This confirms the L3 runner is still being tuned immediately before launch and the tree is not a frozen release state.
- **Confirmed high-risk implementation defect:** `stage_train` swaps one shared backbone between arms by loading each arm's previous checkpoint, but in round 0 neither previous checkpoint exists. It trains Naive first, mutating the shared parameters, then trains RFO-Gold without restoring the base weights. Therefore RFO-Gold round 0 starts from Naive's updated weights, contradicting the module contract that both arms start from the same base weights.
- The existing v4 training tests cover schedules, pairing, abstention, selection, manifests, and resume sentinels, but do not execute/test the full `stage_train` cross-arm weight lifecycle. The defect is therefore not caught by current unit coverage.
- This matters immediately: `run_l3_preview.sh` calls this `stage_train` after batch 3 and Gate B reporting. Unless fixed before launch, the preview's paired comparison would be invalid from its first checkpoint.
- The current uncommitted observer-loading edit does not address this weight-state issue.

### Orchestration chain

- `run_conf2_pipeline.sh` completes crop/verify, Tier-B deletion planning/check/accept/questions, two observation conditions, then writes the preregistered `conf_fork.txt` sentinel.
- `run_gate_b_report_after.sh` waits for that sentinel, reruns the pure-read Gate B report over three large gradient memmaps, then writes `REPORT_DONE`.
- `run_l3_preview.sh` waits for both upstream sentinels, chooses GPU placement, runs the paired training, adjudicates every arm/checkpoint with the same dual-detector ladder, folds checkpoint metrics, estimates divergence, applies a mechanical preregistered terminal rule, and writes `TERMINAL.txt`.

### Gate B has already been expanded, but not re-reported

- `gate_b.json` is the older n=138 report (timestamp 19:02), while `gradients.meta.json` now records **232 prompts** and the run list includes both main and first confirmatory batches. Three float32 memmaps are 17.1 GB each, consistent with 232 × 18,464,768 gradients.
- The queued report will recompute Gate B at n=232; it is intentionally waiting for the disk-intensive third-batch pipeline because the report performs roughly 51 GB of near-random reads.
- A simple 1/sqrt(n) projection from width 0.1728 at n=138 gives about **0.133 at n=232**, so Gate B is likely still above the ≤0.10 threshold unless the added prompts materially reduce variance. This is an estimate, not a result; the queued report has not run.
- The third confirmatory batch is not part of those gradient files. Adding it to Gate B would require pool/observation/selection/gradient stages, not merely rerunning the report.

### Additional implementation details

- Outcome evaluation seeds intentionally ignore the `arm` parameter, so both arms draw the same latent for each `(step, prompt)`; this part of pairing is implemented correctly.
- External correctness excludes unresolved `pending_human`/`unnameable` cases rather than counting them wrong, and reports the unresolved denominator separately to avoid fabricating a D* from increasing detector disagreement.
- D* reporting uses a noise-derived minimum slope (median internal SEM divided by step span), preventing floating-point drift on a flat internal curve from being misclassified as “still rising.”

### Additional tooling error

- A second PowerShell `foreach (...) { ... } | Format-Table` command hit the same parser limitation while counting Gate B rows; use an intermediate `$rows` variable if the count is needed.

### Verification status

- The repository collects **300 tests**.
- Full CPU/file-level suite result on the current worktree: **299 passed, 1 failed** (inferred from one failure out of 300; pytest's double-quiet config suppresses the aggregate pass line).
- The single failure is in `tests/test_gradients_checkpoint.py`: the test calls `pytest.importorskip` without importing `pytest`; it then also references `_Model`, which is not defined in that file. This is a broken test implementation, not a runtime traceback from product code.
- Two warnings come from Matplotlib PDF backend using Pillow's soon-to-be-removed `mode` parameter; they do not fail output generation.
- Passing unit tests do not exonerate the round-0 arm-state bug because no test invokes `scripts/v4_train.py::stage_train` end to end.

### Concrete data example

- First `2+1` spec: prompt `two blue mugs, one white plate`; machine spec contains `(mug, blue, count=2)` and `(plate, white, count=1)` on a marble counter. The generator creates K images, detectors reconstruct the pixel-side multiset, and questions compare the observed image facts to this spec/prompt under image-only and prompted conditions.
- Main `2+1` run preserves extensive immutable artifacts: manifests, two detector outputs, crop resolutions, human labels/review codes, pre/post-human verified files, both observation-condition answers, paired comparison, and per-condition analyses.

### End-to-end completion gaps confirmed

- The v4 runner hardcodes `ARMS = ("naive", "rfo_gold")`; `RFO_SELF` exists only as a selectable helper and is explicitly excluded from the default pairing. There is no CLI argument that launches a Pass-2 training arm. Therefore **Gate D / RFO-Self training is not end-to-end implemented**.
- Legacy configs still list `arms: [naive, rfo_self]`, but `v4_train.py` ignores them and uses the constant. This is an acknowledged v3-era leftover, another documentation/config divergence.
- README/RUNBOOK commands reference missing files: `src/selfsight/cli.py`, `scripts/download_models.py`, `scripts/run_main_seed_e2.py`, `scripts/evaluate_checkpoints.py`, and `scripts/build_v3_scenes.py`. The live v4 flow uses different scripts, so the documented quick start and server runbook cannot be followed literally from the current checkout.

### Gold-standard main results verified from artifacts

- `main-2plus1`: 468 images, p=0.2201, 63 balanced pools, human rate 5.98%, 12 unnameable.
- `main-1plus1plus1`: 444 images, p=0.3131, 75 balanced pools, human rate 9.46%, 11 unnameable.
- Pooled paired result: 3,623 trials (3,581 scored); conflict trials n=667 drop from 0.6537 image-only to 0.3358 prompted, with 221 right→wrong vs 9 wrong→right and p≈5.29e-44.
- Matching trials n=2,281 move in the opposite direction, 0.9794→0.9908. This is why generic prompt distraction is implausible as the main explanation.
- Family conflict results are all same-direction: counting 0.8245→0.2500, binding 0.7971→0.4928, absence 0.5574→0.3552, existence 0.5000→0.2955.
- A real example (`v4a-0000`) requests two blue mugs and a white plate; both detectors agree the rendered image contains exactly that multiset, and the generated existence question (`plate` vs `banana`) is graded deterministically from image-derived gold.

### Further tooling error

- The combined `RFO_SELF` search + missing-file check failed to parse because of the same direct-foreach-pipe pattern; rerun with intermediate variables.

### Static-quality checks

- `ruff check src tests scripts` reports **43 issues**, 19 automatically fixable. Most are style/import hygiene, but actionable correctness signals include the two undefined names in the broken checkpoint test and an unresolved `AtomicQuestion` type name in `v4/questions.py` (safe at ordinary runtime under postponed annotations, but problematic for type-hint introspection).
- Full repository history has 92 commits; 62 are local-only relative to the tracked remote branch. The current worktree has the user-owned `scripts/v4_train.py` modification plus the audit's temporary `.planning/` directory.
- `git diff --check` reports only a CRLF→LF warning for the modified training script, no whitespace-error failure.

### More evidence of current maturity

- The scientific artifact trail is considerably stronger than the release/maintenance polish: main results, preregistrations, immutable failed evidence, and per-stage run outputs are meticulous; tests/docs/entrypoint consistency and the brand-new training orchestrator are not yet at the same level.
- This repository is an active research workbench rather than a packaged application or a reproducible public release. Its internal measurement pipeline is advanced; its public setup/run instructions are stale.

### Repeated tooling error

- The latest sentinel-status command again failed to parse due to direct piping after `foreach`; no state was changed. The final live-state check must use an intermediate array.

### Logged tooling errors

- Function inventory again used a Windows-incompatible wildcard (`scripts/v4_*.py`); corrected with ripgrep `-g 'v4_*.py'`.
- First LOC PowerShell pipeline had a parser error due to piping directly after a `foreach` statement; corrected by assigning rows first, then piping.
