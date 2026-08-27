# SelfSight

Research implementation for **Can You See What You Drew? Visual Self-Confirmation in Unified Multimodal Models**.

The repository is gate-first: exact controlled scenes and measurement audits precede every phenomenon claim. Local dual-RTX-3090 runs are one-seed engineering evidence only; formal inference requires the locked single-A800 80GB, three-seed configuration.

The active `experiment/v2.2-joint-readiness` branch implements the approved conditional design.
The frozen Show-o v1 red result is preserved at tag `v2.1-showo-gate-red`; none of its manifests,
reports, or figures are recalculated by v2.2.

## v2.2 Joint Readiness quick start

Candidate order is fixed: Show-o2-1.5B, then 1.5B-HQ only after a hashed rank-1 failure, then 7B
only after both smaller candidates fail. The commands below therefore download and run rank 1 only.

```powershell
. .\scripts\set_h_env.ps1
$core = "H:\selfsight-envs\core\python.exe"
$showo2 = "H:\selfsight-envs\showo2\python.exe"
$root = "H:\selfsight-runs\readiness\showo2-1p5b"

# Dedicated native-Windows environment; omits FlashAttention/DeepSpeed/xFormers/TF/ONNX/W&B.
.\scripts\bootstrap_showo2_windows.ps1

# Inspect the exact 12.78 GB candidate-1 plan, then download only that group.
& $core .\scripts\download_models.py --group readiness_candidate_1 --plan
& $core .\scripts\download_models.py --group readiness_candidate_1

# Isolated v2.2 namespace: 6 canary + 120 reference + 60 generated-domain records.
& $core .\scripts\build_readiness_data.py

# A1 and A2 use the same locked checkpoint for generation and RGB atomic QA.
& $showo2 .\scripts\run_backbone_readiness.py canary `
  --backbone-config configs\backbones\showo2_1p5b.yaml `
  --manifest H:\selfsight-data\selfsight-v2.2\manifests\canary.jsonl `
  --output "$root\a1-canary.json"

& $showo2 .\scripts\run_backbone_readiness.py reference `
  --backbone-config configs\backbones\showo2_1p5b.yaml `
  --manifest H:\selfsight-data\selfsight-v2.2\manifests\reference.jsonl `
  --output "$root\a2-reference.json"

# A3 runs K=1 for all families and K=4 only for A2-retained families.
& $showo2 .\scripts\audit_generated_precision.py generate `
  --backbone-config configs\backbones\showo2_1p5b.yaml `
  --reference-report "$root\a2-reference.json" `
  --manifest H:\selfsight-data\selfsight-v2.2\manifests\generated.jsonl `
  --output "$root\a3-generated.json"

# If A3's automatic coverage/Oracle/seed-stability checks are red, freeze the red decision here.
# Human precision and A4 are then recorded as preregistered skips, never fabricated as failures.
& $showo2 .\scripts\finalize_joint_readiness.py `
  --backbone-config configs\backbones\showo2_1p5b.yaml `
  --canary-report "$root\a1-canary.json" `
  --reference-report "$root\a2-reference.json" `
  --generated-report "$root\a3-generated.json" `
  --stop-before-human-a4 `
  --output "$root\decision-red.json"

& $core .\scripts\render_readiness_matrix.py `
  --decision "$root\decision-red.json" `
  --evidence-status "local one-seed upstream stop" `
  --output "$root\figure-readiness-red"

# Continue below only if A3's automatic checks are green. The packet shows only RGB plus an
# atomic question. Fill review_blinded.csv before score.
& $showo2 .\scripts\audit_generated_precision.py export `
  --generated-report "$root\a3-generated.json" `
  --output "$root\a3-blind-packet"

& $showo2 .\scripts\audit_generated_precision.py score `
  --review-csv "$root\a3-blind-packet\review_blinded.csv" `
  --answer-key "$root\a3-blind-packet\answer_key.json" `
  --output "$root\a3-human.json"
```

Blind-review rules are fail-closed. Review the contact sheets without opening `answer_key.json` or
searching for the generating prompt. Every CSV row requires: (1) `human_answer`, containing only the
answer visible in the pixels; (2) `parseable_yes_no=yes` when the image supports a definite atomic
answer, otherwise `no` and a non-empty placeholder such as `abstain` in `human_answer`; and (3) a
non-empty pseudonymous `reviewer_id`. `notes` is optional. All rows must be completed by a human;
model-generated annotations are not admissible evidence.

A1 writes `a1-canary-lora-module-tree.json`. Inspect it before selecting suffixes; there is
intentionally no default copied from Show-o v1. The `select` command expands explicit suffixes to
the exact shared-transformer module names and binds them to the A1/tree hashes:

```powershell
& $showo2 .\scripts\run_showo2_lora_canary.py select `
  --canary-report "$root\a1-canary.json" `
  --suffix SELECT_AFTER_TREE_AUDIT `
  --output "$root\a4-lora-targets.json"

& $showo2 .\scripts\run_showo2_lora_canary.py run `
  --backbone-config configs\backbones\showo2_1p5b.yaml `
  --canary-report "$root\a1-canary.json" `
  --reference-report "$root\a2-reference.json" `
  --generated-report "$root\a3-generated.json" `
  --human-report "$root\a3-human.json" `
  --target-config "$root\a4-lora-targets.json" `
  --manifest H:\selfsight-data\selfsight-v2.2\manifests\reference.jsonl `
  --output "$root\a4-lora.json"

& $showo2 .\scripts\finalize_joint_readiness.py `
  --backbone-config configs\backbones\showo2_1p5b.yaml `
  --canary-report "$root\a1-canary.json" `
  --reference-report "$root\a2-reference.json" `
  --generated-report "$root\a3-generated.json" `
  --human-report "$root\a3-human.json" `
  --lora-report "$root\a4-lora.json" `
  --output "$root\decision.json"

& $core .\scripts\render_readiness_matrix.py `
  --decision "$root\decision.json" `
  --evidence-status "local one-seed engineering evidence" `
  --output "$root\figure-readiness"
```

A red decision stops E1/E2 and names only the preregistered next candidate. An automatic A3 failure
stops before human review and A4 because neither can repair failed coverage, Oracle@4, or seed
stability; the decision explicitly stores those evidence fields as absent. HQ/7B configs and download
groups exist for that route, but fallback downloads require `--predecessor-decision` pointing to the
immediately preceding immutable red decision (fallback finalization later uses `--predecessor`).
Readiness figures encode absent measurements as gray dotted `N/T`
cells, never as orange failures. Full rules are in
[`docs/PROPOSAL_V2.2_AMENDMENT.md`](docs/PROPOSAL_V2.2_AMENDMENT.md).

For example, after rank 1 has produced `decision-red.json`, rank 2 is the only authorized download:

```powershell
& $core .\scripts\download_models.py --group readiness_fallback_hq --plan
& $core .\scripts\download_models.py --group readiness_fallback_hq `
  --predecessor-decision "$root\decision-red.json"
```

## Storage and processes

All large state is outside this checkout and under short H-drive paths:

| Variable | Path |
|---|---|
| `SELFSIGHT_CACHE_ROOT` | `H:\selfsight-cache` |
| `SELFSIGHT_DATA_ROOT` | `H:\selfsight-data` |
| `SELFSIGHT_RUN_ROOT` | `H:\selfsight-runs` |
| `SELFSIGHT_MODEL_ROOT` | `H:\selfsight-models` |
| `SELFSIGHT_ENV_ROOT` | `H:\selfsight-envs` |
| `SELFSIGHT_TMP_ROOT` | `H:\selfsight-tmp` |

Start every PowerShell session from the repository root with:

```powershell
. .\scripts\set_h_env.ps1
$core = "H:\selfsight-envs\core\python.exe"
$observer = "H:\selfsight-envs\observer\python.exe"
$janus = "H:\selfsight-envs\janus\Scripts\python.exe"
```

The script also disables the unreliable Xet route seen on this Windows host. Large locked LFS files use resumable aria2 transfer plus exact size/SHA-256 verification; small files still use Hugging Face Hub.
The downloader ignores unrelated ONNX/OpenVINO/CoreML/GGUF/TF/Flax exports unless a model lock
explicitly asks for them.

GPU0 owns Show-o generation/training/backward. GPU1 owns frozen observers and parallel evaluation. The two 24GB cards are never treated as pooled 48GB memory.

## 1. Environment and immutable assets

The current local core environment was cloned from a working CUDA 12.1 environment and therefore uses Torch 2.5.1+cu121. This is a recorded local deviation; the A800 lock remains Torch 2.2.1.

```powershell
# Core plus figure dependencies. Omit CloneCudaEnv if installing CUDA wheels afresh.
.\scripts\bootstrap_windows.ps1 -InstallFigure `
  -CloneCudaEnv C:\Users\Admin\anaconda3\envs\quest-zero-p0

# Sync exact Show-o and Janus commits to H:.
& $core .\scripts\sync_repositories.py

# Create the isolated observer environment.
.\scripts\bootstrap_windows.ps1 -InstallObservers `
  -CloneObserverCudaEnv C:\Users\Admin\anaconda3\envs\bind

# Janus .bin loading requires Torch >=2.6; keep it out of the Qwen/Intern environment.
.\scripts\bootstrap_windows.ps1 -InstallJanusObserver

# Inspect size/revision plan, then obtain core weights.
& $core .\scripts\download_models.py --group core --plan
& $core .\scripts\download_models.py --group core --large-file-transport auto
```

Downloads are registered under `H:\selfsight-models`; `configs/models.lock.yaml` is authoritative. Never replace a locked revision with `main`.
The complete local `core`, `observers`, and `audit` inventories are materialized at their locked
revisions. `late_eval` remains intentionally absent until its upstream Gate is green.

## 2. Deterministic data and Gate 0 smoke evidence

```powershell
& $core -m selfsight.cli doctor --config configs\local_3090.yaml `
  --output H:\selfsight-runs\manifests\local-host.json

& $core -m selfsight.cli build-data --config configs\local_3090.yaml `
  --output H:\selfsight-data\selfsight-v1

& $core -m selfsight.cli audit-data H:\selfsight-data\selfsight-v1 `
  --output H:\selfsight-runs\audits\data-audit.json

& $core -m selfsight.cli audit-tier-b `
  H:\selfsight-data\selfsight-v1\manifests\tier_b.jsonl `
  --output H:\selfsight-runs\audits\tier-b-audit.json

# Deterministic 600-image E4 subset: 300 Tier-A images + 150 complete Tier-B pairs.
& $core -m selfsight.cli build-tier-d H:\selfsight-data\selfsight-v1 --seed 20260827
& $core -m selfsight.cli audit-tier-d `
  H:\selfsight-data\selfsight-v1\manifests\tier_d.jsonl `
  --output H:\selfsight-runs\audits\tier-d-audit.json

# Export the blinded 120-image manual packet (six families x 20).
& $core .\scripts\manual_reference_audit.py export `
  --manifest H:\selfsight-data\selfsight-v1\manifests\tier_a_outcome.jsonl `
  --output H:\selfsight-runs\audits\manual-reference

& $core -m pytest -q
& $core -m selfsight.cli mock-pilot --config configs\local_3090.yaml `
  --output H:\selfsight-runs\mock-pilot
```

The mock output is stamped `synthetic_smoke_only` and must never be cited as scientific evidence.
The programmatic reference gate currently passes 3,200/3,200 scenes, 400/400 Tier-B pairs, and
600/600 registered Tier-D images; manual stratified agreement remains a separate pending audit.

After a blinded reviewer fills `review_blinded.csv`, score it without modifying the separate answer key:

```powershell
& $core .\scripts\manual_reference_audit.py score `
  --review-csv H:\selfsight-runs\audits\manual-reference\review_blinded.csv `
  --answer-key H:\selfsight-runs\audits\manual-reference\answer_key.json `
  --output H:\selfsight-runs\audits\manual-reference\manual-audit-report.json
```

Exact Windows package/CUDA inventories are captured in `H:\selfsight-envs\locks` by the bootstrap script.

## 3. Real model canaries and Gate -1

```powershell
# Show-o: generation, RGB reload/MMU, T2I+replay LoRA step, save/corrupt/exact restore.
& $core .\scripts\canary_showo.py --config configs\local_3090.yaml `
  --output H:\selfsight-runs\canaries\showo-local

# Train-only, six-family generated-RGB parseability audit; no Tier-A held-out prompt is used.
& $core .\scripts\canary_generated_domain.py --config configs\local_3090.yaml `
  --manifest H:\selfsight-data\selfsight-v1\manifests\train.jsonl `
  --limit 60 --verifier generated_cv_v2 `
  --output H:\selfsight-runs\canaries\generated-domain-cv2-dev60-20260827

# Example isolated observer canary; repeat for the locked capability ladder.
& $core .\scripts\canary_observer.py --python $observer --backend smolvlm `
  --model-id HuggingFaceTB/SmolVLM-500M-Instruct `
  --revision a7da5b986cb59b408707209984f360a5f4ad7e47 `
  --manifest H:\selfsight-data\selfsight-v1\manifests\tier_a_probe.jsonl `
  --output H:\selfsight-runs\canaries\smolvlm
```

Run `selfsight.cli audit-observer` for Show-o and every candidate on program-rendered truth, then finalize the matched detector:

```powershell
& $core -m selfsight.cli finalize-gate-minus-1 `
  --reference-audit H:\selfsight-runs\audits\data-audit.json `
  --showo-report H:\selfsight-runs\gate-minus-1\showo-local120.json `
  --candidate-report H:\selfsight-runs\gate-minus-1\smolvlm-local120.json `
  --candidate-report H:\selfsight-runs\gate-minus-1\internvl-local120.json `
  --candidate-report H:\selfsight-runs\gate-minus-1\qwen2vl-local120.json `
  --output H:\selfsight-runs\gate-minus-1\decision.json
```

Gate -1 is a hard stop: Show-o needs at least four question families at 80% or above, and the frozen heterogeneous observer must be within 3 percentage points of Show-o macro accuracy after forced-choice/yes-bias auditing.

When this Gate is red, render the preregistered capability-floor fallback artifact instead of
running E1/E2:

```powershell
& $core .\scripts\render_capability_floor.py `
  --report H:\selfsight-runs\gate-minus-1\showo-local120.json `
  --report H:\selfsight-runs\gate-minus-1\showo-discrete-local120.json `
  --report H:\selfsight-runs\gate-minus-1\janus-local120.json `
  --report H:\selfsight-runs\gate-minus-1\smolvlm-local120.json `
  --report H:\selfsight-runs\gate-minus-1\internvl-local120.json `
  --report H:\selfsight-runs\gate-minus-1\qwen2vl-local120.json `
  --evidence-status "local diagnostic; Gate -1 decision frozen" `
  --output H:\selfsight-runs\gate-minus-1\figure2-all-backbones-local
```

The figure exports PNG/PDF/SVG/grayscale plus tidy CSV and a QA manifest. Accuracy is redundantly
encoded by cell text and a bold outline at the 80% threshold; yes-bias uses a separate axis.

There is an earlier generated-domain stop rule as well: deterministic primary-answer coverage on
generated RGBs must be at least 95%. The exact-palette verifier remains authoritative for
program-rendered references; `generated_cv_v2` is a deterministic contour candidate for approximate
model output and must receive its own blinded human-agreement audit before adoption. Current
train-only canaries are below 95%, so E1, Gate -1b, and real self-training remain blocked even while
the independent reference-image observer ladder is audited.

## 4. E1, Gate -1b, and local one-seed loop

The active v2.2 E1 entry point accepts only a complete green Gate -2 decision. It restricts Tier B
to `selected_eligible_families`, verifies every Gate -2 evidence hash, binds the exact Show-o2
checkpoint and frozen public-observer revision, and rechecks the observer's per-family accuracy,
yes-bias, and abstention floors before loading either model:

```powershell
. .\scripts\set_h_env.ps1
$showo2 = "H:\selfsight-envs\showo2\python.exe"
$observer = "H:\selfsight-envs\observer\python.exe"
$root = "H:\selfsight-runs\readiness\showo2-1p5b"
$observerAudit = "H:\selfsight-runs\gate-minus-1\qwen2vl-local120.json"

& $showo2 .\scripts\run_e1.py --config configs\local_3090_showo2.yaml `
  --manifest H:\selfsight-data\selfsight-v1\manifests\tier_b.jsonl `
  --joint-readiness-decision "$root\decision.json" `
  --backbone-config configs\backbones\showo2_1p5b.yaml `
  --observer-config configs\observers\qwen2vl_2b.yaml `
  --detector-audit-report $observerAudit `
  --detector-python $observer --detector-backend qwen2vl `
  --detector-model-id Qwen/Qwen2-VL-2B-Instruct `
  --detector-revision 895c3a49bc3fa70a340399125c650a463535e71c `
  --output H:\selfsight-runs\e1\showo2-1p5b

& $showo2 .\scripts\run_gradient_gate.py `
  --config configs\local_3090_showo2.yaml `
  --probe-manifest H:\selfsight-data\selfsight-v1\manifests\tier_a_probe.jsonl `
  --joint-readiness-decision "$root\decision.json" `
  --backbone-config configs\backbones\showo2_1p5b.yaml `
  --observer-config configs\observers\qwen2vl_2b.yaml `
  --lora-target-config "$root\a4-lora-targets.json" `
  --detector-audit-report $observerAudit `
  --detector-python $observer --detector-backend qwen2vl `
  --detector-model-id Qwen/Qwen2-VL-2B-Instruct `
  --detector-revision 895c3a49bc3fa70a340399125c650a463535e71c `
  --device cuda:1 --output H:\selfsight-runs\gate-minus-1b\showo2-1p5b

& $showo2 .\scripts\run_local_pilot.py `
  --config configs\local_3090_showo2.yaml `
  --train-manifest H:\selfsight-data\selfsight-v1\manifests\train.jsonl `
  --joint-readiness-decision "$root\decision.json" `
  --backbone-config configs\backbones\showo2_1p5b.yaml `
  --lora-target-config "$root\a4-lora-targets.json" `
  --gradient-gate-report H:\selfsight-runs\gate-minus-1b\showo2-1p5b\gate_minus_1b.json `
  --frozen-observer-python $showo2 `
  --output H:\selfsight-runs\local-pilot\showo2-1p5b --resume

& $showo2 .\scripts\evaluate_pilot.py `
  --config configs\local_3090_showo2.yaml `
  --run-root H:\selfsight-runs\local-pilot\showo2-1p5b `
  --outcome-manifest H:\selfsight-data\selfsight-v1\manifests\tier_a_outcome.jsonl `
  --probe-manifest H:\selfsight-data\selfsight-v1\manifests\tier_a_probe.jsonl `
  --joint-readiness-decision "$root\decision.json" `
  --backbone-config configs\backbones\showo2_1p5b.yaml `
  --observer-config configs\observers\qwen2vl_2b.yaml `
  --lora-target-config "$root\a4-lora-targets.json" `
  --detector-audit-report $observerAudit `
  --detector-python $observer --detector-backend qwen2vl `
  --detector-model-id Qwen/Qwen2-VL-2B-Instruct `
  --detector-revision 895c3a49bc3fa70a340399125c650a463535e71c `
  --device cuda:1
```

These commands must not be run while Gate -2 is red or incomplete. E2 filters the training pool to
the same eligible-family set, launches a frozen step-0 Show-o2 observer on GPU1 through the blind
JSONL boundary, and binds checkpoint resume to the Gate, backbone, and exact LoRA target digest.
The evaluator repeats those bindings, excludes non-eligible outcome/probe families before stable
sampling, and loads every adapter with the training contract digest rather than the base YAML alone.

The following commands are retained only to reproduce the frozen v2.1 Show-o experiment. Its
current `decision.json` is red, so they also remain blocked unless supplied a distinct, immutable
green v2.1 decision; they are not a route around Gate -2:

```powershell
$gate = "H:\selfsight-runs\gate-minus-1\NEW-GREEN-decision.json"
$domain = "H:\selfsight-runs\canaries\NEW-GREEN-generated-domain\generated_domain_report.json"
$detectorAudit = "H:\selfsight-runs\gate-minus-1\SELECTED-detector-audit.json"
$detectorBackend = "SELECTED_BACKEND"
$detectorModel = "SELECTED_MODEL_ID"
$detectorRevision = "SELECTED_REVISION"

& $core .\scripts\run_e1.py --config configs\local_3090.yaml `
  --manifest H:\selfsight-data\selfsight-v1\manifests\tier_b.jsonl `
  --gate-minus-1-report $gate --generated-domain-report $domain `
  --detector-audit-report $detectorAudit `
  --detector-python $observer --detector-backend $detectorBackend `
  --detector-model-id $detectorModel --detector-revision $detectorRevision `
  --output H:\selfsight-runs\e1

& $core .\scripts\run_gradient_gate.py --config configs\local_3090.yaml `
  --probe-manifest H:\selfsight-data\selfsight-v1\manifests\tier_a_probe.jsonl `
  --gate-minus-1-report $gate --generated-domain-report $domain `
  --detector-audit-report $detectorAudit `
  --detector-python $observer --detector-backend $detectorBackend `
  --detector-model-id $detectorModel --detector-revision $detectorRevision `
  --output H:\selfsight-runs\gate-minus-1b

& $core .\scripts\run_local_pilot.py --config configs\local_3090.yaml `
  --train-manifest H:\selfsight-data\selfsight-v1\manifests\train.jsonl `
  --gate-report $gate `
  --gradient-gate-report H:\selfsight-runs\gate-minus-1b\gate_minus_1b.json `
  --generated-domain-report $domain `
  --frozen-observer-python $core --output H:\selfsight-runs\local-pilot --resume
```

The legacy entry points validate the locked Gate identity, selected detector model/revision, exact
SHA-256 of its capability audit, internal consistency, sample basis, and 95% coverage threshold.
A red, mismatched, or malformed report fails closed rather than becoming an accidental training
override.

If Gate -1b fails, E2 remains allowed but GDA reporting is disabled and the preregistered entropy/public-view fallback is activated. Checkpoints are adapter-only and each round is atomically committed, so rerunning with `--resume` is safe.

Evaluate every checkpoint with `scripts/evaluate_pilot.py`. This produces internal/external correctness, SCFR@competent, entropy/public-view signals, optional GDA-free/GDA-gold/noise-floor trajectories, exploratory D*/Dg, and Figure 1 in PNG/PDF/SVG/grayscale. Local nonappearance of D* does not block migration if engineering, stability, gradients, and resume all pass.

## 5. A800 migration and formal E2

The operator-facing, fail-closed handoff procedure is in
[`docs/A800_RUNBOOK.md`](docs/A800_RUNBOOK.md). It includes storage setup, immutable asset checks,
the paired 32-prompt canary, exact acceptance thresholds, restart behavior, and the formal command.

On the Linux A800 host, place the checkout at a short data mount, then:

```bash
export SELFSIGHT_ROOT=/data/selfsight
source scripts/set_a800_env.sh
bash scripts/bootstrap_a800.sh
CORE="${SELFSIGHT_ENV_ROOT}/core/bin/python"
SHOWO2="${SELFSIGHT_ENV_ROOT}/showo2/bin/python"
"${CORE}" scripts/sync_repositories.py
"${CORE}" scripts/download_models.py --group readiness_candidate_1
"${CORE}" scripts/download_models.py --group observers
"${CORE}" scripts/materialize_a800_seed_configs.py --output "${SELFSIGHT_RUN_ROOT}/formal-configs"
```

Run the fixed 32-prompt canary locally and on A800, then compare them with `scripts/compare_migration_canaries.py`. Formal E2 is blocked unless answer/verifier agreement is at least 95% and metric drift is at most 1 point.

After a green Gate -2, `scripts/build_eligible_e2_data.py` creates the decision-bound 2400/200/600
train/probe/outcome splits using only eligible families and excluding readiness signatures. Because
their manifests contain host-absolute RGB paths, run
`scripts/rebase_dataset_manifests.py` after copying data to Linux. It writes a new manifest view and
verifies every original file/RGB SHA-256; it never edits the Windows manifests. The A800 runbook
uses only this rebased view.

`scripts/run_formal_e2.py` executes the three locked seeds resumably. `scripts/aggregate_formal_e2.py` performs paired seed bootstrap, Gate 2/2b decisions, GDA-free versus entropy/public-view comparison, and multi-seed Figure 1. E3, E4, Tier C, and human evaluation remain blocked until their preregistered upstream Gates pass.

## Scientific invariants

- The 200-prompt gradient probe and 600-prompt outcome set never enter training or selector tuning.
- Blind observer subprocesses receive only a hard-reloaded RGB path, atomic questions, and request metadata—never prompt, expected answer, generator state, or source label.
- RFO training uses a frozen step-0 copy of the selected Show-o2 backbone; `g_rfo` detection uses the fixed frozen Qwen2-VL-2B public observer.
- Candidate IDs, prompt IDs, K, and random seeds are paired between Naive and RFO arms; common non-abstained counts are enforced.
- Formal conclusions require three A800 seeds. Local one-seed curves are exploratory regardless of apparent effect size.

Current gate state and exact evidence are tracked in `task_plan.md`, `findings.md`, and `progress.md`.

As of the latest local evidence, Show-o2-1.5B passes A1 and retains five A2 reference families:
existence, count, color, spatial, and binding. Absolute size is excluded. A3-r1 was automatically
red and additionally invalidated by a candidate-path collision discovered during artifact audit.
Collision-safe A3-r2 is running under the same manifest, seeds, revision, and verifier. Until r2
finishes, no blind-human packet, A4, Gate -2 decision, fallback download, E1, or E2 is authorized.
The frozen Show-o v1 negative result remains available at tag `v2.1-showo-gate-red` and is not
recalculated by this branch.
