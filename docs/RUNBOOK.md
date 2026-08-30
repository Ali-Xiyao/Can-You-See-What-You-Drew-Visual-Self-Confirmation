# Runbook — 2 × RTX 3090 (local) + 2 × A800 80GB (server)

取代已删除的 `A800_RUNBOOK.md`。该文档假定单张 A800 与三 seed 正式实验；v3.0 改为
**双 A800 并行 + 单主 seed 先行**。

## 0. 分工

| | 本地 3090 × 2（Windows） | 服务器 A800 × 2（Linux） |
|---|---|---|
| 角色 | 候选库生成、冻结观察者服务、Gate A/B 筛查、E1、E5、图表 QA | E2 主训练曲线、E3、E4 |
| 硬约束 | 两卡独立 worker，非 48GB 池；单模型不跨卡切分；无 NVLink | 两卡并行跑两个 arm；**不启用分布式** |
| 环境入口 | `scripts/set_h_env.ps1` | `scripts/set_a800_env.sh` |

HQ 的 LoRA backward/resume 实测峰值 **10.15 GiB**，两种卡都不受显存约束。因此 A800 的价值是
**吞吐与并行**（同时跑两个 arm、更大 batch、更长曲线），不是「跑得下」。

---

## 1. 服务器初始化

```bash
cd /path/to/Visual-Self-Confirmation
git rev-parse HEAD
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
df -h .

export SELFSIGHT_PROJECT_ROOT="$(pwd)"
export SELFSIGHT_MODEL_ROOT=/data/selfsight-models
source scripts/set_a800_env.sh
```

`set_a800_env.sh` 把 HF / Torch / pip / tmp / data / runs / envs 全部收进
`${SELFSIGHT_PROJECT_ROOT}`；模型目录由 `SELFSIGHT_MODEL_ROOT` 单独指定。
不要用 `sudo pip`，不要让 `HF_HOME` 回落到 home。数据盘需 ≥650GB。

```bash
bash scripts/bootstrap_a800.sh
source scripts/set_a800_env.sh

CORE="${SELFSIGHT_ENV_ROOT}/core/bin/python"
SHOWO2="${SELFSIGHT_ENV_ROOT}/showo2/bin/python"
OBSERVER="${SELFSIGHT_ENV_ROOT}/observer/bin/python"

"${CORE}" scripts/sync_repositories.py
"${CORE}" scripts/download_models.py --group readiness_candidate_2 --large-file-transport auto
"${CORE}" scripts/download_models.py --group observers --large-file-transport auto

"${CORE}" -m selfsight.cli doctor --config configs/a800_80g_showo2.yaml \
  --output "${SELFSIGHT_RUN_ROOT}/manifests/a800-host.json"
"${CORE}" -m pytest -q
```

> **v3.0 变更**：主 backbone 是 `showlab/show-o2-1.5B-HQ`（rev `d3a220ec`），
> 所以服务器下载的是 `readiness_candidate_2` 组，不是 `_1`。
> 若下载脚本报告 revision / size / SHA-256 不一致，停止迁移，不得改用 `main` 或手工换权重。

---

## 2. 数据：本地生成，服务器 rebase

Tier A/B v3 与候选库在**本地 3090** 上生成（生成是 GPU-bound 但可并行、且本地已有全部
v2.3 manifest 与 RGB 哈希链）。然后整目录复制到服务器并 rebase 绝对路径。

本地：

```powershell
. .\scripts\set_h_env.ps1
$core = Join-Path $env:SELFSIGHT_ENV_ROOT "core\python.exe"
& $core scripts\build_v3_scenes.py `
  --output-root "$env:SELFSIGHT_DATA_ROOT\selfsight-v3" `
  --objects-per-scene 2 `
  --split calibration=24 --split tier_a_probe=128 `
  --split train=2400 --split tier_a_outcome=600
```

服务器（复制后）：

```bash
V3_DATA="${SELFSIGHT_DATA_ROOT}/selfsight-v3"
"${CORE}" scripts/rebase_dataset_manifests.py \
  --data-root "${V3_DATA}" --output "${V3_DATA}/manifests-a800"
```

验证 `rebase_report.json` 中每份 manifest 的 row count、source/rebased hash 和逐图 RGB。
manifest 内的图像路径是生成主机的绝对路径；**不得用文本替换或软链接伪装 H 盘**。

若不能复制数据，可用相同 commit 与配置重新生成。由于绝对路径不同，不能直接比较 JSONL 的
SHA；必须比较 registry 的 `split_signature_digest`、全部 scene ID、prompt/atom/seed 和逐图
RGB SHA-256。任一非路径字段不一致视为新数据版本，不能与本地 canary 混用。

---

## 3. 迁移 canary（跨平台一致性）

同一 probe manifest、同一模型 revision、同一 commit，先本地后服务器。输出目录必须是新的。

```powershell
& $showo2 .\scripts\run_migration_canary.py `
  --config configs\local_3090_showo2.yaml `
  --backbone-config configs\backbones\showo2_1p5b_hq.yaml `
  --probe-manifest "$env:SELFSIGHT_DATA_ROOT\selfsight-v3\manifests\tier_a_probe.jsonl" `
  --output "$env:SELFSIGHT_RUN_ROOT\migration\windows-32"
```

```bash
"${SHOWO2}" scripts/run_migration_canary.py \
  --config configs/a800_80g_showo2.yaml \
  --backbone-config configs/backbones/showo2_1p5b_hq.yaml \
  --probe-manifest "${V3_DATA}/manifests-a800/tier_a_probe.jsonl" \
  --output "${SELFSIGHT_RUN_ROOT}/migration/a800-32"

"${CORE}" scripts/compare_migration_canaries.py \
  --local "${SELFSIGHT_RUN_ROOT}/migration/windows-32" \
  --a800  "${SELFSIGHT_RUN_ROOT}/migration/a800-32" \
  --output "${SELFSIGHT_RUN_ROOT}/migration/migration-gate.json"
```

通过条件：观察答案一致率 ≥95%、verifier 标签一致率 ≥95%、三个汇总指标绝对偏差 ≤1 pt、
backbone revision 完全相同。**失败时不要改 seed 或删掉不一致样本重跑；先定位平台差异。**

---

## 4. E2 主 seed：双卡并行，机制优先

主 seed `20260901`。两遍，每遍两个 arm 并行占一张卡。

### Pass 1 — Naive ‖ RFO-Gold（Gate C）

```bash
# 终端 A
CUDA_VISIBLE_DEVICES=0 "${CORE}" scripts/run_main_seed_e2.py \
  --base-config configs/a800_80g_showo2.yaml \
  --arm naive --seed 20260901 \
  --train-manifest   "${V3_DATA}/manifests-a800/train.jsonl" \
  --probe-manifest   "${V3_DATA}/manifests-a800/tier_a_probe.jsonl" \
  --outcome-manifest "${V3_DATA}/manifests-a800/tier_a_outcome.jsonl" \
  --output "${SELFSIGHT_RUN_ROOT}/v3-main-seed/naive"

# 终端 B
CUDA_VISIBLE_DEVICES=1 "${CORE}" scripts/run_main_seed_e2.py \
  ... --arm rfo_gold ... \
  --output "${SELFSIGHT_RUN_ROOT}/v3-main-seed/rfo_gold"
```

两个 arm 必须使用**完全相同**的 prompt ID、候选池 RGB、K、optimizer 步数、microbatch 顺序、
flow latent 种子与 LoRA-dropout 种子。任一选择器 abstain 时，该 prompt 从**所有** arm 对称移除。

**Gate C 判据**（proposal §9）：RFO-Gold 的 external verifiable correctness 高于 Naive，
差距超过 checkpoint 内方差。**不过则停止扩展**——回到修正目标 / 样本权重 / 梯度探针的诊断，
不加 seed、不换 backbone、不加算力。

### Pass 2 — RFO-Self ‖ checkpoint 评测（Gate D）

Gate C 绿后才跑。

```bash
CUDA_VISIBLE_DEVICES=0 "${CORE}" scripts/run_main_seed_e2.py --arm rfo_self ...
CUDA_VISIBLE_DEVICES=1 "${CORE}" scripts/evaluate_checkpoints.py \
  --run-root "${SELFSIGHT_RUN_ROOT}/v3-main-seed" \
  --output   "${SELFSIGHT_RUN_ROOT}/v3-main-seed/evaluations"
```

> **这两个脚本尚未实现**（本节是它们的接口约定）。`run_main_seed_e2.py` 从
> `run_formal_e2.py` 拆出单 arm / 单 seed / 单卡；`evaluate_checkpoints.py` 是
> `evaluate_pilot.py` 的 v3 版本。到 Gate C 之前不需要它们。

每个 checkpoint 记录：internal cycle score、external verifiable correctness、SCFR（附原始分母）、
verifier coverage、public-view consistency、GDA-free 及范数比及 per-block 分解、GDA-gold、
**配对 bootstrap CI**、观察答案熵（竞争基线信号）。

### 追加 seed 的触发条件

只有 Gate C 与 Gate D 在主 seed 上均绿、且 Naive 与 RFO 的 external correctness 差距超过
checkpoint 内方差时，才跑 seed `20260902` / `20260903`。否则先诊断。

---

## 5. 需要改造的脚本

现有 A800 路径是围绕「Gate -2 绿 + 三 seed 正式 E2」设计的，v3.0 需要以下改动。
**这些是已知缺口，不是已完成工作。**

| 脚本 | 现状 | v3.0 需要 |
|---|---|---|
| `run_formal_e2.py` | 要求三份 seed config + 绿色 `decision.json` | 拆出 `run_main_seed_e2.py`：单 arm、单 seed、单卡，可并行启动 |
| `materialize_a800_seed_configs.py` | 固定物化三份 seed config | 支持单 seed；追加 seed 时增量物化 |
| `run_gradient_gate.py` / `run_v23_gradient_gate.py` | 用不相交半批做噪声地板 | 改为 prompt 层配对 bootstrap（EVIDENCE_LOG §5） |
| `build_v23_data.py` | 96 probe prompt，随机 K=4 | 新增 `build_v3_data.py`：有界候选库搜索（M≤16），probe 规模按 informative pool 计 |
| `finalize_joint_readiness.py` | `families_passing_min = 4` | 主族 floor 改 3；hard tier 单独报告 |
| `aggregate_formal_e2.py` | 三 seed 聚合 | 单 seed 描述性报告 + 条件性多 seed 聚合 |

---

## 6. 恢复、监控与归档

- 训练日志：`v3-main-seed/<arm>/logs/`。checkpoint 只含 LoRA、optimizer、scheduler、RNG
  和完整配置快照。
- 进程中断后用**完全相同**的命令重启；不要删除最后一个已原子提交的 round。
- 磁盘不足时只清理下载缓存中可再生的临时分片，**不能**删除 manifest、Gate 报告、checkpoint 或
  `resolved_config.json`。
- 红色或不完整的报告保留为不可变证据，不覆盖、不重解释。

最终归档至少包括：Git commit、`configs/models.lock.yaml`、seed config 及 SHA-256、环境 freeze、
host manifest、全部 Gate 报告、原始 checkpoint metrics、聚合报告、图和日志。

**论文主张不得引用 mock 结果，也不得把单 seed 描述性轨迹写成显著性结论。**
