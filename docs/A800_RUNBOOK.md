# A800 80GB 单卡迁移与正式 E2 运行手册

本手册只描述已经实现的 A800 路径。当前 Windows 证据中的 Gate 0
generated-RGB coverage 与 Gate -1 都是红色，因此**现在把代码复制到 A800 也只能做环境和
32-prompt canary，不能启动正式 E2**。`scripts/run_formal_e2.py` 会在创建输出目录、加载模型
或消耗训练算力之前验证这些报告并失败关闭。

## 1. 主机和存储前提

- Linux x86_64、单张 NVIDIA A800 80GB；不启用分布式训练。
- Python 3.10、可工作的 NVIDIA 驱动；环境使用 CUDA 12.1 PyTorch wheel。
- 选择一个短、独占且至少有 650GB 可用空间的数据根目录。下文用
  `/data/selfsight` 举例，实际路径由操作者显式设置。
- 代码必须来自一个固定 Git commit；模型与外部仓库 revision 由
  `configs/models.lock.yaml` 固定。

```bash
cd /path/to/Visual-Self-Confirmation
git status --short
git rev-parse HEAD
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
df -h /data

export SELFSIGHT_ROOT=/data/selfsight
source scripts/set_a800_env.sh
```

`set_a800_env.sh` 会把 Hugging Face、Torch、pip、临时文件、数据、模型、运行结果和虚拟环境
全部放到 `${SELFSIGHT_ROOT}` 下。不要用 `sudo pip`，也不要让 `HF_HOME` 回落到 home 目录。

## 2. 环境、代码仓库与锁定权重

```bash
bash scripts/bootstrap_a800.sh
source scripts/set_a800_env.sh

CORE="${SELFSIGHT_ENV_ROOT}/core/bin/python"
OBSERVER="${SELFSIGHT_ENV_ROOT}/observer/bin/python"
JANUS="${SELFSIGHT_ENV_ROOT}/janus/bin/python"

"${CORE}" scripts/sync_repositories.py
"${CORE}" scripts/download_models.py --group core --large-file-transport auto
"${CORE}" scripts/download_models.py --group observers --large-file-transport auto

"${CORE}" scripts/capture_environment_lock.py --role core \
  "${SELFSIGHT_RUN_ROOT}/manifests/a800-core-environment.json"
"${OBSERVER}" scripts/capture_environment_lock.py --role observer \
  "${SELFSIGHT_RUN_ROOT}/manifests/a800-observer-environment.json"
"${JANUS}" scripts/capture_environment_lock.py --role janus \
  "${SELFSIGHT_RUN_ROOT}/manifests/a800-janus-environment.json"
"${CORE}" -m selfsight.cli doctor --config configs/a800_80g.yaml \
  --output "${SELFSIGHT_RUN_ROOT}/manifests/a800-host.json"
"${CORE}" -m pytest -q
```

若下载脚本报告 revision、文件尺寸或 SHA-256 不一致，停止迁移，不得改用 `main` 或手工换权重。
`late_eval` 与 `audit` 组不是迁移 canary 或 E2 的前置依赖，不要提前下载。

## 3. 数据清单与 Gate 证据

优先从 Windows 复制整个 `selfsight-v1` 数据目录和以下只读报告，保留文件名及内容不变：

- `manifests/train.jsonl`
- `manifests/tier_a_probe.jsonl`
- `manifests/tier_a_outcome.jsonl`
- `manifests/tier_b.jsonl`
- `manifests/tier_d.jsonl` 与 `manifests/tier_d_selection.json`（仅供 Gate 4 后的 E4）
- Gate -1 最终决策、被其引用的 observer audit，以及 generated-domain 报告

manifest 内的图像路径是生成主机的绝对路径；不得用文本替换或软链接伪装 H 盘。复制后生成一份
新的 Linux 路径视图。该工具不修改 Windows 原始 manifest，并会在写出前逐图验证原文件与 RGB
SHA-256：

```bash
REBASING="${SELFSIGHT_DATA_ROOT}/selfsight-v1/manifests-a800"
"${CORE}" scripts/rebase_dataset_manifests.py \
  --data-root "${SELFSIGHT_DATA_ROOT}/selfsight-v1" \
  --output "${REBASING}"
```

然后只对 rebased manifests 重新跑无泄漏与像素审计：

```bash
"${CORE}" -m selfsight.cli audit-data "${SELFSIGHT_DATA_ROOT}/selfsight-v1" \
  --manifest-dir "${REBASING}" \
  --output "${SELFSIGHT_RUN_ROOT}/audits/a800-data-audit.json"
"${CORE}" -m selfsight.cli audit-tier-b \
  "${REBASING}/tier_b.jsonl" \
  --output "${SELFSIGHT_RUN_ROOT}/audits/a800-tier-b-audit.json"
"${CORE}" -m selfsight.cli audit-tier-d \
  "${REBASING}/tier_d.jsonl" \
  --output "${SELFSIGHT_RUN_ROOT}/audits/a800-tier-d-audit.json"
```

若不能复制数据，可用相同代码 commit 与配置重新生成；由于绝对路径不同，不能直接比较 JSONL
文件 SHA。必须比较 registry 的 `split_signature_digest`、Tier-D `selection_digest`、全部 scene ID、
prompt/atom/seed 和逐图 RGB SHA-256。任一非路径字段不一致都视为新数据版本，不能和 Windows
canary 混用。

当前红色报告应被保留为不可变证据。只有在用户批准了会改变实验设计的修订、重新跑完 Gate，
且新的 `decision.json` 与 generated-domain report 同时为绿色后，才能继续第 5 节。

## 4. 固定 32-prompt Windows/A800 canary

先在 Windows 3090 上、再在 A800 上使用同一 probe manifest、同一模型 revision、同一代码
commit 运行。输出目录必须是新的，脚本会拒绝覆盖已有目录。

Windows：

```powershell
. .\scripts\set_h_env.ps1
$core = "H:\selfsight-envs\core\python.exe"
& $core .\scripts\run_migration_canary.py `
  --config configs\local_3090.yaml `
  --probe-manifest H:\selfsight-data\selfsight-v1\manifests\tier_a_probe.jsonl `
  --output H:\selfsight-runs\migration\windows-32
```

A800：

```bash
"${CORE}" scripts/run_migration_canary.py \
  --config configs/a800_80g.yaml \
  --probe-manifest "${REBASING}/tier_a_probe.jsonl" \
  --output "${SELFSIGHT_RUN_ROOT}/migration/a800-32"
```

把 Windows canary 目录复制到 A800 后作唯一一次接受判定：

```bash
"${CORE}" scripts/compare_migration_canaries.py \
  --local "${SELFSIGHT_RUN_ROOT}/migration/windows-32" \
  --a800 "${SELFSIGHT_RUN_ROOT}/migration/a800-32" \
  --output "${SELFSIGHT_RUN_ROOT}/migration/migration-gate.json"
```

通过条件固定为：观察答案一致率不低于 95%，verifier 标签一致率不低于 95%，三个汇总指标
的绝对偏差均不超过 1 percentage point，并且 backbone revision 完全相同。失败时不要通过改 seed
或只删掉不一致样本重跑；先定位平台差异。

## 5. 物化正式配置与执行 E2

以下命令只在 Gate -1、generated-domain Gate 和 migration Gate 全绿后执行。先物化三份不可
覆盖的 seed 配置：

```bash
FORMAL_CONFIGS="${SELFSIGHT_RUN_ROOT}/formal-configs"
"${CORE}" scripts/materialize_a800_seed_configs.py --output "${FORMAL_CONFIGS}"
sha256sum "${FORMAL_CONFIGS}"/*.yaml > "${FORMAL_CONFIGS}/SHA256SUMS"
```

然后使用 Gate -1 决策中锁定的异构 detector 参数运行。下列占位符必须逐字替换为该绿色决策
记录的 backend、model id 与 revision；不要选择表现更高但能力不匹配的模型。

```bash
"${CORE}" scripts/run_formal_e2.py \
  --base-config configs/a800_80g.yaml \
  --seed-config "${FORMAL_CONFIGS}/a800_seed_20260827.yaml" \
  --seed-config "${FORMAL_CONFIGS}/a800_seed_20260828.yaml" \
  --seed-config "${FORMAL_CONFIGS}/a800_seed_20260829.yaml" \
  --core-python "${CORE}" \
  --observer-python "${OBSERVER}" \
  --train-manifest "${REBASING}/train.jsonl" \
  --outcome-manifest "${REBASING}/tier_a_outcome.jsonl" \
  --probe-manifest "${REBASING}/tier_a_probe.jsonl" \
  --gate-minus-1-report /absolute/path/to/green/decision.json \
  --generated-domain-report /absolute/path/to/green/generated_domain_report.json \
  --migration-report "${SELFSIGHT_RUN_ROOT}/migration/migration-gate.json" \
  --detector-audit-report /absolute/path/to/selected/detector-audit.json \
  --detector-backend SELECTED_BACKEND \
  --detector-model-id SELECTED_MODEL_ID \
  --detector-revision SELECTED_REVISION \
  --output "${SELFSIGHT_RUN_ROOT}/formal-e2"
```

运行器按 seed 顺序完成 Gate -1b、Naive/RFO-Self 训练、checkpoint 评测和三 seed 聚合；已有完整
checkpoint 时自动跳过，半成品通过 `--resume` 继续。单卡上不要同时启动两个 seed，observer 与
训练都使用 `cuda:0`，并以阶段串行换取可复现的显存边界。

## 6. 恢复、监控与完成判定

- 训练日志位于 `formal-e2/seed-*/logs/`；每轮 checkpoint 只包含 LoRA、optimizer、scheduler、
  RNG 和完整配置快照。
- 进程中断后，用完全相同的命令重启；不要删除最后一个已原子提交的 round。
- 磁盘不足时只清理下载缓存中的可再生临时分片，不能删除 manifest、Gate 报告、checkpoint 或
  `resolved_config.json`。
- 正式输出以 `formal-e2/formal-aggregate/formal_gate_2_2b.json` 为准。Gate 2/2b 红色时立即停止，
  不启动 E3/E4/E5；绿色也只授权 Proposal 中紧接着的下一个阶段。

最终归档至少包括：Git commit、`configs/models.lock.yaml`、三份 seed 配置及 SHA-256、环境 freeze、
host manifest、所有 Gate 报告、原始 checkpoint metrics、聚合报告、图和日志。论文主张不得引用
Windows 单 seed 或 mock 结果。
