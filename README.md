# SelfSight

**Can You See What You Drew? Visual Self-Confirmation in Unified Multimodal Models** 的研究实现。

## 文档地图

| 文件 | 内容 |
|---|---|
| **`STATUS.md`** | **从这里开始。** 现状、锁定选择、决策树、成立/作废的结论、Hard Stops |
| `Can You See What You Drew Visual Self-Confirmation.md` | **Proposal v3.0，唯一规范来源**（含 Gate A–D 判据） |
| `docs/EVIDENCE_LOG.md` | 全部实测证据（只读，不重新解释）。开头有逐节状态表 |
| `docs/RUNBOOK.md` | 2×3090 + 2×A800 操作手册 |
| `docs/source-notes/` | 改变过设计的用户输入原件 |

四份文档各有一个职责，互不重复：**规范**（proposal）/ **现状**（STATUS）/ **证据**（EVIDENCE_LOG）/ **操作**（RUNBOOK）。
不要再新增计划类文档——`task_plan.md`、`docs/EXPERIMENT_PLAN.md`、`docs/archive/*` 已于
2026-08-30 删除，它们的内容要么过期、要么与 proposal 重复且已开始互相矛盾。

## 当前状态

见 **`STATUS.md`** —— 正在跑什么、哪些结论还成立、哪些已作废，只在那一页维护，避免多处副本
互相矛盾。

已冻结的红色结论保留为不可变证据，不覆盖、不重新决策。发现仪器缺陷时的正确做法是：旧记录原样
保留 + 缺陷作为新证据登记 + 重新测量并注明来源。

## 硬件与路径

| | 本地 | 服务器 |
|---|---|---|
| GPU | 2 × RTX 3090 24GB，无 NVLink | 2 × A800 80GB |
| 系统 | Windows（原生） | Linux |
| 入口 | `scripts/set_h_env.ps1` | `scripts/set_a800_env.sh` |
| 角色 | 数据/候选库、观察者服务、E1、E5、图表 | E2 / E3 / E4 |

两张卡都是**独立 worker，不是显存池**；单模型不跨卡切分。HQ 的 LoRA backward/resume 实测峰值
约 10.15 GiB，两种卡都不受显存约束。

除模型权重外，所有状态都在本 checkout 内。环境脚本从仓库位置推导路径，因此 checkout 可以整体
移动而无需改配置。

| 变量 | 路径 |
|---|---|
| `SELFSIGHT_PROJECT_ROOT` | `<repo>` |
| `SELFSIGHT_CACHE_ROOT` | `<repo>/cache` |
| `SELFSIGHT_DATA_ROOT` | `<repo>/data` |
| `SELFSIGHT_RUN_ROOT` | `<repo>/runs` |
| `SELFSIGHT_ENV_ROOT` | `<repo>/envs` |
| `SELFSIGHT_TMP_ROOT` | `<repo>/tmp` |
| `SELFSIGHT_MODEL_ROOT` | `H:\selfsight-models`（唯一外部例外） |

## 本地快速开始

```powershell
. .\scripts\set_h_env.ps1
$core   = Join-Path $env:SELFSIGHT_ENV_ROOT "core\python.exe"
$showo2 = Join-Path $env:SELFSIGHT_ENV_ROOT "showo2\python.exe"

& $core -m pytest -q
& $core -m selfsight.cli doctor --config configs\local_3090_showo2.yaml
```

环境自举（首次）：

```powershell
.\scripts\bootstrap_windows.ps1 -InstallFigure
.\scripts\bootstrap_showo2_windows.ps1
& $core .\scripts\sync_repositories.py
& $core .\scripts\download_models.py --group readiness_candidate_2 --plan
```

> 下载器对本机不可靠的 Xet 路由做了禁用；大 LFS 文件走可续传 aria2、有界并行分片、
> 停滞连接回收与精确 size/SHA-256 校验。无关的 ONNX/OpenVINO/CoreML/GGUF/TF/Flax 导出会被忽略。

服务器流程见 `docs/RUNBOOK.md`。

## 科学不变量

这些约束在任何阶段都不放松：

1. **观察者只收到 RGB 路径/字节 + 原子问题。** 不传 prompt、生成 latent/image token、KV cache
   或来源标签。观察器在独立进程与独立会话中运行。
2. **检测器与训练器解耦。** 构造 `g_rfo` 的冻结异构观察者全程不参与任何训练样本选择。
3. **候选库平衡池只用于梯度探针**；训练与 external correctness 曲线用自然采样池，论文中显式区分。
4. **SCFR 必须伴随观察者准确率与原始分母**报告；梯度余弦必须伴随同 checkpoint 的配对
   bootstrap CI 报告。
5. **不得把单 seed 描述性轨迹写成显著性结论**，不得引用 mock 结果。
6. **红色/不完整报告保留为不可变证据。** 缺失的测量记为 N/T，不伪造为失败值。
7. Qwen2-VL 的输出永远不是「统一 backbone 能看见自己的画」的证据。

## 归档与回收

2026-08-30 已整理（路径变更逐条记录在 `docs/EVIDENCE_LOG.md` §12.3）：

- **已删除**：`tmp/`（5.78 GB，失败 venv 与 pip/pytest 临时文件）、`cache/pip`（2.73 GB，可再生）。
  两者都会由 `scripts/set_h_env.ps1` 在下次激活时重建。
- **已归档到 `runs/_archive/` 与 `data/_archive/`**：`mock-pilot*`、`audits`、`host`、
  `manifests`、`wandb`、`selfsight-v1`、v2.3 失败模板。

已物理删除的代码（都在 git 历史里，但不再维护）：

| 删除 | 原因 |
|---|---|
| `scripts/migrate_project_roots.ps1` | 一次性迁移，目标状态已达成并由 `tests/test_project_path_policy.py` 守住 |
| `src/selfsight/data/eligible_e2.py` + `scripts/build_eligible_e2_data.py` | v2.2 E2 数据构建器，被 `build_v3_scenes.py` + `build_v3_readiness_decision.py` 取代 |
| `src/selfsight/pilot/mock_loop.py` + CLI `mock-pilot` | 非科学烟雾测试；proposal 明令不得引用 mock 结果 |
| `src/selfsight/backbones/showo_v1.py` | 冻结负对照，不再需要可运行实现 |

`src/selfsight/observers/mock.py` **保留**——它是「观察者只读 RGB」这条不变量的实现后端
（`tests/test_questions_and_isolation.py`），不是 mock 实验产物。

`runs/readiness`、`runs/v2.3-rfo-gold`、`runs/exploratory-post-gate`、`runs/v3` 是
`docs/EVIDENCE_LOG.md` 引用的证据，**不要删除**。

> `runs/` 与 `data/` 都在 `.gitignore` 里，**删了不可恢复**。就绪审计的 49 张盲审图就是这么
> 丢的（§12.2），标注还在但已无法复用。整理时一律归档到 `_archive/`，不要 `rm`。

唯一的例外已强制入库：`runs/readiness/showo2-1p5b-hq/a3-human-r1-review.csv`（49 条人工标注，
`git add -f`）。图已丢，标注是仅存的不可再生人工产物，不能再留在 gitignore 的目录里。
