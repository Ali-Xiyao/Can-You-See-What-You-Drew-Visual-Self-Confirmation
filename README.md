# SelfSight

**Can You See What You Drew? Visual Self-Confirmation in Unified Multimodal Models** 的研究实现。

## 文档地图

**现在做的是哪条线:`.planning/2026-09-08-iclr-redesign/` —— Blind Self-Verification,
目标 ICLR 2028。** 2026-09-08 起,它取代了原来围绕 D* / D_g / Lead 的路线
(理由在 `PREREG.md` §1:拐点类估计量 SD 11.3 步而训练跨度只有 88 步,加 checkpoint
救不回来;论文改用比例类估计量)。**被取代的东西一件不删**,原始记录保留,
D* / D_g / Lead 降为附录里的仪器负结果。

下面分三层:**一、现在这条线** / **二、长期规范的四份** / **三、保留不删的历史**。

### 一、现在这条线(2026-09-08 起,活的)

| 文件 | 内容 | 状态 |
|---|---|---|
| `.planning/2026-09-08-iclr-redesign/PROPOSAL.md` | 为什么换路线,依据全部是既有数据上实测的 | 冻结 |
| `.planning/2026-09-08-iclr-redesign/PREREG.md` | **预注册**:E3 三臂协同进化、E4 跨模型复制、失败条件。偏离 1–5 只追加不改写 | 冻结 |
| `.planning/2026-09-08-iclr-redesign/EXECUTION.md` | 每一步在等什么。**§0 是这条线的实时状态** | 活的 |
| `.planning/2026-09-08-iclr-redesign/ICLR-2028.md` | 能不能发:三种结局在看到数字之前先定死。§8 是 09-09 追记 | 活的 |
| `.planning/2026-09-08-iclr-redesign/PAPER.md` | 论文结构(取代 PROPOSAL 里的实验编号) | 活的 |
| `.planning/2026-09-08-iclr-redesign/RUN-HALTED-20260908.md` | 主运行 5.83 h 停机的记录 | 冻结 |

这条线上的运行、产物与代码:

| 位置 | 是什么 | 状态 |
|---|---|---|
| `runs/v4/decoupling-main-20260908` | E3 的 arm A(`naive`)+ arm C(`rfo_gold`) | **在跑** |
| `runs/v4/blind-self-<启动日>`(待建) | E3 的 arm B(`naive` + `blind_self` 配对) | 代码就绪,等卡 |
| worktree `H:/Xiyao_Wang/062_armB`,分支 `codex/blind-self-arm-20260908` | arm B 的全部代码改动 | 未合并 |
| `review-packets/cross-model-20260908/` | E4 跨模型复制,注册的三个模型 3/3 | 已完成 |
| `review-packets/armB-audit-preimage-20260909/` | arm B 的场景排除集,在 arm B 存在之前冻结 | 已冻结 |
| `review-packets/bsv-selection-20260908/` | BSV 离线重放:盲验证选出更好的图 | 已冻结 |
| `review-packets/context-ablation-20260908/` | 上下文消融,论文第一张图的来源 | 已冻结 |
| `review-packets/dg-power-20260908/` | 换路线的依据:D_g 找不到,两个独立原因 | 已冻结 |
| `review-packets/e4-preflight-20260908/` | E4 预检:三个模型都真的读到了图 | 已冻结 |

### 二、长期规范的四份(职责不变)

| 文件 | 内容 |
|---|---|
| **`STATUS.md`** | 现状、锁定选择、决策树、成立/作废的结论、Hard Stops |
| `Can You See What You Drew Visual Self-Confirmation.md` | **Proposal v3.0**(含 Gate A–D 判据) |
| `docs/EVIDENCE_LOG.md` | 全部实测证据(只读,不重新解释)。开头有逐节状态表 |
| `docs/RUNBOOK.md` | 2×3090 + 2×A800 操作手册 |
| `docs/source-notes/` | 改变过设计的用户输入原件 |

四份文档各有一个职责,互不重复:**规范**(proposal)/ **现状**(STATUS)/
**证据**(EVIDENCE_LOG)/ **操作**(RUNBOOK)。**不要再新增计划类文档**——
`task_plan.md`、`docs/EXPERIMENT_PLAN.md`、`docs/archive/*` 已于 2026-08-30 删除,
它们的内容要么过期、要么与 proposal 重复且已开始互相矛盾。
`.planning/<日期>-<主题>/` 是这条规矩的**唯一例外**:一次路线变更一个目录、
带日期、预注册在里面冻结,不与上面四份争职责。

两处要当心:

- **`STATUS.md` 的「正在跑」段落停在 2026-09-07/08 的 generator-repair 阶段**,
  不是当前这条线。当前状态看 `EXECUTION.md` §0。
- Proposal v3.0 仍是规范,但**它的实验编号已被 `PAPER.md` 取代**(E1 不用做了,
  数据在盘上)。PROPOSAL 原样保留,不改写。

### 三、保留不删的历史

| 位置 | 是什么 |
|---|---|
| `docs/prereg/` | 六份更早的预注册(qwen 难度规则、选择器裁定、decoupling pilot、cycle selector、D* 主运行) |
| `.planning/2026-09-04-project-audit/` | 2026-09-04 的项目审计:findings / progress / task_plan |
| `review-packets/*-20260906/`、`*-20260907/` | pilot 期与生成器修复期的产物(step24/step32 复核、replay 消融、选择增益、事实性诊断、生成响应修复验证) |
| `review-packets/no-break-null-20260908/`、`dstar-precondition-fix-20260908/`、`outcome-budget-20260908/` | D* 路线上的最后一批工作,**它们是换路线的证据,不是废纸** |
| `review-packets/bon-coupling-20260908/`、`high-n-bon-20260908/`、`grader-seam-20260908/`、`cycle-selector-resolution-20260908/` | 评分器与 BoN 侧的冻结设计和裁定 |
| 分支 `experiment/v2.2-joint-readiness`、`experiment/v2.3-rfo-gold`;`runs/v2.3-rfo-gold/` | v2.x 时代。**红色结论冻结,不覆盖、不重新决策** |
| `runs/_archive/`、`runs/v4/_archive/` | 归档的运行工件。清理一律归档,不 `rm` |

**其中两份只在本地,不在云端。** 它们在这个 checkout 里,推送时被有意排除:

| 位置 | 体积 | 为什么不推 |
|---|---|---|
| `review-packets/replay-ablation-20260906/` | 2.58 GB,其中 `.pt` 占 2.55 GB | `.pt` 本来就被 `.gitignore` 挡住;剩下的 37 MB json/jsonl 是同一批 checkpoint 的重放中间态,离开那些 `.pt` 复现不了,单独推没有意义 |
| `review-packets/generator-repair-validation-20260907/` | 237 MB,1036 张 PNG | 生成器修复期的逐张目视产物;结论已经落在 `STATUS.md` 与 `docs/EVIDENCE_LOG.md` 里,图本身在仓库里只是死重 |

引用它们时写本地路径 + 结论所在的文档,不要假设 clone 下来就有。其余产物(含
`factual-diagnostics-20260906/`、`selector-measurement-next-20260906/`、
`selection-benefit-side-20260906/`)已在云端。


## 当前状态

**当前这条线正在跑什么:`.planning/2026-09-08-iclr-redesign/EXECUTION.md` §0。**
`runs/v4/decoupling-main-20260908` 在跑 E3 的 arm A + C,arm B 的代码就绪、等卡。

**哪些结论还成立、哪些已作废:`STATUS.md`。** 那一页是长期的实验室笔记
(§1..§43+,机器与方法的事实),不是当前进度摘要;它的「正在跑」段落停在
2026-09-07/08 的 generator-repair 阶段。两页各维护一处,不互相复制。

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
