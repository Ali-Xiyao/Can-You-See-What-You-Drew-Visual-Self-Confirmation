# Evidence Log

本文件是 v2.x 阶段全部**实测证据**的唯一权威汇总。它取代了已删除的
`PROPOSAL_V2.2_AMENDMENT.md`、`PROPOSAL_V2.3_AMENDMENT.md` 与 `V2.3_GRADIENT_GATE_DIAGNOSIS.md`。

原始不可变工件仍在 `runs/` 下，路径与 SHA-256 记录在每节末尾。**本文件只记录测量结果，不重新
解释、不重新决策。** v3.0 的设计结论写在 proposal 正文 §3。

| 阶段 | 日期 | backbone | 结论 |
|---|---|---|---|
| v2.1 | 2026-08-27 | Show-o v1 1.3B | 红。生成域覆盖 55%，binding 覆盖 0%，exact-scene 0% |
| v2.2 rank-1 | 2026-08-28 | Show-o2-1.5B (432px) | 红。A3 自动检查失败，停在人工审计前 |
| v2.2 rank-2 | 2026-08-28 | Show-o2-1.5B-HQ (512px) | 红（仅因 ≥4 族门槛）。**3 族全绿** |
| v2.2 rank-3 | 2026-08-28 | Show-o2-7B | 红。覆盖率与种子稳定性均劣于 HQ |
| 探索闭环 | 2026-08-28 | HQ | 工程闭环打通；单 seed 无机制信号 |
| v2.3 | 2026-08-29 | HQ | 红。候选供给不足，梯度门未完成 |
| v3.0 词汇 | 2026-08-30 | HQ | 红。v3 manifest 声称 box 实用 square，183/256 行；pool-ext 运行作废，见 §11 |

### 逐节状态（读之前先看这张表）

本文件按时间追加，后面的节会作废前面的节。**这张表是唯一的现行状态**。

| 节 | 内容 | 状态 |
|---|---|---|
| §1 | HQ 就绪审计 | 成立（但见 §12：审计问题用的是 square 措辞） |
| §2 | 冻结负对照 | 成立 |
| §3 | 本地探索闭环 | 成立（工程结论，非机制结论） |
| §4 | v2.3 RFO-Gold 机制门 | 成立 |
| §5 | 梯度噪声控制设计缺陷 | 成立（跨阶段，已改配对 bootstrap） |
| §6 §7 | 工程环境 / 设计输入 | 成立 |
| §8 | Gate A 候选供给 RED | 成立（干净 v2.3 manifest，2 物体） |
| §9 | L0 就绪决策复算 | 成立 |
| §10 | 3 物体难度标定 | **已作废** — 物体数与词汇双重混淆 |
| §11 | v3 显示词汇缺陷与作废 | 成立（本身即作废记录） |
| §12 | 仓库整理与就绪审计词汇污染 | 成立 |

---

## 1. Show-o2-1.5B-HQ 就绪审计（最重要的一节）

`showlab/show-o2-1.5B-HQ` rev `d3a220ec55feaacbdfcb053847edee14edd4e69a`，原生 512×512。

### 1.1 六问题族全指标

| 问题族 | 参考观察准确率 | 生成覆盖率 | 盲审 precision | Oracle@K=4 |
|---|---:|---:|---:|---:|
| existence | 1.00 | 1.00 | **1.00** | 1.00 |
| color | 1.00 | 1.00 | **1.00** | 1.00 |
| spatial | 1.00 | 0.80 | **1.00** | 1.00 |
| binding | 1.00 | 0.90 | 0.78 | 1.00 |
| count | 0.85 | 1.00 | **0.40** | 0.50 |
| absolute size | 0.60 | **0.20** | 1.00 | 0.20 |

整体：coverage 0.8167（阈值 0.80 ✓）、Oracle@4 0.90（✓）、固定种子覆盖摆动 10.0 pt（阈值 ≤10 ✓）、
**overall precision 0.8367（阈值 0.95 ✗）**。

### 1.2 为什么判红，以及这个判定的性质

Gate -2 的四项子检查：

- `-2A` 统一功能：**记为 false，但原因是按停止规则跳过**（`skipped_by_stop_rule: [a4_lora_backward_resume]`），
  不是测量失败。该项后来在探索阶段实测通过（见 §3.1）。
- `-2B` 参考观察：**true**。
- `-2C` 生成域可测量性：false，唯一失败子项是 `blind_verifier_precision`（0.8367 < 0.95）。
  覆盖率、Oracle@4、逐族覆盖、种子稳定性四项全部通过。
- `-2D` 联合可用族：false，因为 `families_passing_min = 4`。

**关键事实：整体 precision 被 count（0.40）与 binding（0.78）拉低；existence / color / spatial
三族在全部四项指标上都是 1.00 或达标。** 判红的直接原因是「≥4 族」这个门槛值，而不是三个主族
上的任何测量失败。该门槛在 v2.2 注册时没有推导依据。

v3.0 把主族floor 改为 3 并公开说明这一变更（proposal §3.1）。count / size / binding 的失败原因
可解释——count 是 verifier 计数精度、size 是模型无法渲染绝对尺寸、binding 是 verifier
精度——三者作为 hard tier 如实报告。

### 1.3 盲审人工标签

`runs/readiness/showo2-1p5b-hq/a3-human-r1-review.csv`，49 行，单评审。
主路径子集 28 行（existence n=10、color n=10、spatial n=8），逐族 verifier-human precision 均为 **1.00**。

评审策略：用户明确表示 rectangle-like 输出计为 square 正确。该策略等价于 v2.3 起采用的
`box`（轴对齐外接框长宽比 0.5–2.0）外观容差词汇，因此这批标签可直接复用为 v3.0 的视觉词汇校准
证据（28/28，各族 1.00，≥0.95 门槛通过）——**须标注为复用证据，不是新一轮盲标注。**

### 1.4 工件

- 决策：`runs/readiness/showo2-1p5b-hq/decision-hq-red.json`
- A2 参考审计：`a2-reference-r1.json` / `a2-reference-r1-rows.jsonl`
- A3 生成域：`a3-generated-r1.json`（`c19a7193…86ac`）/ 210 个唯一候选，无 ID/路径碰撞
- 人工：`a3-human-r1.json`、`a3-human-r1-review.csv`（`70fe561e…cb68e`）
- 盲审 ZIP：`e9127794…20d7`
- LoRA 目标：`a4-lora-targets-r1.json`

---

## 2. 其他 backbone（冻结负对照）

### 2.1 Show-o2-7B

rev `3012b1d6aee8b57829b23d02cba9190ef5cc3361`。A1/A2 绿，A3 自动检查红，停在人工审计前
（`family_precision: null`，未测量而非失败）。

| 问题族 | 参考准确率 | 生成覆盖率 | Oracle@4 |
|---|---:|---:|---:|
| existence | 1.00 | 1.00 | 1.00 |
| color | 1.00 | 0.90 | 1.00 |
| spatial | 1.00 | 0.70 | 1.00 |
| binding | 1.00 | 0.90 | 1.00 |
| count | 0.90 | 1.00 | 0.60 |
| size | 0.55 | 0.10 | 0.10 |

整体：coverage **0.7667**（< 0.80 ✗）、Oracle@4 0.92（✓）、种子摆动 **16.0 pt**（> 10 ✗）。

**结论：把统一 checkpoint 从 1.5B-HQ 放大到 7B 没有修复生成域可测量性。** Oracle@4 提升到
0.92，但首候选覆盖率下降、种子稳定性恶化。这直接反驳「换更大模型就能解决」，是 v3.0 选择 HQ
的实测依据。

### 2.2 Show-o2-1.5B（rank-1）与 Show-o v1

- rank-1（432px）：A3 自动红，按预注册停止规则停在人工/A4 之前。r1 因候选 ID 碰撞作废，
  r2 完成 210 个唯一 ID/路径并复现 r1 的自动指标。
- Show-o v1：冻结于 tag `v2.1-showo-gate-red`。覆盖 55%、binding 覆盖 0%、exact-scene 0%。
  作为能力下界负对照，**不得被静默提回主线**。

---

## 3. 本地探索闭环（2026-08-28，非预注册）

用户授权的工程诊断，用已下载权重、写在 `runs/exploratory-post-gate/` 下，不触碰冻结的 readiness
目录。seed `20260828`，族 existence/color/spatial，2 轮 × 12 配对 prompt，K=2，每轮每 arm 5 步。

### 3.1 A4：LoRA 训练可行性（实测通过）

七项 backward/update/restore/resume 检查全部通过。采用 in-place LoRA 注入 + non-reentrant
checkpointing 后，**峰值显存约 10.15 GiB**（10,902,030,336 B），24GB 3090 与 80GB A800 都不受
显存约束。适配器精确恢复、optimizer/scheduler 续跑均验证。

这一项弥补了 §1.2 中 `-2A` 被跳过的空缺。

### 3.2 E1（12 对 Tier-B 反事实）

硬渲染一致性 1.0，像素敏感反馈偏好 1.0，observation gain 0.75，SCFR 与 POE 均 0.333。
**n=12，仅为管线连通性证据。**

### 3.3 checkpoint 曲线（24 outcome prompts × 2 图）

| checkpoint | Internal | External | Verifier coverage | Public-view consistency | SCFR |
|---|---:|---:|---:|---:|---|
| Base | 0.8750 | 0.5208 | 0.5208 | 0.8958 | N/A（分母 0） |
| Naive, round 2 | 0.8542 | 0.5625 | 0.5833 | 0.8333 | N/A（分母 0） |
| RFO-Self, round 2 | 0.8542 | 0.5417 | 0.5833 | 0.8542 | 0.50（分母 2） |

**RFO-Self 没有超过 Naive**（+2.08 pt vs +4.17 pt）。单 seed、3 个 checkpoint、K=2、每 arm 10 步、
SCFR 分母为 0–2。**这是描述性轨迹，不构成对机制的任何证据或反证。** D\* 不可估，D_g 被 Gate -1b
回退禁用。

### 3.4 Gate -1b（此处发现设计缺陷）

identical 选择余弦 1.0；**GDA-free 0.9913、GDA-gold 0.9635；不相交半批噪声区间 [0.1348, 0.2899]。**

Gate 判红的依据是「噪声地板 95% 区间宽度 0.155 > 预期效应量 0.20 的一半」。但更重要的是这组
数字暴露的**尺度错配**：跨准则余弦（0.96–0.99）远**高于**半批余弦（0.13–0.29），而门的设计
假定前者会**跌破**后者。原因见 §5。

---

## 4. v2.3 RFO-Gold 机制门（2026-08-29）

在 HQ 上加入第三个 arm `rfo_gold`（程序化 verifier 作为完美选择器），作为机制阳性对照。
GPU0 可训练实例，GPU1 冻结 step-0 观察者，无模型下载，A800 未使用。

### 4.1 候选供给失败（本阶段的核心发现）

计划：96 prompt × 3 次独立重复，每次需 ≥64 个 common informative pool
（informative = K=4 池中同时含 verifier-正确与 verifier-错误候选）。

实际：**筛完 42 个 prompt 只得到 9 个 informative pool**；即使剩余 54 个全中也只有 63 < 64，
门在数学边界处 fail-closed。重复 2/3 与全部三 seed 训练未运行（N/T，非失败）。

分族 informative 率：existence 6/16、color 2/13、spatial 1/13。**总体 21%。**

**含义：在这个尺度上，模型对某个 prompt 的正确性接近确定性——池要么全对、要么全错。
基于选择的自训练没有可利用的方差。** 这是 v3.0 把候选供给提升为 Gate A、并改用有界候选库
搜索的直接依据（proposal §6.1）。

### 4.2 九池小样本诊断（不构成门通过）

| 量 | 结果 |
|---|---:|
| identical-selection 余弦 | **1.000** |
| Naive vs RFO-Self 余弦 | **1.000** |
| Naive vs RFO-Gold 余弦 | 0.274 |
| Naive/Gold 范数比 | 0.758 |
| Naive/Gold 选择一致率 | 0.444 |
| Naive/Self 选择一致率 | **1.000** |
| Gold 选中分数优势 | +0.25（8 个有限比较） |
| 不相交半批余弦区间 | [-0.082, 0.322]，n=4，mean 0.077 |

两点：

1. **实现路径已验证。** identical 精确等于 1.000，证明 flow-noise 与 LoRA-dropout 的 RNG 控制
   正确（早期版本因 dropout 走全局 CUDA RNG 而降到 0.9889，已修复为按 batch seed 派生的
   forked RNG context）。
2. **RFO-Self 在 9 个池上选择与 Naive 完全一致（1.000）**，而 Gold 只有 0.444。这些池足以区分
   不同选择准则，RFO-Self 只是没找出差别。注意这是 step-0、冻结 step-0 观察者，早期一致本就是
   设计预期，因此**不构成对机制的证伪**，但它意味着信号必须在训练中长出来。

### 4.3 证据审计（绿）

42 个原子 packet、168 个唯一 K=4 候选记录及 RGB 哈希、三 arm 决策对齐、168 个
schema-valid 的 RGB/问题-only 观察者请求（无 prompt、无生成状态、无选择器标签泄漏）、
v2.3 可见文本无 `square`。峰值 GPU0 分配 8,892,721,152 B（≈8.28 GiB）。
测试 106/106 通过，Ruff 无问题。

### 4.4 工件

- `runs/v2.3-rfo-gold/gradient-survival/gradient_gate.json`（`03be9f8c…6697`）
- `repeat-01/candidate_manifest.jsonl`（`3779adeb…9140`）
- `repeat-01/informative_filter.json`（`f2e1862e…1894`）
- `evidence_audit.json`（`ba12bed5…4dc20`）
- v2.3 数据 registry digest `f1fd4de4…76a9`：432 train / 96 probe / 90 primary outcome / 60 hard，
  678 条引用全部通过 RGB/atom 校验，三者签名/模板/prompt 零重叠
- v2.3 config digest `d9b7fca7…f746c`

---

## 5. 梯度噪声控制的设计缺陷（跨阶段结论）

v2.x 用「同准则、**不相交**半批之间的梯度余弦」作噪声地板。两次实测：
`[-0.082, 0.322]`（§4.2）与 `[0.135, 0.290]`（§3.4）。

这个控制量与被测量不在同一尺度：

- **跨准则余弦**在同一批 prompt 上比较，两个 arm 共享 prompt 与大部分候选，噪声配对抵消 →
  实测 identical = 1.000、naive/self = 1.000。
- **半批余弦**在不相交样本上比较，噪声独立且加倍 → 实测 0.08–0.2。

因此「跨准则余弦跌破半批地板」这个判据是拿一把量错东西的尺子。它同时解释了两次 Gate -1b 判红
的表面矛盾（§3.4 中跨准则 0.99 远高于地板，§4.2 中 naive/gold 0.274 恰落在地板内）。

**v3.0 的修正**：噪声控制改为 **prompt 层配对 bootstrap**——对 prompt 集有放回重采样，在同一
重采样集上重算 `cos(g_naive, g_rfo)`，直接给出该余弦自身的 CI。不相交半批余弦降级为独立报告的
「单样本梯度信噪比」诊断量，不再充当门槛。

**这项修正使主终点比 v2.x 设想的更可测。** 配对测量的精度（identical 精确 1.000）远高于不相交
测量，此前的红灯有相当部分来自控制量选错，而非信号不存在。

---

## 6. 工程与环境事实

- 本地：2 × RTX 3090 24GB，driver 560.94 / CUDA 12.6，**无 NVLink**。两卡是独立 worker，
  不是 48GB 池；单模型不跨卡切分。63.8GB RAM，20 逻辑核。
- 模型权重固定在 `H:\selfsight-models`；其余全部（envs / data / runs / cache / tmp）位于仓库根下。
  2026-08-28 完成项目根迁移，三个锚点哈希逐字节存活：rank-1 决策 `d2bd4190…d685`、
  HQ A3 报告 `c19a7193…86ac`、盲审 ZIP `e9127794…20d7`。
- Show-o2 7B 官方快照缺少 JSON shard index，需按完整编号 shard 集校验后重建；
  低 RAM 分片加载把进程私有内存从约 45GB 降到 23GB。
- 探索阶段 Tier B 只覆盖 count/color/spatial/binding，**不含 existence 对**。Gate 族集合必须与
  Tier-B 支持集取交，而不能当作必需的 manifest schema。
- `square` 与 rectangle 的分歧是**构念定义问题**，不是 verifier 噪声。内部几何枚举保留
  `Shape.SQUARE`（四顶点、长宽比 0.5–2.0）以免破坏冻结 manifest；对外词汇统一为 `box`。

---

## 7. 设计输入来源

`docs/source-notes/model-backbone-revision-20260828.txt`（SHA-256 `32fddaba…c41a`）是用户提供的
backbone 选型说明原文，确立了核心约束：

> 主模型必须满足——**同一个可训练模型既能生成图像，又能重新读取 RGB 图像并回答视觉问题。**

它同时说明了为什么 `Qwen-Image + Qwen3-VL` 组合不可接受：那已经是「一个图像生成器 + 一个独立
视觉评审器」，会改变论文的核心科学问题。**这条约束在 v3.0 中保持不变。**

---

# v3.0 Measurements

## 8. Gate A — 候选供给（2026-08-30，本地双 3090）

`showlab/show-o2-1.5B-HQ` rev `d3a220ec`，v2.3 gradient-probe 的 96 个 prompt（32/族），
有界候选库 M=16，K=4，每侧至少 2 个。共 1536 次生成，分两 shard 并行。

### 8.1 结果：RED（注册阈值 0.40，实测 0.344）

| 量 | 值 |
|---|---:|
| balanced_rate | **0.344**（阈值 0.40） |
| informative_rate | 0.469 |
| natural_rate（随机 K=4） | 0.135 |
| bank_lift | **+0.208**（2.5×） |

| 问题族 | 平衡池 | 产出率 | 判定 |
|---|---:|---:|:---:|
| existence | 15/32 | 0.469 | PASS |
| spatial | 13/32 | 0.406 | PASS |
| color | 5/32 | **0.156** | FAIL |

**候选库机制本身是有效的**：自然池 0.135 → 库平衡池 0.344，2.5 倍提升，与 v2.3 实测的
0.21 自然率同量级。失败的是绝对水平，差 0.056。

### 8.2 失败模式与 v2.x 的预期相反

拒绝原因（96 池）：`no_verifier_incorrect_candidate` 48、`informative_but_unbalanced` 12、
`too_many_abstentions` 2、`no_verifier_correct_candidate` **1**。

| 问题族 | 平衡 | 不平衡 | **全对** | 全错 | 弃答 | 库内每候选 verifier 准确率 |
|---|---:|---:|---:|---:|---:|---:|
| existence | 15 | 1 | 16 | 0 | 0 | 0.865 |
| color | 5 | 5 | **22** | 0 | 0 | **0.949** |
| spatial | 13 | 6 | 10 | 1 | 2 | 0.671 |

**96 个池里 48 个全对、只有 1 个全错。模型不是太弱，是太准。** color 的每候选准确率 0.949，
16 个候选里期望只有 0.8 个错的，凑不出 2 个来配平。

平衡池产出率与难度呈倒 U 型：spatial（0.671）与 existence（0.865）过线，color（0.949）
因过易而废。**在一个模型 95% 都做对的 prompt 集上，不存在可观测的自确认失败。**

### 8.3 未采取的动作

- **未提高库上限。** config 与报告的 `action_if_failed` 都写明上限在看到梯度前固定、
  之后不得提高。按 q=0.051 估算 color 需要 M≈40，但那是治症状：真正的约束是 prompt 难度。
- **未下调 0.40 阈值。**
- **未把主族 floor 从 3 再降到 2**（existence+spatial 本可过线）。floor 已在 L0 从 4 降到 3
  并公开声明；连续两次看到结果再降阈值不可接受。

### 8.4 采取的动作

把主族场景从 2 物体提高到 3 物体（`objects_per_scene`，受 `len(SHAPES)=3` 限制以保证每个
物体形状唯一、冻结的问题模板不产生歧义）。难度在**独立的 calibration split** 上标定，
标定完冻结后再在 held-out probe split 上重测 Gate A，因此 Gate A 的测量不取自用于选难度的那一份。

### 8.5 工件

- `runs/v3/gate-a/gate_a_merged.json`（含 `diagnosis` 段）
- `runs/v3/gate-a/shard-gpu{0,1}/{gate_a.json,pools.jsonl,natural_pools.jsonl,candidate_manifest.jsonl}`
- 96 个 packet，每个含 16 条候选记录与 RGB 哈希

## 9. L0 — v3.0 就绪决策复算（2026-08-30，无 GPU）

在公开声明的 floor=3 下重算冻结的 HQ 证据，**非新测量**（`derived_measurement: false`）。

四项子检查全绿，`selected_eligible_families = [existence, color, spatial]`：

| 族 | 参考准确率 | 生成覆盖 | 盲审 precision | Oracle@4 |
|---|---:|---:|---:|---:|
| existence | 1.00 | 1.00 | 1.00 | 1.00 |
| color | 1.00 | 1.00 | 1.00 | 1.00 |
| spatial | 1.00 | 0.80 | 1.00 | 1.00 |

两处声明式改动写入决策文件本身：主族 floor 4→3（附理由，`declared_before_any_v3_result: true`）；
`-2A` 的 LoRA backward/resume 由 `skipped_by_stop_rule` 补为 `measured_pass`
（Phase 8 canary，标记 `evidence_is_non_preregistered: true`）。五份输入 SHA 绑定，
冻结的 v2.2 决策只读。

`require_joint_readiness` 的 floor 现为参数，**默认仍是 4**，v2.2 消费者契约未松动。

- `runs/v3/readiness/decision-v3.json`，digest `408f7acf04f8a62c`

## 10. 难度标定：3 物体（2026-08-30，calibration split，n=24）— **已作废，结论不成立**

> **更正（见 §11）**：本节的 calibration split 24/24 行可见文本使用 `square` 而非注册的 `box`
> 词汇，而 §8 的 2 物体基线用的是干净的 v2.3 manifest。因此下表**同时改变了物体数量与显示词汇**，
> 不能把差异归因于物体数量。「3 物体没能推动 color」这一结论**未被建立**，需在修复后的
> box 词汇下重测才能判断。以下数字保留为记录。

独立 calibration split，24 prompt（8/族），M=16，共 384 次生成。**目的是标定难度，不是 Gate A 测量。**

| 问题族 | 2 物体准确率 | 3 物体准确率 | 变化 |
|---|---:|---:|---:|
| existence | 0.865 | 0.898 | +0.033 |
| color | **0.949** | **0.922** | **−0.027** |
| spatial | 0.671 | 0.595 | −0.075 |

合并 balanced_rate **0.292**，比 2 物体的 0.344 **更差**。拒绝原因：全对 9、
`informative_but_unbalanced` **6**（25%，2 物体时为 12.5%）、弃答 2。

### 10.1 结论：`objects_per_scene` 对 color 无效

- **color 几乎没被推动**（0.949 → 0.922）。原因是 color 问题问「the {shape} 是什么颜色」，
  而场景内形状唯一；再加一个不同形状的物体不制造颜色-形状绑定混淆，模型只需给每个形状涂对颜色。
- **spatial 被推过了峰值**（0.671 → 0.595），产生更多「全错」与「只有 1 个对」的池，
  `informative_but_unbalanced` 翻倍。

难度与平衡池产出率是**倒 U 型**：把 spatial 推过峰值的同时没能推动 color，净效果是变差。

**n=24（每族 8），结论是方向性的，不是精确估计。**

### 10.2 工件

- `runs/v3/calib-3obj/calib_merged.json`
- `runs/v3/calib-3obj/gpu{0,1}/`，24 个 packet
- 数据：`data/selfsight-v3-cal/manifests/calibration.jsonl`，
  signature digest `b0941a37bc438f20`

---

## 11. v3 显示词汇缺陷与作废（2026-08-30）

### 11.1 缺陷

`scripts/build_v3_scenes.py` 在每一行写入元数据 `{"square_display_name": "box"}`，
**但从未调用替换**。v2.3 的两道保障在 v3 都缺失：`v23/data.py::display_text()`
（改写 prompt 与 question）与 `v23/audit.py` 的 `box_vocabulary_has_no_square_wording`。

元数据**声称**用 box 词汇而可见文本仍用 square，所有下游消费者读的是元数据，因此无声。

`scripts/audit_v3_vocabulary.py` 实测：

| manifest | 可见文本含 `square` | 其中声称 `box` |
|---|---:|---:|
| `selfsight-v3-pool/tier_a_probe` | **183 / 256** | 183 |
| `selfsight-v3-cal/calibration` | **24 / 24** | 24 |
| v2.3 `gradient_probe` | 0 / 96 | — |
| v2.3 `train` | 0 / 432 | — |

**v2.3 干净；这是 v3 引入的回归。**

### 11.2 影响量：39.5%

verifier 把「4 顶点 + 外接框长宽比 0.5–2.0」判为 `Shape.SQUARE`
（`data/generated_verifier.py`）。对作废运行中的 200 张图、157 个 SQUARE 物体测长宽比
（长/短）：

| 区间 | 数量 | 占比 |
|---|---:|---:|
| 1.00–1.10 | 40 | 25.5% |
| 1.10–1.25 | 55 | 35.0% |
| 1.25–1.50 | 39 | 24.8% |
| 1.50–2.00 | 23 | 14.6% |

中位数 **1.19**，p75 1.33，p90 **1.62**，最大 1.84。
**严格读「square」会拒掉 62/157 = 39.5%，而 gold 全部判对。**

三项后果：

1. 盲审 precision 1.00（n=28）建立在「rectangle-like 计为正确」即 box 读法上
   （§1.3），**不能转移**到严格读法；
2. gold 读宽松类别、全部模型臂被问严格词，**系统性放大 `oracle − naive`**，
   偏差方向会把 Gate C 的预检推向假绿灯；
3. 「external verifiable correctness」不再度量「有没有画出被要求的东西」——
   只有 prompt 说 `box` 时宽松 verifier 才是合法裁判。

### 11.3 修复与结构等价性

新增 `src/selfsight/v3/vocabulary.py`（改写规则 + 声称/内容一致性审计 + fail-closed 断言）；
`build_v3_scenes.py` 对 prompt 与 question 真实替换、在 `scene.metadata` 记录
`shape_display_alias` 与 `quadrilateral_aspect_ratio_range`、写盘前 fail closed。
内部几何码（`atom.subject`、`scene.metadata.*_shape`）不改写，verifier 查找不受影响。

用修复后的构建器以相同参数（`objects_per_scene=2`、`seed=20260901`、
existence+spatial、256 行）重建：

- signature digest **`26b1e97813aa2312…`，与作废 manifest 逐字节相同**；
- scene_ids 相同、atoms 相同；
- 仅 175 条 prompt 与 134 条 question 改词。

**结构未变，只改可见措辞。**

### 11.4 作废与接替

停止时进度 gpu0 84/128、gpu1 70/128 packet（409MB）。

| 作废 | 接替 |
|---|---|
| `runs/v3/pool-ext.square-vocabulary-invalid-20260830/` | `runs/v3/pool-box/` |
| `data/selfsight-v3-pool.square-vocabulary-20260830/` | `data/selfsight-v3-pool/` |
| `runs/v3/selection-headroom/stage1.square-vocabulary-invalid-20260830/`（natural 0.754 / oracle@4 0.879 / n=66） | 待重测 |

**不得把修正后的 probe 续跑进旧目录**：`v3.supply.run_bank_probe` 按
`packets/{index}-{scene_id}.json` 断点续跑，而 scene_id 不受词汇修复影响，
会静默复用 square-prompt 的旧图配 box-prompt 的新 manifest。

**未做**：`data/selfsight-v3-cal` 保留为标定当时的原始记录，未重建。
因此 §10 的难度标定结论（`objects_per_scene=2`）是在 square 措辞下取得的，
换成 box 后模型输出是否改变尚未测量。

### 11.5 工件

- 审计：`runs/v3/vocabulary-audit/report.json`（缺陷）、`post-fix.json`（0/256 干净）
- 作废说明：两个作废目录下的 `INVALIDATED.md`
- 代码：`src/selfsight/v3/vocabulary.py`、`scripts/audit_v3_vocabulary.py`、
  `tests/test_v3_vocabulary.py`（14 项）

### 11.6 词汇缺陷的可比性后果（补记）

缺陷不只影响单次测量的正确性，还决定哪些测量之间**可以相互比较**：

| 测量 | manifest | 物体/场景 | 词汇 |
|---|---|---:|:---:|
| §8 Gate A（balanced_rate 0.344） | v2.3 `gradient_probe`，96 行 | 2 | CLEAN `box` |
| §10 三物体标定 | v3-cal `calibration`，24 行 | **3** | **DIRTY `square`** |
| 作废的 `pool-ext`（256 行扩样本） | v3 pool 旧 | 2 | **DIRTY `square`** |
| `pool-box`（重跑） | v3 pool 新 | 2 | CLEAN `box` |

两点：

1. **§10 的比较同时改变了物体数量与显示词汇**，因此「3 物体没能推动 color」未被建立
   （该节顶部的更正已记录此事）。
2. **作废的 256 行扩样本运行出于同一理由也无法与 §8 的 Gate A 基线比较。** 重跑后
   `pool-box` 与 §8 在物体数与词汇上一致（prompt 集不同：256 条 v3 新 prompt、
   仅 existence/spatial，对 96 条 v2.3 prompt、三族），因此是同类可比，而非同一集合重测。

这项可比性是修复的直接收益，独立于单个数字是否变化。

---

## 12. 就绪审计的词汇污染，与仓库整理（2026-08-30）

### 12.1 §1 的盲审问题也用了 `square` 措辞

`runs/readiness/showo2-1p5b-hq/a3-human-r1-review.csv` 的 49 条问题里，**16 条含 `square`，0 条含 `box`**。该审计跑在 v2.2 manifest 上，**早于 v2.3 引入 `box` 词汇**。也就是说：决定保留/淘汰哪些问题族的那份证据，本身带有 §11 所述的同一类构造歧义。

把 8 条人机分歧逐条对照措辞后：

| id | 族 | 人 | 机器 | 措辞 | 问题 |
|---|---|---:|---:|:---:|---|
| generated-025 | binding | 弃权 | green | **square** | What color is the square? |
| generated-035 | binding | 弃权 | red | **square** | What color is the square? |
| generated-014 | count | 2 | 7 | **square** | How many squares? |
| generated-022 | count | 0 | 4 | **square** | How many squares? |
| generated-007 | count | 8 | 7 | — | How many **circles**? |
| generated-023 | count | 3 | 2 | — | How many **triangles**? |
| generated-048 | count | 1 | 3 | — | How many **triangles**? |
| generated-049 | count | 8 | 6 | — | How many **circles**? |

三项测量结果：

1. **binding 的 precision 0.778 完全由词汇造成。** 2 条分歧都是 `What color is the square?`，且人工标注为 `parseable=no`（弃权），即无法确定被问的是哪个物体——verifier 的 SQUARE 含长宽比至 2:1 的矩形。剔除后 7/7 = 1.00（n=7，只能说「未被证明有问题」）。
2. **count 的 precision 0.40 大部分是真的。** 6 条分歧中 4 条与 `square` 无关（数圆、数三角形均错）。剔除 2 条词汇相关项后仍为 8 题错 4 题，precision ≈ **0.50**。轮廓计数在图形粘连时不可靠。
3. **size 的淘汰与词汇无关。** coverage 0.20、observation 0.60 两项独立失败。

**本节不改变 §1 的任何记录。** 按 §11 先例，缺陷在此登记，binding 是否复核由后续决定。

### 12.2 盲审图像已丢失，标注无法复用

上述 CSV 的 49 条标注仍在，但 `image_path` 全部指向**迁移前的仓库外 runs 根目录**（`…\selfsight-runs\readiness\showo2-1p5b-hq\a3-generated-r1-images\`；完整原文在该 CSV 自身的 `image_path` 列里）。**该目录已不存在，0/49 可访问**。因此这些标注只能作为历史记录，**不能用于验证任何重建后的 verifier**。任何 binding 复核都必须重新生成图像并重新标注。

工件不在 git 里（`/runs/` 在 `.gitignore` 中）且无备份。已在 STATUS.md 增加 hard stop：不删除 `runs/` 下的研究工件。

### 12.3 仓库整理（路径变更记录）

为保持工件路径可追溯，记录本次移动：

| 原路径 | 新路径 | 处置 |
|---|---|---|
| `runs/mock-pilot`, `-balanced`, `-balanced-v2` | `runs/_archive/` | 归档（mock 数据，已被真实运行取代） |
| `runs/audits`, `runs/host`, `runs/manifests`, `runs/wandb` | `runs/_archive/` | 归档（未被本文件任何节引用） |
| `data/selfsight-v1` | `data/_archive/` | 归档（v1 为冻结负对照） |
| `data/selfsight-v2.3.failed-template-overlap-20260829` | `data/_archive/` | 归档（已隔离的失败生成） |
| `tmp/`（5.78 GB，34641 文件） | 已删除 | 失败的 venv、pip/pytest 临时文件，无研究内容 |
| `cache/pip`（2.73 GB） | 已删除 | 可再生 |
| `src/selfsight/backbones/showo_v1.py` | 已删除 | 零 importer；可从 `8371a92` 或 tag `v2.1-showo-gate-red` 恢复 |

未移动：`runs/readiness`、`exploratory-post-gate`、`v2.3-rfo-gold`、`v3`、`canaries`、`review-packets`、`gate-minus-1`（最后一项被 `configs/observers/qwen2vl_2b.yaml` 引用）。
