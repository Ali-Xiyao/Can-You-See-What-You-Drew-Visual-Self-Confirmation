# Can You See What You Drew? Visual Self-Confirmation in Unified Multimodal Models

**Version 3.0 · 2026-08-30 · 目标会议 ICLR 2028**

---

## 摘要

统一多模态模型正在用自身的理解能力评价自己的生成结果，把「模型能从自己画的图中读回原意」当作图像正确的证据，并以此作为自训练奖励 [1–8]。本文检验这个奖励信号何时失效。

我们首先建立一把标尺：**分叉点 D\***——内部循环一致性仍在上升、而外部可验证正确率停止上升的最早训练步。但 D\* 的测量需要外部真值，而真正需要这个结论的从业者恰恰没有真值可用。因此本文的核心交付是一个**不依赖 ground-truth verifier 的预警信号**：把「某个 reward 的梯度」定义为该 reward 所选样本上 SFT loss 的梯度，在同一候选池上只改变选择准则，监控泄漏式自一致性与隔离盲观察两条更新方向的夹角。我们证明该夹角的转折点 **D_g 早于 D\***，给出可操作的早停规则。

随后我们把分叉归因到意图泄漏，并评估反馈原则 **Render–Forget–Observe（RFO）** 能否将其推后：强制自我评价穿过真实 RGB 瓶颈，移除对原始意图的访问，只接受盲观察者能稳定确认的原子事实。RFO 在本文中同时扮演两个角色——它既是修法，其隔离观察方向也正是构造上述检测器所需的对照。

核心主张：**自一致性只在模型仍允许自己被实际渲染结果「惊讶」的那段区间内值得信任，而这段区间的位置可以被无标签地测量。**

---

## 1. 一级主张

论文按三步推进，顺序即叙事顺序：

| | 主张 | 主要证据 |
|---|---|---|
| **①  预测** | 分叉可被**不依赖 ground-truth verifier** 地提前检测：`cos(g_naive, g_rfo)` 的转折点 D_g 早于 D\* | Lead = D\* − D_g > 0，配对 bootstrap CI 不跨 0 |
| **②  机理** | 分叉主要由意图泄漏驱动，共享感知盲点为次因 | 上下文消融对 D_g / D\* 的位移量分解；per-block 梯度余弦定位 |
| **③  延后** | RFO 把 D\* 推后或消除，位移量与②的归因同量级 | compute-matched 下 external correctness 提升，D\* 右移 |

D\* 是**标尺不是卖点**：「代理指标升、真指标不升」的曲线形状在 reward overoptimization 文献中已是常识 [18]。D\* 的唯一作用是给①的无标签信号提供一把用程序化真值刻好的尺子。

**这项工作值得做的原因**：现有 overoptimization 早停方法全部依赖 ground-truth queries [22]，无标签替代（loss plateau 等）已被证明与正常收敛不可区分 [23]；梯度层检测在文本/RLVR 上刚起步 [19–21]，在统一多模态模型上是空白。而文本 overoptimization 的 gold 是一个更大的 reward model，它自己就在漂 [18]；本文受控层的真值由 scene graph 构造，是精确的。**想到测梯度分歧不难，但只有在这里才测得准。**

---

## 2. 关于「无标签」的准确表述

检测器需要构造两个方向：`g_naive`（泄漏式自一致性所选样本）与 `g_rfo`（隔离盲观察所选样本）。为避免循环论证，构造 `g_rfo` 的观察配置必须与训练所用的配置解耦（§7.3）。

因此本文**不主张零外部依赖**。准确表述是：

> 检测器不需要 ground-truth verifier、不需要任务标签、不需要人工标注。它需要一个冻结的异构观察者。这严格弱于需要 verifier——观察者只需回答原子视觉问题，不需要判定「是否满足目标」，也不需要比生成器更强。

**我们把这一点做成一个实验而不是一段辩解**：报告检测器在观察者比生成器**更弱**时是否仍然有效（§8.4）。若在弱观察者下 Lead 仍显著为正，主张①的适用范围比"需要 verifier"宽得多；若不成立，如实报告检测器对观察者质量的依赖曲线。

---

## 3. 已完成的前置工作与它决定的设计

v2.x 阶段已在本地完成三个 backbone 的就绪审计与一次工程闭环。完整数据见 `docs/EVIDENCE_LOG.md`。三个结果直接决定了 v3.0 的设计，写在这里而不是藏在 appendix：

### 3.1 backbone 与问题族已由实测选定

`showlab/show-o2-1.5B-HQ`（rev `d3a220ec`）在 6 个问题族上的实测：

| 问题族 | 参考观察准确率 | 生成覆盖率 | 盲审 verifier precision | Oracle@K=4 | 采用 |
|---|---:|---:|---:|---:|:---:|
| existence | 1.00 | 1.00 | 1.00 | 1.00 | ✅ 主 |
| color | 1.00 | 1.00 | 1.00 | 1.00 | ✅ 主 |
| spatial | 1.00 | 0.80 | 1.00 | 1.00 | ✅ 主 |
| binding | 1.00 | 0.90 | 0.78 | 1.00 | ⬜ hard tier |
| count | 0.85 | 1.00 | 0.40 | 0.50 | ⬜ hard tier |
| absolute size | 0.60 | 0.20 | 1.00 | 0.20 | ❌ 排除 |

**主问题族固定为 existence / color / spatial**，这三族在四项指标上全部为 1.00 或以上阈值，且 precision 由盲审人工标签（n=28）验证。count 的失败原因是 verifier 计数精度（0.40），size 的失败原因是模型无法渲染绝对尺寸（覆盖率 0.20）——两者都是可解释的仪器/能力边界，作为 hard tier 如实报告，构成附带贡献（能力地板表）。

Show-o2-7B 已审计：Oracle@4 提升到 0.92，但首候选整体覆盖率降到 0.767，固定种子覆盖摆动恶化到 16pt（HQ 为 10pt）。**放大规模没有修复生成域可测量性**，7B 与 Show-o v1 一并作为冻结负对照。

### 3.2 候选供给是这条路线的真实瓶颈

在 HQ 上用随机 K=4 采样构造候选池，**42 个 prompt 中只有 9 个池同时包含 verifier-正确与 verifier-错误的候选**（21%）。分族：existence 6/16，color 2/13，spatial 1/13。

含义比"采样效率低"更强：在这个尺度上，模型对某个 prompt 的正确性接近确定性。**如果候选池是退化的（全对或全错），基于选择的自训练就没有可利用的方差，也就不存在可观测的分叉。** 这是整条路线的前提条件，因此 v3.0 把它提升为**第一道门（Gate A）**，并用有界候选库搜索来构造，而不是靠随机采样碰运气（§6.1）。

### 3.3 原设计的梯度噪声地板控制是错的

v2.x 用「同准则、**不相交**半批之间的梯度余弦」作为噪声地板。实测两次：`[-0.082, 0.322]`（n=9 池）与 `[0.135, 0.290]`（n=12 对）。

这个控制量与被测量不在同一尺度上：跨准则余弦是在**同一批 prompt**上比较（噪声大部分共享、配对抵消），半批余弦是在**不相交样本**上比较（噪声独立且加倍）。证据是同一次运行里 identical 与 naive/rfo 的配对余弦精确等于 1.000，而半批余弦只有 0.08。**用后者给前者划地板，是拿一把量错东西的尺子。**

v3.0 的噪声控制改为**在 prompt 层做配对 bootstrap**：对 prompt 集有放回重采样，在同一重采样集上重算 `cos(g_naive, g_rfo)`，直接给出该余弦自身的 CI。不相交半批余弦降级为独立报告的「单样本梯度信噪比」诊断量，不再充当门槛。

这项修正使主终点比 v2.x 设想的**更可测**，而不是更难测。

---

## 4. 问题形式化

**Visual Self-Confirming Failure**：对某张生成图像，(a) 程序化 verifier 判定它违反至少一个目标属性，且 (b) 生成模型自身在该属性上给出与原始意图一致、而与像素不一致的回答。

**意图泄漏（Intention Leakage）**：观察阶段仍可访问原始 prompt、同一会话历史、生成 latent / image tokens / KV cache，或来源标签。

**共享感知盲点（Shared Perceptual Blind Spot）**：即使在 fresh context + image-only 条件下，同模型或同家族观察者仍在相同视觉模式上共同失误。

### 4.1 指标

| 指标 | 定义 | 角色 |
|---|---|---|
| **GDA-free** | `cos(g_naive, g_rfo)`：泄漏式自一致性与隔离盲观察两条更新方向的夹角余弦 | **核心信号** |
| **D_g** | GDA-free 的分段回归断点，且该点余弦的配对 bootstrap CI 不含起点值 | 主终点分量 |
| **Lead = D\* − D_g** | 无标签信号相对指标分叉的领先步数 | **主终点** |
| GDA-gold | `cos(g_naive, g_gold)`：需要 verifier，不可部署 | 机制参照，①→②的桥梁 |
| **D\*** | internal cycle score 仍显著上升、external verifiable correctness 停止上升或下降的最早训练步；若预算内未出现则取 T_max | **次主终点 / 标尺** |
| ΔSCFR | 从训练起点到 D\*+Δ 步，错误确认率的变化 | 次主终点 |
| SCFR | 外部判错图像中模型仍判正确的比例 | 主诊断量 |
| POE | 已知像素反事实上，回答跟随原意而非修改后像素的比例 | 因果证据 |
| CSR | 收到原子反馈后目标错误被修复且正确属性未被破坏的比例 | 实用价值 |

**D\* 的兜底定义**（`T_max`）使 Lead 在「未观察到分叉」时依然有定义。这解决了 v2.x 的一个结构缺陷：主终点不能依赖一个可能不存在的量。

### 4.2 梯度量的等价定义

主线不使用可微 reward，因此不存在显式 reward gradient。采用：

> **某个 reward 的梯度 ≔ 该 reward 所选出的样本集上 SFT loss 的梯度。**

这是模型实际收到的参数更新方向，比 policy gradient 更直接、方差更低。关键设计：在**同一批 prompt、同一个候选池**上只改变选择准则，得到完美配对的三个梯度，唯一差异来源是「谁选的样本」。

| 记号 | 选择准则 | 需要 verifier |
|---|---|---|
| `g_gold` | 程序化 verifier | 是 |
| `g_naive` | 朴素 cycle score | 否 |
| `g_rfo` | 隔离盲观察 | 否 |

### 4.3 测量协议

1. **固定 probe set**，全程不变，规模以**informative pool 数**计（不是 prompt 数）；
2. **在 LoRA 子空间计算**，同时报告 per-block 余弦以定位分歧层段；
3. **checkpoint 序列 EMA 平滑**后再做断点估计，窗口预先固定；
4. **报告范数比** `‖g_naive‖/‖g_rfo‖`，区分方向分歧与幅度分歧；
5. **配对 bootstrap** 给出每个 checkpoint 余弦的 CI（§3.3）。

---

## 5. 数据：SelfSight-Bench v3

主问题族 existence / color / spatial，plain white background，视觉词汇使用 `box`（四边填充区域，轴对齐外接框长宽比 0.5–2.0，允许旋转），避免 `square` 的构念歧义。该词汇已通过盲审人工/verifier 一致性校准（28/28，各族均为 1.00）。

| 层 | 规模 | 作用 |
|---|---:|---|
| Tier A 受控生成 | 800 prompts（train/probe/outcome 零重叠） | 主测量场 |
| Tier B 像素反事实 | 300 对 | 因果证据（RQ-①的前置现象） |
| Tier C 自然场景 | 300 prompts 抽样 | 外部有效性方向抽查，非贡献点 |
| Hard tier | count 30 + binding 30 | 单独报告，不进主决策 |

**Tier B 构造**：选外部验证为正确的简单生成图 → 对一个原子属性做最小可控编辑（删除对象、改颜色、左右互换）→ 保持原 prompt 不变 → 用开放式原子问题询问实际事实。真值完全由构造过程给出。

**提问设计**：全部为开放式原子问题或平衡强制选择；二元提问仅用于诊断 yes-bias，不进主指标。同一问题在所有观察条件与观察者之间完全一致。

**观察条件矩阵**：

| 条件 | 原始 prompt | 同一会话 | 生成隐状态 |
|---|---|---|---|
| Same-session self-check | 有 | 有 | 可访问 |
| Fresh-context prompted | 有 | 无 | 无 |
| Fresh-context image-only | 无 | 无 | 无 |
| Counterfactual prompt | 冲突 | 无 | 无 |
| 独立冻结观察者 | 无 | 无 | 无 |

---

## 6. 方法：Render–Forget–Observe

RFO 不改变基础架构，只重新设计反馈通道。

- **Render**：图像 tokens / latent 必须真正 decode 为 RGB、写入文件、从磁盘重新读取、断开原生成计算图。禁止把 hidden state、semantic token 或 KV cache 传给观察器。
- **Forget**：观察器在独立进程与独立会话中运行；不提供原始 prompt、生成历史、来源标签；不复用 cache；不使用「是否满足这段 prompt」式诱导提问。
- **Observe**：观察器参数冻结，生成器可更新（避免两者共同演化出私有协议）。`RFO-Self` = 同基础模型的冻结 step-0 副本，image-only fresh context。
- **Compare**：目标事实直接来自 scene graph。比较器不输出模糊总分，只输出明确差异（目标：两个蓝色 box / 观察：一个 / 错误类型：数量 / 其余属性：正确）。
- **Repair**：三种部署形态——`RFO-Rerank`（免训练，按事实一致性择优）、`RFO-Repair`（推理时，最小修正指令）、`RFO-PostTrain`（主线，rejection-sampling SFT）。主结果不含 RL。

**训练原则**：只奖励能够通过硬渲染、上下文隔离与盲观察之后仍被确认的事实，而不是奖励模型对原始意图的内部恢复。

### 6.1 候选库搜索（v3.0 新增，Gate A 的实现）

随机 K=4 只有 21% 的池是 informative（§3.2）。v3.0 改为有界候选库搜索：

1. 对每个 prompt，用与梯度/dropout **不相交**的搜索种子生成 M=16 个候选（上限预先固定）；
2. 用程序化 verifier 给每个候选打分；
3. 从库中取出 K=4 呈现给全部选择准则：尽可能 2 个 verifier-正确 + 2 个 verifier-错误；
4. 无法凑齐平衡池的 prompt 从**全部 arm 中对称移除**；
5. 同一个 K=4 池原封不动地交给 naive / rfo / gold 三个选择器。

**定位说明**：候选库是**测量仪器的标定刺激**，不是训练分布。因此——

- **梯度探针**使用库平衡池（保证 GDA 有动态范围）；
- **训练与外部正确率曲线**使用模型的自然 K=4 采样（保证 D\* 反映真实自训练）。

两者必须在论文中显式区分，并报告库平衡池上估出的 D_g 是否预测了自然池上测出的 D\*。**这个跨分布的验证本身就是①最强的证据**：如果一个用标定刺激测出的信号能预警自然训练中的退化，那它就是一个真正的仪器。

---

## 7. 实验计划

### 7.1 实验组

| 组别 | 反馈方式 | 看到 prompt | hard render | 冻结观察器 |
|---|---|---|---|---|
| Base | 无自反馈 | – | – | – |
| **Naive Cycle** | prompt→image→恢复 prompt | 间接 | 是 | 否 |
| **RFO-Gold** | 程序化 verifier 选择 | 否 | 是 | 是（真值） |
| **RFO-Self** | 冻结 step-0 盲原子观察 | 否 | 是 | 是 |
| Fresh Blind | fresh context image-only | 否 | 是 | 是 |
| Same-Context Judge | 同会话总分判断 | 是 | 否 | 否 |

**Base / Fresh Blind / Same-Context Judge 为推理时或消融组，不占主训练预算。**

### 7.2 机制优先的执行顺序

先跑 **RFO-Gold vs Naive**，而不是 RFO-Self vs Naive。理由：RFO-Gold 是**阳性对照**——它用完美选择器、相同候选池、相同更新预算。若一个完美选择器都无法在这个 setup 里产生可测的训练优势，那么限制因素不是观察者质量，而是修正目标 / 样本权重 / 梯度探针，此时增加算力或换 backbone 都是浪费。

```
Pass 1   A800-0: Naive        ‖  A800-1: RFO-Gold      → Gate C
Pass 2   A800-0: RFO-Self     ‖  A800-1: checkpoint 评测 → Gate D
```

### 7.3 检测器与训练器的解耦（硬性要求）

| 用途 | 使用的隔离观察配置 |
|---|---|
| 训练（RFO-PostTrain 样本选择） | `RFO-Self`：同基础模型冻结 step-0 副本，image-only fresh context |
| **检测（构造 `g_rfo`）** | **冻结异构 VLM（Qwen2-VL-2B），全程不参与任何训练样本选择** |

若不解耦，「检测器报警 → 换 RFO 有效」构成同义反复。此外，非平凡内容不在「两个梯度不同」——它们在任何时刻都不同——而在**时间结构**：夹角在训练早期落在其配对 CI 内，随后单调扩大，且这一扩大发生在外部正确率退化**之前**。这个时序命题无法由定义差异推出，并且可以真实失败（Gate D）。

### 7.4 实验清单

| 编号 | 内容 | 硬件 |
|---|---|---|
| **E0** | 候选库搜索与 informative pool 供给统计（Gate A） | 3090 ×2 |
| **E1** | 五种观察条件下的 SCFR / POE / PSFP；Tier B 因果证据 | 3090 ×2 |
| **E2** | 主训练曲线 + 梯度分歧监控（**第一主实验**） | A800 ×2 |
| **E3** | compute-matched 价值验证：rerank / repair / posttrain vs 全部 baseline | A800 ×2 |
| **E4** | 机制消融：去 hard render、保留 prompt、观察器共同更新、原子问题→整体打分 | A800 ×2 |
| **E5** | 理解能力保持 + Tier C 方向抽查 | 3090 ×2 |

**E2 每个 checkpoint 记录**：internal cycle score / external verifiable correctness（→D\*）/ SCFR / GDA-free 及范数比及 per-block 分解（→D_g）/ GDA-gold（机制上界）/ 配对 bootstrap CI / 观察答案熵（作为竞争基线信号，须被 GDA-free 击败）。

### 7.5 基线

无自反馈 Base；same-context self-judge；fresh-context self-judge；naive cycle consistency（主对手）；冻结异构 VLM judge；CLIP / VQAScore rerank；random best-of-K；UniRL / UniCorn / ASG 风格自生成反馈的最接近可复现版本 [5–7]。全部在**相同 K 与相同算力**下比较。

### 7.6 消融

| 消融项 | 验证的问题 | 预期方向 |
|---|---|---|
| 去掉 hard render | 渲染屏障是否必要 | D\* 提前 |
| 保留 prompt（不 Forget） | 意图泄漏的贡献 | SCFR 上升、D\* 提前 |
| 观察器与生成器共同更新 | 冻结观察器是否必要 | internal 升更快、external 不升 |
| 原子问题 → 整体打分 | 反馈粒度 | CSR 下降、collateral damage 上升 |
| GDA-free vs 熵类无标签信号 | 梯度信号是否优于更简单替代 | GDA-free 的 Lead 更大 |
| GDA 全参数 vs LoRA 子空间 | 投影是否必要 | 全参数余弦退化到无信息量 |
| 检测器 = 训练器 vs 解耦 | 循环论证的影响量 | 不解耦时 Lead 被系统性高估 |
| 弱观察者构造 `g_rfo` | §2 的适用范围 | Lead 随观察者能力单调变化 |
| 库平衡池 vs 自然池 | §6.1 的跨分布有效性 | D_g 一致，D\* 不同 |

---

## 8. 统计方案

**主线先跑单主 seed**（`20260901`）。主 seed 的作用是判断曲线形状是否分离、仪器是否工作；**不从单 seed 做显著性主张**。只有主 seed 显示可分离形状后，才投入 seed 2/3 做区间估计。

- 主比较以 prompt 为配对单位，报告 paired bootstrap 95% CI；
- 二元错误使用 McNemar 检验；
- **D\* 与 D_g 的估计**：分段线性回归 + profile likelihood 断点，估计流程与超参在实验开始前写入代码并冻结；
- **Lead 的推断**：直接对 `(D* − D_g)` 做配对 bootstrap，不分别给 CI 再目测重叠（那会严重低估功效）；
- **GDA 余弦的所有报告必须伴随同 checkpoint 的配对 bootstrap CI**，不得单独出现；
- **SCFR 必须伴随观察者准确率与原始分母**一同报告；
- 所有曲线图绘出单 seed 而非只画均值；
- 不把 1 个百分点以内的无方差差异写成确定提升；
- 主终点 `Lead`、次主终点 `D*` 与 `ΔSCFR` 在实验前指定，结果出来后不得替换。

**多 seed 的触发条件**：主 seed 上 Gate C 与 Gate D 均为绿，且 Naive 与 RFO 曲线在 external correctness 上的差距超过 checkpoint 内方差。否则先诊断，不加 seed。

---

## 9. Go / No-Go

四道门，每道门只在有真实失败可能时才存在。

### Gate A — 候选供给（第一道门）

有界候选库搜索（M ≤ 16）能否在主问题族上稳定产出平衡的 K=4 池。

- 平衡池产出率 ≥ 40%；
- 梯度探针可凑齐 ≥ 128 个 informative pool；
- 搜索种子与梯度/dropout 种子不相交，库上限在看到任何梯度输出之前固定。

**不过：** 基于选择的自训练在此 backbone 上没有可利用的方差。停止，产出物是能力地板报告，不是主线论文。

### Gate B — 仪器有效

- 主问题族的 verifier precision 由盲审人工标签验证 ≥ 0.95（已达成：existence/color/spatial 均为 1.00）；
- identical-selection 梯度余弦 ≥ 0.999（已达成：1.000）；
- **GDA-free 的配对 bootstrap CI 宽度 ≤ 0.10**（v3.0 新控制量，见 §3.3）；
- LoRA 子空间投影下余弦不退化到 0 附近。

**不过：** 增大 probe 的 informative pool 数直到 CI 收窄；若增大到预算上限仍不收窄，主张①退回熵类替代信号，如实报告梯度信号在此规模下不可测。

### Gate C — 机制阳性对照（**生死实验**）

主 seed 上，RFO-Gold 在相同候选池与相同更新预算下：

- external verifiable correctness 高于 Naive，差距超过 checkpoint 内方差；
- 选择一致率与 Naive 显著不同（已知：0.444）。

**不过：** 完美选择器都无法产生训练优势 ⇒ 限制因素是修正目标/样本权重/梯度探针，**不是**观察者质量、模型规模或算力。停止扩展，回到目标函数诊断。这是项目的**真实失败条件**。

### Gate D — 主张成立

主 seed 上：

- **①** `Lead = D* − D_g > 0`，且 D_g 处 GDA-free 的配对 CI 不含起点值；早期 GDA-free 落在起点 CI 内（分歧是后天出现的）；GDA-free 与 GDA-gold 的 checkpoint 级 Spearman ρ ≥ 0.7；GDA-free 的 Lead 优于熵类基线；
- **②** 至少一项上下文消融显著提前 D\* 与 D_g，且「移除 Forget」的效应量大于「移除 hard render」；
- **③** compute-matched 下 RFO-Self 的 D\* 晚于 Naive（或预算内不出现），external correctness 提升 ≥ 3 pt。

**不过：** ①不成立 ⇒ 如实报告梯度信号在本设置下无预警价值，论文重心回落到②③。②不成立 ⇒ RFO 失去理论动机，§6 改写为纯经验方法，限制一节明写「我们能预警但不能解释」。

---

## 10. 硬件与预算

| | 3090 × 2（本地 Windows，24GB，无 NVLink） | A800 × 2（服务器 Linux，80GB） |
|---|---|---|
| 角色 | 数据/候选库生成、冻结观察者服务、Gate A/B 筛查、E1、E5、图表 QA | E2 主训练曲线、E3、E4 |
| 约束 | 两卡是独立 worker，不是 48GB 池；单模型不跨卡切分 | 两卡并行跑两个 arm，不做分布式 |

HQ 的 LoRA backward/resume 实测峰值约 **10.15 GiB**，两种卡都不受显存约束。

| 阶段 | 内容 | 估算 GPU-hours |
|---|---|---:|
| P0 | 候选库搜索 + Gate A/B（3090） | 40–70 |
| P1 | Tier A/B 构建 + E1（3090） | 60–100 |
| P2 | **E2 主 seed：Naive ‖ RFO-Gold，然后 RFO-Self**（A800） | 200–320 |
| P3 | E3 compute-matched + 全部 baseline（A800） | 90–150 |
| P4 | E4 机制消融（**每项消融 = 一次完整训练**） | 250–400 |
| P5 | E5 + Tier C + 图表 + 补实验 | 60–100 |
| | **主 seed 完成度合计** | **700–1140** |
| | 追加 seed 2/3（仅 P2，条件触发） | +400–640 |

**关于预算的诚实说明**：v2.x 报的 450–750 小时把 E4 严重低估了——「报告消融对 D\* 的影响」意味着**每项消融一次完整训练**，不是一次评测。上表按完整训练计。P4 是最大的单项，可在①②③确立后按需裁剪消融项数。

**MVP（先做这一段）**：P0 + P1 + E2 短程（60–100）= **160–270 GPU-hours**。它回答三个决定性问题：候选库能不能供上（Gate A）、GDA 的配对 CI 够不够窄（Gate B）、以及 **RFO-Gold 在短程内是否已经拉开 external correctness**（Gate C 预告）。**没有看到这三个结果之前不投入剩余算力。**

存储：生成图像与公共视图 100–200GB，checkpoints/日志 150–300GB，模型权重 100–150GB，建议 400–650GB。

---

## 11. 时间线

| 时间 | 里程碑 | 门 |
|---|---|---|
| 2026.09 | 候选库搜索实现；Gate A 供给统计；梯度仪器配对 CI 改造 | **A / B** |
| 2026.10 | Tier A/B v3 冻结；E1 五种观察条件 | – |
| 2026.11 | A800 环境与迁移 canary；**E2 主 seed Pass 1（Naive ‖ RFO-Gold）** | **C** |
| 2026.12 | E2 主 seed Pass 2（RFO-Self）；D\*/D_g/Lead 估计与 Figure 1 | **D** |
| 2027.01–02 | 条件触发：seed 2/3；E3 compute-matched | – |
| 2027.03–05 | E4 机制消融；per-block 定位 | – |
| 2027.06–07 | E5、Tier C、人类评测、failure cases | – |
| 2027.08–09 | 写作、内部审稿、补实验、复现包 | – |

---

## 12. 关键图表

| 图 | 内容 | 来源 |
|---|---|---|
| **Figure 1** | 预警图：internal / external / GDA-free 三条曲线随训练步叠加；两条竖线标出 D_g 与 D\*，中间阴影为 Lead；下方平行轴为 SCFR | E2 |
| Figure 1b | 梯度仪器有效性：配对 bootstrap CI 带 + GDA-free/GDA-gold 双曲线 + per-block 余弦热力图 | E2 |
| Figure 2 | 能力地板表：观察者 × 问题族的四项指标，标出主族/hard tier/排除 | §3.1 |
| Figure 3 | RFO 流程示意 | – |
| Figure 4 | 像素覆盖因果图：固定像素改 prompt vs 固定 prompt 改像素 | E1 |
| Figure 5 | 候选库供给：自然池 vs 库平衡池的 informative 率与 D\* / D_g | E0 + E2 |

**Figure 1 必须来自真实短程数据。** MVP 阶段就要能画出雏形；如果短程中两条曲线完全重叠且无分开迹象，立即进入 Gate C 诊断，而不是先把 200 GPU-hours 跑完。

---

## 13. 计划贡献

最终论文只在实验支持时主张：

1. **【① 预测】** 一个不需要 ground-truth verifier 的分叉预警信号 GDA-free，证明它领先于外部指标退化（Lead > 0），可直接作为自训练系统的早停规则，并给出它对观察者质量的依赖曲线；
2. **【② 机理】** 把分叉归因到意图泄漏为主、共享感知盲点为辅，给出可加分解与层级定位；
3. **【③ 延后】** Render–Forget–Observe 原则与后训练流程，展示它将分叉点推后或消除；
4. 统一多模态模型中自一致性奖励**分叉点**的可重复测量（作为①的标尺）；
5. 一套受控自反馈测量协议：能力地板表、去偏提问库、配对梯度 CI、候选库供给统计；
6. SelfSight-Bench v3，用像素反事实与跨观察者控制区分反馈失败来源。

---

## 14. 预期审稿问题

**Q1：你怎么知道不是模型太小看不清？**
主问题族的观察准确率与 verifier precision 都是 1.00（盲审人工验证），且主结论是**同一个模型在训练过程中的时间序列变化**——能力地板在各 checkpoint 上是公共因子，无法解释两条曲线的**反向**运动。看不清的问题族（count / size / binding）作为 hard tier 单独报告。

**Q2：GDA-free 既由 RFO 构造、RFO 又是你的修法，这不是循环论证吗？**
检测器与训练器在设计上强制解耦（§7.3）。且非平凡内容不在「两个梯度不同」——它们在任何时刻都不同——而在**时间结构**：夹角早期落在其配对 CI 内，随后单调扩大，且扩大发生在外部正确率退化**之前**。这个时序命题无法由定义差异推出，可以真实失败（Gate D）。

**Q3：候选库搜索是不是在制造不存在的方差？**
是的，而且我们明说了。候选库是**仪器的标定刺激**，只用于梯度探针；训练与 external correctness 曲线用自然采样池。跨分布验证（库上估的 D_g 是否预测自然池上的 D\*）本身是①的核心证据（§6.1）。

**Q4：只训了一个 1.5B 模型。**
backbone 由预注册的就绪审计选定而非事先指定，7B 已审计且更差（§3.1）。这是计算约束下的选择，不对更大规模做定量外推。

**Q5：RFO 是不是只是 rejection sampling 加集成？**
RFO-Self 只用同模型冻结副本，不引入额外能力。贡献在于反馈信息的**约束**：hard render、blind observation、原子比较。全部 baseline 在相同 K 与相同算力下比较。

**Q6：分叉会不会只是过拟合？**
外部正确率在 held-out prompts 上测量，模板与属性组合严格隔离。若只是普通过拟合，internal cycle score 应当同时在 held-out 上下降；我们预期它继续上升。这两种情形在图上可直接区分。

---

## 15. 参考文献

[1] Xie et al. *Show-o: One Single Transformer to Unify Multimodal Understanding and Generation*. arXiv:2408.12528, 2024/2025.
[2] Wu et al. *Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation*. arXiv:2410.13848, 2024.
[3] Chen et al. *Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling*. arXiv:2501.17811, 2025.
[4] Deng et al. *Emerging Properties in Unified Multimodal Pretraining (BAGEL)*. arXiv:2505.14683, 2025.
[5] Mao et al. *UniRL: Self-Improving Unified Multimodal Models via Supervised and Reinforcement Learning*. arXiv:2505.23380, 2025.
[6] Han et al. *UniCorn: Towards Self-Improving Unified Multimodal Models through Self-Generated Supervision*. arXiv:2601.03193, 2026.
[7] Thawkar et al. *Ask, Solve, Generate: Self-Evolving Unified Multimodal Understanding and Generation via Self-Consistency Rewards*. arXiv:2606.27376, 2026.
[8] Shan et al. *CyCLeGen: Cycle-Consistent Layout Prediction and Image Generation in Vision Foundation Models*. arXiv:2603.14957, 2026.
[9] Mollah et al. *The Telephone Game: Evaluating Semantic Drift in Unified Models*. arXiv:2509.04438, 2025.
[10] Chu et al. *CycleGAN, a Master of Steganography*. arXiv:1712.02950, 2017.
[11] Spiliopoulou et al. *Play Favorites: A Statistical Method to Measure Self-Bias in LLM-as-a-Judge*. arXiv:2508.06709, 2025.
[12] Wataoka et al. *Self-Preference Bias in LLM-as-a-Judge*. arXiv:2410.21819, 2024.
[13] Ghosh et al. *GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment*. arXiv:2310.11513, 2023.
[14] Xie et al. *MME-Unify: A Comprehensive Benchmark for Unified Multimodal Understanding and Generation Models*. arXiv:2504.03641, 2025.
[16] Amodei et al. *Concrete Problems in AI Safety*. arXiv:1606.06565, 2016.
[17] Skalse et al. *Defining and Characterizing Reward Hacking*. arXiv:2209.13085, 2022.
[18] Gao et al. *Scaling Laws for Reward Model Overoptimization*. arXiv:2210.10760, 2022.
[19] Ackermann et al. *Gradient Regularization Prevents Reward Hacking in RLHF and Verifiable Rewards*. arXiv:2602.18037, 2026.
[20] *Detecting and Suppressing Reward Hacking with Gradient Fingerprints (GRIFT)*. arXiv:2604.16242, 2026.
[21] *Proxy Reward Internalization and Mechanistic Exploitation (PRIME)*. arXiv:2606.09711, 2026.
[22] Moskovitz et al. *Confronting Reward Model Overoptimization with Constrained RLHF*. arXiv:2310.04373, 2023.
[23] *EvalStop: Using World Feedback to Detect and Correct Reward Overoptimization*. arXiv:2606.04145, 2026.

> **注：** [6][7][8][15][19][20][21][23] 的 arXiv 编号为 2026 年条目，投稿前需逐条核验编号、作者与标题。[15] EvoTok 在 v3.0 正文中已无引用点，若不恢复引用则删除。
