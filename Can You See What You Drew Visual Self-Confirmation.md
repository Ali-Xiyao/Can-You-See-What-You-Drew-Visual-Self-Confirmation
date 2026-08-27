# Can You See What You Drew? Visual Self-Confirmation in Unified Multimodal Models

<aside>
🧭

**一句话故事**

统一多模态模型可以在自我训练中变得越来越「自洽」，同时越来越不正确：内部循环一致性分数一路上升，而外部可验证的图像正确率停在原地、甚至下滑。

</aside>

<aside>
📌

**研究定位速览**

- **目标会议：** ICLR 2028
- **论文类型：** 训练信号失效的**无标签提前预警** + 机理归因 + 轻量修正原则
- **主线主张：** 朴素自一致性奖励存在**分叉点**，且该分叉可以在**不使用任何外部标签**的前提下被梯度层信号提前检测
- **三步叙事：** ① 提前预测 → ② 机理归因 → ③ 延后分叉
- **Primary endpoint：** **Lead = D* − D_g**（无标签梯度信号相对分叉点的领先步数）
- **Co-primary：** Divergence Point **D*** 及该点处的 ΔSCFR@competent
- **Benchmark：** SelfSight-Bench v2（约 800 受控 prompts + 400 像素反事实对）
- **核心方法：** Render–Forget–Observe（RFO）
- **Backbone：** 1 个可训练 + 1 个推理审计（第三个 backbone 不在主线内）
- **预算：** 450–750 A100 GPU-hours（MVP 125–215）
- **版本：** Version 2.1 · 2026-08-27
</aside>

<aside>
💡

**核心命题**

当一个系统同时充当行动者与评价者时，它的内部一致性只在一段有限的训练区间内是外部正确性的代理；越过这个区间，优化自一致性会开始**系统性地**奖励错误。

**对应的方法原则：**

可信的自反馈必须穿过真实 RGB 渲染、隔离原始意图，并且只接受由一个**已被证明有能力看清的**公共观察者稳定确认的事实。

</aside>

<aside>
📝

**标题方案（随主线重排一并调整）**

- **方案 A（推荐）：** *When Does Self-Consistency Start Lying? Diagnosing the Divergence of Internal Agreement and External Correctness in Unified Multimodal Models*
- **方案 B：** *Can You See What You Drew? Visual Self-Confirmation and the Failure of Self-Consistency Rewards*

方案 A 把论文卖点放在「奖励信号何时开始骗你」，方案 B 保留原有记忆点。两者都不再把「模型相信自己」作为一级主张。

</aside>

---

# 0. 项目定位与硬约束

| 维度 | Version 2.1 的决定 |
| --- | --- |
| 论文一句话主张 | 朴素 cycle / self-reward 训练会提高内部一致性而不提高外部可验证正确率；这一分叉可被**无标签的梯度分歧信号**提前检测，其主因是意图泄漏，其对策是 RFO |
| 三步叙事顺序 | **① 不依赖外部结果提前预测分叉 → ② 归因分叉机理 → ③ 提出方法延后分叉** |
| 第一主实验 | E2 训练动态分叉 + 梯度分歧监控（internal / external / cos 三条曲线同图） |
| Primary endpoint | **Lead = D* − D_g**：无标签梯度信号相对指标分叉的领先步数，3 seeds，paired bootstrap 95% CI |
| Co-primary endpoint | Divergence Point D* 及该点处 ΔSCFR@competent |
| D* 的角色 | **标尺，不是卖点**：由程序化真值测得的 gold-standard 锚点，唯一用途是验证无标签信号确实预示了分叉 |
| 前置必过检验 | Gate −1：观察者能力地板 + 去 yes-bias + 同能力观察者对照 + 能力分层
Gate −1b：梯度余弦的噪声地板与零效应基线 |
| 测量工具层（原 C1/C2） | 降为第 5–6 节的前置章节，用于建立 SCFR 的可信度，不作为独立卖点 |
| 私有视觉通道（原 C5 / Tier E） | 整体移入 appendix，作为条件性负结果，预算上限 5% |
| 可训练 backbone | 1 个（Show-o 1.3B 级），Naive Cycle 与 RFO 各 3 seeds × 1 配置 |
| 第二 backbone | 仅推理审计（Janus-Pro-1B 级），不训练 |
| 第三 backbone | 不在本项目范围内 |
| 人类评测 | 投稿前 6 周，300–500 样本 × 3 评审，只覆盖最终主结论 |
| 总预算 | 450–750 A100 GPU-hours（梯度监控边际成本 5–15） |
| 失败退路 | Gate −1 不过 → 换观察者规模并如实报告能力地板；Gate −1b 不过 → 主张 ① 退回熵类信号；Gate 2b 不过 → 主张 ① 改为严谨负结果，重心回落到 ②③；Gate 2 不过 → 转写「自一致性奖励在何种条件下是安全的」正面结论论文 |

<aside>
🚫

**0.1 本项目明确不做什么**

1. **不做第三个泛化 backbone**（BAGEL / Show-o2 类），也不做任何 7B 以上模型的后训练。
2. **不把私有视觉通道写进主线**。Tier E 与 PCG 指标只出现在 appendix，摘要与标题中不得出现「private visual channel」。
3. **不使用 yes/no 诱导式提问**作为任何主指标的输入。全部原子问题为开放式或平衡强制选择。
4. **不在观察者未通过能力地板的问题族上报告 SCFR**，也不用「更强观察者 vs 生成模型」这种能力不匹配的比较去支撑 Observer Gap。
5. **不做大规模自然场景 benchmark**。Tier C 只作为外部有效性抽查（约 300 prompts），不作为贡献点。
6. **不做复杂 RL**。主线只用 rejection-sampling SFT；preference optimization 作为可选增强。
7. **不声称「统一模型偏爱自己」**，除非该效应在能力分层后仍然存在。
</aside>

---

# 1. 研究摘要

统一多模态模型（UMMs）正在把图像理解与生成合并进同一个系统，并且越来越多地用自己的理解能力评价自己的生成结果，把这种「自一致性」当作自我训练或强化学习的奖励信号。[1–8] 这些工作共享一个未被检验的假设：

> **如果模型能从自己生成的图片中恢复原始语义，那么这张图片就更可能是正确的。**
>

本项目检验这个假设**何时失效**。我们提出并量化 **Visual Self-Confirmation**：模型生成了一张与目标不符的图像，却在重新观察时按原始意图解释像素，因而认为自己画对了。但本项目的一级主张不是「这个现象存在」，而是它的**训练后果**：

> 在朴素 cycle / self-reward 训练中，internal cycle score 与 external verifiable correctness 会在某个可测量的训练步 **D*** 之后分叉；越过 D* 之后，继续优化自一致性会同时提高错误确认率（SCFR）。
>

为了让这个主张站得住，本项目先建立一套**能力地板受控**的测量协议：所有观察者必须先证明自己在受控原子问题上有 ≥80% 的准确率，所有提问必须去除 yes-bias，所有跨观察者比较必须包含一个**能力匹配**的异构模型，所有 SCFR 都按问题族的观察者准确率分层报告。这一步不是形式主义——它是把「模型确认了自己的意图」与「模型根本看不清并倾向说 yes」这两个解释区分开的唯一办法。

随后我们提出 **Render–Forget–Observe（RFO）**：所有自反馈必须经过真实 RGB 渲染并从磁盘重新读取（Render），在新上下文中移除原始意图、生成历史与来源标签（Forget），只通过中性原子问题从公共可见像素中提取事实（Observe），再用确定性规则比较意图事实与观察事实并只修正冲突属性（Compare and Repair）。RFO 的作用在本文中被定义为一个**可测量的量**：它应当把 D* 推后或消除，而不只是提高某个自评分数。

<aside>
🔧

**相对 Version 1.0 的三处结构性修改**

1. **主线从「现象」换成「训练后果」。** 原 Experiment 3 升为 E2 第一主实验，primary endpoint 从 SCFR 绝对值换成分叉点 D*。理由：SCFR 的绝对值永远可以被「这就是 prompt 泄漏」或「小模型看不清」解释掉，而两条曲线的**反向运动**不能。
2. **新增 Gate −1 能力地板层。** 在任何现象性结论之前，先证明测量工具本身有效。这一层同时消灭了 yes-bias 与能力不匹配两个致命替代解释。
3. **规模与范围收缩到单卡可完成。** Tier E、第三 backbone、大规模自然场景与人类评测全部下线或后置，算力从 1,080–2,160 压到 450–700。
</aside>

<aside>
⚡

**Version 2.1 的核心修改：从「事后诊断」升级为「提前预警」**

v2.0 的 D* 存在一个逻辑闭环问题：**要测出 D*，必须先拥有 external verifiable correctness；而一旦拥有它，直接用它做 early stopping 即可，D* 本身没有可操作价值。** 真正需要这个结论的人——正在跑 self-reward 的从业者 [5–8]——恰恰是因为没有 verifier 才使用自一致性的。

v2.1 因此把主线重心从「分叉存在」移到「**分叉可以被无标签地提前检测**」，并把论文重排为三步：

1. **① 提前预测。** 新增**梯度分歧信号**（§5.3）：定义 **g_naive** 为泄漏式自一致性所选样本上的更新方向，**g_rfo** 为隔离盲观察所选样本上的更新方向，监控 cos(g_naive, g_rfo)。该量**不需要任何外部标签**。Primary endpoint 改为 **Lead = D* − D_g**。
2. **② 机理归因。** 原 RQ4 上升为一级主张：用上下文消融把分叉分解为意图泄漏与共享感知盲点两个成分，并报告各自对 D_g / D* 的位移量。
3. **③ 延后分叉。** 原 RQ3（RFO 的价值）下移为第三步，作为前两步的自然推论——既然机理是泄漏，切断泄漏就应当推后分叉。

**D* 降级为 gold-standard 标尺**，只用于验证无标签信号的有效性，不再作为卖点。新增 **Gate −1b**：使用梯度余弦之前必须先报告它的噪声地板——与 Gate −1 对观察者的要求同源，先证明仪器有效，再用仪器下结论。

**为什么受控视觉场景是做这件事最好的实验台：**文本 overoptimization 文献里的 gold 是一个更大的 reward model，它自己就在漂 [18]；而 Tier A 的真值由 scene graph **构造**而来，是精确的。想到测梯度分歧并不难，但在文本里**测不准**。这是本项目相对该文献线的结构性优势，应写进 intro。

</aside>

---

# 2. 科学动机

## 2.1 闭环中最脆弱的一步是反馈，而不是生成

统一模型的长期价值在于闭环：生成一个视觉结果，重新理解它，发现错误，继续修正。这条链路支撑多模态 Agent、自训练、自动数据生成与视觉规划。但闭环的脆弱点不在生成质量，而在**反馈是否可信**。

设提示词为「两个蓝色方块位于一个红色球体左侧」，模型实际只画出一个方块。如果同一个模型随后回答「图中有两个蓝色方块」，系统面对的不是普通生成失败，而是**反馈通道失败**：错误不可被发现，并且在自训练中会被当作正样本强化。

## 2.2 为什么这不是一个评测问题，而是一个学习问题

GenEval、T2I-CompBench 等指标只回答「图像是否正确」。它们无法回答：

- 自一致性作为奖励，在什么条件下与外部正确性同向？
- 这种同向关系会不会随训练步数**失效**？
- 失效是否可以被提前检测（早停信号）？
- 什么样的反馈约束能延后失效？

这些是标准的「代理目标何时开始与真目标背离」问题（reward hacking / Goodhart），只是第一次被放在**视觉像素**这个可精确构造反事实的空间里研究。文本上的 self-bias 研究无法构造「固定像素、只改意图」这样的干预，这是本项目相对文本 LLM-as-a-judge 文献的结构性差异。[11–12]

## 2.3 与 The Telephone Game 的区别

The Telephone Game 研究多轮循环中的语义漂移，核心问题是信息经过多次转换后丢失多少。[9] 本项目关注更危险的相反情形：**信息在模型内部被恢复得很好，但最终像素并没有正确表达它**——而且这种情形会被自一致性奖励主动放大。

---

# 3. 主张阶梯（已重排）

| 层级 | 角色 | 可主张内容 | 所需证据 |
| --- | --- | --- | --- |
| D0 | **测量工具** | 观察者有能力看清；SCFR 在去偏后仍可测量 | Gate −1 全部四项通过 |
| D1 | 前置现象 | 固定像素下，原始意图会牵引模型的事实回答 | Pixel Override 上 POE 显著高于能力匹配观察者 |
| D2 | **锚点** | 朴素 cycle 训练存在分叉点 D*：内部一致性升、外部正确性不升、SCFR 升 | 3 seeds 下 D* 稳定存在，CI 不跨 0 |
| D3 | **一级主张 ①（预测）** | 分叉可被**无标签**提前预测：cos(g_naive, g_rfo) 的转折点 D_g 显著早于 D* | 3 seeds 下 Lead = D* − D_g 显著为正，且信号幅度超过 Gate −1b 噪声地板 |
| D4 | **一级主张 ②（机理）** | 分叉主要由意图泄漏驱动，共享感知盲点为次要成分 | 上下文消融对 D_g / D* 的位移量可分解 + 能力分层后的跨观察者矩阵 |
| D5 | **一级主张 ③（延后）** | RFO 把 D* 推后或消除，且在 compute-matched 下提高外部正确率 | 同预算下 external accuracy 提升 ≥3 pt 且 SCFR 下降 |
| D6 | appendix | 私有视觉通道（条件性） | PCG 显著为正且人类标签保持；否则报告为未被支持 |

<aside>
⚖️

**为什么这个顺序更安全**

D1 单独成立时，审稿人可以说「same-session 泄漏而已，理所当然」。D2 成立时，这个反驳失效：分叉现象与「泄漏是否理所当然」无关，它直接说明**一整类正在被广泛使用的训练信号**会在某个点开始造成损害。D2 也天然免疫「这只是 self-bias 的视觉版」——文本 self-bias 文献里没有对应的训练分叉结果。

**但 D2 单独仍然不够。** 「内部指标升、外部指标不升」这条曲线的形状在 reward overoptimization 文献中已是常识 [18]，审稿人完全可以说这只是换了个模态。**D3 才是不可替代的那一层**：现有 overoptimization 早停方法无一例外依赖 ground-truth queries [22]，而无标签替代（如 loss plateau）已被证明无法与正常收敛区分 [23]。如果 D3 成立，本文交付的就不是一个已知现象的复现，而是一个**在没有 verifier 的真实自训练场景里可以直接用的检测器**。

**D3 → D4 → D5 的顺序也不是随意排的：**先证明信号能预警（有用），再证明它为什么能预警（可信），最后证明按机理设计的干预确实延后了分叉（闭环）。倒过来讲——先讲方法再讲现象——会让 RFO 看起来像一个没有动机的工程 trick。

</aside>

---

# 4. 致命混淆与 Gate −1：能力地板与 yes-bias

## 4.1 替代解释

<aside>
🚨

**必须预先排除的解释**

模型不是「确认了自己的意图」，而是**根本看不清，并且倾向说 yes**；独立观察者用的是更强的异构 VLM，所以它更常说 no。

这一个解释能同时覆盖 SCFR、Observer Gap 与全部前置 RQ 的预期结果。1–1.5B 级统一模型在计数、空间关系、属性绑定上的准确率本来就接近「随机 + 偏置」基线，并且普遍存在强 yes-bias。**如果这一条不在方法一节被主动排除，论文在第一轮审稿就结束。**

</aside>

## 4.2 四道防线

### 防线 1：观察者能力地板

所有充当观察者的冻结模型，必须先在 Tier A 受控图（**外部程序验证为正确**的图）上完成原子问题体检：

- 每个问题族（存在、计数、颜色、大小、左右、属性绑定）分别报告准确率；
- **准入线 ≥80%，目标 ≥90%**；
- 未达线的问题族**整族排除**在该观察者的 SCFR 统计之外；
- 若主 backbone 在多数问题族达不到 80%，则不能用它自己充当观察者——此时改用同家族更大的冻结版本，并在论文中如实报告这一约束。

这一步同时给出了一个附带贡献：**一张「哪些统一模型在哪些视觉事实上根本没有观察能力」的能力地板表**，这本身对整个 self-reward 方向有直接使用价值。

### 防线 2：去 yes-bias 的提问设计

- 主指标的全部提问改为**开放式原子问题**（「图中有几个方块？」「方块是什么颜色？」）或**平衡强制选择**（选项集包含真值与等量干扰项，选项顺序对抗平衡）；
- 报告每个模型在每个问题族上的**答案分布先验**与 yes-rate；
- 对二元子集额外报告**偏置校正后的判别力**（log-odds 或 d'），而不是原始接受率；
- SCFR 只在**观察者委员会高置信一致**的样本上统计，记作 **SCFR@competent**。

### 防线 3：同能力观察者对照

独立观察者集合必须包含三类，缺一不可：

| 观察者类型 | 作用 | 匹配要求 |
| --- | --- | --- |
| 能力匹配异构模型 | **主对照**：分离「自我偏好」与「模型强弱」 | Tier A 原子问答总准确率与生成模型差 ≤3 pt |
| 更强异构模型 | 上界参考，不用于主结论 | 准确率显著更高，需明确标注 |
| 程序化 verifier | ground truth，只在受控层可用 | scene graph 直接给出真值 |

**Observer Gap 的主报告版本只允许使用能力匹配对照。** 使用更强模型得到的 gap 单独报告，并显式标注它混杂了能力差异。

### 防线 4：能力分层报告

把所有 SCFR / OG 结果按「该问题族上观察者委员会的准确率」分成三层（80–85%、85–92%、>92%）分别报告。核心检验是：

> gap 在**最高准确率层**依然显著存在。
>

如果 gap 随观察者准确率上升而单调消失，那么它就是能力假象，必须如实报告并放弃自我确认叙事。这一条是**可以真实失败的**，也正因如此它才有说服力。

## 4.3 Gate −1 通过判据

| 检验 | 通过条件 | 不通过时的动作 |
| --- | --- | --- |
| 观察者能力地板 | 至少 4 个问题族准确率 ≥80% | 换更大的冻结观察者；若仍不达线，项目改为「能力地板报告 + 分析论文」 |
| yes-bias 残留 | 平衡强制选择下 yes-rate 偏离 0.5 不超过 0.1 | 改用开放式提问并重新校准 |
| 能力匹配对照可得 | 找到一个准确率差 ≤3 pt 的异构模型 | 用同家族不同规模的模型构造匹配点，并降级主张强度 |
| 分层后 gap 存活 | 最高准确率层 gap ≥3 pt 且 CI 不跨 0 | 放弃 D1，直接以 D2 训练分叉为唯一主线 |

**Gate −1 的算力代价约 30–50 GPU-hours，占总预算不到 10%，但它决定整篇论文能否被相信。**

---

# 5. 问题形式化与指标

## 5.1 核心定义

**Visual Self-Confirming Failure**：对某张生成图像，(a) 程序化 verifier 或高置信观察者委员会判定它违反至少一个目标属性，且 (b) 生成模型自身在该属性上给出与原始意图一致、而与像素不一致的回答。

**意图泄漏（Intention Leakage）**：观察阶段仍可访问原始 prompt、同一会话历史、生成 latent / image tokens / KV cache，或来源标签。

**共享感知盲点（Shared Perceptual Blind Spot）**：即使在 fresh context + image-only 条件下，同模型或同家族观察者仍在相同视觉模式上共同失误。

## 5.2 指标

| 指标 | 定义 | 角色 |
| --- | --- | --- |
| **GDA-free** | cos(g_naive, g_rfo)：泄漏式自一致性与隔离盲观察两条更新方向的夹角余弦；**不需要任何外部标签** | **核心信号** |
| **D_g** | GDA-free 显著跌出其 Gate −1b 噪声地板并开始单调下降的最早训练步 | **Primary endpoint 分量** |
| **Lead = D* − D_g** | 无标签梯度信号相对指标分叉的领先步数；正值代表可提前预警 | **Primary endpoint** |
| GDA-gold | cos(g_self, g_gold)：自一致性所选样本与程序化 verifier 所选样本的更新方向夹角余弦；**需要标签，不可部署** | 机制证据（①→②的桥梁） |
| **D*（Divergence Point）** | internal cycle score 仍显著上升、而 external verifiable correctness 停止上升或开始下降的最早训练步 | **Co-primary / gold-standard 锚点** |
| **ΔSCFR@competent** | 从训练起点到 D* 之后固定步数，观察者高置信子集上错误确认率的变化 | **Co-primary** |
| SCFR@competent | 外部判错图像中模型仍判正确的比例，限定在能力达标问题族与高置信样本 | 主诊断量 |
| POE | 已知像素反事实上，回答跟随原意而非修改后像素的比例 | 因果证据 |
| OG-matched | 自身观察与**能力匹配**异构观察者在同图上的错误接受率差，按质量与能力分层 | 机制证据 |
| PSFP | 固定图像、只替换 prompt 时事实回答改变的概率 | 机制证据 |
| CSR | 收到原子反馈后目标错误被修复且正确属性未被破坏的比例 | 实用价值 |
| PCG | 公共变换前后同模型恢复下降幅度减去参照观察者下降幅度 | **仅 appendix** |

<aside>
📐

**预注册式声明**

主终点为 **Lead = D* − D_g**；次主终点为 **D*** 与 **ΔSCFR@competent**。全部在 Naive Cycle 与 RFO-PostTrain 两条训练曲线上、3 seeds、能力达标问题族上测量。所有其他指标为次要或诊断指标，不得在结果出来后替换主终点。**D* 与 D_g 的估计方式（分段回归断点 + profile likelihood + seed-level bootstrap CI）以及 GDA 的投影子空间、probe set 与 EMA 窗口，全部在实验开始前固定并写入代码**，实验开始后不得调整。

</aside>

## 5.3 梯度分歧信号：定义、无标签版本与噪声地板

### 5.3.1 为什么必须有这一层

D* 的定义中包含 external verifiable correctness，因此测量 D* 必须先有外部真值——而一旦有了外部真值，直接用它做 early stopping 即可。这使 D* 成为**事后诊断量**而非可用信号。现有 reward overoptimization 文献中所有早停方法都受同一约束（「we rely on some access to ground-truth queries」[22]）；而 training loss plateau 之类的无标签替代已被证明与正常收敛不可区分 [23]。本节给出一个不依赖外部标签的替代信号。

### 5.3.2 把「reward 的梯度」翻译到 rejection-sampling SFT

本项目主线不使用可微 reward，因此不存在显式的 reward gradient。我们采用如下等价定义：

> **某个 reward 的梯度 ≔ 该 reward 所选出的样本集上 SFT loss 的梯度。**
>

这正是模型实际收到的参数更新方向，比 policy gradient 更直接、方差更低。关键设计是：在**同一批 prompt、同一个候选池**上只改变选择准则，得到完美配对的两个梯度，二者唯一的差异来源就是「谁选的样本」。

### 5.3.3 三个梯度量

| 记号 | 定义 | 需要外部标签 | 角色 |
| --- | --- | --- | --- |
| g_gold | ∇θ L_SFT（程序化 verifier 选出的候选） | **是** | 机制参照，建立「梯度分歧确实对应错误方向」 |
| g_naive | ∇θ L_SFT（朴素 cycle score 选出的候选） | 否 | 被检测对象 |
| g_rfo | ∇θ L_SFT（隔离盲观察选出的候选） | 否 | **无标签对照方向** |

**GDA-gold = cos(g_naive, g_gold)** 用于建立机制并给出效应上界；**GDA-free = cos(g_naive, g_rfo)** 是实际交付的可部署检测器。核心桥梁命题是：**GDA-free 与 GDA-gold 在时间上高度相关**（Spearman ρ ≥0.7），因此前者可以在没有 gold 的场景里替代后者。

### 5.3.4 测量协议

1. **固定 probe set。** 每个 checkpoint 用完全相同的 prompt 子集（建议 200 条 Tier A held-out）与相同候选池计算梯度，消除采样噪声，使余弦变化只来自参数演化与选择准则差异。
2. **在 LoRA 子空间计算。** 全参数空间余弦会退化到接近 0 而无信息量；本项目使用 LoRA，可训练参数仅百万量级，正好适配。同时报告 per-block 余弦，定位分歧发生在哪些层。
3. **EMA 平滑。** 单 checkpoint 余弦方差较大，对 checkpoint 序列做指数滑动平均后再做断点估计。
4. **报告范数比。** 除夹角外同时记录 ‖g_naive‖ / ‖g_rfo‖，区分「方向分歧」与「幅度分歧」两种失效形态。

### 5.3.5 Gate −1b：先证明仪器有效，再用仪器下结论

与 Gate −1 对观察者的要求同源。在任何 GDA 结论之前，必须先建立**零效应基线**：

- 把同一 probe batch 随机对半分，计算两半**同一准则**梯度之间的余弦，得到纯采样噪声下的余弦分布；
- 报告该分布的均值与 95% 区间，作为**噪声地板**；
- GDA-free 只有在**持续跌出噪声地板下界**时才判定为真实分歧；
- 训练早期（前 10% steps）GDA-free 必须落在噪声地板内——若一开始就分歧，说明两个选择准则在任何时刻都不可比，信号无效。

### 5.3.6 检测器与训练器必须解耦

<aside>
⚠️

**这是审稿人一定会提的循环论证问题**

如果用 RFO-Self 训练、又用同一个 RFO-Self 构造 g_rfo 来检测，那么「检测器报警 → 换 RFO 有效」几乎是同义反复。

**硬性设计要求：**用于构造 g_rfo 的隔离观察配置，必须**不同于**用于训练的那一个。具体做法是从隔离观察者集合中留出一个**纯检测用**的配置（例如训练用 RFO-Self，检测用 RFO-Committee 中的能力匹配异构观察者），并在 §9.5 中显式记录这一划分。

**同时要说清楚非平凡内容在哪里：**g_naive 与 g_rfo 在任何时刻都不相同，这一点毫无信息量。真正的主张是**时间结构**——夹角在训练早期落在噪声地板内（即早期自一致性确实是安全的），随后单调扩大，且这一扩大**发生在外部正确率退化之前**。这个时间序列命题无法由「两者定义不同」推出。

</aside>

---

# 6. 研究问题与可证伪假设

<aside>
🧭

**RQ 顺序即论文顺序：预测 → 机理 → 延后**

RQ0 与 RQ1 是**前置**（证明工具可信、现象可测），RQ2 是**锚点**（提供 gold-standard 标尺），RQ3–RQ5 是三个一级主张，严格对应论文的三步叙事：

1. **RQ3 ①预测：** 不依赖最终结果，提前预警分叉；
2. **RQ4 ②机理：** 分叉从哪来；
3. **RQ5 ③延后：** 按机理设计的干预是否真的推后了分叉。

三者的依赖是单向的：没有 RQ2 的标尺就无法验证 RQ3 的信号；没有 RQ4 的归因，RQ5 的 RFO 就只是一个没有动机的工程 trick。

</aside>

## RQ0（前置）：测量工具是否有效？

**H-a（观察者）：** 存在至少 4 个问题族，使冻结观察者准确率 ≥80% 且平衡提问下无显著 yes-bias。

**H-b（梯度仪器）：** 同准则半批梯度余弦的噪声地板足够窄，使得跨准则余弦的变化可被区分（噪声地板 95% 区间宽度 **小于** 预期效应量的一半）。

**可证伪条件：** 多数问题族达不到 80%，或去偏后判别力接近随机；或梯度余弦噪声地板过宽以致任何跨准则差异都无法分辨。前者使项目转为能力地板分析论文，后者使 RQ3 退回熵类替代信号。

## RQ1（前置）：原始意图是否覆盖像素证据？

**H：** 固定像素、替换 prompt 时，模型事实回答显著被 prompt 牵引；POE 显著高于能力匹配观察者。

**可证伪条件：** 固定像素后回答基本不随 prompt 改变，或与能力匹配观察者无差异。

## RQ2（**锚点**）：自一致性奖励是否存在分叉点？

**H：** Naive Cycle 训练中存在可稳定估计的 D*，此后 internal cycle score 继续上升而 external correctness 停滞或下降，且 SCFR@competent 上升 ≥3 pt。

**可证伪条件：** 3 seeds 下两条曲线同步上升，断点回归找不到显著断点，或 SCFR 不升。此时主线改写为「朴素自一致性在本设置下是安全的」，并给出安全边界条件——这依然是一篇有价值的论文。

**定位说明：** RQ2 本身不是卖点。「代理指标升、真指标不升」的曲线形状在 reward overoptimization 文献中已是常识 [18]。RQ2 的唯一作用是**提供一把用程序化真值刻好的标尺**，让 RQ3 的无标签信号有东西可以对齐。论文中它应占 Figure 1 的背景层，而不是标题。

## RQ3（**一级 ①：预测**）：分叉能否在不使用外部标签的前提下被提前检测？

**H-1（领先性）：** GDA-free = cos(g_naive, g_rfo) 的转折点 D_g 显著早于 D*，即 **Lead = D* − D_g > 0**，3 seeds 下 paired bootstrap 95% CI 不跨 0。

**H-2（早期安全性）：** 训练早期 GDA-free 落在 Gate −1b 噪声地板内，随后单调下降——即分歧是**后天出现**的，而不是两个准则定义不同造成的常数偏移。

**H-3（可替代性）：** GDA-free 与需要标签的 GDA-gold 在时间序列上显著相关（Spearman ρ ≥0.7），从而支持在无 verifier 场景中用前者替代后者。

**H-4（可操作性）：** 以 D_g 作为早停规则，相比训练到预算耗尽，能在同等算力下取得更高的最终 external correctness。

**可证伪条件：** Lead 不显著为正（信号与分叉同时发生或滞后）；或 GDA-free 从第一步就跌出噪声地板（说明它测的是常数差异而非训练动态）；或它与 GDA-gold 不相关。此时降级为熵类替代信号，并如实报告梯度信号在本设置下无预警价值——这是一个真实可能发生的负结果。

## RQ4（**一级 ②：机理**）：分叉由什么驱动？

**H-1（主因是意图泄漏）：** 移除 Forget（保留 prompt）显著提前 D* 与 D_g，且效应量大于移除 hard render。

**H-2（次因是共享盲点）：** 同家族观察者比**能力匹配**异构观察者更容易接受错误，且该差异在最高能力分层中依然存在。

**H-3（可分解性）：** 两个成分对 D_g 位移量的贡献可加性分解，残差项小于总效应的 30%。

**H-4（层级定位）：** per-block 梯度余弦显示分歧优先出现在特定层段（预期为跨模态融合层），为机理提供内部证据。

**可证伪条件：** 上下文消融不改变 D* 与 D_g，且能力分层后跨观察者差异消失。此时无法归因，RQ5 的 RFO 失去理论动机，须降级为纯经验方法。

## RQ5（**一级 ③：延后**）：RFO 能否延后或消除分叉？

**H-1：** 在 compute-matched 条件下，RFO-PostTrain 的 D* 显著晚于 Naive Cycle（或在预算内不出现），且最终 external correctness 高 ≥3 pt。

**H-2（机理一致性）：** RFO 带来的 D* 位移量，应与 RQ4 中「移除 Forget」造成的反向位移量在同一量级。若两者数量级不符，说明 RFO 的收益来自机理之外的因素（例如更多候选），须如实报告。

**可证伪条件：** D* 位置无差异，或 external 提升只来自更多候选与更多推理算力。

## RQ6（appendix）：是否存在私有视觉通道？

**H：** 语义保持的公共变换选择性破坏同模型恢复，PCG 显著为正。

**可证伪条件：** 所有观察者同等受变换影响。默认预期为负结果，按负结果篇幅处理。

---

# 7. SelfSight-Bench v2：规模已按立项标准收缩

| 层 | v1.0 规模 | **v2.0 规模** | 作用 |
| --- | --- | --- | --- |
| Tier A 受控生成 | 4,000 prompts × 4 图 | **800 held-out prompts × 4 图** | 能力地板体检 + 主测量场 |
| Tier B 像素反事实 | 2,000 对 | **400 对** | RQ1 因果证据 |
| Tier C 自然场景 | 完整 GenEval / T2I-CompBench / MME-Unify | **300 prompts 抽样** | 外部有效性抽查，非贡献点 |
| Tier D 跨观察者矩阵 | 全量 | **Tier A/B 子集 600 图** | RQ4 机制 |
| Tier E 公共通道 | 完整层 | **appendix，200 图 × 3 变换** | 条件性负结果 |
| 人类评测 | 600–1,000 × 3 | **300–500 × 3，投稿前 6 周** | 只验证最终主结论 |

## 7.1 Tier A：受控自生成

程序化 scene graph 生成提示词，覆盖对象存在、数量、颜色属性、大小、左右/上下/前后、属性绑定、二至三项组合约束。模板、对象组合与属性组合在 train/dev/test 之间严格隔离。**每个问题族至少 100 个 test prompts**，以支撑能力分层统计。

## 7.2 Tier B：像素覆盖集（最重要的因果层）

1. 选择外部验证为正确的简单生成图；
2. 对一个原子属性做最小可控编辑（删除一个对象、蓝改绿、左移到右、交换属性、改变大小关系）；
3. 保持原始 prompt 不变；
4. 用开放式原子问题询问模型图中的实际事实。

400 对已足够在 4 类编辑上各获得约 100 对配对样本，配对 McNemar 检验在 5 pt 效应量下有充足功效。**真值完全由构造过程给出，不依赖任何 judge。**

## 7.3 提问库设计

每张图配一组固定的原子问题，每个问题有三种形式：开放式、平衡强制选择、二元（仅用于诊断 yes-bias，不进主指标）。同一问题在所有观察条件与所有观察者之间完全一致，保证唯一变化的是**观察者与其可见的上下文**。

## 7.4 观察条件矩阵

| 条件 | 原始 prompt | 同一会话 | 生成隐状态 | 来源标签 |
| --- | --- | --- | --- | --- |
| Same-session self-check | 有 | 有 | 可访问 | 可见 |
| Fresh-context prompted | 有 | 无 | 无 | 不可见 |
| Fresh-context image-only | 无 | 无 | 无 | 不可见 |
| Counterfactual prompt | 冲突 prompt | 无 | 无 | 不可见 |
| 能力匹配独立观察者 | 无 | 无 | 无 | 不可见 |

---

# 8. 方法：Render–Forget–Observe

RFO 不改变基础架构，只重新设计反馈通道。v2.0 相对 v1.0 增加了一条硬性前置条件：**观察者必须先通过 Gate −1 能力地板**，否则 RFO 只是在放大噪声。

## 8.1 Render：硬渲染屏障

图像 tokens / latent 必须真正 decode 为 RGB、写入标准图像文件、从磁盘重新读取、断开原生成计算图。禁止把生成 hidden state、semantic token 或 KV cache 传给观察器。

## 8.2 Forget：上下文隔离

观察器在独立进程与独立会话中运行：不提供原始 prompt、不提供生成历史、不说明图片来源、不复用 cache、不使用「是否满足这段 prompt」式诱导提问。

## 8.3 Observe：有能力的盲观察

- 观察器参数**冻结**，生成器可更新——避免两者共同演化出新的私有协议；
- **RFO-Self**：同基础模型的冻结副本，image-only fresh context；
- **RFO-Committee**：加入一个能力匹配异构冻结 VLM 与程序化 verifier，保守共识；
- 事实只有在观察者委员会高置信一致时才被接受，其余样本进入 abstain 池并单独报告比例。

## 8.4 Compare：原子证据比较

受控 prompt 的目标事实直接来自 scene graph；自然 prompt 通过结构化 parser 转为原子约束。比较器不输出模糊总分，只输出明确差异（目标：两个蓝色方块 / 观察：一个 / 错误类型：数量 / 其余属性：正确）。

## 8.5 Repair：三种部署形态

- **RFO-Rerank（免训练）**：每 prompt 生成 K 个候选，按 RFO 事实一致性择优；必须与 random best-of-K、CLIP/VQAScore rerank、same-context self-rerank 在相同 K 与相同算力下比较。
- **RFO-Repair（推理时）**：把原子差异转为最小修正指令，只重生成或编辑错误属性，报告目标修复率与 collateral damage。
- **RFO-PostTrain（主线学习方法）**：用 RFO 选出的候选与成功修复样本做 rejection-sampling SFT。**主结果不含 RL**；preference optimization 作为可选增强单独报告。

<aside>
🎯

**RFO 的训练原则**

只奖励能够通过硬渲染、上下文隔离与有能力的公共观察之后仍被确认的事实，而不是奖励模型对原始意图的内部恢复。

</aside>

---

# 9. 实验计划

## 9.1 模型

- **可训练主 backbone：** Show-o 1.3B 级统一模型，公开代码与训练路径，单卡 A100 可做 LoRA / 部分参数后训练。[1]
- **推理审计 backbone：** Janus-Pro-1B 级，只做推理，不训练。[3]
- **能力匹配异构观察者：** 一个非统一架构的通用 VLM，规模按 Gate −1 的准确率匹配结果选定。
- **更强观察者（上界参考）：** 一个明显更强的 VLM，只用于标注过的辅助分析。

## 9.2 实验组

| 组别 | 反馈方式 | 看到 prompt | hard render | 冻结观察器 |
| --- | --- | --- | --- | --- |
| Base | 无自反馈 | - | - | - |
| Same-Context Judge | 同会话总分判断 | 是 | 否 | 否 |
| **Naive Cycle** | prompt→image→恢复 prompt | 间接 | 是 | 否 |
| Fresh Blind | fresh context image-only | 否 | 是 | 是 |
| **RFO-Self** | 盲原子观察 | 否 | 是 | 是 |
| RFO-Committee | 能力匹配异构共识 | 否 | 是 | 是 |

主训练比较为 **Naive Cycle vs RFO-Self**，各 3 seeds × 1 配置，compute-matched。其余组为推理时或消融组。

## 9.3 E0：Gate −1 能力地板审计

所有候选观察者 × 6 个问题族 × Tier A 受控图，报告准确率、答案先验、yes-rate、去偏判别力。**这是项目的第一个实验，也是唯一一个必须在任何其他实验之前完成的实验。**

## 9.4 E1：测量工具层（前置现象）

在通过能力地板的问题族上，测量 5 种观察条件下的 SCFR@competent、POE、OG-matched、PSFP，并做三层能力分层报告。此实验建立 D1，同时给出 SCFR 的可信区间。

## 9.5 E2：训练动态分叉与梯度分歧监控（**第一主实验**）

Base / Naive Cycle / RFO-Self 三条训练曲线，固定评测检查点间隔，每个 checkpoint 全程记录：

1. internal cycle score（模型自己的一致性分数）；
2. external verifiable correctness（程序化 verifier）→ 用于估计 **D***；
3. SCFR@competent；
4. **GDA-free = cos(g_naive, g_rfo)** 及其范数比与 per-block 分解 → 用于估计 **D_g**；
5. **GDA-gold = cos(g_naive, g_gold)**（仅 Naive Cycle 分支，用于建立机制与效应上界）；
6. Gate −1b 噪声地板（同准则半批余弦），每个 checkpoint 重算，不复用初始值；
7. 观察答案熵与公共视图一致性（作为 RQ3 的**竞争基线信号**，须被 GDA-free 击败）。

### 9.5.1 梯度计算的固定配置（预注册）

- **Probe set：** 200 条 Tier A held-out prompts，全程固定不变；每 prompt 固定候选池 K 个，三种选择准则从**同一候选池**中挑选，保证配对；
- **投影子空间：** 仅 LoRA 参数，另报 per-block 余弦；
- **平滑：** checkpoint 序列 EMA，窗口预先固定；
- **成本：** 每 checkpoint 额外 3 次 backward（naive / rfo / gold 各一），相对一次完整评测可忽略。

### 9.5.2 检测器与训练器的解耦划分（硬性要求）

| 用途 | 使用的隔离观察配置 |
| --- | --- |
| 训练（RFO-PostTrain 的样本选择） | RFO-Self：同基础模型冻结副本，image-only fresh context |
| **检测（构造 g_rfo）** | **能力匹配异构冻结 VLM，全程不参与任何训练样本选择** |

该划分必须在论文方法一节显式声明，否则「检测器报警 → RFO 有效」构成循环论证（见 §5.3.6）。

### 9.5.3 主图

用预先固定的分段回归分别估计 **D*** 与 **D_g**，seed-level bootstrap 给出各自 CI 与 Lead 的 CI。**论文的 Figure 1 就是这张图：三条曲线叠加，两条竖线，中间的阴影区就是这篇论文的卖点。**

## 9.6 E3：compute-matched 价值验证

在相同 GPU-hours 与相同候选数 K 下比较 RFO-Rerank / RFO-Repair / RFO-PostTrain 与全部 rerank baseline，报告 external correctness、CSR、gain/cost 与 latency。

## 9.7 E4：机制与因果消融

去掉 hard render、保留 prompt（不 Forget）、只看原图不看公共视图、观察器与生成器共同更新、同家族 vs 能力匹配异构观察者、原子问题改整体打分、来源标签随机化。每一项都报告对 D* 与 SCFR 的影响。

## 9.8 E5：理解能力保持与外部有效性

检查后训练是否损害标准理解能力（重点关注计数、属性、空间关系），并在 Tier C 的 300 个自然 prompt 上抽查主结论方向是否一致。**此实验只求方向一致性，不求数值一致性。**

---

# 10. 基线、消融与关键对照

## 10.1 必需基线

- 无自反馈 Base；
- same-context self-judge；
- fresh-context self-judge；
- naive cycle consistency（主对手）；
- 能力匹配异构 VLM judge；
- CLIP / VQAScore rerank；
- random best-of-K；
- UniRL / UniCorn / ASG 风格自生成反馈的最接近可复现版本。[5–7]

## 10.2 消融表

| 消融项 | 验证的问题 | 预期方向 |
| --- | --- | --- |
| 去掉 hard render | 渲染屏障是否必要 | D* 显著提前 |
| 保留 prompt（不 Forget） | 意图泄漏的贡献 | SCFR 上升、D* 提前 |
| 只看原图（无公共视图） | 公共视图不变性的价值 | 小幅变差（预期效应最弱） |
| 观察器与生成器共同更新 | 冻结观察器是否必要 | internal score 升得更快、external 不升 |
| 同家族 vs 能力匹配异构观察者 | 共享盲点是否存在 | 同家族更容易接受错误 |
| 原子问题 → 整体打分 | 反馈粒度的作用 | CSR 下降、collateral damage 上升 |
| 二元提问 vs 开放式提问 | **yes-bias 的影响量** | 二元提问系统性高估 SCFR |
| 能力分层 vs 不分层 | **能力地板的影响量** | 不分层时 gap 被高估 |
| 不同 K 候选数 | 提升是否只来自算力 | RFO 在同 K 下仍领先 |
| **GDA-free vs 熵类无标签信号** | **梯度信号是否优于更简单的替代** | GDA-free 的 Lead 更大、跨 seed 方差更小 |
| **GDA 全参数空间 vs LoRA 子空间** | **投影是否必要** | 全参数余弦退化到噪声地板内，无信息量 |
| **检测器 = 训练器 vs 解耦** | **循环论证的影响量** | 不解耦时 Lead 被系统性高估 |
| **固定 probe set vs 每步重采样** | **采样噪声的影响量** | 重采样时噪声地板变宽，D_g 不可估 |

最后两行是 v2.0 新增的。它们不是为了完整性，而是为了把「审稿人会提出的致命替代解释」本身变成**论文里的一个定量结果**：我们不只声称控制了 yes-bias 与能力差异，而是报告它们分别能黏高多少个百分点。

## 10.3 关键对照

- **来源标签对照**：同一张图随机标注为「你生成的」或「其他模型生成的」，检验纯标签是否产生 self-preference；
- **固定像素、替换提示词**：正确 / 错误 / 无 prompt 三种条件；
- **固定提示词、替换像素**：原图 vs 已知反事实编辑图；
- **质量匹配对照**：只在外部质量匹配的图上比较不同观察者；
- **能力匹配对照**（v2.0 新增，最重要）：见 §4.2 防线 3。

---

# 11. 统计方案

- 主比较以 prompt 为配对单位，报告 paired bootstrap 95% CI；
- 二元错误使用 McNemar 检验；
- 使用 mixed-effects logistic regression，模型与条件为固定效应，prompt / template 为随机效应；
- **D* 的估计**：对 internal 与 external 两条曲线做分段线性回归，断点位置以 profile likelihood 估计，再用 seed-level bootstrap 给出 CI；若两条曲线断点 CI 重叠且跨 0，则判定无分叉；
- **D_g 的估计**：对 EMA 平滑后的 GDA-free 序列使用同一套分段回归流程；额外要求 D_g 处的余弦值跌出 Gate −1b 噪声地板 95% 下界；
- **Lead 的推断**：以 seed 为单位配对，直接对 (D* − D_g) 做 bootstrap，**不要**分别给 D* 与 D_g 的 CI 再目测是否重叠——那会严重低估功效；
- **GDA-free 与 GDA-gold 的一致性**：按 checkpoint 配对计算 Spearman ρ，并报告 seed 间的 ρ 分布而非合并值；
- 梯度余弦的所有报告必须**强制伴随同 checkpoint 的噪声地板区间**，不得单独出现（与 SCFR 必须伴随观察者准确率同理）；
- 训练主结果至少 3 seeds，所有曲线图绘出单 seed 而非只画均值；
- 不把 0.5–1 个百分点的无方差差异写成确定提升；
- 所有 SCFR / OG 结果**强制伴随观察者准确率与 yes-rate**一同报告，不得单独出现；
- 预先指定 **Lead = D* − D_g** 为主终点、D* 与 ΔSCFR@competent 为次主终点，避免事后选择最有利指标；
- 人类评测 300–500 样本 × 3 评审，报告 inter-rater agreement。

---

# 12. Go / No-Go 判据

## Gate −1：测量工具有效（新增、最高优先级）

完整判据见 §4.3。四项全部通过才能进入任何现象性实验。

**这道 gate 存在真实失败可能：** 1–1.5B 统一模型在计数与空间关系上很可能达不到 80%。如果真的达不到，那么本项目的正确产出就是一篇**能力地板报告**：“why self-reward cannot work at this scale”。这也是一个对社区有用的结论，不是项目失败。

## Gate −1b：梯度仪器有效（新增）

在任何 GDA 结论之前必须通过：

- 同准则半批梯度余弦的噪声地板已建立，95% 区间宽度小于预期效应量的一半；
- 训练前 10% steps 内，GDA-free 落在噪声地板区间内（证明分歧是后天出现的，而非常数偏移）；
- LoRA 子空间投影下余弦不退化到 0 附近；
- 检测器与训练器的解耦划分已在代码中固定（§9.5.2）。

**不通过时的动作：** 放弃梯度信号，RQ3 退回观察答案熵 / 公共视图一致性等替代信号，并在论文中如实报告「梯度层信号在此规模下不可测」。这会削弱主张 ① 但不影响 ②③。

## Gate 0：工程可行性

主 backbone 能在单张 A100 上完成稳定生成、理解与 LoRA 后训练；硬渲染管道与程序化 verifier 在 Tier A 上一致率 ≥98%；同一图重复观察的自一致率 ≥90%（测量噪声上限）。

## Gate 1：意图覆盖像素成立（D1）

Pixel Override 上，模型跟随原意而非像素的错误率比**能力匹配**观察者高 ≥5 pt，至少在 2 种属性类型上成立，且在最高能力层仍然存在。

不过则放弃 D1，直接以训练分叉为唯一主线（不阻塞项目）。

## Gate 2：分叉点存在（**生死实验**）

Naive Cycle 训练中：

- 分段回归在 3 seeds 中至少 2 个找到显著断点；
- 断点后 external verifiable correctness 斜率 ≤0，而 internal cycle score 斜率显著 >0；
- ΔSCFR@competent ≥3 pt 且 CI 不跨 0。

**不过时的动作：** 不强行挖掘。直接转写成**正面结论论文**：「在能力地板以上、上下文隔离的条件下，自一致性奖励在 X 步内是安全的」，并给出安全边界与适用条件。这与发现分叉同等重要，且使用完全相同的实验数据。

**注意：** Gate 2 现在的角色是**为 Gate 2b 提供标尺**。它通过与否决定的是「有没有东西可以预测」，而不是论文的卖点。

## Gate 2b：无标签信号具备领先性（**本版本新的生死实验**）

在 Naive Cycle 训练中：

- **Lead = D* − D_g** 在 3 seeds 配对 bootstrap 下 95% CI 不跨 0，且中位领先量 ≥ 总训练步数的 10%；
- D_g 处 GDA-free 已跌出 Gate −1b 噪声地板 95% 下界；
- 前 10% steps 的 GDA-free 落在噪声地板内（早期安全性成立）；
- GDA-free 与 GDA-gold 的 checkpoint 级 Spearman ρ ≥0.7；
- GDA-free 的 Lead 优于熵类基线信号。

**不过时的动作：** 主张 ① 降级为「我们检验了四类无标签早停信号，均无显著领先性」的严谨负结果，论文重心回落到 Gate 2 + Gate 3（分叉现象 + RFO）。这条退路使加入梯度信号成为**纯上行改动**：成功则论文换一个量级，失败则回到 v2.0 的形态，不损失任何东西。

## Gate 3：RFO 有实际价值

compute-matched 条件下：

- RFO 的 D* 显著晚于 Naive Cycle，或在预算内不出现；
- external compositional accuracy 提升 ≥3 pt；
- SCFR@competent 绝对下降 ≥5 pt 或相对下降 ≥30%；
- CSR 提高，标准理解能力无明显退化。

若提升只在更大 K 或更多算力下出现，则 RFO 降为「推理时可选工具」，不作为主方法主张。

## Gate 4：机制可归因（**升为一级**）

- 至少一项上下文消融显著提前 D* 与 D_g，且「移除 Forget」的效应量大于「移除 hard render」；
- 能力分层后同家族 vs 能力匹配异构观察者的差异依然存在；
- 两个成分对 D_g 位移的可加性分解残差 <30%。

**不过时的动作：** 无法归因时，RFO 失去理论动机，§8 须重写为纯经验方法，并在限制一节明写「我们能预警但不能解释」。主张 ①③ 不受影响。

## Gate 5：appendix 私有通道

只有 PCG 显著为正、跨模型重复、且人类标签保持率 ≥95% 时，才在 appendix 中作为条件性发现报告。摘要与标题不得使用。预期结果为负。

---

# 13. 算力与资源预算

以下为工程估算。主训练模型约 1–1.5B，BF16 + LoRA，以 A100 80GB 为参考。

| 阶段 | 内容 | A100 GPU-hours |
| --- | --- | --- |
| P0 | 统一推理接口、硬渲染管道、程序 verifier、**Gate −1 能力地板审计**、**Gate −1b 梯度噪声地板** | 35–55 |
| P1 | Tier A/B 构建 + E1 测量工具层（五种观察条件 × 多观察者） | 70–120 |
| P2 | **E2 训练动态分叉 + 梯度分歧监控**：Naive Cycle vs RFO-Self，各 3 seeds × 1 配置（梯度监控边际成本 5–15，已含在区间内） | 185–290 |
| P3 | E3 compute-matched 价值验证（rerank / repair / posttrain 与全部 baseline） | 60–100 |
| P4 | E4 机制消融 + Tier D 跨观察者矩阵 | 60–100 |
| P5 | E5 理解保持 + Tier C 抽查 + appendix Tier E + 补实验 | 40–70 |
| **总计** | ICLR 主线完成度 | **450–735（报 450–750）** |

<aside>
💰

**MVP：125–215 GPU-hours**

P0（35–55）+ P1（70–120）+ E2 短程试跑含梯度监控（20–40）。单张 A100 约 **5–9 天**即可回答三个决定性问题：观察者能不能看清（Gate −1）、梯度余弦有没有可用信噪比（Gate −1b）、以及**短程训练中 GDA-free 是否已经先于 external correctness 开始下降**（Gate 2 + Gate 2b 预告）。建议在没有看到这三个结果之前不投入剩余算力。

**梯度监控的边际成本极低**——每 checkpoint 额外 3 次 backward，全程约 5–15 GPU-hours——因此它不挤占任何现有实验，纯属上行改动。

</aside>

### 存储与配套

- 生成图像与公共视图：约 100–200GB（规模收缩后）；
- checkpoints / 优化器 / 日志：约 150–300GB；
- 模型权重与缓存：约 100–150GB；
- **建议总存储：400–650GB**；
- 人类评测：300–500 样本 × 3 评审（投稿前 6 周）。

---

# 14. 执行时间线

| 时间 | 里程碑 | Gate |
| --- | --- | --- |
| 2026.09–10 | 统一推理接口、硬渲染管道、scene graph 与程序 verifier、提问库三种形式 | Gate 0 |
| 2026.11 | **能力地板审计**：全部候选观察者 × 6 问题族；选定能力匹配对照；**建立梯度余弦噪声地板与检测器解耦划分** | **Gate −1 / −1b** |
| 2026.12 | Tier A/B 冻结；E1 五种观察条件 + 能力分层报告 | Gate 1 |
| 2027.01–02 | **E2 训练动态分叉主实验**，3 seeds，D* 与 D_g 估计、Lead 推断与 Figure 1 | **Gate 2 / 2b** |
| 2027.03–04 | E3 compute-matched 价值验证与全部 rerank baseline | Gate 3 |
| 2027.05–06 | E4 机制消融、Tier D 矩阵、第二 backbone 推理审计 | Gate 4 |
| 2027.07 | E5、Tier C 抽查、appendix Tier E、人类评测、failure cases | Gate 5 |
| 2027.08–09 | 写作、内部审稿、补实验、复现包 | - |

ICLR 2028 官方时间尚未公布，以上按 2027 年下半年完成投稿准备倒排。

---

# 15. 审稿风险与防御

| 风险 | 审稿人的原话 | 防御 |
| --- | --- | --- |
| **1. 能力地板（最致命）** | 「模型不是确认意图，它只是看不清并且倾向说 yes」 | Gate −1 四道防线；能力分层报告；并把「二元 vs 开放式」与「分层 vs 不分层」作为消融行定量报告 |
| 2. 现象理所当然 | 「same-session 看得到 prompt，当然会跟随意图」 | 一级主张不是现象而是**训练分叉**；分叉发生在 fresh-context 评测上，与泄漏是否理所当然无关 |
| 3. self-bias 的视觉版 | 「文本 LLM-judge 已经知道这一点」 | 文本文献无法固定像素只改意图，也没有对应的**训练分叉点**结果；且本文给出早停信号 |
| 4. 玩具世界 | 「只在方块与球体上成立」 | 受控层用于因果识别（真值无需 judge），Tier C 与人类评测用于方向抽查；并在限制一节明写这一边界 |
| 5. 单 backbone 泛化性 | 「只训了一个 1.3B 模型」 | 预先声明这是计算约束下的选择；第二 backbone 做推理审计以验证现象层泛化；不对更大规模做定量外推 |
| 6. RFO = 多模型投票 | 「这就是 rejection sampling 加集成」 | 主版 RFO-Self 只用同模型冻结副本，不引入额外能力；所有 rerank baseline 在相同 K 与相同算力下比较 |
| 7. 分叉不存在 | 「你的两条曲线同步上升」 | Gate 2 预先定义了这种结果下的论文形态（安全边界论文），不依赖阳性结果 |
| **9. 检测器循环论证（新增风险）** | 「你用 RFO 做检测器，又用 RFO 做修法，当然自圆其说」 | 检测器与训练器强制解耦（§9.5.2）；且主张是**时序结构**（早期落在噪声地板内、后期扩大、且早于外部退化），无法由定义差异推出；并把「不解耦时 Lead 被高估多少」作为消融行定量报告 |
| **10. 梯度检测已有人做（新增风险）** | 「GRIFT / PRIME / gradient regularization 已经做过了」 | 那些工作全部在文本 / RLVR / coding [19–21]，且多数仍需 gold 或 hack 标签；本文的 GDA-free **零标签**，且只有在真值可构造的受控视觉空间里才能被精确标定 |
| 8. 后训练损害理解能力 | 「你把统一模型训坏了」 | E5 报告理解侧 Pareto；混合理解 replay；冻结部分参数 |

---

# 16. 论文叙事与关键图表

| 图 | 内容 | 数据来源 |
| --- | --- | --- |
| **Figure 1** | **预警图**：internal cycle score、external verifiable correctness 与 GDA-free 三条曲线随训练步数叠加；两条竖线标出 **D_g** 与 **D***，中间阴影区为 **Lead**；下方平行轴为 SCFR@competent。**阴影区就是这篇论文的卖点** | E2 |
| **Figure 1b** | **梯度仪器有效性**：Gate −1b 噪声地板带 + GDA-free / GDA-gold 双曲线 + per-block 余弦热力图 | E2 |
| Figure 2 | **能力地板热力图**：观察者 × 问题族准确率，标出 80% 准入线 | E0 |
| Figure 3 | RFO 流程：hard render barrier / context isolation / 有能力的公共观察 / 原子比较 / 定向修复 | 示意图 |
| Figure 4 | 像素覆盖因果图：固定像素改 prompt vs 固定 prompt 改像素 | E1 |
| Figure 5 | 跨观察者矩阵，按能力分层分面板 | E4 |
| Figure 6 | appendix：公共变换下的恢复下降（预期为负结果） | Tier E |

<aside>
📊

**Figure 1 必须来自真实短程数据**

在 MVP 阶段（P0+P1+E2 短程）就必须能画出这张图的雏形。如果短程训练中两条曲线完全重叠且无任何分开迹象，应立即进入 Gate 2 的备选叙事，而不是先花 200 GPU-hours 把它跑完。

</aside>

---

# 17. 面向审稿人的创新定位

本项目的 novelty 不建立在新 loss 上，而建立在四点：

1. **新的可部署工具（最重要）**：一个**不需要任何外部标签**的分叉预警信号。现有 reward overoptimization 早停方法无一例外依赖 ground-truth queries [22]，而无标签替代（loss plateau 等）已被证明与正常收敛不可区分 [23]。梯度层检测在文本 / RLVR 上刚刚兴起 [19–21]，在统一多模态模型上是空白；
2. **更干净的实验台**：文本 overoptimization 文献里的 gold 是一个更大的 reward model，它自己就在漂 [18]；本项目 Tier A 的真值由 scene graph **构造**而来，是精确的。想到测梯度分歧不难，但只有在这里才**测得准**；
3. **新的测量纪律**：能力地板、去 yes-bias、能力匹配对照、能力分层，加上梯度信号自己的噪声地板与检测器解耦——把「小模型看不清」和「循环论证」两类平凡解释从根上排除；
4. **新的反馈原则**：自我评价只在穿过硬渲染屏障、完成上下文隔离，并能被**有能力的**公共观察者稳定确认时才可信；且该原则同时充当**修法与量具**——RFO 的隔离观察方向本身就是构造无标签检测器所需的对照。

> 现有工作主要问「如何利用自一致性提高模型」；本项目问的是「自一致性在何时停止值得被信任，以及怎样提前发现这一刻」。
>

这也把本项目接入了一条比多模态更大的文献线：reward overoptimization 与 reward hacking。[16–18] 本项目的贡献是把它第一次放到一个**真值可以被程序化构造**的视觉空间里。

---

# 18. 预期审稿问题

## Q1：你怎么知道不是模型太小看不清？

四层防御：观察者能力地板（≥80%）、去 yes-bias 提问、能力匹配对照、能力分层报告。而且主结论是**同一个模型在训练过程中的时间序列变化**，能力地板在各个检查点上是公共因子，无法解释两条曲线的**反向**运动。

## Q2：为什么不用更强的观察者就好？

更强观察者只作为上界参考。若用它做 Observer Gap 主报告，测到的是模型强弱而不是自我偏好——这正是本版本修正的主要问题之一。

## Q3：RFO 是否只是多模型投票或 rejection sampling？

主版 RFO-Self 只用同模型冻结副本，不引入额外能力。贡献在于反馈信息的**约束**：hard render、blind observation、公共视图不变性、原子比较。全部 baseline 在相同 K 与相同算力下比较。

## Q4：为什么需要训练，直接 rerank 不就够了？

因为一级主张本身就是一个**训练现象**（分叉点）。不训练就无法观测 D*；rerank 只是验证反馈质量的辅助实验。

## Q5：GDA-free 既由 RFO 构造、RFO 又是你的修法，这不是循环论证吗？

两点回应。第一，**检测器与训练器在设计上强制解耦**：训练用 RFO-Self，构造 g_rfo 用一个全程不参与训练样本选择的能力匹配异构冻结观察者（§9.5.2）。第二，非平凡内容不在「两个梯度不同」——它们在任何时刻都不同，这毫无信息量——而在**时间结构**：夹角在训练早期落在噪声地板内，随后单调扩大，且这一扩大发生在外部正确率退化**之前**。这个时序命题无法由定义差异推出，并且可以真实失败（Gate 2b）。

## Q6：分叉会不会只是过拟合？

外部正确率在 held-out prompts 上测量，且模板与属性组合严格隔离。如果只是普通过拟合，internal cycle score 应当同时在 held-out 上下降；我们预期的是它继续上升。这两种情形在图上可直接区分。

---

# 19. 计划贡献

最终论文只在实验支持时主张以下内容，顺序即叙事顺序：

1. **【① 预测】**给出一个**无需外部标签**的分叉预警信号 GDA-free，并证明它显著领先于外部指标的退化（Lead > 0），可直接作为真实自训练系统的早停规则；
2. **【② 机理】**把分叉归因到意图泄漏为主、共享感知盲点为辅，并给出可加性分解与层级定位；
3. **【③ 延后】**提出 Render–Forget–Observe 原则与后训练流程，展示它能将分叉点推后或消除，且位移量与机理归因在同一量级；
4. 首次给出统一多模态模型中**自一致性奖励分叉点**的可重复测量（作为①的 gold-standard 标尺）；
5. 提出一套**能力地板受控**的自反馈测量协议（含能力地板表、去偏提问库、能力匹配对照、梯度噪声地板）；
6. 提出 SelfSight-Bench v2，用像素反事实与跨观察者控制区分反馈失败来源；
7. （appendix）关于私有视觉通道的条件性证据或严格负结果。

---

# 20. 最终摘要草稿

统一多模态模型越来越多地用自身的理解能力评价并改进图像生成，即把「模型能从自己的图中读回原意」当作图像正确的证据。本文系统检验这一奖励信号**何时失效**。

我们首先指出，此类结论存在一个平凡而致命的替代解释：小规模统一模型在计数、空间关系与属性绑定上本来就接近随机，并带有强 yes-bias。为此我们提出一套**能力地板受控**的测量协议：观察者必须先在程序验证的受控图上达到准确率阈值，所有提问去除 yes-bias，所有跨观察者比较使用能力匹配的异构模型，所有结果按能力分层报告。

在这套协议下，我们首先建立一把标尺：**分叉点 D***——内部循环一致性仍在上升、而外部可验证正确率停止上升的最早训练步。但 D* 的测量需要外部真值，而真正需要这个结论的从业者恰恰没有真值可用。因此本文的核心交付是一个**不依赖任何外部标签的预警信号**：把「某个 reward 的梯度」定义为该 reward 所选样本上 SFT loss 的梯度，在同一候选池上只改变选择准则，监控泄漏式自一致性与隔离盲观察两条更新方向的夹角。我们证明该夹角的转折点 **D_g 显著早于 D***，从而给出可操作的早停规则。

随后我们把分叉归因到意图泄漏，并评估一个简单的反馈原则 **Render–Forget–Observe** 能否将其推后：强制自我评价穿过真实 RGB 瓶颈，移除对原始意图的访问，并只接受有能力的公共观察者能稳定确认的原子事实。RFO 在本文中同时扮演两个角色——它既是修法，其隔离观察方向也正是构造上述无标签检测器所需的对照。

本文并不否定自一致性，而是主张：**自一致性只在模型仍然允许自己被实际渲染结果“惊讶”的那段区间内值得信任，而这段区间的绝对置是可以被测量的。**

---

# 21. 最小实验清单

- [ ]  主 backbone 统一推理接口 + 硬渲染管道完成（Gate 0）
- [ ]  程序化 verifier 与人工校核一致率 ≥98%
- [ ]  **能力地板审计完成，至少 4 个问题族 ≥80%（Gate −1）**
- [ ]  **平衡强制选择提问库完成，yes-rate 偏离 ≤0.1**
- [ ]  **能力匹配异构观察者选定（准确率差 ≤3 pt）**
- [ ]  **梯度余弦噪声地板建立，区间宽度 < 预期效应量一半（Gate −1b）**
- [ ]  **检测器与训练器解耦划分写入代码并固定**
- [ ]  **固定 probe set（200 条 held-out）与 LoRA 投影配置冻结**
- [ ]  Tier A 800 prompts / Tier B 400 对冻结
- [ ]  E1 五种观察条件 + 三层能力分层报告完成
- [ ]  **E2 Naive Cycle vs RFO-Self 各 3 seeds，D* 与 D_g 估计与 CI 完成**
- [ ]  **Lead = D* − D_g 的配对 bootstrap CI 完成（Gate 2b）**
- [ ]  **GDA-free 与 GDA-gold 的 checkpoint 级 Spearman ρ 报告完成**
- [ ]  **GDA-free 对熵类基线信号的优势验证完成**
- [ ]  E3 compute-matched rerank / repair / posttrain 比较完成
- [ ]  E4 上下文消融 + Tier D 矩阵完成
- [ ]  E5 理解能力保持 + Tier C 方向抽查完成
- [ ]  人类评测 300–500 样本完成，报告 agreement
- [ ]  appendix Tier E 完成并如实报告（预期为负）
- [ ]  主要结论逐条通过 D0–D5 阶梯审查
- [ ]  未支持的结论从标题与摘要中移除
- [ ]  代码、prompt、图像哈希、评测脚本、随机种子发布

---

# 22. 参考文献

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

[15] *EvoTok: A Unified Image Tokenizer via Residual Latent Evolution for Visual Understanding and Generation*. arXiv:2603.12108, 2026.

[16] Amodei et al. *Concrete Problems in AI Safety*. arXiv:1606.06565, 2016.

[17] Skalse et al. *Defining and Characterizing Reward Hacking*. arXiv:2209.13085, 2022.

[18] Gao et al. *Scaling Laws for Reward Model Overoptimization*. arXiv:2210.10760, 2022.

[19] Ackermann et al. *Gradient Regularization Prevents Reward Hacking in Reinforcement Learning from Human Feedback and Verifiable Rewards*. arXiv:2602.18037, 2026.

[20] *Detecting and Suppressing Reward Hacking with Gradient Fingerprints (GRIFT)*. arXiv:2604.16242, 2026.

[21] *Proxy Reward Internalization and Mechanistic Exploitation (PRIME): A Learned Precursor to Reward Hacking and Its Generalization*. arXiv:2606.09711, 2026.

[22] Moskovitz et al. *Confronting Reward Model Overoptimization with Constrained RLHF*. arXiv:2310.04373, 2023.

[23] *EvalStop: Using World Feedback to Detect and Correct Reward Overoptimization*. arXiv:2606.04145, 2026.
