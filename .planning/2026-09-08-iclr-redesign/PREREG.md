# 预注册:Blind Self-Verification(2026-09-08)

**本节写于 `runs/v4/decoupling-main-20260908` 产出任何一行
`checkpoint_metrics.csv` 之前,也写于 arm B 生成任何一张图之前。此后不改。
任何偏离以「偏离」的名义追加,不得改写本文件。**

按项目的缺陷登记流程:**§25 / §27 / §34 原样保留,一个字不改。**
本文件说明它们被取代的范围和依据。

---

## 0. 写作时我已经看过什么(利益冲突声明)

预注册只有在「还没看到结果」时才值钱,所以先说清哪些已经看过:

**已看过,因此本文件里的对应条目是描述性的、不是预注册的:**
- `review-packets/context-ablation-20260908/`:上下文消融(盲 0.654 / 告知 0.336)。
- `review-packets/bsv-selection-20260908/`:离线选择重放(+0.106,以及两个 corpus
  之间的异质性)。
- `review-packets/dg-power-20260908/`:Lead 分辨率、D_g 估计量的功效。
- pilot(`runs/v4/decoupling-pilot-20260906`)的弃权率与 s_select gap 塌陷。

**尚未看过,因此下面 E3 / E4 是真预注册:**
- 主运行任何 checkpoint 的外部正确率、内部曲线、s_select 逐点数值。
- arm B 的任何输出(尚未训练)。
- Show-o v1 / Show-o2-7B 上的任何复制结果。

---

## 1. 取代了什么,依据是什么

| 冻结条目 | 内容 | 取代范围 | 依据 |
|---|---|---|---|
| §25 | §24(b) 服从分界的确证检验 | **不取代**。已由 §26 / §35 判定为不确证,§24(b) 永久降为探索性。本文件不碰它 | — |
| §27 | 第三批语料按实测效应量重新定 n | **不取代**。同上,已执行完毕并已判定 | — |
| §34 | L3 预告的停止规则(围绕 D\*) | **取代其终态定义**:论文的主张不再依赖 D\* 或 Lead | 见下 |

**§34 被取代的具体理由**(全部可复算,见 `review-packets/dg-power-20260908/`):

1. `Lead = D* − D_g` 是**两个搜索出来的拐点之差**,SE 16–18 步,而整个训练跨度
   只有 88 步。要让它显著,真实 Lead 得大于约 35 步。
2. 这个 SE **不能靠加 checkpoint 救**:12→24→48 个 checkpoint,拐点 SD 只从
   11.4 走到 11.0 再到 7.2。噪声在曲线本身,不在采样密度。
3. 已注册的 `paired_bootstrap_lead` 是**反保守的**(实测 type-I 9.5%),
   而它报的「95% CI」实际上只是 3 个 seed 的 lead 极差。
4. D_g 的出厂估计量在真实数据上**从未跑通过**:缺 `gda_*` / `noise_*` 列、
   缺 `seed` 列、缺 `gradient_probe.ema_alpha`。原先记录的 0/2000 误报率是**模拟**的。

**结论**:拐点类估计量(searched knot,SD 11.3 步)与比例类估计量
(SEM 0.0150)差一个数量级。论文改用后者。
D\* / D_g / Lead **降为附录里的仪器负结果**,原始记录保留,不删。

---

## 2. 论文主张(一句话)

**在统一模型里,验证者与生成者共享权重,而且验证时生成 prompt 还留在上下文里,
所以它不是独立验证者,是一个 prompt 条件下的确认器。把 prompt 移出上下文
(BSV)是零训练成本的修复。**

---

## E3. 主实验:三臂协同进化(预注册)

### 设计

| arm | 选择信号 | 来源 |
|---|---|---|
| A `naive` | prompt 在上下文里 | `runs/v4/decoupling-main-20260908`(在跑) |
| B `blind_self` | prompt 移出上下文 | **待跑** |
| C `rfo_gold` | 外部 gold | 同 A 的运行(在跑) |

**arm B 的运行方式已定,写在这里以免事后可选:**
新建输出目录,用**同一份 config**(`configs/v4_decoupling_main_20260908.yaml`),
跑 `--arms naive blind_self`,即 **B 与一列新的 A 在同一次运行里配对**。

理由(pilot 实测,`rounds/round-*/selection.json`):`naive` 12/12 从不弃权,
`rfo_gold` 只选 7–9/12。`pair_decisions` 取交集,所以**主运行的 A 列实际只在
约 75% 的 prompt 上训练**。若 B 单臂跑,它会在 100% 上训练,A–B 对比就带上
prompt 集混淆——而 A vs B 正是本文的主张所在。
split 会自动一致:`split_prompts` 是 (prompt_ids, outcome, probe, seed) 的纯函数,
同 config 必然复现同一份划分,digest 校验会确认。

主运行里的 A 列因此成为**独立复制**,不是唯一比较对象。

### 端点 1(主):最终外部正确率

**假设(方向已定,单侧)**:B > A。

**统计**:同一批 outcome prompt、同一批固定 latent,逐 (prompt, draw) 配对。
64 outcome prompt × R=4 draw = 每 checkpoint 每臂 256 张。
配对比例差,bootstrap over prompt(不是 over image——同 prompt 的 4 draw 不独立),
20000 次重采样,报 95% CI 与配对精确 McNemar。

**功效**:已实测配对比例差的 SEM ≈ 0.0150(`review-packets/dg-power-20260908/lead_resolution.py`)。
80% 功效可测到约 **+0.042** 的差。**这个数字在看结果之前写下**:
若 B−A 的点估计小于 0.042 且 CI 含 0,判为**未测到**,不得改口径抢救。

### 端点 2(主):协同——逐 checkpoint 的盲测判别 gap

对每个 checkpoint,在**盲**条件下计算 s_select 对
「实际正确」与「实际错误」候选的判别 gap(即
`review-packets/dg-power-20260908/prompt_recital.py` 的口径,但去掉 preamble)。

**假设**:A 的 gap 随训练塌向 0(pilot 已观测:+0.047 → 0.000,32 步内,两臂皆然),
B 的 gap 不塌。

**判定**:对每个臂做 gap 对 step 的 OLS 斜率,配对 bootstrap over prompt。
**确证条件**:A 的斜率显著为负,且 B 的斜率显著大于 A 的斜率。
只有 A 塌而 B 也塌 → 判为「BSV 不能阻止感知被吃掉」,如实写。

### 端点 3(次要,预注册):剂量-反应

离线重放发现两个 corpus 的选择增益差 5 倍(+0.190 vs +0.034),
而**来源在确认偏差本身的强度**:冲突试次上的上下文效应
−0.440 vs −0.214,盲侧几乎不动(0.638 / 0.667)。

**预注册的预测**:把主运行的每个 (arm, checkpoint) 当一个点
(13 checkpoint × 2 臂 = 26 点),横轴 = 该 checkpoint 的上下文效应强度
(冲突试次上 blind 减 prompted 的准确率),纵轴 = 该 checkpoint 的选择增益
(盲选相对带 prompt 选的外部正确率之差)。
**预测斜率 > 0**,cluster bootstrap over checkpoint。

若斜率不显著为正:异质性在论文里写成**未解释的调节效应**,不得说成剂量-反应。

### 预先声明的失败条件(任一成立即降级,不抢救)

1. 端点 1 未测到 → 论文不得声称「BSV 提升生成」,只能声称
   「BSV 提升选择信号质量」(离线重放已支持),主张收缩到推理期。
2. 端点 2 里 B 也塌 → §2 的机制解释被证伪,必须在正文写明。
3. arm B 的 `blind()` 守卫抛错 → 说明题目里混进了 prompt,该轮作废重跑,
   **不得**关掉守卫。
4. 墙钟超过 60 h,或同一阶段连续两次不可恢复失败 → 停,登记 traceback。

**不构成停止理由**:结果是 null;曲线难看;别人的作业占卡(等就行)。

**无论结果如何都不做**:不加 seed / 不加算力 / 不换 backbone 去抢救;
单 seed 轨迹不写成显著性结论;不回头改本文件。

---

## E4. 跨规模复制(预注册,纯推理)

在 **Show-o v1** 与 **Show-o2-7B** 上复制 §1 的上下文消融。

**这两个模型是冻结负对照**(Hard Stop)。本条**不**把它们提回主线:
它们不参与任何训练、不参与 A/B/C 任何比较、不进主结果表。
它们只回答一个问题:**「共享上下文的验证者不是独立验证者」这条是不是
只在 Show-o2-1.5B 上成立。** 论文中必须标注为负对照模型上的推理期复制。

**做法**:同一批图(`runs/v4/main-*/images`)、同一批题、同两个条件。
不重新生成、不下载新权重(两者本地已有;若本地没有则本条不做,不申请下载)。

**判定**:`image_differs_from_spec` 上 prompted 低于 image_only,单侧。
三个模型里至少两个复制成功 → §6 的泛化主张成立;
只有 1.5B 有 → 写成「在我们测到的最小模型上出现,更大模型上没有」,
这同样是有价值的结果,但主张必须收缩。

---

## 3. 分析代码与冻结

- 离线重放:`review-packets/bsv-selection-20260908/bsv_selection.py`(已冻结,
  E3 端点 3 复用其 `spec_agreement` 口径)。
- 平局处理:均匀随机打破的期望值为主,首索引为稳健性对照。**两者都报。**
- 所有 bootstrap 的种子固定为 `20260908`。

## 4. 已知会被审稿人打的点(先写下来,不当作发现)

1. **题目是 detections 派生的**,离线重放的绝对水平偏乐观。
   arm B 用的是 spec-only 题(`v4/probe.py::spec_questions`),不带这个问题;
   两者不可直接比绝对值,论文中分开列。
2. **单 backbone、单 seed**。E4 处理规模,seed 不处理——
   项目的算力预算里没有第二个 seed,这条如实写进 limitations。
3. **BSV 不是新架构**,是把一段文本从上下文里删掉。
   论文的贡献主张必须是「测量 + 机制 + 后果」,不是「新模块」。

---

## 偏离 1(2026-09-08,同日追加,不改写上文)

**E4 写的「本地没有权重」是错的。** 我当时只查了 HF cache
(`cache/huggingface` 与 `~/.cache/huggingface`),没查 `SELFSIGHT_MODEL_ROOT`。
实际全部在 `H:\selfsight-models`:

```
show-o2-7B                  17 G   revision 3012b1d6  <- 与 configs/backbones/showo2_7b.yaml 对得上
show-o-512x512             5.4 G
show-o-w-clip-vit-512x512  5.6 G
magvitv2                   365 M
deepseek-ai/Janus-Pro-1B   3.9 G   <- 另一个家族的统一模型
```

7B 的三个依赖(Wan2.1 VAE、SigLIP-so400m、Qwen2.5-7B-Instruct)也都在。

**因此 E4 不需要任何下载,原文里「若本地没有则本条不做」这句作废。**
`Showo2Adapter` 本来就由 `backbone_config` 参数化,而且里面已经有一段
`_ensure_legacy_shard_index`,注释直说是为 Show-o2 7B 的分片布局写的
——这条路以前走通过。

**修订后的 E4 成本**:
- **Show-o2-7B**:`observe --backbone-config configs/backbones/showo2_7b.yaml`,
  纯推理。已实现(worktree 提交 279df47),零新增权重、零新增适配器。
- **Show-o v1**:权重和 magvitv2 都在,但适配器在 8f155ae
  (「Collapse to v3」)里被删了。要从那个提交的父节点恢复。工作量真实存在,
  但不是下载问题。
- **Janus-Pro-1B**:另一个家族,能挡住「这是 Show-o 家族的怪癖」这句审稿意见。
  没有适配器,需要新写。

判定标准不变(三个模型里至少两个复制成功)。
「负对照模型、必须标注」这条约束也不变。
