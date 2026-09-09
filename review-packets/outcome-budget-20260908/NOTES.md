# 「240 题 × 24 checkpoint」这个配置不存在,以及能跑的那个是什么

2026-09-08。起草运行配置时先查了三件事,三件都推翻了那个提法的一半。
产出:`configs/v4_decoupling_main_20260908.yaml`。

## 1. 两个数字都被语料卡住,不是被预算卡住

**240 个 outcome prompt:不可能。** 语料一共 **228 个 spec**。
`split_prompts` 在 `outcome + probe >= 228` 时直接 raise,而冻结的划分已经
花掉 64 + 32,剩 132 个训练 prompt。

检出力表里的 `outcome_n` 数的是**被裁定的图**,不是 prompt。**这是我上一条消息里
写错的地方**——我写成了「240 个 prompt」。240 张图在这个语料上只能靠
「每个 prompt 画多张」拿到。

**24 个 checkpoint:不可能。** `build_schedule` 强制 `max_epochs=1`,
即 `rounds × prompts_per_round ≤ 132`。每轮 12 个 prompt 就是 **11 轮 = 12 个 checkpoint**
(含未训练基线)。要到 24,只有两条路:

- 把 `prompts_per_round` 砍到 6 → 22 轮。但每轮 8 次 optimizer step × 8 次梯度累积
  = 64 个 microbatch 摊到 6 张图上,是每张图重复十遍。
  `v4_l3_preview.yaml` 的注释已经点名这件事:**那训练的是记住六张图,
  正是 D\* 要排除的混淆。**
- 写更多 spec。

**加多少 wall-clock 都不解决。这不是预算决定。**

## 2. 但 12 个 checkpoint 可能已经够,因为噪声模型错了(朝有利方向)

`calibrate.py` 和 `power_outcome_n.py` 都把外部曲线画成 `binomial(n,p)/n`。
**试点自己的数据说这高估了约 2.6 倍。**

`runs/v4/decoupling-pilot-20260906/checkpoint_metrics.csv`,两臂各 5 个 checkpoint:

| 臂 | 外部曲线 | 拟合直线后的残差 SD | 二项模型预测 | 比值 |
|---|---|---:|---:|---:|
| naive | .281 .259 .344 .339 .333 | **2.92 pt** | 6.08 pt | 0.48 |
| rfo_gold | .281 .293 .322 .351 .333 | **1.49 pt** | 6.12 pt | 0.24 |

两臂合并 6 个自由度:`Σ res²/binomial_var = 0.87`,若二项噪声正确则服从 χ²(6),
**p = 0.010**。比值 0.38,**95% CI [0.25, 0.84]**——CI 很宽,所以配置按悲观端定,
不按点估计定。

**原因写在配置里:`evaluation.fixed_latents: true`。**
`scripts/v4_train.py:521` 把 `evaluation_seed` 的 step 分量钉死为 0,
所以每个 checkpoint 重画的是**同一批 (prompt, latent) 对**。
spec 之间的难度差(228 个 spec 里 21 个恒错、中位正确率 0.25——这是二项方差的主要来源)
因此是**固定效应,被配对消掉了**。只有模型真正改变的部分能翻转结果。

顺带核对同一件事的另一条路径:按 spec 分解 5,459 张已裁定图,
`mean_j p_j(1−p_j) = 0.1576` 对 `p̄(1−p̄) = 0.1955`,方差压缩 0.806。
那只是「固定 prompt 集」这一项;配对还额外消掉了 latent 那一项,所以实测的 0.38 更小。
**两条路径方向一致。**

## 3. 唯一还自由的旋钮:每个 outcome prompt 画几张

R 张/prompt/checkpoint,每张自己的 latent,R 张都跨 checkpoint 固定。
64 × 4 = **256 张/checkpoint/臂**,对试点的 64 张。

预算(试点实测:5 个 checkpoint/臂 用了 14.8 h,两臂并行占两张卡,
即 **2.96 h/checkpoint**;按图计价的部分是 **38.8 s/图** =
生成 20.5 + internvl 9.0 + qwen3vl 7.6 + crop 1.8):

| R | 图/checkpoint | 12 个 checkpoint 的墙钟 |
|---:|---:|---:|
| 1 | 64 | ~36 h |
| 2 | 128 | ~44 h |
| 3 | 192 | ~52 h |
| **4** | **256** | **~60 h** |

R=4 正好压在旧的 `max_wall_hours: 60` 上,所以**改的是那个上限(→72),
而不是让设计悄悄少画图**。

**裁定损耗要算进去:试点每 checkpoint 请求 64 张,平均只有 57.6 张拿到裁定
(掉 10.0%,最差的一个 checkpoint 掉 16%)。** 256 张 → 约 230 张可用,
最差情况 215 张。这就是检出力表里的 240 那一格。

## 4. 需要的代码改动(尚未做)

`data.images_per_outcome_prompt` 现在是**惰性键**,不改代码不起作用。
manifest 的 schema 已经带 `candidate_index`(语料池 K=8 就用它),
所以改动是纯增量的,**detect / verify 一行都不用动**。

四处:

1. `evaluate.evaluation_seed` 加 `candidate_index=0` 参数。
   **index 0 必须继续 hash 原来那几个分量**,不能追加 `0`——
   `_seed_from_parts` 是 `":".join(...)` 后取 sha256,追加会改哈希。
   这样第一张图与试点逐字节相同,**新 run 的 R=1 子集仍与冻结的 2026-09-06 曲线可比**。
2. `evaluate.write_evaluation_manifest` 的 `images` / `seeds` 改成按
   `(prompt_id, candidate_index)` 索引,`candidate_index` 写真值而不是常量 0。
3. `scripts/v4_train.py` 的 `stage_generate` 按 R 循环取种子与出图。
4. 内部曲线(`cycle_scores` / `self_selection_scores`)**默认只吃 index 0 的 64 张**,
   使其与试点同定义;另报一份 R 张全量的值。
   内部曲线本来就不缺检出力(斜率 +9.14/1000 步,SEM 0.09),不需要这些图。

## 5. 没有动的两件事

- **选择器不变。** cycle 已在 2026-09-08 过了冻结门槛(92.7% / 0.0% / 0.0%,
  见 `review-packets/cycle-selector-resolution-20260908`),但换它是
  `docs/prereg/2026-09-08-cycle-selector.md` §5 登记的训练回路改动,**尚未批准**,
  而且不是配置项——`src/selfsight/v4/train.py:488` 直接调用评分器。
- **划分不变。** `seed` / `local_outcome` / `local_probe` 与试点逐字节相同,
  `split_digest` 是这四项(加 run 列表)的 sha256,所以新 run **复用冻结的划分**,
  不重抽。改 `seed` 会静默重新划分。

## 6. 仍然是 rise 说了算

以上全部只动分母。**分子——外部正确率在折点前实际移动多少——仍然没有答案,
而且是三个旋钮里唯一不保证有效的那个。**

试点自己的曲线 32 步涨了 +7.4 pt(naive)/ +6.5 pt(rfo_gold),两臂内部曲线都单调上升。
**这个形状对**,但 n=57 时噪声 2.3 pt、信号 7.4 pt,5 个点拟合不出折点。
12 个 checkpoint × 256 张图是在问:**把这条已经在动的曲线画清楚,它会不会在某处停住。**
