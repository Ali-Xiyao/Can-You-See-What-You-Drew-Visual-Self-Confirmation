# E3 端点 2 的仪器:三处口径缺陷 + 一处端点 1 会崩的洞(2026-09-09)

**结论先说**:端点 2 按现在的口径**算不出来**,不是难算,是数据不在。
另外顺带发现端点 1 的最终脚本会在真实数据上抛异常。四条都可复现:

    envs/core/python.exe review-packets/e3-endpoint2-instrument-20260909/gap_pairing.py

输出存在同目录 `output.txt`。脚本只读已存盘的产物,不碰 GPU,不写任何 run 目录。

**我在写规则之前已经看过的东西**(照偏离 10.0 的规矩先说):主运行 `naive` 与
`rfo_gold` 在 step 0 / 8 / 16 的判别 gap(两种配对都看了),以及 pilot 全部
0–32 的同一条曲线。**没有** arm B 的任何数据——它还不存在。

---

## D1 标签取错了 draw

`s_select` 每个 prompt 只记一行,打分的是 **candidate 0** 那张图。
`prompt_recital.py` 用 `correct[spec_id] = ...` 逐行灌进字典,R=4 时活下来的是
**candidate 3** 的判决。pilot 是 R=1,这个错误在那儿观测不到;主运行是 R=4。

| | step 0 | step 8 | step 16 |
|---|---|---|---|
| naive,candidate 0(对) | +0.0697 | +0.0770 | +0.0795 |
| naive,最后一行(错) | −0.0091 | −0.0111 | +0.0231 |
| 差 | +0.0788 | +0.0881 | +0.0564 |

差值比端点 1 有功效测到的效应(+0.042)还大,而且它把符号都改了。
两种标签在 64 个 prompt 上只有 41 个一致。

## D2 端点 2 假设里引的 pilot 轨迹,引用的脚本印不出来

预注册端点 2 写:「pilot 已观测:+0.047 → 0.000,32 步内,两臂皆然」,
出处经 `review-packets/context-ablation-20260908/RESULTS.md` 指到
`review-packets/dg-power-20260908/prompt_recital.py`。

今天原样跑那个脚本:

| arm | step 0 | 8 | 16 | 24 | 32 |
|---|---|---|---|---|---|
| naive | +0.070 | +0.092 | +0.083 | +0.113 | **+0.049** |
| rfo_gold | +0.070 | +0.122 | +0.083 | +0.102 | **+0.046** |

**没有 0.000,也没有塌**:从 +0.070 到 +0.047,降三分之一。
脚本自己的结论文字说的是「gap 一直很小,像在背 prompt」,不是「gap 塌掉了」。

数据没动过:pilot 的 `verified.jsonl` 最后写于 09-05/09-06,
`RESULTS.md` 与 `prompt_recital.py` 都是 09-08 09:08。所以这不是重新裁决造成的漂移。
看起来 **+0.047 是 step 32 两臂的均值(+0.0493 与 +0.0456),而 0.000 是表里
`1.000 +- 0.000` 那一列的 SEM**——两个数被读成了一条轨迹的两端。

**这不改端点 2 的判定条件。** 判定是斜率检验(A 显著为负,且 B 显著大于 A),
不依赖这个被引错的量级。被改的是它的动机陈述。

## D3 盲条件根本没记

端点 2 要的是**盲**条件下的 gap(去掉 preamble)。
`src/selfsight/v4/evaluate.py` 里逐 checkpoint 的评测**只调 `observe_naive`**,
它把每道题都套进 `PROMPTED_PREAMBLE`。所以:

- `naive` 的逐 checkpoint `s_select` = 带 prompt 的分
- `rfo_gold` 的 = 带 prompt 的分
- `blind_self` 的将来也会是带 prompt 的分

三个臂一致地记了**另一个条件**。`observe_rfo`(盲线)只在 `train.py` 的选择步用,
不进 checkpoint 评测。

**好消息:可以离线补。** 图和每轮的 adapter 都存着
(`evaluations/<arm>/step-*/images/`、`checkpoints/<arm>/round-*/adapter.pt`),
所以补测是一次**重新观察**,不用重训。代价按实测每图约 1.4 s / 4 题估:
64 图 × 13 checkpoint × 2 臂 ≈ 每个 run 半小时的观察 + adapter 装载。

## D4 一行可以是「不在」,而不是「没裁决」

`verified.summary.json` 里有 `skipped_no_detection`:检测器在这张图上什么都没找到时,
这张图**不产生 `verified.jsonl` 行**。

偏离 10 登记的成对剔除处理的是「行在、但没裁决」(`pending_human` / `unnameable`),
而 `endpoint1.load_checkpoint` 对「一臂有、另一臂没有的 key」是**抛异常**的——
我当时写的理由是「运行没产出的图是仪器的洞」。它已经发生过一次:

    rfo_gold step 16:manifest 要 256 张,verified.jsonl 只有 255 行,
    缺的是 (v4a-0001, 1)。图在盘上(196 KB),
    detections.qwen3vl / internvl / crops 三个文件里都没有它。

现在的频率是 16 个 (arm, checkpoint) 里 1 次。13 checkpoint × 2 臂 × 5 seed = 130,
它一定还会发生,而端点 1 的最终脚本会在那儿崩。

同一张表还给出 `pending_human` 的量级:主运行每 256 张里 14–23 张(5.5–9.0%),
而且两臂不等(step 8:naive 17、rfo_gold 23)。成对剔除删的是并集。

**跳过不是随机缺失**:检测器找不到任何东西的图,大概率是生成失败的图。
悄悄删掉它等于把一张多半是错的图移出分母,而两臂不必跳过同样的图。
所以规则要连**每臂跳过数**一起报,不只是删。

---

## 处置

按项目纪律:旧记录原样保留,缺陷登记为新证据,重新测量并注明来源。

- 不改 `review-packets/context-ablation-20260908/RESULTS.md`,不改
  `review-packets/dg-power-20260908/prompt_recital.py`,不改预注册端点 2 的原文。
- 追加**偏离 11**(端点 2 的口径:D1 配对、D2 更正动机陈述、D3 盲测补测协议)。
- 追加**偏离 12**(D4:缺行按未裁决处理并逐臂上报,修正偏离 10 留下的洞)。
