# 执行计划:剩下的每一步,以及每一步在等什么

**2026-09-08 写。** 预注册在 `PREREG.md`,本文不改写它,只把它落成命令。
状态用「能不能现在跑」分类,不用「重要不重要」。

---

## 0. 现在的真实状态

| | 状态 | 卡上 |
|---|---|---|
| E3 arm A + C(`decoupling-main-20260908`)| **在跑**,round 0 / 11,07:33 起 | cuda:0 观察 + cuda:1 生成,两张都占 |
| E3 arm B(`naive` + `blind_self` 配对新跑)| 代码就绪,**等卡** | — |
| E4 三个新模型作答 | 代码就绪、预检通过,**等卡** | — |
| 评分器缺陷 | **已修**(worktree 99b7c0f,登记为偏离 3) | 无 |
| arm B 的轮次恢复路径 | **已修**(worktree 93379c8) | 无 |

两张 3090 都被主运行占着,而且它按 checkpoint 在两张卡之间来回。
`nvidia-smi` 显示 GPU 0 空闲**不代表它是空的**——那是 detect 和 crop
之间的间隙。所有 CPU 能做的事已经在 CPU 上做完了。

---

## 1. arm B 的启动命令,以及它还差的那一处代码

预注册钉死了跑法:**同一份 config,`--arms naive blind_self`,新输出目录**。
理由不是省事——pilot 实测 `rfo_gold` 只在 7–9/12 个 prompt 上给出选择,
`pair_decisions` 取交集,所以主运行的 A 列实际只训练了约 75% 的 prompt。
B 若单臂跑会训练 100%,A–B 差异就带上 prompt 集混淆,而 A vs B 正是主张所在。

`scripts/v4_train.py` 已经有 `--arms`(worktree 41a9170,choices 取自 `SELECTORS`),
`prepare_round` 也已经按传入的臂集归档(93379c8)。

**还差的是 supervisor**:`scripts/run_decoupling_pilot.py` 有自己的
模块常量 `ARMS = ["naive", "rfo_gold"]` 和 `ARM_DEVICE`,不往下传 `--arms`。
主树的这份文件正被另一个 agent 改着(未提交,而且正在监督主运行),
**不要现在动它**。等它提交后按下面改,四处:

1. `--arms nargs="+" default=["naive","rfo_gold"]`,加进 argparse。
2. `ARMS` 常量 → 实例属性 `self.arms`(`validate_round` 的
   `sorted(...) != sorted(ARMS)` 一并改)。
3. `ARM_DEVICE` 由臂集派生:`dict(zip(self.arms, ["cuda:0", "cuda:1"]))`,
   而不是写死的两个键——`blind_self` 不在现在的字典里,会 KeyError。
4. `self.run("round-....train", ..., "scripts/v4_train.py", [... "--arms", *self.arms])`。

改完要有测试:臂集为 `naive blind_self` 时,(a) 两条链各拿到一张卡,
(b) `validate_round` 要求的正是这两个臂的报告,(c) 传给 `v4_train.py`
的命令行里带 `--arms`。前两条对照旧实现会红。

**启动**(等主运行释放两张卡之后):

```
envs/core/python.exe -u scripts/run_decoupling_pilot.py   --outdir runs/v4/blind-self-20260910   --config configs/v4_decoupling_main_20260908.yaml   --arms naive blind_self
```

启动前必查:`split.json` 的 digest 要与主运行一致(预注册说它是
(prompt_ids, outcome, probe, seed) 的纯函数,同 config 必然复现;
**要核对,不要相信**)。

---

## 2. E4 的三条命令

预检已在 CPU 上过了(偏离 2):Show-o v1 与 Janus-Pro 都对不同的图给不同的答案。
评分器缺陷已修,所以现在跑出来的行才是可用的。

```
python scripts/v4_cross_model.py preflight --device cuda:0
python scripts/v4_cross_model.py observe   --device cuda:0
python scripts/v4_cross_model.py report    --out review-packets/cross-model-20260910/cross_model.json
```

`observe` 会为每个模型起它自己 config 里 `environment` 指定的解释器。
判定按注册的三个模型算,Janus 单列不计入——这条在 `stage_report` 里,
`tests/test_v4_cross_model.py` 有对照测试。

**先跑 E4 还是先跑 arm B**:E4 只要一张卡、纯推理、几小时;arm B 要两张卡、
约 52 小时。主运行一释放,先把 E4 跑完再起 arm B,总墙钟不变而 E4 提前落地。

---

## 3. 合并顺序

worktree `codex/blind-self-arm-20260908` 领先主树 5 个提交。
`selfsight` 在每个环境里都是 editable 装到**主树**的 `src`,
而 `scripts/v4_run_pipeline.py` 不自己插 src 路径,
所以**必须先合并再跑**,否则 worktree 的脚本 import 的是主树的旧库。

合并前提:另一个 agent 的 `run_decoupling_pilot.py` 改动先提交。
预期冲突只有那一个文件。

---

## 4. 不做的事

- 不因为 `nvidia-smi` 显示 0% 就去抢卡。
- 不在主运行还没写出 `checkpoint_metrics.csv` 之前看它的任何数值。
- 不为了让 E4 凑够 2/3 而增加模型。
- 不动 `runs/` 下的任何研究工件;要清理就归档到 `_archive/`。
