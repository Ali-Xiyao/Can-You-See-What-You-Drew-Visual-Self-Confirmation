# 连接粒度:把主运行协议 §3 已经登记的口径写成可执行的判据

**2026-09-10 起草。** 这不是新的科学决定。
`docs/prereg/2026-09-08-dstar-main-run.md`
(sha256 `e276f3c74681df4459f2d0efa877002d4f61bf0d48db61ca2226426cd26b46ef`)
的假设、终态、判定阈值与禁止事项**原样继承,一个字不改**;
`docs/prereg/2026-09-06-decoupling-pilot.md` 经由它继承。
本文只登记一件事:**读手与那份协议 §3 已经写死的粒度不符**,以及修复后的判据长什么样。

供 replicate 的 `run_manifest.json` 的 `protocol_sha256` 指向(见 §7)。

## 1. 协议 §3 早就定了粒度,读手没有实现它

2026-09-08 的主运行协议 §3「明确不变的」里有这一条,逐字:

> **内部曲线。** 只用每个 prompt 的第一张图,与试点同定义。
> 多出的 3 张只服务外部曲线。内部曲线不缺检出力(斜率 +9.14/1000 步,SEM 0.09)。

写手照做了。`scripts/v4_train.py:551-556` 只对 `first_draw` 打 `s_select`,
并把口径盖进每一行的 metadata:`internal_curve_scope="first_draw_only"`、
`outcome_draws_per_prompt=4`。主运行磁盘上 12 个 checkpoint × 2 臂,
每份 `s_select.jsonl` 都是 64 行,被打分的 64 张图 **candidate_index 全为 0**。

读手没有。`scripts/v4_decoupling_report.py:106-107` 的判据是:

> Keep a spec only when all its outcome images have both measurements.

`read_outcome` 因此要求一个 spec 的**每一张**图同时有 `s_select` 和裁定。
R=4 时没有任何 spec 能满足——**这个条件在 R>1 下由构造不可满足**。

后果:`decoupling_report.json` 的 8 个窗口全部 `n_paired_specs: 0`、
`external.delta: null`、`first_candidate_step: None`。
**预注册的主检验从来没有过一个可测的输入。**
它看起来像阴性结果,实际上是没有结果。空集不是证据。

缺陷登记与重分析:`review-packets/dstar-join-granularity-20260910/`
与 `.planning/2026-09-08-iclr-redesign/EXECUTION.md` §3.9(commit `8b1b46a`)。

## 2. 登记的连接规则

以下是唯一的规范陈述。`read_outcome` 的完备性判据改为——
一个 spec 计入配对集,当且仅当:

1. 它的图**至少有一张**有有限的 `s_select`,且
2. 它的图**全部**已裁定(`resolution != "pending_human"` 且 `image_correct` 是 bool)。

计入的 spec 上:

- `internal` = 有分的那些图的 `s_select` 均值(R=4 时按 §1 必然只有第 0 张一张);
- `external` = **全部** `n_images` 张的 `image_correct` 均值;
- `n_images` = 该 spec 在 `manifest.jsonl` 里的图数,不是有分的图数。

两条曲线的分母**不同,而且是故意不同的**:内部曲线保持与试点同口径,
外部曲线吃满 R 张以换检出力。这正是协议 §2 定规模时给的理由。

## 3. 不改的

- **D\* 判定规则一个字不改。** `internal_delta_min=0.02`、`external_delta_max=0.02`、
  `window_checkpoints=3`、`bootstrap_samples=2000`、`bootstrap_seed=20260906`、
  `NUMERIC_TOLERANCE=1e-12`,以及 `candidate` / `supported` 两式的形状,
  全部沿用 `scripts/v4_decoupling_report.py:533-538` 冻结的那一份。
- **不重测。** 本条只改读法,不改任何已写下的 `s_select` 或裁定。
- **不放宽裁定要求。** 一张图没裁定,整个 spec 出局(§2 第 2 条)。代价见 §5。

## 4. 这条规则没有自由参数,而我已经看过结果了

必须写明:**这条规则是在跑过修复后的读手、看过主运行重分析之后落笔的。**
按常规这足以让它变成事后调参。它不是,理由要经得起查:

1. 粒度的目标是 2026-09-08 在任何 outcome 存在之前定的(协议 §3,上引)。
   本文没有选择粒度,只是把那句话写成判据。
2. 判据里没有可调的数。「至少一张有分」「全部已裁定」都不是阈值。
3. 唯一的备选读法——要求每张图都有分——在 R=4 下等价于现状,即空集。
   它不是一个能给出不同结论的分支,而是同一个不可满足条件。
4. 修复后的读手做过一次与 D\* 结论无关的核对:每个 spec 的 `s_select` 均值
   与 `checkpoint_metrics.csv` 的同名列在 12 个 checkpoint 上差都 < 0.01。

仍然要记着:重分析给出 `first_candidate_step=16`(两臂同),
`first_bootstrap_rule_supported_step=None`(两臂同)。**后者不许被争辩掉。**
naive 在 step-40 的内部 CI 下界是 `+5.05e-18`,对 `1e-12` 的门槛而言就是零;
下界压在零上的置信区间不排除零,判「不支持」是对的。

## 5. 这条规则驱逐了什么

裁定缺口:主运行每个 checkpoint 有 14–23 / 256 张图停在 `pending_human`,
按 §2 第 2 条会带走 15–20 / 64 个 spec,修复后每个窗口实得 44–49 对。
**这是真实的检出力损失,写论文时随 n 一起报,不许只报 n。**

放宽这一条(例如「只在已裁定的图上取外部均值」)会让 `external` 的分母
随裁定进度浮动,把裁定顺序变成一个混淆变量。**不采用。**

## 6. 代码落地的时机

`scripts/v4_decoupling_report.py` 在 `SOURCES` 里,而
`run_decoupling_pilot.py:193-196` **在每个 stage 开头**核 `source_sha256`。
主运行 `runs/v4/decoupling-main-20260908` 在 2026-09-10 06:04 仍停在 round-005.train。
**现在改它会当场打断主运行。** 因此:

- 主运行到达 `pilot_complete` 之前,修复只存在于 review packet 的重分析脚本里,
  **不进 `scripts/`**。
- 主运行结束后再改 `scripts/v4_decoupling_report.py`;五个 replicate 于是在构造时
  自然冻结修复后的指纹,**不需要 `--accept-code-update`**。
- 主运行此后若要 resume,需要 `--accept-code-update`,并把这次改动记进 `code_repairs`。

**主运行既有的 `decoupling_report.json` 不重跑、不覆盖。** 它是冻结记录,
连同那 8 个 `n_paired_specs: 0` 一起保留。修复后的数字在 review packet 里,注明来源。

## 7. 五个 replicate 指向本文

`--protocol` 只收一个路径,而 replicate 跑的是修复后的读手。
让它们继续指向 `2026-09-08-dstar-main-run.md`,等于把一份不描述其读手的协议
冻进 `run_manifest.json`——那是 §1 同一个毛病换个地方犯。因此:

- 五条启动命令改为 `--protocol docs/prereg/2026-09-10-join-granularity.md`
  (EXECUTION.md §1.3 更正 §1.2;**§1.2 原样保留**,
  `tests/test_section_1_launch_commands.py` 改读 §1.3)。
- 本文以 sha256 引用 2026-09-08 协议,该文件**逐字节不动**,
  主运行的 provenance 因此完好。

## 8. 事先声明

- 修复只把主检验从「无输入」变成「有输入」。**它不提高 D\* 成立的概率**,
  也不是为了让 D\* 成立而写的。重分析里 `supported` 仍然全是 False。
- **单 seed 与五 seed 的界线。** 2026-09-08 §4 的「单 seed,描述性轨迹,
  不得写成显著性结论」是就主运行**那一次运行**说的,对**每一个 replicate 单独看**
  同样有效:任何单个 seed 的轨迹仍然只是描述性的。显著性只能来自 §34 登记的
  五 seed 符号检验,**不得对 seed 做 bootstrap**。这就是 EXECUTION.md §1.2
  点出的那处不吻合的处置,写在这里而不是留着。
- 若五个 replicate 上 `first_bootstrap_rule_supported_step` 仍为 None,
  那是阴性结果,**要按阴性结果写**,并同时报检出力(每窗约 44–49 对,不是 64)。
- 本文不许可任何其他读手改动。`read_outcome` 之外的函数若也与写手不符,
  按同样流程另行登记,**不搭这次的便车**。

## 9. 补:同一处缺陷还有第二个出口(2026-09-10 09:3x 追加)

**追加时点与可核查性**:本节写于 2026-09-10 09:3x。此时**没有任何 run 冻结过本文的摘要**
——`runs/*/*/run_manifest.json` 共 3 份,均不引用本文,五个 replicate 尚未启动。
所以这是一次追加而不是改写。追加的内容与 §2 同源、同理由、**同样没有自由参数**。

**§2 只覆盖了「配对集」,而 `read_outcome` 里还有第二处用同一个判据。**
`measurements[spec_id]["s_select"]` 写作
`sum(scores) / size if len(scores) == size else None`。
R=4 而按协议 §3 只有第 0 张有分时,`len(scores)=1 != size=4`,
**该字段对每个 spec 恒为 None**。它喂 `full_population_screen` 的 `internal` 列表
(条件是 `a["s_select"] is not None and b["s_select"] is not None`),于是那条链整条为空。

主运行冻结报告的 robustness block 里可以逐字看到:

```
"n_external_paired_specs": 44,   "n_internal_paired_specs": 0,
"internal_pair_coverage": 0.0,   "internal_all_available_delta": null,
"internal_all_available_bootstrap_ci": null,   "internal_supported": false
```

外部那半是有输入的(44/64),内部那半**由构造为空**。
`internal_supported: false` 看起来像测量结果,实际是结构性的零。
这个 block 自称 `supplementary_robustness_check_not_registered_confirmation`,
不是登记的主检验,**但它在报告里,会被当成证据读**。

**登记的修复**:`measurements[spec_id]["s_select"]` 改为

```python
sum(scores) / len(scores) if scores else None
```

理由与 §2 逐字相同——内部曲线按协议 §3 只在第一张图上测,分母就该是有分的那些图。
同一字典里的 `external` 保持 `sum(verdicts) / size if len(verdicts) == size else None`,
**不改**:那四张图本来就都该裁定,缺的是真缺。

**明确不改的**:`s_select_low` / `s_select_high` 保持原样。它们把没打分的图当缺失值,
在 R>1 下给出近乎无信息的区间(单张 0.95 时是 `[0.2375, 0.9875]`)。
但**全仓库没有任何地方读它们**——只在 `read_outcome` 里写入、
在 `_measurements` 的兼容分支和 `unknown` 缺省里构造,再无消费者。
它们是死字段,**改死字段等于在没有检验的地方动手,不做**。
将来若有人要读,那时按同一流程另行登记。

**§8 那句「不搭便车」仍然有效。** 本节改的两个出口都在 `read_outcome` 内部,
都是协议 §3 同一句话的同一次落地。`read_outcome` 之外仍然一个字节不动。
