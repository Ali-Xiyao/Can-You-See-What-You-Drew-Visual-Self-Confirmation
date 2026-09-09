# 执行计划:剩下的每一步,以及每一步在等什么

**2026-09-08 写。** 预注册在 `PREREG.md`,本文不改写它,只把它落成命令。
状态用「能不能现在跑」分类,不用「重要不重要」。

---

## 0. 现在的真实状态

| | 状态 | 卡上 |
|---|---|---|
| E3 arm A + C(`decoupling-main-20260908`)| **在跑**,**round 0 / 11 训完**,正在测它的 step-00008;09-08 07:33 起 | cuda:0 观察 + cuda:1 生成,两张都占 |
| E3 arm B(`naive` + `blind_self` 配对新跑)| 代码就绪(含新的 audit stage,§1),**等卡** | — |
| E4 三个新模型作答 | **已完成**,3/3 复现,见 §2 | 已释放 |
| 评分器缺陷 | **已修**(worktree 99b7c0f,登记为偏离 3) | 无 |
| arm B 的轮次恢复路径 | **已修**(worktree 93379c8) | 无 |
| 排卡的空载基准(§0.3)| 脚本就绪并已预注册判据,**等两张卡都空** | — |

两张 3090 都被主运行占着,而且它按 checkpoint 在两张卡之间来回。
`nvidia-smi` 显示 GPU 0 空闲**不代表它是空的**——那是 detect 和 crop
之间的间隙。所有 CPU 能做的事已经在 CPU 上做完了。

## 0.1 墙钟外推:约 70 h,而两份预注册给的上限不一样

**只有 round 0 一个数据点,下面全是外推,不是测量。** 但它已经够指出一个冲突。

`stage-completion/` 的时间戳(起点 07:33):

```
round-000.train                    +1.77 h
naive    step-00000 整条链          +1.90 h(train 结束后)
rfo_gold step-00000 整条链          +3.95 h(同上,两臂本应并行)
```

两臂各占一张卡,但**一次双臂评测的墙钟由慢的那条决定,3.95 h,
而不是单臂的 1.90 h**。§2b 的并行化把两条链拆开了,可它们在检测阶段
仍然没有真正同时跑满:naive 的 internvl 用 0.38 h,rfo_gold 的用 1.19 h,
而后者跑的时候 naive 整条链已经结束了——所以这不是简单的抢卡。
**原因已于 2026-09-08 深夜查明,见 §0.2。**(查的过程只用只读查询,
没有动正在跑的作业。)

按 11 轮训练 + 12 次评测外推:

```
11 × 1.77 (train) + 12 × 3.95 (evaluate)  ≈  67 h
```

再加 gradient 探针和 report,实际会更高一点。

协议 §1 预计的是「85 h → 约 52 h」。**差距在于那个估计假设两臂完全并行,
实测没有。**

### 冲突

| 出处 | 上限 |
|---|---|
| `configs/v4_decoupling_main_20260908.yaml` `max_wall_hours` | **96** |
| `docs/prereg/2026-09-08-dstar-main-run.md` §1 表 | **96**(明确从 60 提上来) |
| `.planning/2026-09-08-iclr-redesign/PREREG.md` E3 失败条件 4 | **60**,「停,登记 traceback」 |

67 h 落在两者之间。**supervisor 不会停(配置是 96),但 E3 注册的失败条件 4
会在 60 h 触发。** 这两份是同一天写的,E3 那条大概是从 pilot 的 60 继承下来、
没跟着主运行协议一起提到 96。

**这需要你裁定,而且要在 60 h 之前裁**(约 2026-09-10 19:30)。
两个选择都合规,但必须现在选、写成偏离,不能等跑到 61 h 再决定:

- 认 96:追加一条偏离,说明 E3 失败条件 4 的 60 沿用了 pilot 的数,
  主运行协议已按 R=4 的实测代价改为 96,E3 随之对齐。
- 认 60:60 h 时停,登记 traceback,按失败条件处理。

**我不替你选**,因为事后按结果挑上限就是在挑停止规则。

**已裁定(2026-09-08):认 96。** 写成 `PREREG.md` 偏离 5,失败条件 4 的原文
一字未动。同时登记的两件事:96 h 是停止线不是预算;墙钟按 run 计,
重启不清零(`started_unix` 从 manifest 读回),停机时间照算。
截止时刻 2026-09-12 07:33。

---

## 0.2 两臂不对称的原因:卡 1 挂在 PCIe 3.0 x4 上——**这条 STATUS 早就记着**

**先更正一句**:我 2026-09-08 深夜把这条当成新发现报了,它不是。
`STATUS.md` 第 19 行和 §32(2026-09-03)已经写着「GPU1 挂在 PCIe gen3 x4、
GPU0 是 gen4 x16,逐图检测慢 7.5 倍但生成只慢 7%」,连操作规则都定了:
**传输受限的活(detect / observe)留给 GPU0,算力受限的活(生成、梯度)才给 GPU1。**
今晚的只读复核(`nvidia-smi`,没有碰任何作业)只是把它再确认一遍:

```
index  name       temp  power        clocks.sm  clocks.max.sm  pcie.width  pcie.gen
0      RTX 3090   36C     9.05 W /350      0 MHz       2100 MHz    16 / 16     4 / 4
1      RTX 3090   42C    88.11 W /350   1245 MHz       2130 MHz     4 / 16     3 / 3
```

`clocks_throttle_reasons` 全部 `Not Active`,88 W / 350 W、1245 / 2130 MHz——
在等数据不在算,和 §32 的判断一致。

### 新的部分之一:§0.1 的问号和这条事实之间的线

§0.1 写「原因未查明」,是因为没人把这条已知事实接到那个具体现象上。接上就是:
`run_decoupling_pilot.py` 的 `ARM_DEVICE` 把 **一整条链**(generate → detect ×2 →
crop → verify → probe)按臂钉死在一张卡上,rfo_gold 整条都在 cuda:1。
于是 detect 也跑在慢卡上,而一个 checkpoint 的墙钟取两臂的大者。
§0.1 的 3.95 h vs 1.90 h 就是这么来的。

### 新的部分之二:§32 说缺的那个对照,现在有了一半

§32 记的是「没有干净的同机对照」,7.5 倍和 36% 两个数都在不干净的竞争条件下测的,
「要定这件事需要两卡各跑一次空载基准,现在卡没空」。今晚拿到的不是空载基准,
但是**同一个检测器、同样 256 张图、同一次运行、同时在跑**的两臂对照:

| 阶段 | cuda:0 (naive) | cuda:1 (rfo_gold) | 比 |
|---|---|---|---|
| internvl | 10.0 s/img | 14.6 s/img | 1.46x |
| qwen3vl | 12.6 s/img | 19.8 s/img | 1.57x |

**1.46 倍,不是 7.5 倍。** 差别不在卡变快了,在分母变慢了:两臂同时检测时
cuda:0 自己也被拖到 10.0 s/img,而 §0.1 记的 round 0(那时 rfo_gold 还没开始检测)
naive 的 internvl 是 0.38 h / 256 = **5.3 s/img**。所以正确的读法是——
**当 cuda:0 本身已经饱和时,把 detect 放到 cuda:1 的边际代价远小于 7.5 倍。**
7.5 倍那个数描述的是「空载 GPU0 vs 满载 GPU1」,不是排期时会遇到的情形。

### 由此产生的一个待裁定的问题(**不改正在跑的运行**)

按 §32 的规则,两臂的 detect 都应该在 cuda:0 上、串行,generate 才给 cuda:1。
用本运行自己的数字估一下 internvl 那一段:

```
现在(两臂各一张卡,并行)      max(256×10.0, 256×14.6) = 3738 s = 1.04 h
都在 cuda:0(串行,按 5.3 s/img) 2 × 256 × 5.3         = 2714 s = 0.75 h
```

**串行可能反而快约 28%**,而且能把 cuda:1 空出来跑生成。这只是外推,5.3 s/img
是单点;要坐实需要一次真正的对照。但它足以说明现在的 `ARM_DEVICE` 与 §32 的规则
是冲突的,而冲突的那一边是代码。

**现在不动它。** 三个理由:`run_decoupling_pilot.py` 在自己的 `SOURCES` 里,
改了正在跑的主运行下一个阶段就会拒绝继续;主运行已经过半;而这个改动的收益
是估出来的、不是测出来的。**arm B 是验证它的地方**——起 arm B 之前可以先花
十几分钟各测一次空载速率,拿到数再决定,那时两张卡本来就是空的。

---

## 0.4 先更正一句:这是 round 0 / 11,不是 step 8 / 11(2026-09-09)

`state.json` 里的 `step-00008` 是**累计优化器步数**,不是轮次。
`optimizer_steps_per_round: 8`、`rounds: 11`,所以最后一个检查点是 step-00088,
而 `rounds/` 底下到现在只有 `round-000`。我此前把它读成「11 轮里的第 8 轮」,
据此说过「96 h 上限还很宽」——**那句话是错的**,下面是重算。

### 重算的办法,比数完成标记准

`stage-completion/*.json` 的时间戳只给结束时刻,相邻标记的差包含了排队和停机。
`logs/<stage>.log` 的**创建时间到修改时间**才是那个 stage 自己的墙钟
(`run()` 在起子进程前就建好日志文件)。两者对 `step-00000.report` 差了 6.4 h,
差的正是 09-08 那次因缺 audit 的停机。

### 两个数据点,第二个明显更慢

| | round 0 的基线评测(step-00000)| round 0 的检查点评测(step-00008)|
|---|---|---|
| naive 链(cuda:0)整条 | 1.89 h | 3.32 h |
| rfo_gold 链(cuda:1)整条 | 3.95 h | 5.25 h |
| **块墙钟 = 慢的那条** | **4.06 h** | **5.30 h** |

逐 stage 的检测速率,**两次都是同样的 256 张图**(日志里 `251/256` 一致):

| | step-00000 | step-00008 | 我 09-09 深夜的空载单跑 |
|---|---|---|---|
| naive internvl(cuda:0)| 4.4 s/img | 10.0 s/img | 6.04 s/img |
| naive qwen3vl(cuda:0)| 4.1 s/img | 12.6 s/img | 4.26 s/img |
| rfo_gold internvl(cuda:1)| 15.7 s/img | 19.4 s/img | — |

**cuda:0 慢了 2.3–3.1 倍,cuda:1 只慢了 1.24 倍。** 排除掉的解释:

- 不是工作量。两次都是 256 张,同一份 prompt 集,同一份代码(SOURCES 每个
  stage 都重核过摘要,中途没人改)。
- 不是降频。09-09 00:5x 查:卡 1 42 °C、88 W / 350 W、SM 1290 / 2130 MHz,
  卡 0 36 °C、8.94 W;`clocks_event_reasons.active` 没有热或功耗位。
- 不是 CPU 打满。20 个逻辑核,`_Total` 三次采样 18.8 / 26.6 / 27.0 %。
- 不是机器坏了。空载单跑还能跑到 4.26 s/img(qwen3vl),和 step-00000 同级。
- 也不是本次运行自己的并发结构:两个块里 rfo_gold 的 generate 都压着
  naive 的 detect,拓扑一样。

**原因未定。** 唯一站得住的说法是:代价落在 cuda:0 上远多于 cuda:1,
而 cuda:0 正是本项目之外的东西最容易碰到的那张卡(它是主显示输出所在、
PCIe 4.0 x16 的那张,见 §0.2 / STATUS §32)。这把 §0.2 里我撤回的那个猜测
**部分恢复**:不是「有东西在占 cuda:1」,而是**有东西在占 cuda:0**,
两个评测块之间它出现了。这仍然是猜测,没有证据锁定是哪个进程。

### 对 96 h 的影响:不宽

现在 +17.5 h,剩下 round 1..10 共 10 个块,每块 = train(1.77 h)+ 评测 + report。

```
乐观(评测按 4.06 h):17.5 + 10 × 5.83  ≈ 76 h   → 09-11 15:20
悲观(评测按 5.30 h):17.5 + 10 × 7.07  ≈ 88 h   → 09-12 03:45
96 h 停机线                                        → 09-12 07:33
```

也就是说:**按最近这一个块的速度,余量只有 8 h。** 而且第二个块比第一个块慢
31%;如果那是趋势而不是一次性台阶,第三个块就会把预算吃穿。
**已裁定的 failure condition 4 不动**——这里只是把外推更新到实测,
不是提议改判据。第三个块(round-001 的 step-00016)结束时会有第三个数据点,
那时才谈得上「趋势」。

### 对 §0.3 的一个口径提醒(不改 §0.3)

`card_benchmark.py` 的 `require_idle()` 要求两张卡都空,所以它测的是
**干净机器**上的排卡。而主运行实际跑在一台不干净的机器上,上面这组数字
就是证据。§0.3 的判据已经冻结,照跑照用;但它给出的结论适用于干净条件,
arm B 若也在不干净的条件下跑,实际收益可能与判据算出来的不同。
这一条写在这里,是为了将来不要把「干净条件下的 1.28x」当成「实际会快 28%」。

---

## 0.3 排卡方案:在 arm B 上定,先测空载(用户 2026-09-09 裁定)

§0.2 那个「串行可能快 28%」是从单点外推的,不足以拿来改一个 52 h 的运行。
**裁定:不改主运行;起 arm B 之前先做一次空载对照,按测出来的数决定。**

脚本已写好:`review-packets/card-scheduling-20260909/card_benchmark.py`(armB 分支)。
它调的是真的 `v4_run_pipeline.py detect`,环境和 supervisor 给自己 detect 阶段的一样,
计时是整个子进程的墙钟(含模型加载——两臂两次调用,两种方案都付两次)。
每个检测器、每张卡 N=64 张图,测四个数:

```
A   cuda:0 单独
B   cuda:1 单独
C   两张卡同时      ->  wall_P = max(C0, C1)      今天 arm B 每个 detect 阶段的代价
                        wall_S = 2 × A            两臂串到快卡上的代价
```

### 判据(**在测之前定死**)

> **当且仅当 `wall_S ≤ 0.90 × wall_P` 在两个检测器上都成立**,arm B 改用「两臂的
> detect 都串在快卡上」。两个检测器方向不一致 → 保持现在的按臂分卡。

10% 的余量是因为 N 比运行里的 256 小,而且基准复现不了全部竞争条件。
这个比较对串行方案是**偏保守**的:它没给「卡 1 空出来可以并行生成」记任何分,
而那才是更大的那块收益,也是更难测的。

### 顺带查一件不是排期的事

A 和 B 是同一批图在两张卡上检测。**输出必须逐字节一致**;不一致就意味着换卡
是在改测量结果而不只是改排期,那时排期问题作废,先查那个。
同卡确定性已经验过:2026-09-09 在 cuda:0 上重测 48 张,与主运行自己的
`detections.internvl.jsonl` **48/48 完全相同**。

### 已经有的一半数据(不是空载,记在这里备查)

主运行占着卡 1 时,在空着的卡 0 上跑同样 48 张图:

| 检测器 | 卡 0,本次实测 | 卡 0,主运行日志 | 卡 1,主运行日志 |
|---|---|---|---|
| internvl | **6.04** s/img | 10.0 s/img | 14.6 s/img |
| qwen3vl | **4.26** s/img | 12.6 s/img | 21.7 s/img |

(本次实测取第 2–26 张的稳态,把模型加载和第一张的预热剔掉:
internvl `(26×6.2 − 10.3)/25`,qwen3vl `(26×4.4 − 7.8)/25`。)

**同一张卡、同一个检测器,差 1.7 倍和 3.0 倍,原因未定。** 三个候选:
(a) 主运行那两段时间 ChatGPT 桌面端还开着(你 23:2x 才关),而它很可能占的是卡 0
——我之前猜它占卡 1,那个猜测已经被关掉后的测量否掉了;
(b) 卡 1 当时的负载不同(naive 跑 qwen3vl 时,卡 1 在**生成**,不是检测);
(c) 我只取了 manifest 的前 48 张。

**不要现在下结论。** 但方向已经很清楚:**卡 0 的真实吞吐是主运行实际拿到的
2–3 倍**,所以「两臂 detect 都串到卡 0」这个方案比 §0.2 估的还要更有利。
正因为如此更要按 §0.3 的判据老老实实测一次空载,而不是拿这张脏表去改设计。

---

## 1. arm B 的启动:代码已就位,只等主运行让出卡

预注册钉死了跑法:**同一份 config,`--arms naive blind_self`,新输出目录**。
理由不是省事——pilot 实测 `rfo_gold` 只在 7–9/12 个 prompt 上给出选择,
`pair_decisions` 取交集,所以主运行的 A 列实际只训练了约 75% 的 prompt。
B 若单臂跑会训练 100%,A–B 差异就带上 prompt 集混淆,而 A vs B 正是主张所在。

`scripts/v4_train.py` 已经有 `--arms`(worktree 41a9170,choices 取自 `SELECTORS`),
`prepare_round` 也已经按传入的臂集归档(93379c8)。

**supervisor 的那四处已经改完**(worktree `39f4d83`),原样落地:

1. `--arms nargs="+" default=["naive","rfo_gold"]` 进了 argparse。
2. `ARMS` 常量 → 实例属性 `self.arms`,`validate_round` 的比较一并改。
3. `ARM_DEVICE` 由臂集派生:`dict(zip(self.arms, ARM_CARDS))`。
4. train stage 带上 `"--arms", *self.arms`。

改的时候另外加了两条原计划里没有的护栏:

- **臂数不等于卡数就在构造时拒绝**。`zip` 会静默丢掉第三个臂,而丢掉的那个
  不训练、却仍然出现在 `validate_round` 的期望里,要到第一个 round 结束才炸。
- **重复臂拒绝,不折叠**;`run_manifest.json` 冻结臂集,relaunch 时臂集不一致
  就拒绝 resume。旧运行(`--arms` 之前起的)没有这个键,回退到注册的配对,
  所以 `decoupling-main-20260908` 仍然 resume 得了。

**scene audit 现在是 supervisor 的一个 stage**(worktree `3b4ef0f` → `a78f5b9` →
`9095560`),不再是启动前要记得手跑的一步:

- 位置在 `freeze-probe` 之后、第一个 round 之前。它读的 split 和 probe bank
  正是前两个 stage 的产物,而第一个 round 之后就不再有「什么都还没产生」这回事。
- **只自动造 prospective 那一种**(`audit-splits/scene_overlap.json`)。已经跑过的
  运行造不出来,`v4_scene_audit.py` 会当场拒绝;退而用 retrospective 是
  「这批证据允许支持什么」的决定,保持人工。
- 目录里已有任一种 audit 就不重造——resume 时重造会把一份本该早于 outcome 的
  freeze 重新盖上晚于 outcome 的时间。
- 因此 `scripts/v4_scene_audit.py` 进了 `SOURCES`。**副作用**:在此之前冻结的运行
  (包括 `decoupling-main-20260908`,它的 manifest 里没有这个键)合并后若要 resume,
  需要 `--accept-code-update`,那会把这次改动记进 `code_repairs`。主运行现在跑的是
  主树那一份,§3 的合并顺序保证它跑完之前碰不到这件事。

测试:66 个通过(supervisor 34 + scene audit 32)。新写的都先对旧代码验红过
(`--arms` 11 红、audit 解析 5 红、audit stage 5 红),对新增代码的变异测试 11/11 击杀。

**启动**(等主运行释放两张卡、并且 §3 的合并做完之后):

**在主树跑,不要在 worktree 跑。** `run()` 拼的是 `ROOT/envs/<环境>/python.exe`,
`ROOT` 是脚本自己所在的树;`H:/Xiyao_Wang/062_armB` 底下没有 `envs/`,
在那里启动会在第一个 stage 就找不到解释器。这也是合并必须排在启动前面的原因。

```
envs/core/python.exe -u scripts/run_decoupling_pilot.py --outdir runs/v4/blind-self-<启动日> --config configs/v4_decoupling_main_20260908.yaml --arms naive blind_self
```

启动前必查:

1. `split.json` 的 digest 要与主运行一致(预注册说它是 (prompt_ids, outcome,
   probe, seed) 的纯函数,同 config 必然复现;**要核对,不要相信**)。主运行的值是
   `83b7eaa696726a1f6261327a5ac70c2d494629e5e736bcae7799eed0482f7094`,
   它既是 `runs/v4/decoupling-main-20260908/split.json` 的 sha256,也是那份
   audit 里记的 `provenance.split_sha256`。另有一个 `provenance.split_digest`
   `9bab14d427f2b61bea001b710126dcf18d610972985f723665534209231f9d58`,
   那是 split **内容**的规范摘要,和文件字节摘要是两个数,别混。
   arm B 的 split stage 一跑完就核对,不一致就停,不要往下走。
2. `audit-splits/scene_overlap.json` 落地后,`kind` 是
   `prospective_scene_overlap_sensitivity_freeze`、
   `created_before_any_outcome_evaluation_artifact` 是 `true`。supervisor 的
   `resolve_audit()` 已经查了旗标与目录一致这一条;这里重复,是因为它是主运行
   没有、arm B 独有的东西,而 `v4_decoupling_report.py` 会拿它去挑 outcome 子集。

   **这份 audit 的内容已经先算出来了**,见
   `review-packets/armB-audit-preimage-20260909/`:排除集是 split 的纯函数,
   而 arm B 现在什么都还没产生,所以现在是「prospective」唯一成立的时点。
   要核对的是 `scene_disjoint_outcome_sensitivity.spec_ids_sha256` =
   `8c638fb3efd86258fbc6e51146e18086e5c517acaa48d017a208a4e24bcb5264`,
   以及 `summary` 的 outcome 64 / scene-disjoint 54 / probe bank 排除 2 /
   train 132-of-132。那份 NOTE 里还记着一件顺带验到的事:主 config 排满
   132/132,所以保守口径和实际日程口径重合,排除集不依赖日程种子——
   pilot 不是这样(120/132)。
3. `configs/v4_decoupling_main_20260908.yaml` 至今仍未纳入 git(主树 `??`)。
   manifest 会冻结它的 sha256,所以两个运行用的是不是同一份 config 事后查得出来,
   但文件本身没有版本记录。

---

## 2. E4 的三条命令 —— 已跑完,3/3 复现

**结果在 `review-packets/cross-model-20260908/`。** 注册的三个模型全部复现,
PREREG E4 达成;Janus-Pro-1B 单列不计入。要带进论文的两条限定:
showo_v1 的效应小一个数量级(−0.079 对 −0.318 / −0.271);
showo2_1p5b 与 janus_pro_1b 在两个条件下的弃答率有显著差异
(p=6.1e−5 / 1.8e−4),两个 n 都必须报。第三张表 `blind only` 列的含义
见该目录的 `NOTE.md`。下面的命令原样保留,是复跑用的。

预检已在 CPU 上过了(偏离 2):Show-o v1 与 Janus-Pro 都对不同的图给不同的答案。
评分器缺陷已修,所以现在跑出来的行才是可用的。

**从主树跑,但调 worktree 的脚本。** 两边各有对方没有的东西:
分支的 `showo_v1.yaml` / `janus_pro_1b.yaml` 和修好的 grader 只在 worktree,
`envs/` 和 `runs/` 只在主树,而 worktree 没有 `envs/`。
驱动现在按「代码和配置跟脚本走,数据和环境跟工作目录走」解析,所以这样可行:

```
cd "H:/Xiyao_Wang/062_Can You See What You Drew Visual Self-Confirmation"
SELFSIGHT_MODEL_ROOT="H:\selfsight-models" envs/core/python.exe -u H:/Xiyao_Wang/062_armB/scripts/v4_cross_model.py preflight --device cuda:0
```

`observe` 与 `report` 同样写法。`--out` 给 report:
`review-packets/cross-model-20260908/cross_model.json`。

`observe` 会为每个模型起它自己 config 里 `environment` 指定的解释器,
子进程的 `PYTHONPATH` 由 `child_env()` 钉到 worktree 的 `src`,
所以 `v4_run_pipeline.py observe` 用的是修好的 grader,不是主树的旧的。
`HF_HUB_OFFLINE=1` 也在那里设,缺快照会响亮地失败而不是去下载。

判定按注册的三个模型算,Janus 单列不计入——这条在 `stage_report` 里,
`tests/test_v4_cross_model.py` 有对照测试。

**先跑 E4 还是先跑 arm B**:E4 只要一张卡、纯推理、几小时;arm B 要两张卡、
约 52 小时。主运行一释放,先把 E4 跑完再起 arm B,总墙钟不变而 E4 提前落地。

---

## 3. 合并顺序:现在被主运行挡着,但 E4 不再需要它

原先这里写「必须先合并再跑」。**不再成立,而且方向反了。**

不需要:`v4_cross_model.py` / `v4_context_curve.py` 给子进程显式传
`PYTHONPATH`(见 §2),父子两端都读 worktree。修这个之前是真的会错——
父进程读分支、`observe` 子进程读主树的旧 grader,静默。

被挡着:`run_decoupling_pilot.py` 的 `run()` **每个阶段**都重核 SOURCES 摘要,

```python
if any(digest(ROOT / name) != expected
       for name, expected in self.frozen["source_sha256"].items()):
    raise RuntimeError("Experiment source changed during execution; review before resume")
```

`src/selfsight/v4/train.py` 和 `scripts/v4_train.py` 都在 SOURCES 里、
也都在分支里改了。所以**主运行重启之前合并,重启会被拒**,
除非带 `--accept-code-update`。分支后来又往 `SOURCES` 里加了
`scripts/v4_scene_audit.py`(§1),这条只会让上面那句更硬:主运行的 manifest
里根本没有这个键,合并后再 resume 一定要 `--accept-code-update`。

顺序因此是:先裁定主运行(见 RUN-HALTED-20260908.md)→ 重启并跑完 →
再合并 → 再起 arm B。E4 不在这条链上,随时可跑。

「再合并 → 再起 arm B」这一步现在是**硬依赖**,不只是整洁:supervisor 用
`ROOT/envs/<环境>/python.exe` 起每个子进程,而 worktree 里没有 `envs/`,
所以 arm B 只能从主树启动,也就只能在合并之后。

合并前提不变:另一个 agent 的 `run_decoupling_pilot.py` 改动先提交。

---

## 4. 不做的事

- 不因为 `nvidia-smi` 显示 0% 就去抢卡。
- 不在主运行还没写出 `checkpoint_metrics.csv` 之前看它的任何数值。
- 不为了让 E4 凑够 2/3 而增加模型。
- 不动 `runs/` 下的任何研究工件;要清理就归档到 `_archive/`。
