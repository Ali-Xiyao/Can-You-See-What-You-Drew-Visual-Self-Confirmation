# 执行计划:剩下的每一步,以及每一步在等什么

**2026-09-08 写。** 预注册在 `PREREG.md`,本文不改写它,只把它落成命令。
状态用「能不能现在跑」分类,不用「重要不重要」。

---

## 0. 现在的真实状态

| | 状态 | 卡上 |
|---|---|---|
| E3 arm A + C(`decoupling-main-20260908`)| **在跑**,**round 1 / 11 训完**,正在测它的 step-00016;09-08 07:33 起,已用 **23.0 h**。按实测吞吐外推 **~92 h,余量不到 4%**(§0.5)| 两张都占,但卡 0 每块空转约 37%(§0.5)|
| E3 arm B(`naive` + `blind_self` 配对新跑)| 代码就绪(含新的 audit stage,§1),**等卡** | — |
| E4 三个新模型作答 | **已完成**,3/3 复现,见 §2 | 已释放 |
| 评分器缺陷 | **已修**(worktree 99b7c0f,登记为偏离 3) | 无 |
| arm B 的轮次恢复路径 | **已修**(worktree 93379c8) | 无 |
| 排卡的空载基准(§0.3)| 脚本就绪并已预注册判据,**等两张卡都空**。窗口唯一:主运行结束到 replicate 1 启动之间,约 1.5 h,换 55 个块的排期收益(§0.5)| — |
| seed 数 | **已定:5 seed**(2026-09-09;偏离 9 取代 8.1)。取值定死 20260906 / 20260907 / 20260909 / 20260910 / 20260911(跳过 20260908,§3 已占)。增量 4 次运行,304–352 h | 等 A800 |
| `training.seed` 旋钮 | **已实现并验证**(偏离 8.5;arm B worktree `da5ce71`):10 个测试先验红、10 个变异全杀。split / 评测 latent 归 `partition_seed`,轮次调度 / LoRA init / 优化器 RNG 归 `training_seed`,缺省逐位不变 | 完成 |
| A800 服务器 | 09-09 不可用,**明后天可用**(用户告知)。3 个 seed 靠它并行 | — |
| 主运行源码未提交 | **已解决**(`90431bf`):`scripts/v4_train.py` 与 `src/selfsight/v4/evaluate.py` 的工作树字节与 `run_manifest.json` 的 launch 摘要逐位相同,却不在任何分支上。提交不改磁盘字节(已复核 23adcc98 / 636c498a 未变),同时解开了 arm B 合并的唯一阻塞 | 不影响在跑的运行 |
| arm B 合并 | **已预演并验通**(探针 worktree `H:/Xiyao_Wang/062_mt`):唯一冲突在 `scripts/v4_train.py` 一处,解法见 §3.1;`test_v4_training_seed` / `test_v4_evaluate` / `test_v4_verify_replicates` / `test_v4_pilot_supervisor` / `test_v4_blind_self_arm` 共 103 通过 2 跳过 | — |
| E3 端点 1 最终分析脚本 | **已写并验通**:`src/selfsight/analysis/endpoint1.py` + `scripts/v4_e3_endpoint1.py` + 29 个测试,**20/20 变异全杀**。按偏离 6.4 的 20000 / 20260908(不复用 report 的 2000 / 20260906),按偏离 9.3 的五 seed 精确符号检验(拒绝对 5 个值 bootstrap、拒绝非注册 seed 集冒充确证),按偏离 10 的成对剔除。在合成的五 seed 夹具上端到端跑通 | CPU,不占卡 |
| bootstrap 种子/次数不一致 | **已登记,代码不动**(偏离 6.4):在跑的 report 是 20260906 / 2000,§3 注册的 20260908 只活在已降级的 breakpoints 模块里。端点 1 的最终脚本按 20000 / 20260908 新写 | 不动在跑的脚本 |

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

## 0.5 96 h 只剩约 4 h 余量,卡 0 每块空转约三分之一(2026-09-09 07:10 实测)

§0.1 的外推是 `11 x 1.77 + 12 x 3.95 ~ 67 h`。**两个因子实测都偏低。**
下面的数全部来自 `supervisor-relaunch-20260908.log` 的 stage 起始时刻,
只读,没碰运行。

注意日志有两行被两个线程同时写而粘在一起
(`naive.step-00008.generate2026-09-08`)——被吃掉的是 rfo_gold 的
`generate` 起始行,两臂的 generate 本来就同刻启动,所以时刻仍可还原。

| | 注册外推 | round 0 检查点块 | round 1 检查点块 |
|---|---|---|---|
| train | 1.77 h | — | **2.18 h** |
| 块墙钟(= 慢的那条链) | 3.95 h | **5.48 h** | **~5.3 h**(在跑) |

一轮的真实代价是 **2.18 + 5.3 ~ 7.5 h**,不是 5.7 h。

### 到期日推算(**08:22 有了第一个干净的整轮周期,重算过**)

`round-001.train` 01:16:57 → `round-002.train` 08:22,**整轮 7.06 h**
(train 2.18 + 块 4.89)。这是第一个可比的周期:round 0 那一轮夹着基线评测
和缺 audit 的停机,不能拿来外推。块本身从 5.48 h 降到 4.89 h。

此时 elapsed 24.8 h,还剩 9 轮。用两个节奏夹一下:

| 节奏 | 外推 | 完成 | 余量 |
|---|---|---|---|
| round 1 周期 7.06 h | 24.8 + 9 x 7.06 = **88.3 h** | 09-11 23:54 | **+7.7 h** |
| block 8 节奏 7.66 h | 24.8 + 9 x 7.66 = **93.7 h** | 09-12 05:17 | **+2.3 h** |

上限 96 h,截止 2026-09-12 07:33。**余量在 +2.3 h 到 +7.7 h 之间**,
比本节初稿写的「不到 4 h」宽,但乐观的那头也只有 8%。
下一个轮边界(`round-003.train`)会把区间再收窄一次。 96 h 是停止线不是预算(§0.1 已裁定),
`run()` 到点在下一个 stage 边界 raise,主运行会停在半轮上。
**这不需要现在做任何事**——没有合规的加速手段可以用在正在跑的运行上,
§0.2 三条理由仍然成立。但它把「主运行完成」从默认结果变成了约 4% 余量的事件,
所以「主运行完了就自动合并起 seed 1」这条指令要带一个前置判据:
`state.json` 的 `status` 必须是 `pilot_complete` 且 `completed_rounds == 11`,
`canary_complete` 或 traceback 都不触发。

### 卡 0 的空转,量出来了

一条链按臂钉死在一张卡上(§0.2),两条链在 `chains()` 的 `join()` 汇合,
所以块墙钟取大者,快的那条打完就干等。round 1 块里:

| | naive(cuda:0)| rfo_gold(cuda:1)|
|---|---|---|
| generate | 76.6 min | 102.6 min |
| detect qwen3vl | 38.4 min | 85.1 min |
| detect internvl | 30.0 min | 在跑(06:35 起)|
| crop + verify | 4.2 min | — |
| gradient | 05:56 起,已完 | — |

07:10 查 `nvidia-smi`:**卡 0 = 0 MiB / 0% / 9.2 W,卡 1 = 16965 MiB / 38% / 99 W。**
naive 整条链约 06:30 就打完了,rfo_gold 还剩 internvl + crop + verify + gradient,
约到 08:20。**卡 0 这一块空转约 1.8 h,占块墙钟约 37%。**

这不是「没用上两张卡」——两张卡都在用,是**两条链的长度不等**,
而不等的原因就是 §0.2 那条:整条链(含传输受限的 detect)跟着臂走,
rfo_gold 那条整条压在 gen3 x4 的卡 1 上。

### 对 5 个 replicate 的意义

一个 replicate 自己就要两张卡:`ARM_CARDS` 定死两张、`__init__` 拒绝别的 arity,
而 `train` 阶段 backbone 在 cuda:1、裁决梯队在 cuda:0 且拒绝同卡(STATUS 37)。
**所以 5 个 replicate 在 2x3090 上只能串行**,并行不是排期问题,是结构问题。

能省的是每个块里那 37%。按上表,5 个 replicate x 11 个块 = 55 个块,
每块哪怕只收回 1 h 就是 **55 h**。这正是 §0.3 已注册的判据要裁的事,
而它要求两张卡都空——**唯一的窗口就是主运行结束到 replicate 1 启动之间**。
`card_benchmark.py` 按 N=64 x 2 检测器 x 3 组合估约 1.5 h,
换 55 h 的期望值,先测再起。

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

**启动前的三道闸,按顺序**(2026-09-09 补,见 §0.5):

1. **主运行必须是正常跑完的**。`runs/v4/decoupling-main-20260908/state.json`
   的 `status` 要是 `pilot_complete` 且 `completed_rounds == 11`。
   `canary_complete`(96 h 停止线触发)或 traceback 都**不**触发自动启动——
   96 h 余量只剩不到 4 h,这不再是形式上的判据。
2. **合并**(§3),然后跑下面第 0 条的 `training_seed` 源码闸。
3. **两张卡都空的那一刻先跑 §0.3 的空载基准**,按预注册判据决定 detect 排期,
   再起 replicate 1。这一步约 1.5 h;它换的是 5 x 11 = 55 个检查点块的排期,
   §0.5 估每块可收回约 1 h。基准要求两卡皆空,而这是唯一的窗口:
   replicate 1 一起,两张卡又满了,直到 5 个 replicate 全部跑完。

**启动**(等主运行释放两张卡、并且 §3 的合并做完之后):

**在主树跑,不要在 worktree 跑。** `run()` 拼的是 `ROOT/envs/<环境>/python.exe`,
`ROOT` 是脚本自己所在的树;`H:/Xiyao_Wang/062_armB` 底下没有 `envs/`,
在那里启动会在第一个 stage 就找不到解释器。这也是合并必须排在启动前面的原因。

**这条命令现在跑 5 次,不是 1 次**(偏离 9:5 个 seed)。五份 config 已备好,
彼此只差 `profile` 与 `training.seed`,顶层 `seed` 一律 20260906——
所以五次共享同一份 split、同一批 64 个 outcome spec、同一批固定 latent。
已核对:五份的 split 输入摘要完全相同,五个 `training.seed` 互不相同。

```
envs/core/python.exe -u scripts/run_decoupling_pilot.py --outdir runs/v4/e3-s20260906 --config configs/v4_e3_replicate_s20260906.yaml --arms naive blind_self
envs/core/python.exe -u scripts/run_decoupling_pilot.py --outdir runs/v4/e3-s20260907 --config configs/v4_e3_replicate_s20260907.yaml --arms naive blind_self
envs/core/python.exe -u scripts/run_decoupling_pilot.py --outdir runs/v4/e3-s20260909 --config configs/v4_e3_replicate_s20260909.yaml --arms naive blind_self
envs/core/python.exe -u scripts/run_decoupling_pilot.py --outdir runs/v4/e3-s20260910 --config configs/v4_e3_replicate_s20260910.yaml --arms naive blind_self
envs/core/python.exe -u scripts/run_decoupling_pilot.py --outdir runs/v4/e3-s20260911 --config configs/v4_e3_replicate_s20260911.yaml --arms naive blind_self
```

**0. 先查这一条,它排在所有检查前面。** 合并前的 `scripts/v4_train.py` 不读
`training.seed`,读不到不会报错——它会**静默忽略**,五个 replicate 于是全部
训练在 20260906 上,而目录名、config 和日志会一致地声称它们是五个 seed。
那是本计划里最难事后发现的一种失败:

```bash
envs/core/python.exe -c "import pathlib,sys; sys.exit(0 if 'def training_seed(' in pathlib.Path('scripts/v4_train.py').read_text(encoding='utf-8') else 'no training_seed: every replicate would train on the same seed')"
```

退出码非 0 就停,回去做 §3 的合并。跑完之后再验一次,这次验产物而不是源码:
五个 run 的 `rounds/round-000/done.json` 里 `initialization_seed` 必须是五个
不同的值,而 `split.json` 的 `digest` 必须五个完全相同。**两个都对才算五个 seed。**


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
   train 132-of-132。

   同一条命令还会写出 `audit-splits/scene_representatives.json`(worktree
   `8019e3f` 起)。这个文件报告一直会读、读到就多一条行间独立的敏感性分析,
   pilot 有,而在此之前**树里没有脚本能造它**——它是手工做的。arm B 的这份
   应当是 52 条,`spec_ids_sha256` =
   `7f44b386ed6490544b49c411b17783e1e2edd34b2ffac8eadf6f5e6476e5a2c1`。
   新写的构造函数对着 pilot 那份手工产物验过,除 `created_at` 外逐字段相同。那份 NOTE 里还记着一件顺带验到的事:主 config 排满
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

**2026-09-09 01:1x 空跑过一次合并**,`git merge-tree --write-tree HEAD
codex/blind-self-arm-20260908` 干净出树(`a0169af`),没有冲突。会动 26 个文件、
+4986/−114,其中落在 `SOURCES` 里的正好四个:`run_decoupling_pilot.py`、
`v4_scene_audit.py`、`v4_train.py`、`src/selfsight/v4/train.py`——
和上面那段说的一致。空跑只用 `merge-tree`,没有建分支、没有动工作区。
两边还会再动,合并前要重跑一次。

---

### 3.1 重跑了,现在有一个冲突,解法已经验过(2026-09-09 07:2x)

上面那次空跑之所以干净,是因为当时主树的 `scripts/v4_train.py` 还是脏的、
不在 `HEAD` 上,`merge-tree` 比的是提交而不是工作树。把它提交之后
(`90431bf`,理由见下)再跑,**恰好一个冲突**:

```
git merge-tree --write-tree --name-only paper/iclr-2028 codex/blind-self-arm-20260908
CONFLICT (content): Merge conflict in scripts/v4_train.py
```

冲突就一处,在 `generate` 里造评测 latent 的那几行,两边改的是同一个表达式:

- `paper/iclr-2028`:R 次抽样,键从 `prompt_id` 变成 `(prompt_id, draw)`,
  多传一个 `candidate_index`,种子仍写 `int(config["seed"])`。
- `codex/blind-self-arm-20260908`:键不变,种子改成 `partition_seed(config)`。

**解法是两边都要,不是二选一**:保留 R 次抽样的键和 `candidate_index`,
种子取 `partition_seed`。理由写进了代码注释——这批 latent 是配对设计里配对的
那一半,五个 replicate 必须抽到同一批图,所以它归 partition 不归 training。
`evaluation_seed` 的 `candidate_index: int = 0` 有默认值,而且它定义在
`evaluate.py` 里、arm B 根本没动那个文件,所以签名在合并后仍然成立。

已在探针 worktree `H:/Xiyao_Wang/062_mt` 上真做了一次合并并按此解决,
然后跑测试(主树的解释器 + `PYTHONPATH` 指向探针树,worktree 里没有 `envs/`):
`test_v4_training_seed` / `test_v4_evaluate` / `test_v4_verify_replicates` /
`test_v4_pilot_supervisor` / `test_v4_blind_self_arm` **103 通过 2 跳过**。
其中 `test_every_seed_a_call_site_hands_out_comes_from_one_of_the_two_helpers`
是这个解法的守门人:它按 AST 检查每一处传出去的 seed 都来自那两个 helper,
所以「合并时顺手留下 `int(config["seed"])`」这种解法过不了。
全量套件另有一处失败,`test_v4_checkpoint_probe.py::test_end_to_end_...`,
原因是探针 worktree 没有 `runs/`(未跟踪产物),与合并无关。

**探针 worktree 是一次性的,合并真正落地后删掉。**

### 3.2 为什么主运行的源码先要提交(`90431bf`)

`run_manifest.json` 冻结了 12 个 SOURCES 文件的逐文件 sha256。其中
`scripts/v4_train.py`(23adcc98)和 `src/selfsight/v4/evaluate.py`(636c498a)
与工作树逐位相同,**而与任何分支上的任何提交都不同**——主运行跑的这份代码
不在历史里。它的测试也不在。

提交不改磁盘:`git add` 规范化的是索引里的行尾,不是工作树,
提交前后两个摘要复核过没变,所以每个 stage 重新读盘起的子进程读到的还是同一份。
一并进去的还有 `analysis/breakpoints.py` 的 `break_support` 和
`tests/test_v4_evaluate.py`——`evaluate.py` 在模块级 import 了 `break_support`,
拆成两个提交的话中间那个 import 不了。

先例是 `4fee432`(把一直在跑却没提交的并发重写提交掉)。**第二次发生,
这本身就是发现**:supervisor 会在每个 stage 重核源码摘要、resume 时拒绝改动,
但没有任何东西拒绝「从一棵从未提交过的树上启动」。

---

## 4. 不做的事

- 不因为 `nvidia-smi` 显示 0% 就去抢卡。
- 不在主运行还没写出 `checkpoint_metrics.csv` 之前看它的任何数值。
- 不为了让 E4 凑够 2/3 而增加模型。
- 不动 `runs/` 下的任何研究工件;要清理就归档到 `_archive/`。

## 5. 一个会在论文里露头的不对称,以及它带来的一个陷阱(2026-09-09 登记)

**arm B 会有两条敏感性分析,主运行(A + C)不会。** `v4_decoupling_report.py`
只从 `audit-splits/` 读 outcome 侧的排除集,而那个位置按定义只接受
prospective 的 audit。arm B 现在有(supervisor 在第一个 round 之前就造,§1);
主运行没有,它的 audit 是 09-08 停机时补造的 retrospective,放在
`retrospective-scene-audit/`,只服务梯度探针的排除,不服务 outcome 子集。

**补不上,而且拒绝补是对的。** 给主运行补一份 prospective 的,意思是「这个
子集是在这个运行产生任何东西之前定下的」——那句话现在已经不真。
`v4_scene_audit.py --prospective` 会当场拒绝,`Pilot.resolve_audit()` 也会。

### 能诚实做的一件事(**这是他的决定,我只登记选项**)

把同一套排除集用在主运行上,**显式标注为 post hoc**。支持这样做的事实:

- 排除规则**不读任何 outcome 值**。造 audit 的脚本不打开 `evaluations/`,
  而且它用白名单——运行目录里出现任何不在白名单上的东西就拒绝签 prospective。
- 集合是 split 的确定性函数,两个运行共用同一份 split(§1 的核对项 1)。
  `review-packets/armB-audit-preimage-20260909/` 里那份**就是用主运行自己的
  `split.json` 算出来的**。
- 报告不信任这个文件:representative 规则它自己按父 audit 重算一遍,
  outcome 总体要和 `split.json` 完全一致,不符就拒。
- 这套集合 **2026-09-09 已提交进 git**(`67fad38`)。那时主运行 12 个检查点
  只出了 2 个,所以对后 10 个检查点它确实先于数据。

剩下唯一的污染通道是「人选择什么时候去算这个 audit」。这个通道关不掉,
所以要**标注**,不是要藏。论文里的说法应当是:arm B 的这两条是预注册的,
主运行的同两条是事后的、集合与 arm B 逐字节相同、且冻结于其 12 个检查点中
的第 3 个之前。

### 陷阱:那两个文件不要拷进主运行目录

`review-packets/armB-audit-preimage-20260909/` 里的 `scene_overlap.json` 和
`scene_representatives.json` 拷进 `runs/v4/decoupling-main-20260908/audit-splits/`
会**通过报告的每一道检查**——config_sha256 对得上、split_sha256 对得上、
`created_before_any_outcome_evaluation_artifact` 是 `true`——然后把一份事后的
子集当成开跑前的冻结写进结论。白名单只在**造**的时候查,读的时候不查。

这个洞是我自己造出那份产物之后才存在的,所以补上了守卫(worktree `6f71820`):
audit 的 `provenance.run` 记着它在哪个运行目录里造出来,报告现在核对目录名。
按目录名而不是全路径比,是因为 outcome 存在之后 audit 造不出第二份,
搬一次目录不该让它读不了。没有记 `run` 的 audit 照读——pilot 那份早于这个字段,
而它是唯一一份手工核对过的 prospective 冻结。4 个变异全杀,测试先验红。

---

## 6. 发表判据

**不在这里。** `ICLR-2028.md` 2026-09-08 就写了同一件事(三种结局、单 seed
风险、六条审稿人会打的点),而且写在 E3 出任何数字之前。我 2026-09-09 在这里
又写了一遍四格判据表——那是重复,README 明写过不许再开一份计划类文档,
这条规矩对判据文档同样成立。已删,只在原处追记这两天新出的三条
(见 `ICLR-2028.md` §8)。
