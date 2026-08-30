# STATUS

**唯一入口。** 想知道「现在什么成立」只读这一页；要证据细节再去 `docs/EVIDENCE_LOG.md` 对应节。

最后更新：2026-08-30 23:00 · 分支 `experiment/v2.3-rfo-gold` · backbone `showlab/show-o2-1.5B-HQ` rev `d3a220ec`

---

## 正在跑

| 进程 | 设备 | 输出 | 读取 | 预计完成 |
|---|---|---|---|---|
| PID 3660 | cuda:0 | `runs/v3/pool-box/gpu0` | `data/selfsight-v3-pool/manifests/tier_a_probe.jsonl` | ~01:00 |
| PID 7644 | cuda:1 | `runs/v3/pool-box/gpu1` | 同上 | ~01:45 |

128 prompt/片 × M=16 候选。gpu0 = existence，gpu1 = spatial。完成后跑选择头room 预检。

**这两个目录和那个 manifest 在跑完前不要动。**

---

## 当前成立的结论

按主题组织，不按时间。括号内是证据出处。

### 1. Backbone 已定
`show-o2-1.5B-HQ`（512px 原生）。existence / color / spatial 三族四项指标全达标，程序化 verifier 的盲审 precision = 1.00。就绪门判红**只因为**「主族 ≥4」这条阈值，而该阈值无推导依据——v3 已公开把 floor 降到 3。（§1）

### 2. 负对照全红，不得提回主线
Show-o v1 1.3B、Show-o2-1.5B rank-1（432px）、Show-o2-7B 均判红。（§2）

### 3. 工程闭环已打通
LoRA 训练、E1 反事实、checkpoint 曲线在本地全部跑通。**单 seed 下没有机制信号**——这是工程可行性结论，不是机制结论。（§3）

### 4. 候选供给是真实瓶颈
随机 K=4 池天然 informative 率仅 **21%**。有界候选库搜索把 balanced_rate 从 0.135 提到 **0.344**（2.5×），但仍低于注册阈值 0.40 → Gate A 判红。失败模式与 v2.x 预期**相反**：不是模型画不对，是 color 太准（每候选正确率 0.949）凑不出 2 个错误候选。（§4, §8）

### 5. 梯度噪声控制原设计有缺陷
「不相交半批余弦」测的量与它要把关的配对跨判据余弦**尺度不同**，产生过两次误导性红灯。已改为 prompt 层配对 bootstrap；半批余弦降级为诊断，不再当门槛。（§5）

### 6. 显示词汇缺陷（2026-08-30 发现）
verifier 把长宽比 0.5–2.0 的四边形都判为 SQUARE，而 v3 manifest 声称用 `box` 措辞却有 **183/256 行**仍写 `square`。实测 **39.5%** 的 verifier-SQUARE 物体长宽比 ≥1.25（中位 1.19，p90 1.62）。方向是**系统性抬高 `oracle − naive`**，即把 Gate C 推向假绿。已修复并加 fail-closed 守卫。（§11）

### 7. 选择头room 的判据（已在看数前注册）
`Oracle@K − Naive@K` > 0.05 → go；< 0.03 → stop；之间 → 扩样本。每候选正确率 p 与天花板的关系：p≈0.35 时天花板最大（0.47），p→0 或 1 时坍缩。已测：spatial p=0.194（天花板 0.384，接近最优），existence p=0.864（天花板 0.136，被压缩）。

---

## 已作废 / 已被取代

| 结论 | 状态 | 原因 |
|---|---|---|
| §10 三物体难度标定 | **作废** | 双重混淆：把 3 物体脏词汇数据和 §8 的 2 物体干净数据对比，物体数与词汇同时变化 |
| `runs/v3/pool-ext…` 池扩展运行 | **作废** | 词汇缺陷（§11.4）；且与 §8 基线不可比（§11.6） |
| stage-1 头room 数（0.754 / 0.879 / 0.125） | **作废** | 测在作废数据上，须在 `pool-box` 重测 |
| v2.2 就绪审计的 binding / count 淘汰 | **待复核** | 见下方未决问题 1 |

### 不可恢复的损失
`runs/readiness/showo2-1p5b-hq/a3-human-r1-review.csv` 的 49 条人工标注**仍在**，但它们指向的 49 张图位于已删除的 `H:\selfsight-runs\`，**0/49 可访问**。标注无法用于验证任何重建后的 verifier。这是本项目最严重的流程失误：**gitignore 的 `runs/` 里的研究工件没有任何备份**。

标注本身已 `git add -f` 强制入库（`runs/readiness/showo2-1p5b-hq/a3-human-r1-review.csv`），不再依赖 gitignore 目录存活。

---

## 未决问题

1. **binding 是否该重建。** v2.2 把 binding 判出局（verifier precision 0.778），但那 2 条分歧**全部**是 `What color is the square?` ——同一个词汇缺陷。剔掉后是 7/7。相对地 count 的 6 条分歧里 4 条与 square 无关（数圆、数三角形都错），剔掉词汇因素后 precision 仍 ≈0.50，是真坏。size 独立失败（coverage 0.20 / observation 0.60）。
   → 需要决定：是否按 §11 的先例「旧记录保留 + 缺陷登记 + 重新测量」来复核 binding。
2. **Gate C 是否可达。** 今晚预检给出。
3. **人工标定必须重做**（图已丢）。重做时要改三点：图留在仓库内、用 `box` 措辞、把「图看不清」与「你的词和我看到的对不上」拆成两个选项。
4. **`data/selfsight-v3-cal` 未在 box 词汇下重生成**，§10 的 `objects_per_scene` 决策仍未测。

---

## 锁定的选择

改动这些需要新的明确决定，不能顺手改。

| 项 | 值 | 依据 |
|---|---|---|
| 主 backbone | `showlab/show-o2-1.5B-HQ` rev `d3a220ec`（512px 原生） | §1 |
| 主问题族（目标） | existence / color / spatial | 四项指标全达标，盲审 precision 1.00 |
| **当前作用域** | **existence / spatial** | Gate A：color 供给 0.156（§8） |
| hard tier | count（单独报告） | precision ≈0.50，剔除词汇因素后仍真坏（§12.1） |
| **待复核** | **binding** | 原判 0.778 全部由 `square` 措辞造成，剔除后 7/7（§12.1） |
| 排除 | absolute size | 生成覆盖 0.20 / observation 0.60，与词汇无关 |
| 冻结负对照 | Show-o v1、Show-o2-7B | 不得静默提回主线 |
| 检测器（构造 `g_rfo`） | **Qwen3-VL-8B-Instruct**（冻结，不参与训练选样） | 检测器/训练器解耦。**必须先过自己那轮 120 图参考观察审计**——v2.1 那份只覆盖 Qwen2-VL-2B，不继承。后端拒绝 Thinking 变体 |
| 弱观察者消融臂 | Qwen2-VL-2B（已审计） | proposal §7.6 |
| 视觉词汇 | `box`（长宽比 0.5–2.0） | v2.3 盲审 28/28。v3 曾丢失重写已修复（§11）；v2.2 就绪审计从未有过该词汇（§12.1） |
| 主 seed | `20260901` | — |
| Gate A/B/C/D 判据 | proposal §9 | 唯一规范来源，本页不复制 |

## 决策树

```
L1 Gate A ──红──> 能力地板报告，项目转向
   │绿
   L2 Gate B ──红──> 加 pool 数；仍红则主张①退熵类信号，②③继续
   │绿
   L3 Gate C 预告 ─重叠─> 不上云，诊断修正目标/样本权重/梯度探针
   │分离
   C0 canary ──红──> 定位平台差异
   │绿
   C1 Gate C ──红──> ★ 项目失败条件：停止扩展，回到目标函数诊断
   │绿
   C2 Gate D ──红──> 降级：①无预警价值 → 重心回②③
   │绿                      ②无法归因 → RFO 改写为纯经验方法
   C3 + 追加 seed
```

**尚未实现的运行器**：`scripts/run_main_seed_e2.py`（从 `run_formal_e2.py` 拆出单 arm / 单 seed / 单卡，供 C1/C2 并行启动）。其余 L0–L2 的脚本均已存在。

**原则**：所有能杀死项目的实验都在本地跑完，再碰 A800。v2.x 的教训是把最贵的东西排在最前面——先审计观察者、再审计生成，最后才发现**根本没有可选的东西**（79% 的候选池是退化的）。

### 已注册的规则（在看到结果之前定的，不能事后改）

| 规则 | 内容 |
|---|---|
| 种子域分离 | 候选搜索 `700000000+`，梯度/dropout `900000000+`，两域不相交。代码里由 `v3.bank.assert_disjoint_seed_domains` 强制 |
| 候选库上限 | M=16。**只允许提到 24 一次**，且必须在报告中记录这次提升；仍不过则停 |
| 自然采样对照 | 每个 bank 的前 K 个候选即未筛选抽样，因此自然率与库率在**同一批 prompt** 上比较 |
| Gate B 资源 | 18.4M LoRA 参数 × 128 prompt × 3 准则，float16 memmap ≈ 14 GB 磁盘；Gram 分块峰值内存 < 1 GB |
| Gate B 现值参照 | identical 选择余弦 1.000（要求 ≥0.999）；不相交半批余弦 [−0.082, 0.322]，**只报告不当门** |
| L3 判据 | Gold 明显高于 Naive → 上云；两条曲线重叠 → **不上云**，回目标函数诊断；Gold 低于 Naive → 同上，且优先怀疑 SFT 目标对「正确候选」的加权方式 |

L3 这一步能省下 120–200 A800 小时。参照：v2.2 探索闭环里 Naive +4.17 pt、RFO-Self +2.08 pt（单 seed、K=2、10 步，只是工程证据）。

## Hard Stops

- 未经明确批准，不下载任何新模型或权重。需要时先暂停并报告 model/revision、磁盘占用、用途、替代方案。
- 不重写、不重新决定 v2.1 / v2.2 / v2.3 的已冻结证据。红色报告保留为不可变记录。**发现仪器缺陷时的正确做法是：旧记录原样保留 + 缺陷作为新证据登记 + 重新测量并注明来源**（先例见 §11）。
- Show-o v1 与 Show-o2-7B 是冻结负对照，不得静默提回主线。
- 不把 Qwen2-VL / Qwen3-VL 的输出当作「统一 backbone 能看见自己的画」的证据；它们始终是冻结外部观察者/检测器。
- 不在 Gate C 红之后靠加算力、加 seed 或换 backbone 来抢救。
- 单 seed 的描述性轨迹不得写成显著性结论。
- 候选库平衡池只用于梯度探针；训练与 external correctness 曲线必须用自然采样池，论文中显式区分。
- **绝不把已修正的探针 resume 进被隔离的输出目录**：`run_bank_probe` 按 `packets/{index}-{scene_id}.json` 断点续跑，而词汇修复不改变 scene_id，会静默复用旧图。
- **不删除 `runs/` 下的研究工件**（它们不在 git 里，删了就没了）。归档到 `_archive/`，不要 `rm`。

---

## 目录导航

四份文档各有一个职责，互不重复。**不要再新增计划类文档。**

```
Can You See What You Drew….md   规范 —— proposal v3.0，含 Gate A–D 判据
STATUS.md                       现状 —— 本页
docs/EVIDENCE_LOG.md            证据 —— 全部实测（§1–§12），只记录不重解释
docs/RUNBOOK.md                 操作 —— 3090 / A800 手册
docs/source-notes/              改变过设计的用户输入原件

src/selfsight/v3/               当前代 —— 全部新工作写在这里
src/selfsight/v23/              v2.3 冻结实现，不要修改
src/selfsight/{data,analysis,backbones,observers,pilot,training}/  共用基础设施

runs/readiness/                 §1 §2 就绪审计证据（冻结）
runs/exploratory-post-gate/     §3 探索闭环
runs/v2.3-rfo-gold/             §4 v2.3 机制门
runs/v3/                        §8–§11 当前代
runs/canaries/ review-packets/ gate-minus-1/   环境 canary 与盲审包
runs/_archive/                  已归档：mock-pilot ×3、audits、host、manifests、wandb
data/_archive/                  已归档：selfsight-v1、v2.3 失败模板
```
