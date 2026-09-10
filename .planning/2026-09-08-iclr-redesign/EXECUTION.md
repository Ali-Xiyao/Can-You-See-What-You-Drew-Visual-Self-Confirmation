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
| `training.arms` 是死配置 | **已修**(`staging/arm-b-merged` `1eccc20`):五份 replicate config 都写了 `training.arms`,而 supervisor 的臂集只来自 `--arms`(缺省是注册的 naive+rfo_gold)。漏了这个 flag 就会在 `e3-s20260906` 目录下训 naive+rfo_gold,config、日志、目录名一致地说它是 arm B,而 manifest 只在 **resume** 时才发现。现在构造时就拒绝矛盾,顺序也查(ARM_CARDS 是位置相关的)。4 个测试,拒绝那两个已对旧代码验红 | 这是本项目第三次同型失败 |
| 五份 replicate config | **已修字面缺陷**:`s20260906` 的头注释块重复了一遍(生成脚本没剥掉继承的头),`s20260906` 与 `s20260907` 都标着「Replicate 2 of 5」,头注释还漏数了 `training.arms` 这处差异。现在五份各 193 行,编号 1–5,与主 config 的差异是 `profile` + `training.{arms, seed, seeds}` | — |
| E3 端点 1 最终分析脚本 | **已写并验通**:`src/selfsight/analysis/endpoint1.py` + `scripts/v4_e3_endpoint1.py` + 29 个测试,**20/20 变异全杀**。按偏离 6.4 的 20000 / 20260908(不复用 report 的 2000 / 20260906),按偏离 9.3 的五 seed 精确符号检验(拒绝对 5 个值 bootstrap、拒绝非注册 seed 集冒充确证),按偏离 10 的成对剔除。在合成的五 seed 夹具上端到端跑通 | CPU,不占卡 |
| 启动闸把 96 h 停止线认错了 | **已修**(`v4_e3_launch_preflight.py` 与本文 §1 的判据文字一起改):`canary_complete` 在这份代码里只表示 supervisor 是带 `--through-round N`(N < `training.rounds`)起的,停在被告知的地方——STATUS §43.15 早就这么记着,是我在 §1 和闸里把它写成「96 h 停止线触发」。真的 96 h 停止线走 `run()` 里那次 `raise`,异常从轮循环出去,走不到第 380 行的终态 `state()`,`state.json` 永远停在 `running`;stage traceback、被 kill、重启留下的文件一模一样。三种情况裁定相同(拒绝)、现场不同,所以现在按 supervisor pid 还在不在,把「停了」和「在跑」分开报。**存活探针不能用 `os.kill(pid, 0)`**:Windows 上 CPython 的 `os.kill` 对非控制台信号是开进程 + `TerminateProcess`,那条可移植写法会杀掉它要保护的运行;用 `tasklist`,并有一条 AST 测试钉住。13 个测试,3 条对旧代码验红,8/8 变异全杀 | CPU,不占卡 |
| 端点 2 **算不出来** | **已登记(偏离 11),数据待补**:端点 2 要盲条件的判别 gap,而 `src/selfsight/v4/evaluate.py` 的逐 checkpoint 评测只调 `observe_naive`,三个臂一致地记的是**带 prompt** 的分(`observe_rfo` 只在 `train.py` 的选择步用)。补测协议已在偏离 11.3 写死:同 backbone、同 adapter、同图、同题,题面不套 `PROMPTED_PREAMBLE`,只测 candidate 0 那 64 张,落在 review-packet 里不写 run 目录。图和每轮 adapter 都存着,所以是**重新观察**不是重训。按实测 1.4 s / 4 题估,每个 run 约半小时观察 + adapter 装载 | 等卡,GPU 轻 |
| 端点 2 口径的另两处 | **已登记(偏离 11.1 / 11.2)**:(1) `s_select` 打分的是 candidate 0,而 `prompt_recital.py` 的标签字典逐行覆盖、R=4 时留下 candidate 3——主运行上两种标签 64 个 prompt 只有 41 个一致,gap 差 +0.056~+0.094,**比 +0.042 还大且会翻符号**;pilot 是 R=1 所以看不到。(2) 端点 2 假设里引的「pilot +0.047 → 0.000」在它引的脚本上印不出来:实际是 +0.070 → +0.047/+0.046,不塌,数据也没动过。判定条件是斜率检验,两条都不改判定,**但动机变弱了,要写进正文** | CPU |
| 端点 1 会在真实数据上崩 | **已修(偏离 12)**:`verified.summary.json` 有 `skipped_no_detection`——检测器什么都没找到时那张图**不产生行**。偏离 10 只覆盖「行在、没裁决」,`load_checkpoint` 对缺 key 是抛异常的。已经发生过一次(主运行 rfo_gold step 16 缺 `(v4a-0001, 1)`,图在盘上,三个检测文件都没有它)。130 个 (arm, checkpoint) 里它还会来,而崩的时机是最后一步。现在缺行按未裁决走成对剔除、逐臂分开记账、分母改从 **manifest** 取(否则跳过的图会连分母一起消失,剔除率永远读 0),两臂 **manifest** 不一致仍然抛。5 个新测试全部对旧代码验红 | CPU |
| E3 三个端点的数据可得性 | **已审计并量化**(`review-packets/e3-data-availability-20260909/`,脚本自证):逐 checkpoint 评测只记**一个条件**(带 prompt)、**一个候选**(candidate 0)、**一套题**(spec 派生)。端点 1 够用;端点 2 缺条件,补 **2.8 h**/5 replicate;端点 3 缺条件+候选+题集,补 **20.3 h**。一次装 adapter 同时补两个 = 10.7 min/(arm, checkpoint),主运行 4.6 h、五个 replicate **23.1 h**(约 400 GPU-h 的 6%)。观察者是 1.5B,**一张卡就够**——不受 replicate「一份占两张卡」的结构约束,所以它是卡 0 空转窗口的候选负载,但能不能和 17 GB 裁决梯队共卡是**要实测的显存问题**。端点 3 的口径**还没登记**,跑之前要追加一条偏离 | 等卡,单卡 |
| 端点 2 与端点 3 的口径写死了 | **已登记(偏离 13)**:注册原文「对每个臂做 gap 对 step 的 OLS 斜率,配对 bootstrap over prompt」有四处两个人会写成两样的地方——配对配的是什么、「显著」读哪个分位、常数走 偏离 6.4 的哪一支、退化重采样怎么办。13.1 七条把它们定死:**一份重采样后的 prompt 多重集,两臂每个 checkpoint 共用**(臂内一个 prompt 只在 gap 的一侧,配不了自己);A 显著为负 = A 斜率分布 97.5 分位 < 0;B 显著大于 A = 差值分布 2.5 分位 > 0;20000 / 20260908 是对 6.4 最终分析支的**一个阅读**,所以登记而不是默默继承;横轴是 `step`;缺 checkpoint 照实报。13.2 把端点 3 钉在 `bsv_selection.py` 上,不新造口径。**判定条件与注册预测一个字没改** | CPU |
| 端点 2 从「算不出来」变成「能算」 | **代码已写并验通**(`0877a62` / `f8103f3` / `feaf56b`):`src/selfsight/analysis/endpoint2.py`(46 测试,**35/35 变异全杀**)+ `scripts/v4_e3_blind_observe.py`(13 测试,11/11 全杀)+ `scripts/v4_e3_endpoint2.py`(8 测试)。补测脚本的题面**从 `s_select.jsonl` 里逐字读回**,不是照 spec 重生成——「同一套题」于是是字节事实,连 `choice_order_seed` 都一样;一条 AST 测试钉住它从不套 preamble(grep 会被模块 docstring 绊倒,那里故意提了 preamble)。两处 off-by-one 会印出一条完整、可信、错的曲线:step 8 是 round-000 不是 round-001,step 0 是两臂共享的未训 base 不是各自第一轮——都钉住了。`--dry-run` 对在跑的主运行解析全通,8 个 (arm, checkpoint) 全部映射到正确 adapter。**还没跑**:两张卡都在主运行上 | 等卡,单卡 |
| 「未做」是有代码路径的 | 偏离 11.3 把「补测跑不完 → 端点 2 判为未做」写成了注册结果,驱动脚本因此有三条拒绝路径且**任何一条都不印斜率**:没有盲测、不足 3 个 checkpoint、盲测分数配不到已裁决的判决(这是过期输出目录的形状)。退出码 **2**——不是 0,免得外层当它有答案继续走;不是 1,它不是崩溃。少一个 seed 就是整个端点少:9.4 的检验是五 seed 的,13.1 第 7 条没给四 seed 的规则,报已完成的那几个就是现编一条 | CPU |
| 检查点数是 12 不是 13 | **我自己的计数错,已登记为新证据**(`review-packets/e3-data-availability-20260909/CORRECTION-checkpoint-count.md`,旧文件一个字节没动):可得性审计把两个补测按 **26** 个 (arm, checkpoint) / 五个 replicate **130** 计价,数字是从预注册的「13 checkpoint × 2 臂 = 26 点」抄的,我没对着 supervisor 核。`run_decoupling_pilot.py:355` 是 `range(11)`,`index == 0` 时额外测一次 step 0,所以每臂 **12** 步(0…88),全run **24**,五个 **120**。盘上的 `step-00000/8/16/24`(三轮之后)正好是 1+3 而不是 1+3+1,本文 §0.1 的「12 次评测」也一直是对的。**只会更便宜**:端点 3 从 20.3 h 降到 18.7 h,两个一起从 23.1 h 降到 21.3 h;每单位的分钟数是实测的,没动。预注册的 26 / 130 **不改**——没有任何注册判据在数 checkpoint | — |
| bootstrap 种子/次数不一致 | **已登记,代码不动**(偏离 6.4):在跑的 report 是 20260906 / 2000,§3 注册的 20260908 只活在已降级的 breakpoints 模块里。端点 1 的最终脚本按 20000 / 20260908 新写 | 不动在跑的脚本 |
| 端点 3 的两个轴也能测了 | **代码已写并验通**(`2963fd6` / `dc41671`):`scripts/v4_e3_selection_observe.py`(16 测试,**21/21 变异全杀**)+ `scripts/v4_e3_endpoint3.py`(16 测试,**15/15 变异全杀**)。补测是 `v4_run_pipeline.py` 的 observe 阶段指向 checkpoint:同 `build_questions`、同 `grade`、同 `has_unnameable` 跳过、同行形状,加上语料版不需要的 adapter 装载。断点续跑**按计划数行数**而不是看有没有行——一个 (图, 条件) 是四行,被 kill 会留下半对,续写就让那个候选用自己一半的题打了两次分。最要紧的一条测试是端到端的:一个「盲读像素、带 prompt 背描述」的漫画式 backbone 过写入端再过 `endpoint3`,必须得出 x=+1、y=+0.5、first-index y=+1.0(docstring 里手算)。符号错或跨文件改名都会印出一个完全正常的数,所以两半一起验。**端点 3 比端点 2 严**:偏离 13.1 第 6 条准端点 2 拟合更短的序列,13.2 没有这一条,缺一个 (臂, checkpoint) 就整个端点未做 | 等卡,单卡 |
| 不可提问的图**全是失败图** | **已登记为新证据**(`49d81f8`,`review-packets/e3-data-availability-20260909/UNASKABLE-IMAGES.md`,旧文件一个字节没动):跑 GPU 之前的 dry-run 发现,一张支撑不了任何题的图会退出候选池——这是 `stage_observe` 的既有规则,`bsv_selection.py` 也这么算——但主运行上**这些图外部正确率是 0.000**,六个 (臂, checkpoint) 共 70 张无一正确。不是巧合:`build_counting` 在一个被要求的类别都没检出时返回 None,而这也正是 miss 的定义,所以剔除和结果**确定性混淆**。**协议不改**:这个剔除就是 13.2 钉住的 口径,+0.190 / +0.034 也是这么算的,现在拿掉是另立口径不是修 bug;而且它按构造在两个条件间抵消(题集只取决于图,不取决于条件)。改的是让规模不再是断言:每个 (臂, checkpoint) 记四个计数,120 个点都能读出剔除前后的天花板。前三个 checkpoint 上代价是 64 个 prompt 里丢 0–3 个、天花板最多 +0.033,而 step 16 上 naive 丢 3 个、rfo_gold 丢 0——**两臂受影响不等**,88 步之后不保证还这么小 | 要写进正文 |

| 偏离 7.2 / 7.3 也有代码路径了 | **代码已写并验通**(`4d8d3a5`):`src/selfsight/analysis/drift.py` + `scripts/v4_e3_drift.py`,38 个测试,**50/50 变异全杀**。这是预注册里最后两条「有判定、无代码」的规则——端点 2 / 3 在偏离 13 之前就是这个状态,教训是这种规则等着在看见数字之后由人手工执行。偏离 14 把 7.2 留白的四处钉死:θ 取端点 1 的口径**连聚合方式一起**、三个 θ 用同一批键、五个 replicate 逐个判且**不注册聚合规则**、step 取三列各自最大里的最小。写 14.1 时抓到一处真的口径不符:端点 1 是「prompt 内先平均、再对 prompt 平均」(偏离 10.2 第 2 条),我第一版写成了键上的平铺均值——四个 draw 齐全时两者相等,一旦剔除让 draw 数不齐就分开,而 7.2 跑的正是有剔除的那个区间。第一轮变异有两个活口,都值得留:钉「边界」的那个测试实际是 drift 0.5 对 effect 0.0,根本没到边界,`<` 和 `<=` 对它没差别;out-of-force 的测试拿退出码去比它本该钉住的那个常数。现在边界两侧各有一个测试,退出码按字面的 2 断言(supervisor 就是按它分支的)| CPU,不占卡 |
| 冻结输入的仓库版本 != 磁盘版本 | **已查清并钉住**(§3.5):`run_manifest.json` 冻结 14 个输入,其中 4 个(主 config、主 protocol、`v4_train.py`、`evaluate.py`)的已提交 blob 是 LF、工作树是 CRLF,差别**只有行尾**,没有内容分歧。一次 `git checkout` 这四个路径中的任何一个,都会改掉在跑的运行读的字节:源码可用 `--accept-code-update` 记一条 repair 续上,**config 与 protocol 没有豁免**——config 当场停且 resume 抛「use a new run version」,protocol 安静到下次 resume 才发现。自动合并的变更集不含这四个路径(已核,是运气不是设计)。`tests/test_frozen_inputs_vs_git.py` 6 测试,**14/14 变异全杀**(跑在隔离复制品里——在真树上造这些变异就是要防的那个事故) | 不影响在跑的运行,但主运行活着时不许 checkout 这四个路径 |
| ^ 上一行末句更正 | **上一行说「自动合并的变更集不含这四个路径」,这半句错了**(§3.6)。合并在当前 HEAD 上无冲突,但变更集 31 个路径,与冻结的 14 个交出 **4 个**:`run_decoupling_pilot.py` / `v4_decoupling_report.py` / `v4_train.py` / `src/selfsight/v4/train.py`——全都在 **source** 那一类,`--accept-code-update` 可修复。两个**无豁免**的(主 config、主 protocol)确实不在变更集里,所以合并不会让主运行不可 resume。由此的操作纪律:**要补跑的 stage 必须在合并前补完**。上一行原文保留 | 不影响在跑的运行 |
| 启动闸 | **六个闸全部有测试了**(§3.7):原来只有 `gate_main_run_finished` 和 §3.4 补的 `gate_card_schedule_decided` 有,`gate_merge_landed` / `gate_configs` / `gate_cards_free` / `gate_model_root` **一个都没有**。补 26 个,**19/19 变异全杀**;夹具照写方造(五份真 config、`staging/arm-b-merged` 的三个真文件)。真树上现在 5 拒 1 过,退出码 1,`gate_merge_landed` 的拒路第一次被证明走得通 | CPU,不占卡 |
| 五条启动命令少了 `--protocol` | **已更正并钉住**(§1.2):`run_decoupling_pilot.py` 的缺省是**试点**协议 `2026-09-06-decoupling-pilot.md`,而主运行冻的是 `2026-09-08-dstar-main-run.md`。照 §1 原样起,五个 replicate 会把试点协议冻进 `run_manifest.json` 的 `protocol_sha256`(构造时)和第一份 report 的 `provenance.protocol`,**两处都没有回头路**(都抛「use a new run version」);而命令不报错,目录名 / config / 日志 / seed 全对,要到写论文查 provenance 才看得见,那时是 5 × 约 80 GPU-h。同一个坑 2026-09-08 踩过一次,那次修的是「两处一致」而不是「那一处对」。更正后的五条在 §1.2,`tests/test_section_1_launch_commands.py` 把它们**从文档里读出来**逐条核,12 测试、**10/10 变异全杀**——第一个变异就是 §1 的原命令 | CPU,不占卡 |

| 四条失败条件里有两条**印不出来** | **已登记并修好**(偏离 15,`2481a91`):偏离 14 收尾后做了一遍**覆盖清查**——把预注册里每条有约束力的判定逐条对到执行它的代码。端点 1/2/3 的判定、偏离 7.2/7.3/9.3/9.4/10/12/13 都有代码;**「预先声明的失败条件」四条里 1 和 2 没有**。第 2 条更糟:缺的不是那句话,是那个量——「B 也塌」不是「B 超过 A」的否定(B 可以一边塌一边超过 A,那样端点 2 确证而 §2 的机制同时被证伪),而 `endpoint2.py` **从来没对 B 自己的抽样取过分位**,这条子句根本算不出来。15.1 用 13.1 给 A 的同一个单侧读法补上,跨 seed 用同一个 4/5(14.3 拒绝过把这个 4 借给 7.2,差别写在 docstring 里:同端点、同斜率、同 seed,只换臂;而且 4 和 5 里 4 是让不利结论更容易成立的那个)。第 1 条的后果句(不得声称「BSV 提升生成」)只活在预注册里——端点 3 有 `DOWNGRADE`、偏离 7.2 有两句,最重的那条反而没有;现在按 14.6 的层读研究级并印出。**端点 1 的驱动此前一个测试都没有**,注册句子就是这么进到没测过的代码里的,已补 6 个测试。另外 §3 的「两者都报」在端点 3 上只报了两列、只拟合了一支,15.3 读作两支都拟合(同一批重采样),判定仍只看期望值那支,两支不一致时正文必须写明。失败条件 3 / 4 **有**代码但比注册**更严**(一次不可恢复失败就停、守卫抛错停机而不是作废重跑),都在保守方向,15.4 登记,主运行期间不改。27 变异全杀(第一轮 7 个活口,形状全一样:夹具里注册读法和错读法碰巧同值)| CPU,不占卡 |
| arm B 合并(**重跑,取代上面那一行的结论**)| **不是快进,是真合并;已重演并验通**(§3.3):两条分支已分叉 25 / 30,`paper/iclr-2028` 那 25 个是偏离 14 / 15 与 §12。`merge-tree` 干净出树 `10b0e74`(§3.1 的那一个冲突没了——`staging/arm-b-merged` 相对 `codex/blind-self-arm-20260908` 是 23 领先 0 落后,解法已在那条分支上)。核过合并后的字节:评测 latent 那处 `partition_seed` + `candidate_index` 两边都在。探针树 detached 真做一次得 `f33555a`,tree 与空跑逐位相同。**全量** 1031 测试,只两处失败,都是探针树没有 `runs/`,主树单独跑都绿——其中一处 §3.1 没见过,因为它只跑了五个文件 | CPU,不占卡 |
| 启动 preflight 彩排过了 | `v4_e3_launch_preflight.py` 现在跑,**六个闸拦下五个**(主运行未完成 / 合并未落地 / 两卡不空 / 排卡未裁定 / `SELFSIGHT_MODEL_ROOT` 未设),退出 1。就是它该有的样子。顺带核清两件事:(一) 它印的 `31.0 h` 是 `state.json` 里**上次状态写入时的快照**,不是活钟,而 96 h 那条线在 supervisor 里是活的(`deadline = started + max_wall_hours*3600`,阶段间查一次、阶段内用 `process.wait(timeout=...)` 兜住),消息里同时印了「距上次写入 76 min」,两个数加起来自洽,**不是缺陷,不改**;(二) 那一刻两张卡上的两个进程**都是主运行自己的**——卡 1 是 `v4_train.py`(就是 `state.json` 记的 `child_pid`),卡 0 是 round-003 rfo_gold 的 detect,没有外来作业,但也意味着**整个主运行期间不存在两卡皆空的窗口**,§0.3 的空载基准只能等它结束 | CPU,不占卡 |
| 排卡的那个闸读错了文件(**上一行写早了**)| 上一行「就是它该有的样子」是在**读 `card_benchmark.py` 之前**写的,不改,记在这里:六个闸里 `gate_card_schedule_decided` 读的是 `verdict.json` 的顶层 `serialise_detect`,而基准写的是 `card_benchmark.json` 的 `verdict.adopt_serial`,**文件名和键名两处都对不上**,而且这两个名字在预注册和任何别的脚本里都不存在。基准在 arm B 的 review packet 里、闸在这条分支上,**今天预演合并之前两者没在同一棵树里待过**,加上这个闸零测试,所以没有任何一次运行能碰到。它不会误放行,它会**永远拦着**——代价落在只开一次的那 1.5 h 窗口里,唯一出路是手写决定文件。已修(`8f5304d`,只动闸不动 packet),9 测试 12/12 变异全杀,详见 §3.4。**不追加偏离**:判据没有含糊,`card_benchmark.py` 已经实现对了,这是接线错误不是口径缺口 | CPU,不占卡 |
| 偏离 7.2 的守卫会拒绝每一对合法运行 | **已修**(`4d852dc`):`drift.split_digest` 把整份 `split.json` **连 `"created"` 时间戳一起**做规范化哈希,而 `load_columns` 拿它比主运行和 replicate。两个留出 prompt 逐个相同的运行,split stage 在两个时刻跑、字节必不同——**守卫会对每一对合法的 (主运行, replicate) 抛错**,而且是在分析那一刻、400 GPU-h 花完之后。套件看不见,因为每个夹具的 split 都是合成 dict、没有 `created` 键。新测试**对旧代码先验红**;identity 改成「除时间戳外的全部内容」,比读 `digest` 字段更严(配方 + 配方产生的名单);9/9 变异全杀。顺带更正 §1「启动前必查 1」:它点名的 `83b7eaa6...` 是**文件字节**的 sha256,照字面比会把五个正确的 replicate 全停掉;三个数已钉在 `tests/test_main_run_split_identity.py`,详见 §1.1 | CPU,不占卡 |

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

## 0.6 余量从 +2.3~+7.7 h 放宽到 +8.2~+14.2 h(2026-09-09 12:00,第二次重算)

§0.5 承诺「下一个轮边界会把区间再收窄一次」。现在有了一半:
`round-002.train` 已经跑完,块还在飞。**收窄的方向是好的一头。**

新测到的量只有一个,但它是区间里最不确定的那个:

| | round 1 | round 2 |
|---|---|---|
| train | 2.18 h | **1.67 h**(−23%)|
| 块墙钟(慢链 = rfo_gold)| 4.89 h | 在跑,naive 的 generate 是 1.22 h vs 1.28 h(−4.7%)|

`step-00016.report` 落在 elapsed 24.79 h,`round-002.train` 09-09 10:01 收工。
剩下的 = 当前 step-24 块的余下部分 + 8 个整轮(round 3–10)。三个节奏:

| 节奏 | 完成 | elapsed | 余量 |
|---|---|---|---|
| round 1 原样保持(块 4.89 + train 2.18 = 7.06)| 09-11 23:23 | 87.8 h | **+8.2 h** |
| round 2 的 train 实测 + round 1 的块(1.67 + 4.89 = 6.57)| 09-11 19:28 | 83.9 h | **+12.1 h** |
| 两个都按 generate 的 −4.7% 缩(1.67 + 4.66 = 6.34)| 09-11 17:23 | 81.8 h | **+14.2 h** |

96 h 停止线是 09-12 07:33。**§0.5 的悲观端(7.66 h「block 8 节奏」,余量 +2.3 h)
被观测排除了**:那条外推的锚是 round 0 的块,而 round 0 夹着基线评测和缺 audit 的停机。
现在最保守的一条是 round 1 原样保持,余量 +8.2 h(8.5%)。

**这不改任何决定。** 启动判据仍然是 `pilot_complete` + `completed_rounds == 11`
(§1 第 1 条,判据不看估计值);§0.2 的三条理由仍然成立,卡 0 该空转还是空转。
它只是把「主运行完成」从一个约 4% 余量的事件,变回一个大概率事件。

**下一次收窄**:`step-00024.report` 落地(预计 14:41–14:54)会给出第二个完整的块,
那时块和 train 都有两个可比样本,区间可以按实测的两端夹,不用再靠 generate 外推。

## 0.7 第三次重算:两端都是实测,余量 +8.5~+15.2 h(2026-09-09 14:33)

§0.6 承诺的收窄到了。`step-00024.report` 落在 elapsed **31.01 h**
(预计 14:41–14:54,实到 14:33,早 8 分钟)。现在 train 和块各有两个可比样本,
区间不用再靠 generate 外推。

| | round 1 | round 2 | 变化 |
|---|---|---|---|
| train | 2.18 h | 1.67 h | −23% |
| 块(慢链 = rfo_gold,train 收工 → 该 step 的 report)| 4.89 h | 4.55 h | −7.0% |
| 整轮 | **7.06 h** | **6.22 h** | −12% |

剩下的是 8 个整轮(round 3–10),step-24 的块已经收在 31.01 里。

| 节奏 | 完成 | elapsed | 余量 |
|---|---|---|---|
| round 1 原样保持(7.06)| 09-11 23:03 | 87.5 h | **+8.5 h** |
| 两轮均值(6.64)| 09-11 19:39 | 84.1 h | **+11.9 h** |
| round 2 原样保持(6.22)| 09-11 16:21 | 80.8 h | **+15.2 h** |

**和 §0.6 比,区间几乎没动(+8.2~+14.2 → +8.5~+15.2),但性质变了**:
§0.6 的两端一端是实测一端是 −4.7% 外推,现在两端都是实测过的整轮。

**不外推那条下降趋势。** train 掉了 23%、块掉了 7%,两个点连不出趋势,
而且 train 的耗时本来就随 `pair_decisions` 取交集后的样本数波动——
round 1 和 round 2 的训练集不一样大。把 −23% 再乘一遍会得到 75.2 h / 余量 +20.8 h,
那个数没有依据,不写进表里。

96 h 停止线仍是 09-12 07:33。最保守的一条有 8.9% 余量,**任何一轮多花 1 h 就吃掉
1/8 的余量**——这是一个大概率完成、但不宽裕的事件。

**这不改任何决定。** 启动判据仍是 `pilot_complete` + `completed_rounds == 11`
(§1 第 1 条,判据不看估计值);卡 0 该空转还是空转(§0 与 §0.2)。

**下一次收窄**:`step-00032.report`,预计 elapsed 37.2~38.1 h,即 09-09 20:45–21:39。
第三个整轮会让「6.22 是不是偶然」这件事有个答案。

## 0.8 第四次重算,迟了三轮:余量 +0.9~+14.6 h(2026-09-10 07:0x)

§0.7 把下一次收窄挂在 `step-00032.report`。它 09-09 20:26 就到了,
**这一节欠了三个整轮才补上**。先写明这件事,再写数。

`step-00032.report` 落在 elapsed **36.89 h**;§0.7 预计 37.2–38.1 h,
**比预计区间的快端还快 0.3 h,落在区间之外**。

下表全部重算自 `runs/v4/decoupling-main-20260908/stage-completion/`(103 条时间戳),
单位与 §0.7 相同:整轮 = 上一个 report → 这一个 report。

| | round 1 | round 2 | round 3 | round 4 | round 5 |
|---|---|---|---|---|---|
| train | 2.18 h | 1.67 h | 1.49 h | **3.98 h** | 1.81 h |
| 块(慢链 = rfo_gold,train 收工 → 该 step 的 report)| 4.89 h | 4.54 h | 4.39 h | 4.65 h | 在跑 |
| 整轮 | 7.06 h | 6.22 h | **5.89 h** | **8.63 h** | — |

**§0.7 拒绝外推那条下降趋势,那个决定值得记一笔,因为两边都被验了。**
round 3 是 5.89 h,趋势确实还在往下——§0.7 算过但没写进表的 75.2 h,方向是对的。
然后 round 4 是 8.63 h,趋势一轮之内就没了。
拒绝外推**不是运气好**:§0.7 给的理由是「两个点连不出趋势,而且 train 的耗时
随 `pair_decisions` 取交集后的样本数波动」。round 4 的超出恰恰**不是**那种波动,
是另一个 session 抢同一张卡,一个 §0.7 当时没有、也不该有的变量。
**趋势对了不等于外推对了。**

**抢卡代价在这里登记**(先前只在会话里算过,没有落进任何文档):
round-004.train **3.98 h** 对 round-003.train **1.49 h**,同一份 config、
同一个训练规模区间,差 **+2.49 h**。round 4 的块只有 4.65 h,
和 4.39 / 4.54 / 4.89 挤在一起——**抢卡吃掉的时间全在 train 上,块没受影响**。
现场诊断见 `peer-claude-sessions-race-for-gpus`:两个 session 的等待器
(本项目 `await_room()` 60 s 轮询 / 18000 MiB;对方 180 s / 20.5 GiB)
在同一个 40 秒窗口里都读到卡空了,一起启动。

剩下的是 round 5 的块 + 5 个整轮(round 6–10)。`step-00040.report` 收在 45.52 h,
`round-005.train` 已经收在 47.33 h(1.81 h),块按四轮均值 4.62 h 计。

| 节奏 | 完成 | elapsed | 余量 |
|---|---|---|---|
| 抢卡那轮原样重演(8.63)| 09-12 06:39 | 95.10 h | **+0.90 h** |
| 四轮均值(6.95)| 09-11 22:15 | 86.70 h | **+9.30 h** |
| 无抢卡三轮均值(6.39)| 09-11 19:27 | 83.90 h | **+12.10 h** |
| 最快一轮 round 3(5.89)| 09-11 16:57 | 81.40 h | **+14.60 h** |

96 h 停止线仍是 09-12 07:33。

**区间比 §0.7 的 +8.5~+15.2 宽了,这不是收窄。** 悲观端从 +8.5 掉到 +0.9,
因为 round 4 测出了一个 §0.7 不知道的失败模式。悲观端要成立,需要抢卡**再来五次**。
它已经被协调掉一次(对方 session 自己停了,而且是他们的用户让停的,不是我让停的),
但**没有任何机制保证它不再来**:两边都写了正确的礼让逻辑,洞在「读 free」
和「显存真正落地」之间没有一方公布意图。跨项目 lockfile 提过,**未经批准,没有实施**。

**这不改任何决定。** 启动判据仍是 `pilot_complete` + `completed_rounds == 11`
(§1 第 1 条,判据不看估计值)。

**一个算错的数,更正在这里。** 07:00 前后报过「近三轮中位 6.46 h/round,余量 +16.3 h」。
那个数**漏了最后一个块**:train→train 的差分以 train 收工为界,
round 10 训完之后还有一整个评测块(约 4.6 h)才到 `step-00088.report`。
以 report→report 为单位重算就是上表,**没有 +16.3 那一端**。

**下一次收窄**:`step-00048.report`,按块均值预计 elapsed 51.9 h,即 09-10 11:2x。
它给的是「round 4 的 8.63 是不是一次性的」的第一个答案。


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

**启动前的闸现在是一个脚本**:`scripts/v4_e3_launch_preflight.py`。
下面这段散文是它的规格,脚本是它的实现;不接触 GPU,任一条不过就非零退出。
`--skip <闸名>` 可以放行某一条,但会在末尾把放行清单打出来,绿不掉。

**启动前的三道闸,按顺序**(2026-09-09 补,见 §0.5):

1. **主运行必须是正常跑完的**。`runs/v4/decoupling-main-20260908/state.json`
   的 `status` 要是 `pilot_complete` 且 `completed_rounds == 11`。
   「不是」有三种,裁定相同(都不触发自动启动)、现场不同
   (2026-09-09 更正,初稿这里把前两种搞混了):
   - `canary_complete`:supervisor 带 `--through-round N`(N < `training.rounds`)起的,
     停在被告知的地方。**不是** 96 h 停止线。STATUS §43.15 记的「有界运行结束」就是它。
   - `status` 停在 `running` 不动:**这才是 96 h 停止线的样子**。`run()` 在下一个 stage
     边界 `raise`,异常从轮循环里出去,走不到终态 `state()`,`state.json` 再没人改写。
     stage traceback、被 kill、机器重启,留下的文件长得一样。闸按 supervisor pid
     还在不在把这一种和下一种分开。
   - `status` 是 `running` 而且在动:就是还在跑。
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

### 1.1 更正:「启动前必查 1」点名的是错的那个数(2026-09-09 16:1x)

上面「启动前必查」第 1 条写的是:split digest 要与主运行一致,
主运行的值是 `83b7eaa6...`,「它既是 `runs/v4/decoupling-main-20260908/split.json`
的 sha256」。**那句描述对,那个数错。**

`split.json` 的第一个字段是 `"created": "2026-09-08T14:33:14Z"`,一个墙钟戳。
一个 replicate 就算**留出的 prompt 逐个相同**,它的 split stage 也在另一个时刻跑,
写出的字节不同、sha256 必然不同。而第 1 条注册的反应是「不一致就停,不要往下走」。
**照字面执行,它会把五个正确的 replicate 全停掉。**

**有三个数,上面只提了两个:**

| | 值 | 是什么 | 跨运行 |
|---|---|---|---|
| 文件 sha256 | `83b7eaa6...` | 字节,含 `created` | **必不相同** |
| `digest` | `9bab14d4...` | 配方:`sha256_json({runs, seed, outcome, probe})` | 相同 |
| identity | `41c5f1b2...` | 除 `created` 外的全部内容 | 相同 |

**配方那一半已经有代码,而且一直有**:`v4_train.read_split()` 每个 stage 都拿
`payload["digest"]` 和当前 config 重算的值比,不等就 `SystemExit`。
所以「这个 run 的 split 和它自己的 config 一致」不需要人查。
**缺的是「和主运行一致」那一半。**

**identity 这一列今天才是对的。** 偏离 7.2 的守卫
(`selfsight.analysis.drift.split_digest`,`load_columns` 里比 main 和 replicate)
原来把整份 payload **连 `created` 一起**做规范化哈希,也就是说
**它会对每一对合法的 (主运行, replicate) 抛「different splits」**——
在分析那一刻,400 GPU-h 花完之后。已修(`4d852dc`),9/9 变异全杀;
新测试对旧代码**先验红**,缺陷是实测出来的不是读出来的。
夹具看不见它,是因为每个 split 都是合成 dict、根本没有 `created` 键。

**还剩的口子,以及为什么现在不补**:没有任何东西在**启动时**比 main 和 replicate。
7.2 的守卫在分析时比(太晚),`v4_verify_replicates.py` 只比五个 replicate
**彼此**、而且读的是配方 `digest` 不是 identity。补在哪儿是清楚的
——合并之后加进 `v4_verify_replicates.py`;**现在加不了**:那个文件只在
`staging/arm-b-merged` 上,在这条分支上同路径新建会造成 add/add 冲突。
也做不成 preflight 的闸:replicate 的 split 在启动前还不存在。

**现在能做的那一半已经做了**:三个数钉在
`tests/test_main_run_split_identity.py` 里(`runs/` 缺失时 skip),
参考值从此是一条被检查的事实,不是散文里的一串十六进制。
replicate 1 的 split stage 一跑完,拿它的 identity 和 `41c5f1b2...` 比。
`runs` 字段不用担心:supervisor 用的是模块级常量 `RUNS`,
和主运行 split.json 里记的逐字相同。

---

### 1.2 更正上面那五条命令:少了 `--protocol`,而它没有回头路(2026-09-09 18:1x)

§1 的五条启动命令没有 `--protocol`。`run_decoupling_pilot.py` 的缺省是
`docs/prereg/2026-09-06-decoupling-pilot.md`——**试点那份**,不是主运行那份。
主运行冻的是 `docs\prereg\2026-09-08-dstar-main-run.md`(从它自己的 manifest 读的)。
照 §1 原样起,五个 replicate 会各自把**试点协议**冻进两个地方:

| 冻在哪 | 什么时候 | 改得回来吗 |
|---|---|---|
| `run_manifest.json` 的 `protocol_sha256` | 构造时 | **不能**。resume 那条 `for key in (...)` 抛「use a new run version」,无豁免(§3.5)|
| `decoupling_report.json` 的 `provenance.protocol` | **第一份** report | **不能**。后面每份都比对第一份,不符抛「Frozen protocol changed; use a new run version」|

**这件事已经发生过一次。** `run_decoupling_pilot.py` 的 `report()` 里那段注释记着:
2026-09-08 一整天,report 硬编码了试点协议,而 `run_manifest.json` 指着主运行协议,
两边不一致。那次的修法是让 report 改用 `self.protocol_path`——
**修的是「两处一致」,不是「那一处对」**。现在再踩,就是一致地错。

**而且它长得完全正常**:命令不报错,目录名、config、日志、seed 全对,
只有 provenance 那一块指着一份描述**另一个规模**的文档
(试点是 10 round / 每 prompt 1 张图;replicate 是 11 round / 4 张图)。
要到写论文查 provenance 时才看得见,而那时候已经是 5 × 约 80 GPU-h。

**改成这五条**,唯一的差别是每条末尾多一个 `--protocol`:

```
envs/core/python.exe -u scripts/run_decoupling_pilot.py --outdir runs/v4/e3-s20260906 --config configs/v4_e3_replicate_s20260906.yaml --arms naive blind_self --protocol docs/prereg/2026-09-08-dstar-main-run.md
envs/core/python.exe -u scripts/run_decoupling_pilot.py --outdir runs/v4/e3-s20260907 --config configs/v4_e3_replicate_s20260907.yaml --arms naive blind_self --protocol docs/prereg/2026-09-08-dstar-main-run.md
envs/core/python.exe -u scripts/run_decoupling_pilot.py --outdir runs/v4/e3-s20260909 --config configs/v4_e3_replicate_s20260909.yaml --arms naive blind_self --protocol docs/prereg/2026-09-08-dstar-main-run.md
envs/core/python.exe -u scripts/run_decoupling_pilot.py --outdir runs/v4/e3-s20260910 --config configs/v4_e3_replicate_s20260910.yaml --arms naive blind_self --protocol docs/prereg/2026-09-08-dstar-main-run.md
envs/core/python.exe -u scripts/run_decoupling_pilot.py --outdir runs/v4/e3-s20260911 --config configs/v4_e3_replicate_s20260911.yaml --arms naive blind_self --protocol docs/prereg/2026-09-08-dstar-main-run.md
```

**为什么是 `2026-09-08-dstar-main-run.md`。** 它开头自己就写着「这不是新的科学预注册,
`2026-09-06-decoupling-pilot.md` 的假设、终态与禁止事项原样继承」,登记的是**规模**:
11 round / 12 checkpoint / 每 prompt 4 张图 / 64 个 outcome prompt。
**五个 replicate 跑的正是这个规模**;试点那份描述的是另一个规模。
而且端点 1 要把主运行和五个 replicate 放在一起读,provenance 指同一份文档才对得上。

**一处不吻合,写在这里而不是藏着**:那份文档末尾有一句
「单 seed,描述性轨迹,**不得写成显著性结论**」——那是就主运行**这一次运行**说的;
偏离 9 的五 seed 在**研究层面**取代了它(§34 的符号检验)。
文档本身不改,它是冻结记录。**若要另写一份 replicate 专用协议,必须在启动之前写完**,
因为启动之后这个选择就没有回头路。这是一个可以做的决定,不做也走得通。

**钉住的办法**:`tests/test_section_1_launch_commands.py` 把上面这五条**从本文件里读出来**,
逐条核:`--outdir` 的 seed 与 `--config` 的 `training.seed` 一致、config 文件在、
`--arms` 是注册的那一对、`--protocol` **写明了**而且文件在、五条指同一份协议、
并且那份**不是 supervisor 的缺省**——缺省哪天被改成对的,这条会提醒连本节一起改,
而不是让两处悄悄分家。**§1 原文的五条保留**,更正登记在这里。

### 1.3 再更正一次:replicate 的 `--protocol` 指向粒度修订(2026-09-10 06:1x)

§1.2 结尾留了一句「**若要另写一份 replicate 专用协议,必须在启动之前写完**」。
写完了:`docs/prereg/2026-09-10-join-granularity.md`。**§1.2 的五条原样保留**,
更正登记在这里——理由正是 §1.2 自己钉住的那条:协议要描述这次运行实际跑的东西。

§3.9 查出预注册的主检验从来没有过一个可测的输入。`read_outcome` 要求一个 spec 的
每张图都同时有 `s_select` 和裁定,而主运行协议 §3 早就登记了内部曲线**只打第一张图**;
两者在 R=4 下不可能同时成立。修复必须赶在五个 replicate 起跑前进 `scripts/`
(时机见新文档 §6),**于是 replicate 跑的读手,不是 2026-09-08 那份文档描述的那个**。
让它们继续指向 `2026-09-08-dstar-main-run.md`,就是把一份不描述其读手的协议冻进
`run_manifest.json`——和 §1.2 修的是同一类错,换个地方犯,而且同样没有回头路。

新文档以 sha256 引用 2026-09-08 那份(`e276f3c7…6b46ef`),该文件**逐字节不动**:
主运行的 provenance 因此完好,replicate 冻的是描述自己的那一份。
§1.2 点出的「单 seed / 五 seed」不吻合,也在新文档 §8 里当场处置了,不再悬着。

**改成这五条**,与 §1.2 唯一的差别是 `--protocol` 换了文件:

```
envs/core/python.exe -u scripts/run_decoupling_pilot.py --outdir runs/v4/e3-s20260906 --config configs/v4_e3_replicate_s20260906.yaml --arms naive blind_self --protocol docs/prereg/2026-09-10-join-granularity.md
envs/core/python.exe -u scripts/run_decoupling_pilot.py --outdir runs/v4/e3-s20260907 --config configs/v4_e3_replicate_s20260907.yaml --arms naive blind_self --protocol docs/prereg/2026-09-10-join-granularity.md
envs/core/python.exe -u scripts/run_decoupling_pilot.py --outdir runs/v4/e3-s20260909 --config configs/v4_e3_replicate_s20260909.yaml --arms naive blind_self --protocol docs/prereg/2026-09-10-join-granularity.md
envs/core/python.exe -u scripts/run_decoupling_pilot.py --outdir runs/v4/e3-s20260910 --config configs/v4_e3_replicate_s20260910.yaml --arms naive blind_self --protocol docs/prereg/2026-09-10-join-granularity.md
envs/core/python.exe -u scripts/run_decoupling_pilot.py --outdir runs/v4/e3-s20260911 --config configs/v4_e3_replicate_s20260911.yaml --arms naive blind_self --protocol docs/prereg/2026-09-10-join-granularity.md
```

`tests/test_section_1_launch_commands.py` 改读本节(`SECTION = "### 1.3 "`,
`PROTOCOL` 跟着换),12 个测试原样有效。它仍然核「那份不是 supervisor 的缺省」,
所以缺省哪天被改成 2026-09-10 这份,这条会提醒连本节一起改。

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

### 3.3 又重跑了一次:冲突没了,而且这不是快进(2026-09-09 15:5x)

§3 末尾写着「两边还会再动,合并前要重跑一次」。跑了。

**先更正一处转述。** 执行计划在传递中被转述成「把 `paper/iclr-2028` **快进**到
`staging/arm-b-merged`」。本文件原文一直写的是「合并」,没写过快进,
但那句转述是会被照着执行的。两条分支**已经分叉**:

```
git rev-list --left-right --count paper/iclr-2028...staging/arm-b-merged
25      30
```

左边 25 个是偏离 14 / 15、§12 和它们的测试。**没有快进这个选项**,
方向是把 `staging/arm-b-merged` 合进 `paper/iclr-2028`。

**§3.1 的那一个冲突没有了。** `staging/arm-b-merged` 相对
`codex/blind-self-arm-20260908` 是 23 领先、0 落后——§3.1 的解法已经落在
那条分支上了。所以现在:

```
git merge-tree --write-tree --name-only paper/iclr-2028 staging/arm-b-merged
10b0e741a2312bc7180a8dc91332524cad1c3653      (退出 0,无冲突)
```

**干净不等于对,所以核了合并后的字节。** 合并树里的 `scripts/v4_train.py`,
造评测 latent 那处是 `seed=partition_seed(config)`,键 `(key[0], key[1])`
带 `candidate_index`——§3.1 要的「两边都要」在;每个 call site 都没有裸的
`int(config["seed"])`(它只出现在 `partition_seed` 自己的定义里)。

**然后在探针 worktree 上真做了一次**(detached HEAD,不动任何分支):
合并出 `f33555a`,它的 tree 与上面空跑的 OID **逐位相同**。

**跑的是全量套件,不只是 §3.1 的那五个文件。** 61 文件 817 个测试
(`paper/iclr-2028`)→ 69 文件 1031 个(合并后,arm B 加 8 文件 214 个)。
退出 1,失败**恰好两个**:

| 失败 | 原因 | 主树单独跑 |
|---|---|---|
| `test_v4_checkpoint_probe.py::test_end_to_end_small_backbone_keeps_grams_and_removes_vectors` | 探针树没有 `runs/`(未跟踪产物) | 绿 |
| `test_checkpoint_reload.py::test_the_lora_targets_path_is_absolute` | 同上;读 `runs/readiness/showo2-1p5b/a4-lora-targets-r1.json` | 绿 |

第一个 §3.1 已经登记过。**第二个是这次才看见的**——§3.1 只跑了五个文件,
跑全量才碰到。两个都当场在主树上单独复跑过,都过。

**合并仍然不能现在落地**,理由不变(§3):主运行在跑,合并之后若需要 resume
会被源码摘要拒。落地时机不变:主运行完成之后、起 arm B 之前。

---

### 3.4 合并预演顺手照出一处:启动闸读的文件没人写(2026-09-09 16:0x)

§3.3 跑完之后顺着「主运行完成时会自动跑什么」把那条链读了一遍,
在 `scripts/v4_e3_launch_preflight.py` 的 `gate_card_schedule_decided` 上撞到:

| | 文件 | 键 |
|---|---|---|
| 闸**读**的 | `review-packets/card-scheduling-20260909/verdict.json` | 顶层 `serialise_detect` |
| 基准**写**的 | `review-packets/card-scheduling-20260909/card_benchmark.json` | `verdict.adopt_serial` |

文件名和键名**两处都对不上**,而且 `verdict.json` / `serialise_detect`
在预注册里、在本文件里、在任何别的脚本里**一个字都没有**——是这个闸自己发明的。

**为什么一直没人发现**:`card_benchmark.py` 在 arm B 分支的 review packet 里,
这个闸在 `paper/iclr-2028` 上,**今天做合并预演之前两者从没在同一棵树里待过**,
所以没有任何一次运行能碰到。加上这个闸**一个测试都没有**
(测试文件只覆盖 `gate_main_run_finished`)——和 偏离 15.2 里
「端点 1 的驱动一个测试都没有」是同一个形状。

**代价会付在只开一次的那个窗口里。** 主运行结束到 replicate 1 启动之间约 1.5 h,
基准跑完、把结论印在屏幕上,preflight 仍然说「no verdict.json」,
往下走的唯一路就是**当场手写一份决定文件**——而那正是注册判据最容易被
临场判断顶掉的一刻。

**改的是闸,不是 review packet。** `card_benchmark.py` 一个字节没动
(既有 packet 不可改;往 packet 里加新文件是允许的,`--outdir` 指向它就是加新文件)。
闸现在读 `card_benchmark.json`,**不重新实现判据**——判据归 `card_benchmark.py`,
第二份实现就是两份会漂移的副本(`test_registered_consequences.py` 建起来就是防这个)。
闸另外核两件事,**都不是新条件**:`no_load` 为真(§0.3 标题就是「先测空载」,
脚本自己没有 `--allow-busy-cards` 就拒),和 `margin` 是注册的 0.90;
但**文件里一个没人查的标志位不叫检查**。`no_load` 键缺失按拒绝走,
不按「默认空载」走。

`adopt_serial: false` **算通过**。保持按臂分卡是注册的平局解,不是没做决定。

9 个测试,夹具**照基准自己的 payload 键造,不照闸的期望造**——只有这个顺序
才抓得住原来那个缺陷。12/12 变异全杀,其中一个是把文件名和键名**原样恢复**。

**为什么这条不追加 偏离 16**:偏离 11 / 13 / 14 / 15 都是因为有人**必须选**
一个注册原文没定的读法。这里没有选择——§0.3 的判据本身清楚、
而且已经在 `card_benchmark.py` 里实现对了,`adopt_serial` 和 `serialise_detect`
是同一个命题的两个名字。这是接线错误,不是口径缺口。
登记在这里,是因为「有规则、有代码、两段代码接不上」是本项目的**第五种**
同型失败,值得单独有个名字。

---

---

### 3.5 冻结的 14 个输入里,4 个的仓库版本不是运行读的那份(2026-09-09 16:5x)

§2 的备注写着「`configs/v4_decoupling_main_20260908.yaml` 至今仍未纳入 git(主树 `??`)」。
**这句现在是过期的**:它在 `005b370` 就已经提交在 `paper/iclr-2028` 上。
去核这一句的时候撞到一件更要紧的事。

`run_manifest.json` 冻结 **14** 个输入的 sha256:1 个 config、1 个 protocol、12 个 SOURCES。
`.gitattributes` 对 `*.py` / `*.yaml` / `*.md` 全写着 `eol=lf`,而 `git add` 只规范化**索引**,
不动工作树。于是四个文件的**已提交 blob 与运行正在读的字节不同**:

| 文件 | 工作树 == 冻结摘要 | blob == 工作树 | CRLF 行数 |
|---|---|---|---|
| `configs/v4_decoupling_main_20260908.yaml` | 是 | **否** | 178 |
| `docs/prereg/2026-09-08-dstar-main-run.md` | 是 | **否** | 90 |
| `scripts/v4_train.py` | 是 | **否** | 728 |
| `src/selfsight/v4/evaluate.py` | 是 | **否** | 640 |

其余十个 blob 与工作树逐字节相同。四个的差别**只有行尾**:
`工作树.replace(CRLF, LF) == blob` 四个全为真,没有内容分歧。

**§3.2「提交不改磁盘」是对的,而且当时就复核过。** 这里补的是它的推论,
当时没人写下来:**既然提交没改磁盘,那 blob 就不是磁盘上那份**。
一次 `git checkout <path>` / `git restore <path>` / `git stash` 会把 LF 版落到盘上,
之后 `git status` 干干净净,而运行读的字节已经变了。

**三个摘要的执法强度不一样,名字看起来一样:**

| 摘要 | 每个 stage 前 | resume 时 | 豁免 |
|---|---|---|---|
| `config_sha256` | 查(`RuntimeError`) | 查 | **没有** |
| `source_sha256` | 查(`RuntimeError`) | 查 | `--accept-code-update`,并记一条 repair |
| `protocol_sha256` | **不查** | 查 | **没有** |

于是三种后果:源码被 checkout,下一个 stage 边界停,记一条 repair 可以续;
config 被 checkout,下一个 stage 边界停,**且 resume 抛「use a new run version」,不可续**
——按现在的进度是 33 h GPU 时间;protocol 被 checkout,**当场什么都不发生**,
直到某次需要 resume 才发现早就不可续了。最安静的是这一个,也最容易顺手做掉。

**自动合并不会踩到。** §3.3 那次合并只动 `configs/backbones/janus_pro_1b.yaml` 与
`showo_v1.yaml`,这四个路径一个都不在变更集里(已核)。**这是运气,不是设计**:
变更集是别的原因决定的,没有任何机制保证下一次也躲开。

**更正我自己刚数出来的那个数**:第一遍是 3 个,因为我枚举的是 config + `SOURCES`,
没去看 manifest 的第三个摘要。protocol 是第 4 个,而且它落在**无豁免**那一类里。
现在的数是从 manifest 自己的键重新枚举的,不是从脚本的常量抄的。

**钉住的办法**:`tests/test_frozen_inputs_vs_git.py`,6 个测试,`runs/` 不在就 skip。
钉的是这个集合恰好是这 4 个(**第五个出现要吵,少一个也要吵**)、
差别只有行尾、以及 supervisor 里三种执法强度各自待在哪一段。

**变异测试跑在一个隔离的复制品里,不在真树上**:这四个文件正是在跑的运行的冻结输入,
在工作树上造它们的变异**就是这条规则要防的那个事故**。复制品是新建的一个 git 仓库,
同样的路径、同样的 `.gitattributes`、supervisor 逐字拷贝、
manifest 的摘要照 CRLF 工作树重算,基线 6 通过。**14/14 全杀**:
4 个是「把某个文件 checkout 掉」,2 个是「第五个文件长出 CRLF」,
5 个是把 supervisor 的执法强度改错,其余是摘要与 protocol 指向。

**不追加偏离。** 没有任何注册规则牵涉在内——预注册不管工作树的行尾。
这是一条**操作纪律**,和 §5「那两个文件不要拷进主运行目录」同类:
主运行活着的时候,这四个路径不许从 git 重新落盘。

---

### 3.6 更正 §3.5 的一句:合并确实会动 4 个冻结输入(2026-09-09 17:0x)

§3.5 写着「§3.3 那次合并只动 `configs/backbones/janus_pro_1b.yaml` 与
`showo_v1.yaml`,这四个路径一个都不在变更集里(已核)」。**这句有一半是错的。**

我核的是 `configs/` 底下的变更集,然后把结论推广到了四个路径上。
在当前 HEAD(`1c21e94`)重跑 `git merge-tree --write-tree HEAD staging/arm-b-merged`,
退出 0、无冲突,但变更集是 **31 个路径**,与冻结的 14 个输入交出 **4 个**:

| 冻结路径 | 在合并变更集里 | 执法类别 |
|---|---|---|
| `scripts/run_decoupling_pilot.py` | **是** | source,可 `--accept-code-update` |
| `scripts/v4_decoupling_report.py` | **是** | source,可 `--accept-code-update` |
| `scripts/v4_train.py` | **是** | source,可 `--accept-code-update` |
| `src/selfsight/v4/train.py` | **是** | source,可 `--accept-code-update` |
| `configs/v4_decoupling_main_20260908.yaml` | 否 | config,**无豁免** |
| `docs/prereg/2026-09-08-dstar-main-run.md` | 否 | protocol,**无豁免** |

**对的那一半**:两个无豁免的路径确实不在变更集里,所以合并**不会**把主运行变成不可 resume。
**错的那一半**:`scripts/v4_train.py` 在里面,§3.5 说它不在。这四个合并后的字节
与工作树**内容就不同**,不只是行尾——`合并 == 工作树.replace(CRLF, LF)` 四个全为假,
那正是 §3.1 那处冲突解法本来就要改的东西。

**§3.3 早就写对了**,而且用的是更一般的说法:「主运行在跑,合并之后若需要 resume
会被源码摘要拒」。§3.5 的毛病是我把一次 `configs/` 范围的检查当成了全树检查——
和 §3.4 那处接线错误同一个味道:两个都对,但对的是两件不同的事。
**§3.5 原文保留**,更正登记在这里。

**由此多一条操作纪律**:主运行完成之后、合并之前,**要补跑的 stage 必须在合并前补完**。
合并之后再 resume 主运行,唯一的路是 `--accept-code-update`,而它会就地改写
`run_manifest.json` 的 `source_sha256`(旧指纹进 `code_repairs`,可审计,
但「这 96 h 的结果是这份代码产生的」那份冻结记录不再是原件)。
§1 的启动闸要求 `pilot_complete` + `completed_rounds == 11` 才放行,
正常路径走不到这里;这条是写给不正常路径的。

**顺带一个量**:全树 `git ls-files --eol` 显示 **237** 个文件是「索引 LF、工作树 CRLF」
(172 个在 `review-packets/`,56 个在源码与文档,8 个在 `data/`)。
所以 §3.5 那四个并不特殊,**特殊的只是它们被冻结了**——
replicate 起跑时新冻结的输入按同样的概率也会是 CRLF。
`tests/test_frozen_inputs_vs_git.py` 钉的是「冻结集恰好是这四个」,不是「全树没有 CRLF」;
全树那 237 个**现在不能碰**,因为其中就有主运行正在读的那几份。

---

### 3.7 六个启动闸里有四个从没测过它们会不会拒(2026-09-09 18:0x)

§3.4 抓到的是「闸读的文件没人写」。顺着同一条线把 `v4_e3_launch_preflight.py`
的六个闸点了一遍:

| 闸 | 原有测试 |
|---|---|
| `gate_main_run_finished` | 8 个 |
| `gate_card_schedule_decided` | 9 个(§3.4 补的)|
| `gate_merge_landed` | **0** |
| `gate_configs` | **0** |
| `gate_cards_free` | **0** |
| `gate_model_root` | **0** |

**「今天跑一下是绿的」不等于「该拒的时候会拒」。** 一个永远返回通过的闸比没有闸更坏:
它会在主运行结束的那个钟点被人读一遍,而那一遍会被读成许可。

**先在真树上两头各验了一次**(只读;`gate_cards_free` 只 query `nvidia-smi`)。
现在跑 preflight:`gate_merge_landed` **拒**,六条理由逐条印出来——
`training_seed` 无、`partition_seed` 无、`--arms` 无、`self.arm_device` 无、
`v4_verify_replicates.py` 不在、`registered_arms` 无。**这就是它的验红**,
而且是这个闸头一次被证明拒得动。`gate_cards_free` 拒(两张卡各约 17 GB,是主运行)、
`gate_model_root` 拒(未设)、`gate_card_schedule_decided` 拒(基准还没跑)、
`gate_configs` 通过。整脚本退出码 **1**——第一次我量成 0,是把 `$?` 读成了管道末端
`head` 的退出码,重量过了。

**顺手核清一件本来没在查的事**:五份 replicate config 在 `paper/iclr-2028` 上、
各 193 行、编号 1–5,`gate_configs` 通过。合并的变更集里没有它们,
原因不是「两边一样」,而是 **`staging/arm-b-merged` 自合并基 `90431bf` 之后
一次都没动过这五个文件**——修字面缺陷的是主线的 `9e9084f`。
所以合并保留主线版本,而主线版本正是修过的那个。**方向是对的,但这也是运气**:
若当初把这五份修在 arm B 分支上,同一个合并就会把修丢掉,而且不冲突、不报错。

**补的测试 26 个,夹具一律照写方造:**
- `gate_configs` 的夹具是**把五份真 config 读进来、改一个字段、再写出去**;
  并且先验了「一个字段都不改时夹具本身通过」,否则底下每一条拒
  都可能拒的是这次 YAML 往返而不是那个字段。
- `gate_merge_landed` 的夹具是**从 `staging/arm-b-merged` 取三个真文件**,
  所以「合并到底提不提供这六个标记」是从**写方**验的,不是从闸的期望验的
  ——§3.4 那个缺陷正是从闸的期望造夹具才会漏掉。分支不在时跳过。

三条值得单列:
- **`nvidia-smi` 读不到必须判拒**。「判断不了」倒成通过,会让闸在机器不肯回答
  关于自己的问题时开得最大——而那正是不该动别人卡的时候。
- **空字符串的 `SELFSIGHT_MODEL_ROOT` 判拒**,不判通过:那是 profile 没生效
  留下的形状,不是没设。
- **第二个顶层 `seed` 判拒**。五个 replicate 会落在不同 split 上,
  端点 1 按 (prompt, draw) 跨 replicate 配对,配不上,而且要到分析时才发现。

**19/19 变异全杀。** `v4_e3_launch_preflight.py` 不在 12 个冻结 SOURCES 里,
所以可以就地变异(`-B`、每次清 `__pycache__`、`finally` 还原、
扫完复核字节与基线都对得上,`git diff` 对该文件为空)。
全量 **873** 个测试(+26),退出 0。

**不追加偏离**,理由同 §3.4:这里没有任何注册原文需要人去选一个读法。
六个闸的判据都在 §1 写着,缺的是「拒的那条路从没跑过」。

---

### 3.8 窗口里那 1.5 h 的脚本:接线是对的,但判据从没被跑过(2026-09-09 19:1x)

启动链上还剩一环没查:§0.3 的空载基准 `card_benchmark.py`。
它在主运行结束到 replicate 1 起跑之间那**只开一次的约 1.5 h** 里跑,
两张卡都要空,而 §3.4 之后**闸读的就是它的输出**。

**先查 §3.4 那种接线错误,结论是没有。** 逐条对过,不是假定:

| 它调的 | 在不在 |
|---|---|
| `v4_run_pipeline.py detect --output` | 在(HEAD 581–586 行,arm B 669–674 行,两边都有)|
| `... --overwrite` | 在(同上)|
| `read_detections` 读的 `image_path` / `detections` | 在(对着主运行真的 `detections.internvl.jsonl` 头一行核的:键是 `detections` / `image_path` / `reply`)|

supervisor 自己调 detect 时**不传** `--output` 也不传 `--overwrite`,
所以这两个 flag 是这个脚本独用的——正是最容易只存在于想象里的那种。**它们真的在。**

**但它一次都没跑过**:那个 packet 里只有脚本,没有任何输出。
而且 §3.4 的决定(「闸不重新实现判据,判据归 `card_benchmark.py`」)
让 `verdict()` 成了 §0.3 那条注册规则的**唯一一份实现**,
它**一个测试都没有**。这是「有注册规则、只有一条代码路径、那条路径从没被执行过」。

**能测的一半和不能测的一半分清楚:**
- **不能测的**:计时。四个数要真的在两张空卡上量,只有那个窗口能给。
- **能测的**:拿这四个数做决定。这是纯函数,而它决定 55 个检查点块的排期。

**13 个测试**,钉的是:边界按注册原文是 `wall_S <= 0.90 * wall_P`
(读成 `<` 就是另一条注册规则)、**1-1 分歧按注册的平局解保持按臂分卡**、
**两张卡看到的东西不一样时即使计时大幅胜出也必须拒**
(那时候排期问题让位给正确性问题)、以及 `agree()` 比的是 `detections` 而不是
`reply` 那段散文(比散文会让每张图都报不一致,这个正确性检查就废了)。

**12/12 变异全杀。** 脚本在 `staging/arm-b-merged` 上、又在 review packet 里,
两样都不能改,所以变异是**在 `git show` 之后、import 之前对源码文本做的**,
测试函数直接对着变异模块调用;repo 里没有任何一个字节被动过。

**顺手对上一个数**:`card_benchmark.py` 的 `MARGIN = 0.90` 与
`v4_e3_launch_preflight.py` 的 `CARD_MARGIN = 0.90` 是**同一个数写在两个文件里**,
这是能漂移的最小单位,所以现在有一条测试同时读两处。
(`IDLE_MIB` 两边是 512 与 500,**这个不是缺陷**:一个是基准自己拒绝在有负载时开工,
一个是闸判断卡空没空,两个不同的问题,没有必须相等的理由。)

**不追加偏离**,理由同 §3.4 / §3.7:§0.3 的判据本身清楚,缺的是它从没被执行过。

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

### 3.9 预注册的主检验从来没有过一个可测的输入(2026-09-10 05:3x)

`step-00040.report` 在 05:04:40 落地。八个窗口(两 arm × 四次 look)全部是
`n_paired_specs: 0`、`external.delta: null`、`ci_status:
insufficient_paired_specs`、`candidate: false`,于是
`first_candidate_step: None`。

**这读起来像阴性结果,但它不是。主检验一次都没有被求值过,因为它从来没有输入。**

`read_outcome` 只在一个 spec 的**每一张**图都同时有 `s_select` 和判决时才保留它
(`scripts/v4_decoupling_report.py:165`)。写手不产出这个形状,而且从来没有过。
两个 arm 的每个 checkpoint 上都是:

    images = 256   specs = 64   每 spec 4 张(整齐)
    verified.jsonl = 256 行   每张图都有判决
    s_select.jsonl =  64 行   每 spec 只有被选中的那一张有分
    每 spec 已打分图数分布 = {1: 64}     全部打分的 spec = 0

64 个 `s_select` 路径全在 manifest 里,所以**不是路径对不上,是粒度对不上**。
读手是照着它以为写手会产出的形状写的——与 §3.4、§3.8 同一个失败家族,也是同一条
writer-first 纪律。剩下六轮不会改变它,五个 replicate 会原样继承它。

**修法只改一个判定**:一个 spec 在「至少一张图有分且所有图都已判决」时计入;
`s_select` 改成在有分的那些图上取均值(原为四张取均值),`external` 不变。
`review-packets/dstar-join-granularity-20260910/reanalyze_dstar_join.py`
导入并驱动冻结模块本身——估计量、阈值、bootstrap 次数、种子全部沿用它自己的,
`read_outcome` 先被真实调用一次(重复图检查和 spec-id 校验照跑),之后只重建
`complete_specs`。**运行目录一个字节都没写。**

先用运行自己已经发布过的数验证读手:修好后按 spec 平均的 `s_select` 与
`checkpoint_metrics.csv` 那一列在全部 12 个 checkpoint 上相差 ≤0.01,残差来自
CSV 在 64 个 spec 上平均而修复版在 44–49 个完整 spec 上平均。

修好之后规则能跑了,配对数 44–49。`first_candidate_step` 两个 arm 都变成 **16**;
`first_bootstrap_rule_supported_step` 两个 arm 仍是 **None**。三件事值得记:

1. **两个 arm 卡在规则的相反两半上。** `naive` 在 step 40 通过外部条件
   (`ext ci_high +0.0114 <= 0.02`)而卡在内部条件;`rfo_gold` 在 step 32 和 40
   通过内部条件(`int ci_low +0.0127 / +0.0142 > 0`)而卡在外部条件。
   两半各自都被满足过至少一次,所以**这是精度问题,不是结构性不可达**。
2. **`naive` 在 step 40 差的是算术能表示的最小间距**:它的 2.5 分位是
   `5.05e-18`,即「恰好为零」加上浮点残渣,阈值是 `1e-12`。下界为零的 95% 区间
   不排除零,所以判 not supported **是对的,不许拿去争**。记下来只是因为 44 个
   spec 上的 `s_select` 差分是离散的,「恰好为零」是一个质点而不是巧合。
3. **丢掉的配对就是判决缺口**:每个 checkpoint 有 14–23 张图(共 256)没有判决,
   代价是 64 个 spec 里的 15–20 个。补上它会不会翻转任何一个判定,那是五个
   replicate 上的既定流程要回答的问题,不是这个 packet 能回答的。

**没有动任何冻结文件。** `scripts/v4_decoupling_report.py` 是 12 个冻结 SOURCES
之一,而且 supervisor 在 step-00048 还要调用它;此刻改它会同时踩「改冻结源」和
「改正在跑的脚本」两条。05:04:40 那份 `decoupling_report.json` 原样保留为不可变记录。

**还没定的事(预注册层面,不是我能替他判的)**:修复后的粒度要不要写进预注册、
`s_select` 本来就该是每 spec 一张还是每张图都该打分。这一节只是把缺陷登记为新证据。
