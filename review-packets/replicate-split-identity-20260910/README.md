# replicate 的 split 与主运行,启动时没有任何东西在比

**2026-09-10 10:4x–11:0x,主运行仍在跑(`naive.step-00048`,elapsed 50.4 h)。**
本 packet 不碰工作区里的任何冻结件,也不碰 `staging/arm-b-merged`:
被改的那个文件在这条分支上根本不存在。

登记出处:`.planning/2026-09-08-iclr-redesign/EXECUTION.md` §1 尾部
「**还剩的口子,以及为什么现在不补**」。那一段自己写明了补在哪、以及现在补不了的理由
(同路径新建会造成 add/add 冲突)。本 packet 做的是**把补丁和测试提前做完并验过**,
落地仍等合并。

## 1. 登记时已知的两个口子

`scripts/v4_verify_replicates.py`(只在 `staging/arm-b-merged` 上,
blob `b5fd8b94bde40dcd4861aaf2fcb9b3a9ed63899f`)的 `verify()` 判两件事:
五个 replicate 的 training seed 两两不同、且它们的 split 一致。

- **它只比五个 replicate 彼此。** 五份完美互相复制、但复制的是另一个实验的 split,
  它照样输出 `verdict: "five seeds"`。偏离 7.2 的守卫会比 main 和 replicate,
  但那在 `load_columns` 里,是**分析时**——400 GPU-h 花完之后。
- **它比的是配方 `digest`,不是 identity。** 三个哈希的区别钉在
  `tests/test_main_run_split_identity.py`:`digest` 是配方
  `(runs, seed, outcome, probe)` 的哈希,`identity` 是「除 `created` 外全部字段」的哈希,
  也就是配方**和配方产出的 prompt 列表**。

## 2. 今天新查出来的:那个条件恒为真

`v4_train.split_digest(config, runs)` 是 `sorted(runs)`、`config["seed"]`、
`data.local_outcome`、`data.local_probe` 四项的纯函数——**不读任何数据**。
`run_decoupling_pilot.py:26` 的 `RUNS` 是模块级常量,每个运行都一样。
五份 replicate 配置共用 partition seed(这正是 `v4_e3_launch_preflight.gate_configs`
断言的),`local_outcome`/`local_probe` 也相同。

于是**六份配置(主运行 + 五个 replicate)的配方 digest 是同一个数**,
从配置文本就能算出来,不需要跑过任何一次 split stage:

```
v4_decoupling_main_20260908.yaml       seed=20260906 out=64 probe=32 -> 9bab14d427f2b61b
v4_e3_replicate_s20260906.yaml         seed=20260906 out=64 probe=32 -> 9bab14d427f2b61b
v4_e3_replicate_s20260907.yaml         seed=20260906 out=64 probe=32 -> 9bab14d427f2b61b
v4_e3_replicate_s20260909.yaml         seed=20260906 out=64 probe=32 -> 9bab14d427f2b61b
v4_e3_replicate_s20260910.yaml         seed=20260906 out=64 probe=32 -> 9bab14d427f2b61b
v4_e3_replicate_s20260911.yaml         seed=20260906 out=64 probe=32 -> 9bab14d427f2b61b
```

所以 `split_shared` 不是「较弱的检查」,**是一个不可能为假的条件**。
它唯一能抓到的是配置差异,而配置差异在更早、更好的地方(preflight 的 `gate_configs`)
已经拦了。文件的 docstring 写「Two conditions, and both are required」,
其中一条是同义反复。这和 `read_outcome` 的 R>1 空出口是同一族:
**一段永远不会失败的判据,而没有任何东西会说出来。**

它放过的真实情形是可构造的:`--runs` 下的某个源运行在五次启动之间(§11 登记的
304–352 h 窗口)多了一个 prompt,`split_prompts` 于是产出不同的 outcome 列表,
配方一字未改。实测(第 5 节最后两行):合并树那份对这种数据输出
`verdict='five seeds'`、`split_shared=True`。

## 3. 补丁

`split_check.patch`,−17/+53 行,`VERSION` 从 `1` 升到 `2`。三处实质改动:

- 比较量从 `split.json["digest"]` 换成 `selfsight.analysis.drift.split_digest(run)`
  (identity)。配方 digest 仍然报,但只作为信息:配方同 / identity 异 = 语料动了,
  配方异 = 配置动了,两者要能分开。
- `verify(runs, main=None)` 增加第三个条件 `split_matches_main`。
- **命令行上 `--main` 是必填。** `verify()` 里保持可选,是为了让合并树自带的六个测试
  一个字节不改地继续通过;这只有在人走的那条路无法跳过它时才安全,
  所以有一个测试专门盯着命令行会拒绝。

`v4_verify_replicates.patched.py` 是打完补丁后的完整文件,随包附带,
用来证明补丁和它描述的东西一致。

## 4. 测试

`tests/test_replicate_split_matches_main.py`(**已在本分支落地**,6 个测试)。

- `test_the_recipe_digest_is_one_number_for_all_six_configs` **不跳过**,
  今天就在这条分支上跑。它是第 2 节那张表的可执行形式:哪天有人给某个 replicate
  单独的 partition seed 或不同的 outcome/probe 数,它当场红。
- 另外五个 `skipif(not VERIFIER.exists())`,理由字符串直接写明
  「arrives with the staging/arm-b-merged merge」。合并一落地它们自动生效——
  **补丁没打就是红的**,这就是逼落地的人回来的机制。

## 5. 验过了什么

`verify_split_check.py`,从仓库根跑:

```
envs/core/python.exe -B review-packets/replicate-split-identity-20260910/verify_split_check.py
```

```
staging/arm-b-merged:scripts/v4_verify_replicates.py is blob b5fd8b94bde4 as recorded
git apply reproduces the shipped copy exactly (+2323 bytes)

the 6 tests shipped with the merged file, against the patch:
  PASS  test_five_directories_that_all_trained_on_one_seed_are_refused
  PASS  test_distinct_seeds_on_different_splits_are_refused_too
  PASS  test_distinct_seeds_on_one_split_pass
  PASS  test_a_run_with_no_finished_round_is_an_error_not_a_pass
  PASS  test_a_run_that_changed_seed_midway_is_an_error
  PASS  test_one_run_cannot_be_checked_against_itself

the 5 new tests, against the merged file as it stands:
  FAIL  test_five_replicates_that_agree_only_with_each_other_are_refused
  FAIL  test_one_replicate_off_the_shared_split_is_caught_by_identity
  FAIL  test_the_wall_clock_stamp_is_not_part_of_the_comparison
  FAIL  test_the_command_line_refuses_to_run_without_the_main_run
  PASS  test_the_identity_pinned_for_the_main_run_is_what_this_compares

the 5 new tests, against the patch:  5 PASS

one replicate held out a different prompt set, same recipe:
  merged file : verdict='five seeds'  split_shared=True
  patched     : verdict='NOT independent seeds'  recipe_digests_shared=True
```

四个新测试**先验红**(见 memory `verify-a-new-test-fails-against-the-old-code`)。
第五个 `test_the_identity_pinned_...` 两边都绿,它不碰这个 verifier
——它只把本文件和 `test_main_run_split_identity.py` 钉的第三个哈希绑在一起,
照实说明,不算红先验。

真正的红是最后两行那个判定,不是三个 `TypeError`。签名变了只证明签名变了。

## 6. 顺带查出的一件事:`git apply` 也吃 `.gitattributes`

第一次跑 `verify_split_check.py`,`git apply` 成功、结果却和随包的那份对不上:
沙箱里没有 `.gitattributes`,`core.autocrlf=true` 下 **`git apply` 把 123 行全写成了 CRLF**。

今早 `SECOND-OUTLET.md` §7 加的是 `*.patch text eol=lf`——补丁**自己**必须是 LF 才打得上。
今天这条是另一半:**目标文件也必须被钉成 `eol=lf`,否则 `git apply` 写回去的是 CRLF**。
`.gitattributes` 里 `*.py text eol=lf` 早就有,所以真实落地是安全的;
沙箱现在会把它复制进去,不然沙箱就不是落地路径的模型。

对 `read_outcome_repair.patch` 同样成立:它的目标 `scripts/v4_decoupling_report.py`
是冻结 SOURCE,`run_decoupling_pilot.py:193-196` 每个 stage 开头核它的 sha256。
若 `*.py` 没被钉,打完补丁得到的是整文件 CRLF 重写。

## 7. 落地时机与命令

**不是现在。** `scripts/v4_verify_replicates.py` 在这条分支上不存在,
同路径新建会造成 add/add 冲突。顺序是:主运行 `pilot_complete`
→ 打 `read_outcome_repair.patch` → 摘三个 `NEEDS_REPAIR`
→ 合并 `staging/arm-b-merged`(非 fast-forward)→ **然后**:

```
git apply review-packets/replicate-split-identity-20260910/split_check.patch
envs/core/python.exe -B -m pytest tests/test_replicate_split_matches_main.py tests/test_v4_verify_replicates.py -q
```

合并后 `staging/arm-b-merged:scripts/v4_verify_replicates.py` 若不再是
blob `b5fd8b94…`,`verify_split_check.py` 第一行就会拒绝——补丁不许打在别的东西上。

replicate 1 的 split stage 跑完之后,拿它和主运行比:

```
envs/core/python.exe scripts/v4_verify_replicates.py runs/v4/e3-s20260906 --main runs/v4/decoupling-main-20260908
```

只有一个 replicate 时 `verify()` 会以 "at least two" 拒绝(合并树自带的测试钉了这条),
所以第一次比用 `tests/test_main_run_split_identity.py` 的 `41c5f1b2…`,
五个都跑完再用上面这条命令。

## 8. 这些改动不许可什么

- **不改任何已有判定。** 六个既有测试逐字通过;`verdict` 的两个旧条件原样保留。
- **不碰主运行、不碰 `staging/arm-b-merged`、不碰任何冻结 SOURCE。**
  合并树的那份是从对象库里读出来的,补丁打在临时目录里。
- **不是显著性证据。** 这是一个仪器,不产生任何数。
- **`--main` 必填不等于比过了。** 报告里 `checked_against_main` 是独立字段,
  论文里若引用「五个 seed 落在同一 split 上」,要引 identity 那一栏,不是配方 digest。
