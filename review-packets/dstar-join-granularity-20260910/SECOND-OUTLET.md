# 同一处缺陷的第二个出口,以及可应用的修复

**2026-09-10 09:4x 追加进本 packet。** 原有文件一个字节未动
(`README.md`、`reanalyze_dstar_join.py` 及其输出)。本文加的是三样东西:
缺陷的第二个出口、一个可直接应用的补丁、以及补丁跑在主运行数据上的结果。

登记在 `docs/prereg/2026-09-10-join-granularity.md` §9。

## 1. 第二个出口

`README.md` 记的是 `complete_specs` 那条链——预注册的 D\* 主检验没有输入。
`read_outcome` 里还有第二处用同一个判据:

```python
"s_select": sum(scores) / size if len(scores) == size else None,
```

R=4 而按协议 §3 只有第 0 张有分时,`len(scores)=1 != size=4`,
**`spec_measurements[...]["s_select"]` 对每个 spec 恒为 None**。
它喂 `full_population_screen` 的 `internal` 列表,于是那条链整条为空。

主运行冻结报告的 robustness block 逐字为证:

```
"n_external_paired_specs": 44,   "n_internal_paired_specs": 0,
"internal_pair_coverage": 0.0,   "internal_all_available_delta": null,
"internal_all_available_bootstrap_ci": null,   "internal_supported": false
```

外部那半有 44/64 的输入,内部那半**由构造为空**。
`internal_supported: false` 看起来像测量结果,是结构性的零。

## 2. 补丁

`read_outcome_repair.patch`,两处改动,共 5 行。
`verify_join_repair.py` 把补丁打在源码文本上(**不碰仓库里的文件**——
`scripts/v4_decoupling_report.py` 是冻结 SOURCE,主运行每个 stage 开头都核它的摘要),
再拿 `tests/test_read_outcome_join_granularity.py` 的九个测试逐个跑:

```
both anchors matched exactly once; patch is -109 chars
9 tests, 3 of them currently xfail
PASS  xfail->test_a_spec_scored_on_its_first_draw_and_fully_adjudicated_is_counted
PASS  xfail->test_the_internal_mean_divides_by_the_scored_images_not_the_drawn_ones
PASS  xfail->test_spec_measurements_carries_the_first_draw_score
PASS         test_one_unadjudicated_image_drops_the_whole_spec
PASS         test_a_spec_with_no_scored_image_is_not_counted
PASS         test_spec_measurements_external_still_requires_every_image
PASS         test_the_dead_bounds_are_left_alone
PASS         test_the_pilot_shape_is_untouched
PASS         test_the_writer_still_stamps_first_draw_only
```

那三个 `xfail(strict=True)` 在补丁落地后会变成 XPASS,pytest 报成失败,
**逼落地的人回来摘标记**。一个永远不会 XPASS 的 strict xfail 是废的,所以先验过它会响。

## 3. 补丁复现了本 packet 的重分析

`reanalyze_dstar_join.py` 是 monkeypatch,补丁是改代码。两者若不一致,
补丁就不是被登记的那个东西。把补丁后的 `build_report` 跑在同一个运行目录上:

| arm | end | n | int Δ | int ci_low | ext Δ | ext ci_high | cand | sup |
|---|---|---|---|---|---|---|---|---|
| naive | 16 | 48 | +0.0208 | −0.0243 | +0.0156 | +0.0417 | yes | no |
| naive | 24 | 45 | +0.0056 | −0.0407 | +0.0056 | +0.0333 | no | no |
| naive | 32 | 49 | +0.0170 | −0.0102 | +0.0051 | +0.0306 | no | no |
| naive | 40 | 44 | +0.0398 | +0.0000 | −0.0057 | +0.0114 | yes | no |
| rfo_gold | 16 | 45 | +0.0222 | −0.0241 | +0.0056 | +0.0389 | yes | no |
| rfo_gold | 24 | 45 | +0.0000 | −0.0444 | −0.0111 | +0.0222 | no | no |
| rfo_gold | 32 | 46 | +0.0399 | +0.0127 | +0.0109 | +0.0380 | yes | no |
| rfo_gold | 40 | 47 | +0.0496 | +0.0142 | +0.0160 | +0.0585 | yes | no |

**逐格与 `README.md` 的表一致**,包括 naive/40 那个显示成 `+0.0000` 的
`+5.046468293750712e-18`。`first_candidate_step` 两臂都是 16,
`first_bootstrap_rule_supported_step` 两臂都是 None。**主结论一个字不动。**

## 4. 第二个出口修好之后,新出现的数

`n_internal_paired_specs` 从 **0** 变成 **64**(全体;内部曲线不吃裁定缺口,
因为它只要第一张图有分):

| arm | end | robustness 内部 Δ | internal_supported |
|---|---|---|---|
| naive | 16 | +0.0195 | false |
| naive | 24 | +0.0091 | false |
| naive | 32 | +0.0182 | false |
| naive | 40 | +0.0299 | false |
| rfo_gold | 16 | +0.0156 | false |
| rfo_gold | 24 | +0.0013 | false |
| rfo_gold | 32 | +0.0299 | **true** |
| rfo_gold | 40 | +0.0391 | **true** |

判据是 `internal_n == n` 且 `mean >= 0.02` 且 bootstrap CI 下界 > 0。
naive 在 step-40 也有 +0.0299,但 CI 下界没过零,所以是 false——
和主分析里 naive 那个 `+5.05e-18` 是同一个现象:**naive 的内部效应一直压在零上**。

## 5. 这些数不许可什么

- **不改 D\* 的判定。** `supported` 要内外两半同时成立,外部那半在任何窗口都没过。
  这张表只说明 D\* 的「内部上升」那一半在全体 64 个 spec 上有支撑,**在 rfo_gold 臂**。
- **这是 supplementary,不是登记的确认。** 该 block 自称
  `supplementary_robustness_check_not_registered_confirmation`,写论文时要照此标注。
- **独立性假设未满足。** block 自带
  `"independence_assumption": "independent spec pairs; canonical-scene duplicates
  invalidate this assumption"`。scene audit 没有为这条 bootstrap 背书。
- **单 seed。** 跨 seed 的显著性只能来自 §34 登记的五 seed 符号检验,不得对 seed 做 bootstrap。
- **主运行的 `decoupling_report.json` 不重跑、不覆盖。** 上面所有数字来自
  只读地重跑 `build_report`,输出没有落盘到运行目录。

## 6. 什么时候落地

**不是现在。** `scripts/v4_decoupling_report.py` 在 `SOURCES` 里,
`run_decoupling_pilot.py:193-196` 每个 stage 开头都核 `source_sha256`,
主运行 2026-09-10 09:4x 仍在跑。补丁在主运行到达 `pilot_complete` 之后应用;
五个 replicate 于是在构造时自然冻结修复后的指纹,**不需要 `--accept-code-update`**。
详见预注册 §6。

## 7. 补:补丁差点在需要它的那一刻才失效(2026-09-10 10:0x 追加)

上面第 2 节说「九个测试」,现在是十个。第十个不测 `read_outcome`,测**这个补丁文件本身**。

`git apply` 是把上下文行逐字节比对 `scripts/v4_decoupling_report.py`,
而 `.gitattributes` 把 `*.py` 钉成 `eol=lf`。但 `*.patch` **没被钉**,
落到 `* text=auto` + `core.autocrlf=true`——**新 checkout 出来的补丁会是 CRLF**。
实测:把本补丁转成 CRLF 后 `git apply --check` 死在第一个 hunk,
`error: patch failed: scripts/v4_decoupling_report.py:153`。

危险的地方是时点。工作区里这份现在是 LF(写它的时候就是 LF),所以今天一切正常;
坏掉要等到某次 checkout / clone / stash 之后,也就是 `pilot_complete` 那天
——**补丁唯一要用的那一刻**。这和本 packet 记的缺陷是同一族:
一段永远不会执行或永远不会成功的东西,而没有任何检查会说出来。

改动两处,都不碰任何冻结件:

- `.gitattributes` 追加 `*.patch text eol=lf` 与 `*.diff text eol=lf`(+7/0)。
  本补丁是仓库里唯一的 `.patch`,该规则不波及其它文件。
  验证方式是把工作区副本删掉、`git checkout --` 取回:CRLF=0,与提交前逐字节相同,`git apply --check` 通过。
- `tests/test_read_outcome_join_granularity.py::test_the_patch_is_checked_out_with_lf_endings`。
  它同时断言**字节**与 **`git check-attr` 的属性**:删掉 `.gitattributes` 那行以后,
  工作区字节要到下一次 checkout 才变坏,只查字节的测试会在那个提交上保持绿。
  两个断言分别验过会红(`has CR bytes` / `eol: unspecified`)。

**上面 1–6 节一个字节未改。** 第 3 节那张复现表、第 4 节那八行数、
`README.md` 和 `reanalyze_dstar_join.py` 全部原样。
