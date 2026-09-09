# 补上 D\* 的耦合前置条件

2026-09-08,高 N 轮结束之后。用户指示「先补 slope_before 缺口」。

## 1. 缺陷

`estimate_d_star` 算了 `slope_before` 却从不看它。接受条件只查 `slope_after`:

```python
if internal_candidate.slope_after <= min_positive_internal_slope:
    continue
if external_candidate.slope_after > 0.0:
    continue
```

于是**一条从第 0 步就没在涨的外部曲线,只要内部曲线在某处向上折,就被判为脱钩事件**。
但 D\* 的意思是「训练把耦合打断了」,这预设前面**存在**耦合——正是你补的那个前提。

`.planning/2026-09-04-project-audit/findings.md:131` 早就写过这一条
(「does not require a positive external pre-slope」),
`review-packets/bon-coupling-20260908/dstar-precondition-defect.json` 用合成数据坐实了它:
从不耦合的平曲线和单调下降的曲线**都返回了确信的 D\* = 32**。

## 2. 改法

**`src/selfsight/analysis/breakpoints.py`**

- 新增必要条件 `external_candidate.slope_before > min_coupled_external_slope`,
  与另外两条并列。
- 新增参数 `min_coupled_external_slope`(默认 0.0,与既有的
  `min_positive_internal_slope` 同一约定:估计量自身只做符号检验,调用方给实测下限)。
- `DivergenceEstimate` 新增 `external_pre_slope` / `coupled_candidates` /
  `admissible_candidates`(都带默认值,不破坏既有构造点)。
- **区分两种失败**,这是这次改动里最要紧的一点:
  - 没有任何候选折点存在耦合期 → 「**No coupled phase** … D\* is undefined here,
    which is not the same as a decoupling that was looked for and not found.」
  - 有耦合期但没有脱钩 → 「N of M admissible breakpoints have a coupled phase, but …」

  「前提不成立所以问题无意义」和「找了没找到」是两个不同的科学陈述,不能都写成 `d_star=None`。

**`src/selfsight/v4/evaluate.py`**

- 新增 `external_noise_slope()`,镜像既有的 `internal_noise_slope()`。
  外部正确率是比例而不是自带 SEM 的均值,所以精度取二项式:
  逐 checkpoint `sqrt(p(1−p)/n)`,**取中位数**,除以 step 跨度。
  取中位数是为了不让某个 p=0 或 p=1 的 checkpoint 把下限压塌。
- `divergence_report` 新增 `min_external_slope`(默认用实测下限),
  `DivergenceReport` 与 `to_dict()` 一并带出,`divergence.json` 自动多出这几个字段。

**为什么下限不能是 0**:和内部那条完全同一个理由。一条平的外部曲线拟合出的斜率是
3.9e-05,**是正的**,符号检验放行。实测下限是 5.5e-04,才拦得住。

## 3. 重测(旧记录原样保留)

`verify_fix.py` / `verify-fix.json` 重放同样三个合成案例,**不改动**原缺陷记录,
并在输出里指名其为来源。合成曲线没有裁定人数,下限按 n=120(outcome 集大小)重建,写死在脚本里。

| 案例 | 应接受 | 原记录 | 只做符号检验 | 用实测下限 |
|---|---|---:|---:|---:|
| A 从不耦合(平) | 否 | **32.0** ✗ | **32.0** ✗ | **None** ✓ |
| B 真脱钩 | 是 | 40.0 ✓ | 40.0 ✓ | **40.0** ✓ |
| C 全程下降 | 否 | **32.0** ✗ | None ✓ | **None** ✓ |

**A 只有实测下限拦得住**(pre-slope 3.86e-05 vs 下限 5.51e-04)——所以加参数是必要的,
光加条件不够。**B 拿到的 D\* 一字不差还是 40.0**,加必要条件不会误伤真信号。

## 4. 测试

`tests/test_v4_evaluate.py` 全套 515 项通过。新增四项:

- 全程下降的外部曲线 → 无 D\*,`coupled_candidates == 0`,理由是 "No coupled phase",
  且内部斜率仍在噪声之上(证明拒绝来自前置条件,不是别的)。
- 从头就平的外部曲线 → `min_external_slope=0.0` 仍给出 D\* = 100.0,实测下限拒绝。
- 外部下限 = 二项式 SEM 中位数 / 跨度。
- 没有已裁定 prompt 的曲线 → 拒绝给下限,而不是默默用 0。

**改了一项既有测试。** `test_a_flat_internal_curve_yields_no_d_star_and_says_why`
原来的外部曲线是从第 0 步开始下降的——就是案例 C。它断言
`min_internal_slope=0.0` 时得到 D\* = 75.0,而那个 75.0 正是本次修掉的缺陷行为。
该测试要验的是**内部**下限,外部前提失败会把它变成混淆测试,
所以把 fixture 换成「先涨后停」的真耦合曲线:零下限照样给出 D\* = 75.0
(斜率 2.04e-18),实测下限照样拒绝,**测试原意一字未改,只是不再被前提问题干扰**。
原来那条下降曲线单独成为一项新测试。

## 5. 对已冻结记录的影响

**没有需要改写的。** 加一条必要条件只能**撤回** D\*,不可能造出新的;
本项目从未产出过任何 D\* 数值,所以没有旧结论需要重测。

STATUS §34 的终态 A(「`divergence.json` 里至少一个臂给出非 None 的 `d_star`」)
因此**更难达成**,方向是保守的。§34 本身不改一个字。
`v4_l3_verdict.py` 无需改动:它已经把 `reason` 原样打出来,新的「No coupled phase」
诊断会自动出现在裁定输出里。

**未纳入本次改动**(是缺口,不是这次的任务):`fit_segmented` 没有无折点的零模型对照,
所以它永远会返回某个折点;案例 B 在真折点为 32 的无噪数据上把折点定在了 40.0。
这两条都是折点**位置**的问题,与本次修的前提条件不同层,
仍记在 `.planning/2026-09-04-project-audit/findings.md:131`。
