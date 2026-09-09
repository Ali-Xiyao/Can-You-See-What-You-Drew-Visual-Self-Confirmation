# D_g 找不到,而且有两个彼此独立的原因

2026-09-08。起因是一个问题:已经起跑的主运行能不能把 D* 和 D_g 都找出来。
D* 的检出力在 `review-packets/outcome-budget-20260908/` 里算过了,D_g 的没有。
下面是补上的那一半。结论是 D_g 在当前配置下拿不到,而且拿不到的原因有两个,
它们互相独立,修好其中一个不会让另一个变好。

---

## 1. 管道里根本没有 D_g 需要的输入(不是检出力问题,是没有输出)

`estimate_d_g` 要 `noise_low` / `noise_high`。`divergence_report` 明说:
"A gradient curve needs its measured noise floor to yield D_g"。
这两个值由谁写?没有人写。

实测 pilot 的 `checkpoint_metrics.csv`:

```
columns: arm, round_index, step, internal_cycle, internal_sem, internal_n,
         external_correct, external_n, external_unadjudicated,
         external_coverage_policy, s_select, s_select_sem, s_select_n,
         s_select_available, s_select_total

  seed         present=False
  gda_free     present=False
  gda_gold     present=False
  noise_low    present=False
  noise_high   present=False
```

`analysis/formal.py:56` 直接 `pd.read_csv` 这张表,然后:

- `frame.seed` → `AttributeError: 'DataFrame' object has no attribute 'seed'`
- `values.dropna(subset=["gda_free","gda_gold","noise_low","noise_high"])`
  → `KeyError: ['gda_free', 'gda_gold', 'noise_low', 'noise_high']`

两个都是实跑出来的,不是读代码推的。也就是说 `aggregate_formal_e2` 从来没有
在这个 schema 上跑通过——它是照着一张更宽的表写的,而 `v4_train.py` 写的是这张窄表。

还有第三处,被前两处挡住了所以一直没暴露:`formal.py:100` 读
`config.values["gradient_probe"]["ema_alpha"]`,而 `gradient_probe` 块里
没有 `ema_alpha` 这个键(pilot 配置和主配置都没有)。`if len(valid_gda) >= 7`
永远进不去,所以这个 KeyError 一直没机会抛出来。三个缺陷互相遮蔽。

pilot 的 `decoupling_report.json` 也是一致的:`lead.status = "no_event"`,
gradient 段里的 `warning` 是一句说明文字("marginal cosine intervals are never
subtracted to obtain a difference CI"),不是一个 D_g。**D_g 至今没有在任何真实
数据上被估计过。0/2000 的误报率是模拟出来的。**

## 2. 就算管道补好,n=14 也测不出来

pilot 的 `gradient_sensitivity.json`:`n_bank = 16`,`n_target_retained = 14`,
`retained_scene_clusters = 14`。每个 checkpoint 的 bootstrap 区间:

| arm | step | gda_gold [CI] | 宽 | gda_free [CI] | 宽 |
|---|---|---|---|---|---|
| base | 0 | 0.746 [0.229, 0.979] | 0.750 | 0.715 [0.156, 0.982] | 0.826 |
| naive | 8 | 0.715 [0.216, 0.974] | 0.758 | 0.666 [0.150, 0.976] | 0.826 |
| naive | 16 | 0.500 [0.203, 0.978] | 0.775 | 0.402 [0.164, 0.980] | 0.816 |
| naive | 24 | 0.560 [0.274, 0.987] | 0.713 | 0.527 [0.252, 0.985] | 0.733 |
| naive | 32 | 0.713 [0.445, 0.973] | 0.528 | 0.711 [0.434, 0.986] | 0.551 |
| rfo_gold | 8 | 0.696 [0.212, 0.972] | 0.759 | 0.646 [0.148, 0.975] | 0.827 |
| rfo_gold | 16 | 0.561 [0.162, 0.982] | 0.821 | 0.483 [0.126, 0.983] | 0.858 |
| rfo_gold | 24 | 0.655 [0.031, 0.994] | 0.964 | 0.614 [0.020, 0.993] | 0.973 |
| rfo_gold | 32 | 0.766 [0.527, 0.973] | 0.446 | 0.761 [0.520, 0.986] | 0.466 |

点估计在 0.40–0.77 之间游走,而单点区间宽 0.45–0.97。**每个点都和其他每个点相容。**
这条轨迹目前不能支持任何形状判断。

`dg_power.py` 把这个实测区间直接喂给真正的 `estimate_d_g`(EMA α=0.35,
persistence=2,12 个 checkpoint / 88 步,真实 D_g 在 step 35,4000 次):

```
    n   floor     SD      drop=0.00     0.10     0.20     0.40     0.60
   14   0.156  0.211           0.0%     0.0%     0.0%     0.1%     1.1%
   32   0.345  0.139           0.0%     0.0%     0.0%     1.4%    40.6%
   64   0.454  0.099           0.0%     0.0%     0.1%    29.2%    98.0%
  128   0.530  0.070           0.0%     0.0%     1.2%    94.3%   100.0%
  256   0.584  0.049           0.0%     0.1%    29.4%   100.0%   100.0%
```

`drop=0.00` 那一列是误报率,其余是检出力。**n=14 那一行:即使梯度对齐从 0.715
崩到 0.115,也只有 1.1% 的概率报警。** 0/2000 的误报率和接近零的检出力是同一件
事的两面——噪声地板落在 0.156,规则只有在余弦塌到接近正交时才可能触发。

判据是 `drop > G0 - floor`:

```
  n=  14  需要 gda_free 下跌超过 0.559
  n=  32  需要 gda_free 下跌超过 0.370
  n=  64  需要 gda_free 下跌超过 0.261
  n= 128  需要 gda_free 下跌超过 0.185
  n= 256  需要 gda_free 下跌超过 0.131
```

## 3. 而且 n 被冻结 split 卡死在 32

`configs/v4_decoupling_main_20260908.yaml`:`local_probe: 32`,
`gradient_probe.size: 16`。probe 池一共只有 32 条,现在只用了 16 条。

- 16 → 32 是免费的(池子里现成的),但只把检出力从 1.1% 抬到 40.6%,
  且只在 drop=0.60 这种崩塌情形下。drop=0.40 时还是 1.4%。
- 要到 n=64,必须改 split。而 `split_digest` 是
  `sha256({runs, seed, local_outcome, local_probe})`,改 `local_probe` 就换了
  digest,和 pilot 的 `9bab14d4...` 不再一致,已经跑完的部分不能和新的拼在一起。
- 228 条语料里 outcome 64 / probe 32 / train 132 已经排满,n=64 的 probe 只能从
  train 里挪,train 掉到 100 会同时缩短 D* 的可用步长跨度。

**这不是加算力能解决的,是语料规模的问题。**

---

## 结论与待定

1. 已起跑的主运行(`runs/v4/decoupling-main-20260908`)**不会产出 D_g**。
   `Lead = D* − D_g` 因此也拿不到。这一点在跑之前就成立,和跑得顺不顺无关。
2. D* 不受影响,继续跑有意义。
3. 三个管道缺陷(缺列、缺 `seed`、缺 `ema_alpha`)应登记为新证据,
   不改写任何既有记录。修好它们只是让 D_g 能被"算出来",不会让它变得能被"看出来"。
4. 需要用户裁定的是预注册层面的问题:在 n 最多只能到 32 的前提下,
   D_g 和 `Lead` 是继续作为主结果的一部分,还是降级为探索性描述。
   我不擅自改 §25 / §27 / §34。

---

# 更正(同日,晚些时候):上面第 2 节的诊断是错的,而真正的障碍在别处

上面第 1 节(管道缺三样东西)实测有效,不动。第 2 节的「n=14 检出力 1.1%」
这个结论要收回一半,原因是我把「已发布的那个估计量测不出来」当成了
「梯度预警测不出来」。这两件事不一样,下面是分开之后的数。

## 4. `gda_free` 报的不是它名字说的那个量

`cosine(mean_L, mean_R) = Σᵢⱼ lr / √(Σᵢⱼ ll · Σᵢⱼ rr)`。14×14 = 196 项里只有
14 项在对角线上,所以这个量的绝大部分由**不同 prompt 之间的梯度是否互相对齐**
决定,而不是同一个 prompt 的左右梯度是否对齐。

`gradient-probes/{arm}/step-XXXXX/grams.npz` 存了完整的 16×16 `ll`/`rr`/`lr`,
所以逐 prompt 余弦 `cos_i = lr[i,i]/√(ll[i,i]·rr[i,i])` 可以直接在 CPU 上重算
(`per_prompt_cosine.py`,不需要 GPU,不需要重跑探针):

```
arm       step   cosine-of-means     mean per-prompt cosine     paired delta vs base
base         0            0.7147             0.7570 +- 0.0815         +0.0000
naive        8            0.6661             0.7430 +- 0.0785         -0.0140 +- 0.0166
naive       16            0.4025             0.7449 +- 0.0758         -0.0121 +- 0.0489
naive       24            0.5268             0.7692 +- 0.0763         +0.0122 +- 0.0809
naive       32            0.7113             0.8456 +- 0.0610         +0.0886 +- 0.0560
rfo_gold     8            0.6460             0.7426 +- 0.0785         -0.0144 +- 0.0166
rfo_gold    16            0.4835             0.7370 +- 0.0790         -0.0201 +- 0.0490
rfo_gold    24            0.6139             0.7434 +- 0.0894         -0.0136 +- 0.0834
rfo_gold    32            0.7609             0.8535 +- 0.0563         +0.0965 +- 0.0552
```

**step 16 那个 0.71 → 0.40 的坑,逐 prompt 看根本不存在。** 它是聚合方式的
产物,不是模型的行为。上一节表格里那条「点估计在 0.40–0.77 游走」的轨迹,
以及基于它的所有形状判断,读的是一个名字和内容对不上的量。这一条按缺陷登记流程
处理:旧记录原样保留,缺陷作为新证据登记,重测并注明来源。

配对逐 prompt 的 SEM 中位数 **0.0521**,对比已发布区间隐含的 **0.2108**,紧 4.0 倍,
而且它是 14 个有界量的普通样本均值,不再是病态比值统计量
(病态的直接证据:naive/step-16 的点估计 -0.3122 落在自己 CI 下界 -0.3095 之外)。

## 5. 换成正确统计量 + D* 自己的 knot 机器之后的检出力

规则也一并换掉:不用「连续两点跌破绝对地板」,改用 `fit_segmented` +
`break_support`(校准 knot 搜索的残差 bootstrap)+ 后段斜率为负。
这让 `Lead = D* − D_g` 变成同一个校准过程下两个 knot 的差
(`dg_power_perprompt.py`,12 ck / 88 步,真 D_g 在 step 35,4000 次):

```
    n     SEM     drop=0.00     0.05     0.10     0.15     0.20
   14  0.0521          2.4%     5.5%    10.2%    17.1%    27.8%
   32  0.0345          2.1%     7.6%    16.0%    32.4%    53.4%
   64  0.0244          2.6%    11.0%    28.9%    55.5%    80.8%
```

n=32 是免费的(split 里现成 32 条,现在只用 16 条)。所以不是 1.1%,是
drop=0.20 时 53%。**但也不是「能测」,是「勉强」。** 另外误报率 2.1–2.4% 而不是 5%,
说明合取规则偏保守,改成单边临界值还能再拿回一些检出力。

pilot 自己在 32 步内没有任何下跌可测:-0.014 → -0.012 → +0.012 → **+0.089**,
两个 arm 在 step 32 都是**上升**的。主运行跨到 88 步。

## 6. 真正的障碍:Lead 本身的分辨率(`lead_resolution.py`)

两个 knot 都是在同一个 12 点网格上搜出来的。单个 knot 的抽样 SD:

```
curve                        move     SEM  move/SEM   knot SD  on target   Lead SE
external, rise 0.10 R=4     +0.10  0.0150       6.7      11.3      29.8%      15.9
gradient, drop 0.20 n=32    -0.20  0.0345       5.8      12.1      27.0%      17.1
gradient, drop 0.10 n=32    -0.10  0.0345       2.9      14.5      18.2%      20.5
```

**即使 SNR=6.7,knot 落在正确格点上的概率只有 29.8%。** Lead 的单 seed SE
≈ 16–18 步,而总跨度只有 88 步——要和零区分开,真 Lead 得超过 ~35 步,
也就是整个训练跨度的 40%。

加 checkpoint 救不了,这一点是实测的:

```
 checkpoints   grid   knot SD   Lead SE  need Lead >
          12    8.0      11.4      16.1         32.1
          24    3.8      11.0      15.6         31.2
          48    1.9       7.2      10.2         20.4
```

12 → 24 个 checkpoint,knot SD 从 11.4 只动到 11.0。**knot 的位置受限于 SSE
把它放在哪里,不受限于采样有多密。** 所以这不是预算问题,也不是语料问题,
多花的 GPU 时间不会变成分辨率。

## 7. 已注册的 Lead 端点是反保守的

`paired_bootstrap_lead` 对 3 个数的中位数做 20000 次 bootstrap。3 个值重采样后
中位数只能取那 3 个值之一,所以 2.5%/97.5% 分位就是 min 和 max——
**报出来的「95% CI」就是三个 seed 的极差。** 实测(每 seed SE 17.3,各 400 次):

```
true Lead= 0 -> CI 中位宽 25.7 步, CI 排除零 的比例  9.5%   <-- 不是 5%
true Lead=16 -> CI 中位宽 28.9 步, CI 排除零 的比例 53.5%
true Lead=32 -> CI 中位宽 25.7 步, CI 排除零 的比例 91.0%
```

它同时是反保守的(I 类错误 9.5%)和低效的(真 Lead 要到 32 步才有 91%)。

## 8. 修正后的结论

- 第 1 节的三个管道缺陷:不变,仍需登记和修复。
- 「D_g 找不到」:**收回**。换统计量后 n=32 / drop=0.20 有 53%,不是 1.1%。
- 「D_g 是主要风险」:**收回**。主要风险是 `Lead` 这个端点本身分辨率不够,
  而且加算力、加 checkpoint、加 seed 都改善有限(3 seed 只把 SE 从 17.3 降到 10.0)。
- 换 endpoint 比换算力有用得多:序关系 `P(D̂_g < D̂*)` 在真 Lead=24 时单 seed 已有
  91.7%,而同样条件下报数值差几乎必然给出一个跨零的区间。
