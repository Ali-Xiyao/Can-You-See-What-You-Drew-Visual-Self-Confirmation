# 模型能看见自己画的东西——直到你提醒它本来要画什么

2026-09-08。零新测量:`v4_run_pipeline.py observe` 两种语境都已经跑过,
结果一直在 `runs/v4/main-2plus1/` 和 `runs/v4/main-1plus1plus1/` 里。
`context_ablation.py` 只是读盘。

## 设置

`v4/questions.py` 按 **detections 打分,从不按 spec 打分**,并逐题记录 `gold_source`:

- `spec_matches_image` — spec 和像素一致。「读图」和「背 prompt」预测同一个答案,
  这种试次**分辨不了**两个假设。
- `image_differs_from_spec` — spec 说一样,像素说另一样。**读图的对,背 prompt 的错。**
  这是判决性试次。

两种语境:`image_only`(只给图和问题)vs `prompted`(同一个问题,前面加上生成用的 prompt)。

## 结果

```
gold_source               image_only                      prompted                     delta
spec_matches_image        0.979 [0.973,0.984] n=2281      0.990 [0.986,0.994] n=2306   +0.011
image_differs_from_spec   0.654 [0.617,0.689] n=667       0.336 [0.301,0.372] n=682    -0.318
image                     0.839 [0.808,0.865] n=633       0.772 [0.737,0.803] n=635    -0.067
```

配对(同一张图、同一道题,两种语境),精确 McNemar:

```
gold_source                 pairs  blind only  told only         p
spec_matches_image           2281           9         35  1.06e-04
image_differs_from_spec       667         221          9  5.12e-54
```

**221 道题,模型蒙着 prompt 时答对、告诉它 prompt 之后答错;反方向只有 9 道。**

按 family 拆判决性子集:

```
family        image_only                      prompted                     delta
counting      0.824 [0.764,0.872] n=188       0.256 [0.201,0.321] n=199    -0.568
binding       0.797 [0.688,0.875] n=69        0.493 [0.378,0.608] n=69     -0.304
existence     0.500 [0.358,0.642] n=44        0.292 [0.182,0.432] n=48     -0.208
absence       0.557 [0.506,0.607] n=366       0.355 [0.308,0.405] n=366    -0.202
```

**counting:模型能数清自己画了几个(82.4%);告诉它本来要画几个,掉到 25.6%。**

## 为什么这不是「prompt 让它分心」

对照条件排除了这个解释。在 `spec_matches_image` 上,prompt 是**帮忙**的
(0.979 → 0.990,+1.1 点)。prompt 不是噪声——它**恰好在感知与 prompt 冲突时
压过感知**,而在两者一致时提供有用的先验。

这是一个干净的分离,不是一个梯度效应。

## 与解耦运行的关系

上面是**基座模型**(语料 run)。解耦 pilot 上另有一组:
`s_select` 的判别 gap 在 32 步生成训练内从 +0.047 塌到 **0.000**,两个 arm 都是
(`review-packets/dg-power-20260908/prompt_recital.py`)。

合起来:**缺陷在初始化时就存在,生成训练把它变成彻底的。**

## 直接后果

统一模型的自我改进回路,在验证时 prompt 就在上下文里——`s_select` 就是这么做的。
所以标准回路是在 **33.6%** 的感知准确率上运行,而 **65.4%** 本来就拿得到。
**监督信号在被使用之前就已经被污染了,而污染源是一个可以直接去掉的东西。**

## 局限

- 这是单一模型族(Show-o2-1.5B)。跨规模复制是纯推理,尚未做。
- `image_only` 的 65.4% 远不是天花板,说明去掉 prompt 只恢复了一部分感知。
- 判决性试次按构造偏向生成失败的图,不是 spec 的随机样本;
  它回答「当图错了模型知不知道」,不回答「模型总体感知有多好」。
- 两种语境的题目由同一个 seed 生成、逐题配对,但顺序效应未做随机化检验。
