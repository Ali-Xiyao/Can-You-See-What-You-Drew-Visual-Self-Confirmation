# 评分器在一个会写句子的模型上丢掉多少试次

2026-09-08。`janus_seam_probe.py`,CPU,fp32,零 GPU 时间。
Janus-Pro-1B 回答 `runs/v4/main-2plus1` 里 12 张图的 52 道题(image_only 条件),
同一批真实回复分别用**旧评分器**(HEAD 67bdfb5 时的 `grade`)和**新评分器**
(worktree 99b7c0f)评一遍。

这不是 E4 的测量,是 E4 的**仪器检查**。它不支撑任何主张。

---

## 结果

```
52 试次 / 12 张图
  弃权    旧 14/52 (26.9%)    新 3/52 (5.8%)
  改判    11 条:None -> True 8,None -> False 3
  已判定后又改值的:0 条
```

最后一行是关键:**修复只是把读不出来的读出来,没有把任何已经读出来的读成别的**。

## 回复长什么样

52 条里**没有一条是裸字母**。全部是句子:

```
'The book in the picture is white. The answer is B because the image sh...'
'The tomato in the picture is red.\n\nAnswer: B'
'The image shows two tomatoes, one red and one green.\n\nAnswer: B'
```

这就是为什么旧评分器丢掉四分之一以上:它是照着只回一个字母的
Show-o2-1.5B 写的,而在 7246 条冻结行里那确实是唯一出现过的形状。

## 还读不出来的 3 条,以及为什么不修

三条全部是 absence 家族,全部答对了,全部没提字母也没提选项原文:

```
选项 A='at least one notebook'  B='no notebooks'   gold=B
回复 'The image does not contain any notebooks. It only shows two blue mugs
      placed on a white plate. There are no additional objects or items present.'
```

「The image does not contain any notebooks」在语义上就是「no notebooks」,
但它不是 `no notebooks` 这个字符串。要读懂它,`grade` 得从一个解析器变成
一个否定推理器,而那种规则的误判方式**和旧规则一样是静默的**——
这次的整个教训就是别再装一个会悄悄猜错的东西进去。

**改成报告它。** `conflict_pairs` 丢掉任一条件弃权的试次是对的,
但它注意不到**存活下来的那批试次是被选出来的**:
试次能不能存活取决于模型怎么措辞,而措辞跟条件并不独立。
所以 E4 现在对弃权跑同一个配对精确 McNemar、同一个配对键,
两个条件弃权率不同的模型会被点名(worktree d8ec1a2)。

这是诊断不是闸门:配对检验本身不受影响——它只用两个条件都作答的试次——
但两侧的边际准确率会落在模型自己挑出来的子集上,读者必须被告知。

## 复算

```
SELFSIGHT_MODEL_ROOT=H:\selfsight-models \
envs/janus/Scripts/python.exe -u review-packets/grader-seam-20260908/janus_seam_probe.py \
  --images 12 --device cpu --out review-packets/grader-seam-20260908/janus_seam.json
```

CPU fp32 上约 90 分钟。`janus_seam.json` 存了全部 52 条原始回复与两侧评分,
不重跑模型也能复算上面每一个数字。
