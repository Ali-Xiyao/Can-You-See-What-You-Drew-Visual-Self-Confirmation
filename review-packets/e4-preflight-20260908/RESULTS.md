# E4 预检:三个模型都读到了图(2026-09-08,cuda:0)

`preflight.log` 是完整输出。命令:

```
SELFSIGHT_MODEL_ROOT="H:\selfsight-models" envs/core/python.exe -u H:/Xiyao_Wang/062_armB/scripts/v4_cross_model.py preflight --device cuda:0 --count 4
```

主树作工作目录,worktree 的脚本按路径调用。两边各有对方没有的东西:
分支的 `showo_v1.yaml` / `janus_pro_1b.yaml` 只在 worktree,`envs/` 和 `runs/` 只在主树。

## 结果

| spec 颜色集合 | showo2_7b | showo_v1 | janus_pro_1b |
|---|---|---|---|
| {blue, white} 蓝杯白盘 | Blue | Blue | Blue. |
| {red, green} 红番茄绿椒 | Red | Red | The largest object in the image is a red tomato. |
| {white, black} 白盘黑勺 | White | White | The largest object in the image is a white plate. |
| 第四张 | Blue | Red | Red. |

三个模型的输出都随图变化。`all models read the picture`,exit 0。
`showo2_1p5b` 不参与预检——它的行已经量过了。

## 这一版之前的两次,都是门自己坏的

**第一次:选图。** 门取 `sorted(glob)[:3]`,而语料把同一个 prompt 的多个采样
按文件名连续存放,所以那三张是同一份 spec(一个绿苹果、两根橙胡萝卜)的三次抽样。
三个模型全答 `green`、全答对,门把它读成「图没到达模型」,**三个全判失败**。
现在按 spec 的颜色集合取图。登记为 PREREG 偏离 4。

**第二次:读归一化答案。** `normalize_answer` 对 COLOR 只认
`Color` 枚举的四个值,而两份冻结 manifest 里出现十种颜色,
约三分之一的物体条目在枚举外:

```
spec: blue white red green black yellow orange brown silver gray
enum: blue       red green        yellow
```

所以白盘子那张图,三个模型都正确答了 `White`,三个都被读成 `None`,
那张图对门毫无贡献。现在比原始输出——贪心解码、prompt 固定,
图是唯一变化的输入,原始字符串有差异就直接证明它到达了。

**这个窄枚举不影响任何 v4 数字。** `grade()` 比的是选项文本、从不查枚举,
`stage_observe` 也丢弃 `normalized_answer`。它是 v2 时期的归一化器,
只波及到了预检。**没有动它**,在此登记。

## 顺带印证了偏离 3

Janus 在四张图里有两张写整句。16 token 的预算会在字母出现前截断,
旧评分器会把整句读成弃权——这正是偏离 3 改 grader 和把
`mmu_max_new_tokens` 提到 64 的理由,现在能直接看到那个行为。
