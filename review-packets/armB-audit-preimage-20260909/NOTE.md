# arm B 的 scene audit,在 arm B 存在之前先算出来(2026-09-09)

## 这是什么

`scripts/v4_scene_audit.py --prospective` 在一个**假的 arm B 运行目录**上跑出来的产物。
目录里只放了主运行 `runs/v4/decoupling-main-20260908` 的 `split.json` 和 `probe-bank/`,
外加 supervisor 在 `split` / `freeze-probe` 两个 stage 之后必然会留下的东西
(`run_manifest.json`、空的 `logs/`、`stage-completion/{split,freeze-probe}.json`)。
没有动主运行目录一个字节,全程只读加复制;不需要 GPU。

用意有两条:

1. **烟雾测试。** audit 现在是 supervisor 的一个 stage(EXECUTION §1),
   跑在 `freeze-probe` 之后。如果它在那个时刻会拒绝,arm B 会在启动几秒后死掉。
   这里先用真的 split 和真的 probe bank 验证它不会。
2. **预注册。** 排除集(哪些 outcome prompt 因为和训练 prompt 同场景而被剔掉)
   是 `v4_decoupling_report.py` 挑 outcome 子集的依据。它是 split 的纯函数,
   所以现在就能算,而现在 arm B 还什么都没产生——**这是 prospective 这个词
   唯一能成立的时点**。把数字钉在这里,arm B 跑完之后就没有调整它的余地。

## 钉住的数字

arm B 的 `audit-splits/scene_overlap.json` 落地后,下面几个必须一模一样。
不一样就是 split 没复现,停下来查,不要往下跑。

| 字段 | 值 |
|---|---|
| `scene_disjoint_outcome_sensitivity.spec_ids_sha256` | `8c638fb3efd86258fbc6e51146e18086e5c517acaa48d017a208a4e24bcb5264` |
| `provenance.split_sha256`(split.json 的文件摘要) | `83b7eaa696726a1f6261327a5ac70c2d494629e5e736bcae7799eed0482f7094` |
| `provenance.split_digest`(split 内容的规范摘要) | `9bab14d427f2b61bea001b710126dcf18d610972985f723665534209231f9d58` |
| `provenance.config_sha256` | `620ba17921eb06fc1f3077f021f4d7af392c742f301d4515d6e877389d0e1dee` |
| `provenance.bank_sha256` | `5468b22e74e77f618e0b8ff1221c21d3de7cff1665e0d82929562080772c62f8` |
| `summary.outcome_n` | 64 |
| `summary.scene_disjoint_outcome_n` | 54 |
| `summary.outcome_overlap_n` | 10 |
| `summary.actual_probe_bank_overlap_n` | 2 |
| `summary.scheduled_train_prompt_n` / `train_partition_n` | 132 / 132 |

两次跑的结果除了 `created` 时间戳和 `provenance.run` 路径以外逐字节相同,
所以上面这些数字不是某一次的运气。

## 和 pilot 不一样的一处,是好的方向

pilot(`decoupling-pilot-20260906`)的 132 个训练 prompt 里只有 120 个真的上了
生成日程,所以「保守地把整个 train 分区当作暴露」和「只算实际排上的」两种
敏感性分析给出不同的集合。主 config 是 11 轮 × 12 prompt = 132/132,
`conservative_all_train_partition_sensitivity.equals_scheduled_exposure` 是 `true`,
两种口径重合。也就是说 arm B 的排除集**不依赖生成日程的随机种子**,
少一个要在论文里解释的自由度。

## 已知的口径

- 这份是用**主运行的 split** 算的,不是 arm B 自己的。arm B 的 split 由同一份
  config 和同一个 seed 生成,预注册说它是纯函数——但那是要核对的断言,不是前提。
  上表第二、三行就是核对用的。
- `provenance.rounds_checked_against_executed_selection` 是空的,因为还没有轮次。
  这正是 prospective 应有的样子。
- 建这份东西的过程里 audit 拒绝过一次:我把上一次的输出 `first.json` 留在了运行
  目录里,它认出那是白名单外的文件、拒绝签 prospective。白名单不是黑名单,
  这一条是当场验到的,不是推的。
