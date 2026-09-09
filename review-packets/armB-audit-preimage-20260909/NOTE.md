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
| `scene_representatives.json` → `spec_ids_sha256` | `7f44b386ed6490544b49c411b17783e1e2edd34b2ffac8eadf6f5e6476e5a2c1` |
| `scene_representatives.json` → `n_independent_scene_representatives` | 52 |

两次跑的结果除了 `created` 时间戳和 `provenance.run` 路径以外逐字节相同,
所以上面这些数字不是某一次的运气。

## 顺带补上的一个缺口:`scene_representatives.json` 以前没有脚本能造

`v4_decoupling_report.py` 会读 `audit-splits/scene_representatives.json`,
读到就多出一条 `independent_scene_representative_sensitivity`
(每个场景只留一个 outcome prompt,让那条敏感性分析的行之间互相独立)。
pilot 有这个文件,**而整棵树里没有任何脚本会写它**——它是 2026-09-05 手工造的。
后果有两层:一是 arm B 会缺一条它的对照臂有的分析;二是一份进了结论的产物
没有可复现的来源。

现在 `v4_scene_audit.py --prospective` 会连带写出这个子文件(worktree)。
不做成单独的命令,是因为报告把这一对当作一对来查:子文件里记着父文件的
sha256,父文件一动报告就拒绝。两条命令等于给「只冻结了一半」留了两次机会。

**它对着 pilot 那份手工产物验过**:用 pilot 的 `scene_overlap.json` 喂进去,
新函数产出的 `scene_representatives.json` 除 `created_at` 外**每个字段逐字相同**,
包括 54 个 spec_id、`spec_ids_sha256 = f52826b9…`、54 行 representatives、
父文件摘要,以及那句 `does not replace original64 or sensitivity57`。
选择规则本身也不是新发明的:报告在使用前会自己按父 audit 重算一遍并拒绝不符的文件,
所以这里加的只是产物,不是选择权。

arm B 这份是 **52 个**(59 个场景里,52 个的全部 outcome prompt 都在
scene-disjoint 集内);pilot 是 54 / 59。

变异测试 9/9 击杀,其中一个专门盯着最容易写错的那条规则:
「代表本身在 disjoint 集里就留下」——正确的规则是**整个场景**都要在集内,
否则这个代表就代表了一个训练已经通过别的 prompt 见过的场景。

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
