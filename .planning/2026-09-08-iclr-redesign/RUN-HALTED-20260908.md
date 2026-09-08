# 主运行在 5.83 h 停了,而且是退出,不是暂停

`runs/v4/decoupling-main-20260908`,2026-09-08 13:23:09。
`state.json` 写了 `needs_diagnosis` / `paused`,但 **supervisor PID 19504 已经不在了**。
"paused" 是它退出前留下的最后一句话,不是它正在做的事。没人重启就不会继续。

## 直接原因

```
step-00000.gradient-sensitivity  exit 1
FileNotFoundError: runs/v4/decoupling-main-20260908/audit-splits/scene_overlap.json
```

`v4_gradient_sensitivity.py:207` 在 `--audit` 缺省时取 `<run>/audit-splits/scene_overlap.json`,
supervisor(`run_decoupling_pilot.py:309`)只传 `--outdir`,所以走的就是这个缺省。
**这个目录在主运行里从来没被建过。**

它前面那个 `base.step-00000.gradient` 是成功的(`status: ok`,cosine 0.802)。
挂的是同名前缀的另一个阶段,不要混。

## 这个文件谁产出

没有人。全树扫过 `.py`(排除 `runs/ .git/ envs/ __pycache__`),
提到 `scene_overlap` 的有 6 个文件,**全部是读**:

| 文件 | 读/写 |
|---|---|
| `scripts/v4_gradient_sensitivity.py:207` | 读,缺了就抛 |
| `scripts/v4_decoupling_report.py:663` | 读,缺了记 `None`,不抛 |
| `review-packets/selection-benefit-side-20260906/analyze.py:162` | 读 |
| `review-packets/selector-measurement-next-20260906/prepare.py:98` | 读 |
| `tests/test_v4_*.py` | 造 fixture |

仓库里唯一一份实物在 `runs/v4/decoupling-pilot-20260906/audit-splits/`,
`created_utc: 2026-09-05T18:09:44Z`,`kind: prospective_scene_overlap_sensitivity_freeze`,
**未被 git 跟踪**。当时是手工做的。

## 借不过来

`read_partition` 自己会拒:

```python
if (audit["provenance"]["bank_sha256"] != sha256_file(bank_path)
        or audit["provenance"]["bank_fingerprint"] != bank["fingerprint"]):
    raise ValueError("Scene audit does not describe this frozen main bank")
```

| | bank_sha256 | bank_fingerprint | split_sha256 |
|---|---|---|---|
| pilot 的 audit 期待 | `ebd7fac6…` | `daf31aa3…` | `9b358ecc…` |
| 主运行实际 | `5468b22e…` | `7a90dc39…` | `83b7eaa6…` |

三项全不同。守卫是对的,拷过去只会换一条报错。

## 它会重复发生

`report()` 在**每个计分步**都调这个阶段,12 个 checkpoint 一个不落。
不解决,主运行永远过不了 step-00000。

## 好消息:5.83 h 的算力没白烧

`run()` 开头 `if complete.exists(): return`。
`stage-completion/` 里 split、freeze-probe、round-000.train、两臂 step-00000 全链、
base gradient、score、report 都有标记。**重启会正好从失败的那个阶段接上。**

坏消息:`self.started = manifest["started_unix"]`(148 行),
`deadline = started + max_wall_hours*3600`。**起点不随重启复位,停着的时间照样烧预算。**

## 附带查到的第二个缺陷(不致命,但会污染 provenance)

`report()` 把协议路径写死成 pilot 的:

```python
"--protocol", str(ROOT / "docs/prereg/2026-09-06-decoupling-pilot.md")
```

而 `run_manifest.json` 登记的是对的:`docs\prereg\2026-09-08-dstar-main-run.md`(`e276f3c7…`)。
结果 `decoupling_report.json` 的 `provenance.protocol` 指向 **pilot** 协议(`c1d1e724…`)。
`v4_decoupling_report.py:674` 会把 provenance 跨步冻结比对,所以这个错值一旦写进去,
整个运行都锁死在它上面。

而 pilot 协议正是写 `max_wall_hours: 60` 的那份 —— 所以 §0.1 那个 60/96 冲突
不只是两份计划文档打架,它已经烙进运行自己的产物里了。

## 更正我先前说错的一处

我之前说「中途合并分支会静默换掉代码」。**不对。**
`run()` 每个阶段都重新核 SOURCES 摘要:

```python
if any(digest(ROOT / name) != expected
       for name, expected in self.frozen["source_sha256"].items()):
    raise RuntimeError("Experiment source changed during execution; review before resume")
```

不是静默,是会当场拒绝。但方向反过来变得更要紧:
`src/selfsight/v4/train.py` 和 `scripts/v4_train.py` 都在 SOURCES 里,也都在
`codex/blind-self-arm-20260908` 里改了。**先合并再重启,重启会被拒**,
除非带 `--accept-code-update`。合并顺序现在卡的是主运行的恢复,不只是 arm B。

## 需要裁定的三件事

它们互相咬着:任何代码修复都要求重启时带 `--accept-code-update`,
而那正是重新裁定墙钟的时刻。

1. **audit 文件。** 为主运行重造一份?算法在文件自己的
   `canonical_scene_key_definition` 里写全了,输入(split.json + bank.json)已冻结,
   所以是确定性的、没有可调空间。但它**不能诚实地带
   `created_before_any_outcome_evaluation_artifact: true`** ——
   step-00000 的 score 和 report 已经落地了。这是证据形状的东西,我不替你造。
   另一条路是把这个阶段摘掉,但它是注册过的。
2. **协议写死。** 重启前改掉,还是留着并登记差异。
3. **60 还是 96 h。** 见 EXECUTION.md §0.1,现在又多烧了停机时间。

两张卡当前空闲。
