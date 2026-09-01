"""One page describing what the unattended v3 queue actually measured.

Reads only files the queue wrote. Every threshold quoted here is the registered
one; this script does not decide anything, it reports and compares.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HEADROOM_GO = 0.05
HEADROOM_STOP = 0.03
MIN_BALANCED_RATE = 0.40
MIN_INFORMATIVE_POOLS = 128


def load(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def headroom_section(root: Path) -> list[str]:
    data = load(root / "runs/v3/selection-headroom/box/selection_headroom.json")
    out = [
        "## 1. 选择头room（2 物体 / box 词汇，**改族之前**测的）",
        "",
        "> 这一节测于 gold 原子重设计之前，判分口径已经变了（见第 2 节的弃权列）。",
        "> 保留为历史记录，不要和第 2、3 节的数字放在一起比较。",
        "",
    ]
    if data is None:
        out += ["未产出。", ""]
        return out
    overall = data.get("overall") or {}
    naive = overall.get("naive_cycle_rate")
    oracle = overall.get("oracle_at_k")
    gap = overall.get("headroom_over_naive")
    if naive is None:
        out += [
            "**Naive 仍未被打分**（`naive_cycle_rate` 为空）。这正是 stage-1 作废那次的",
            "同一个缺陷——头room 没有被测出来，不要把它当作结果。",
            "",
        ]
        return out
    out += [
        f"- Oracle@K = {oracle:.4f}",
        f"- Naive@K  = {naive:.4f}",
        f"- **头room = {gap:.4f}**（判据：>{HEADROOM_GO} go / <{HEADROOM_STOP} stop）",
        "",
    ]
    if gap > HEADROOM_GO:
        verdict = "**go** —— 两条臂选出来的东西确实不同，机制有作用空间。"
    elif gap < HEADROOM_STOP:
        verdict = (
            "**stop** —— 在这个难度下 Gold 和 Naive 选择几乎一致，分叉在构造上无法发生。"
            "这不是机制失效，是任务难度未标定；见下一节。"
        )
    else:
        verdict = "**扩样本** —— 落在 stop 与 go 之间。"
    out += [verdict, ""]
    agreement = overall.get("selection_agreement")
    if agreement is not None:
        note = "（=1.000 表示 Naive 修复没生效）" if agreement >= 0.999 else ""
        out += [f"- selection_agreement = {agreement:.4f} {note}", ""]
    return out


def calibration_section(root: Path) -> list[str]:
    data = load(root / "runs/v3/calib-v2/calibration_report.json")
    out = ["## 2. 难度标定（在 calibration 分片上选，判据看数前已注册）", ""]
    if data is None:
        out += ["未产出。", ""]
        return out
    rule = data.get("rule", {})
    out += [
        f"目标 p = {rule.get('target_p')}，有效区间 {rule.get('band')}，K = {rule.get('candidate_k')}。",
        "",
        "| 族 | 选中物体数 | p | 天花板 | informative | 状态 |",
        "|---|---|---|---|---|---|",
    ]
    for family, item in sorted((data.get("chosen") or {}).items()):
        if item.get("status") == "no_measurement":
            out.append(f"| {family} | — | — | — | — | 无测量 |")
            continue
        out.append(
            f"| {family} | {item['objects_per_scene']} | {item['p']:.3f} | "
            f"{item['selection_ceiling']:.3f} | {item['informative_rate']:.3f} | {item['status']} |"
        )
    out.append("")
    # Abstention is the column that matters here. Before the redesign the strict
    # predicates abstained on 60-85% of generated candidates, so `p` was an
    # estimate over whatever minority happened to resolve.
    per_setting = data.get("per_setting") or {}
    out += [
        "全部扫描点（弃权率是这次重设计的核心指标）：",
        "",
        "| 族 | 物体数 | p | 天花板 | 弃权 |",
        "|---|---|---|---|---|",
    ]
    for family, item in sorted((data.get("chosen") or {}).items()):
        for setting, stats in sorted((item.get("swept") or {}).items(), key=lambda kv: int(kv[0])):
            raw = (per_setting.get(str(setting)) or {}).get(family) or {}
            judged = (raw.get("correct") or 0) + (raw.get("incorrect") or 0)
            abstained = raw.get("abstained") or 0
            total = judged + abstained
            rate = f"{abstained / total:.3f}" if total else "—"
            out.append(
                f"| {family} | {setting} | {stats['p']:.3f} | {stats['ceiling']:.3f} | {rate} |"
            )
    out.append("")
    exhausted = [f for f, i in (data.get("chosen") or {}).items() if i.get("status") == "knob_exhausted"]
    if exhausted:
        out += [
            f"**{', '.join(exhausted)} 的物体数旋钮已用尽**：即使 6 个物体 p 仍在区间之上。",
            "这些族需要另一条难度轴（更细的关系粒度、遮挡、更接近的尺寸），加物体没用了。",
            "",
        ]
    out += [
        "物体数这条轴本身是弱的：标定 bank 上的检测直方图显示，要 6 个物体时 Show-o2",
        "最常画 3–4 个，要 3 个时最常画 2 个。真正移动 p 的是原子设计，不是物体数。",
        "",
    ]
    return out


def gate_a_section(root: Path) -> list[str]:
    out = ["## 3. Gate A（held-out，难度不是在这批数据上选的）", ""]
    reports = sorted((root / "runs/v3/gate-a-v2").glob("*/gate_a.json"))
    if not reports:
        out += ["未产出。", ""]
        return out
    out += [
        "| 族 | balanced_rate | informative | 自然 informative | prompts | rate 判据 |",
        "|---|---|---|---|---|---|",
    ]
    total_informative = 0
    for path in reports:
        data = load(path)
        if data is None:
            continue
        family = path.parent.name
        total_informative += int(data.get("informative_pools") or 0)
        ok = "过" if data.get("balanced_rate_ok") else "**不过**"
        natural = data.get("natural_informative_rate")
        out.append(
            f"| {family} | {data.get('balanced_rate'):.3f} | {data.get('informative_pools')} "
            f"({data.get('informative_rate'):.3f}) | "
            f"{natural:.3f} | {data.get('prompts_searched')} | {ok} |"
        )
    out += [
        "",
        f"阈值 balanced_rate ≥ {MIN_BALANCED_RATE}。",
        "",
        f"**informative pool 合计 = {total_informative}**（Gate B 注册的绝对底线是 "
        f"{MIN_INFORMATIVE_POOLS}）"
        + ("，达标。" if total_informative >= MIN_INFORMATIVE_POOLS else "，**未达标**。"),
        "",
    ]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    lines = [
        "# v3 队列结果",
        "",
        "本页由 `scripts/summarize_v3_queue.py` 从队列产物生成，不做任何新判断。",
        "",
    ]
    lines += headroom_section(root)
    lines += calibration_section(root)
    lines += gate_a_section(root)
    lines += [
        "## 下一步取决于上面哪一节是红的",
        "",
        "- 第 1 节 stop + 第 2 节把某个族带进区间 → 旧难度确实是瓶颈，用新难度重测头room。",
        "- 第 2 节全部 `knob_exhausted` → 物体数这条轴到头了，需要换难度维度。",
        "- 第 3 节 balanced_rate 过且 informative 合计 ≥ 128 → Gate A 两条判据首次同时成立，",
        "  可以进 Gate B（梯度仪器）。",
        "",
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n-> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
