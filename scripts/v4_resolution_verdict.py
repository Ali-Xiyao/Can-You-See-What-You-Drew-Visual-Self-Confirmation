"""Judge a rerun against the frozen selector-resolution criteria.

`docs/prereg/2026-09-04-selector-resolution.md` fixed four numbers before the
questions were touched. This script applies them and says passed or not passed.
It does not choose them, and it must never be edited to make a run pass -- the
prereg's declared failure modes say a miss is recorded as a miss.

    envs/core/python.exe scripts/v4_resolution_verdict.py runs/v4/gate-b-openct

The baseline column of the prereg table is the **naive** arm: 34.9 / 61.2 /
60.3 / 83.2 reproduce off `runs/v4/gate-b` on that arm to the digit, and on no
other. So that is the arm the thresholds attach to. The blind `rfo` arm is
printed beside it because leaving it out would hide the more interesting half of
the picture, but its thresholds are not registered and it is not the verdict.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v4_selector_resolution import (  # noqa: E402
    CRITERIA,
    Candidate,
    read_jsonl,
    resolution,
)

FROZEN = Path("runs/v4/gate-b")

# From the prereg, section 2. Copied, not derived, and not to be adjusted.
# (key, human name, direction, threshold, the frozen baseline, the anchor)
REGISTERED = [
    ("sep_higher", "正确 > 错误 的池", ">=", 0.60, 0.349, 0.746, "primary"),
    ("sep_equal", "正确 = 错误 的池", "<=", 0.25, 0.612, 0.119, "primary"),
    ("all_tied", "全部候选同分", "<=", 0.25, 0.603, 0.112, "secondary"),
    ("ceiling", "候选落在满分", "<=", 0.70, 0.832, 0.586, "secondary"),
]


def load_pools(out_dir: Path) -> dict[str, dict[str, Any]]:
    pools = {row["prompt_id"]: row for row in read_jsonl(out_dir / "pools.jsonl")}
    selection = {row["prompt_id"]: row for row in read_jsonl(out_dir / "selection.jsonl")}
    missing = set(pools) ^ set(selection)
    if missing:
        raise SystemExit(f"{out_dir}: {len(missing)} pools without a selection row")
    for prompt_id, row in pools.items():
        row["selection"] = selection[prompt_id]
    return pools


def candidates(pool: dict[str, Any], arm: str) -> list[Candidate]:
    scores = pool["selection"]["scores"][arm]
    return [Candidate(c["candidate_id"], float(scores[c["candidate_id"]]), bool(c["correct"]))
            for c in pool["candidates"]]


def measure(pools: dict[str, dict[str, Any]], arm: str) -> dict[str, Any]:
    return resolution(candidates(pool, arm) for pool in pools.values())


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return (float("nan"), float("nan"))
    p = hits / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def verdict_table(new: dict[str, Any], old: dict[str, Any]) -> bool:
    passed = True
    print(f"  {'判据':22s} {'门槛':>10s} {'冻结现行':>10s} {'本次':>10s}  {'':4s} {'锚点':>8s}")
    for key, name, direction, threshold, baseline, anchor, tier in REGISTERED:
        value = new[key]
        ok = value >= threshold if direction == ">=" else value <= threshold
        passed &= ok
        assert abs(old[key] - baseline) < 5e-4, (
            f"{key}: the frozen run no longer reproduces its own baseline "
            f"({old[key]:.4f} vs {baseline})")
        mark = "PASS" if ok else "FAIL"
        star = "*" if tier == "primary" else " "
        print(f" {star}{name:22s} {direction}{threshold:9.0%} {baseline:10.1%}"
              f" {value:10.1%}  {mark:4s} {anchor:8.1%}")
    print("  (* = 主判据，两条都必须过；次判据两条也必须过)")
    return passed


def movement(new: dict[str, Any], old: dict[str, Any]) -> None:
    """Paired, because it is the same 232 pools and the same images."""
    print(f"\n  同分池 {old['all_tied']:.1%} -> {new['all_tied']:.1%}"
          f"   满分候选 {old['ceiling']:.1%} -> {new['ceiling']:.1%}"
          f"   均分差 {old['mean_gap']:+.4f} -> {new['mean_gap']:+.4f}")
    print(f"  分值分布 {dict(sorted(old['dist'].items()))}")
    print(f"        -> {dict(sorted(new['dist'].items()))}")


def selection_vs_random(pools: dict[str, dict[str, Any]]) -> None:
    """Reported with an interval, and deliberately not gated.

    The prereg says this is a result, not an instrument check: gating on it
    would make the experiment argue in a circle.
    """
    print("\n选中正确图的比例（结果，不是判据）")
    sizes = sorted({len(p["candidates"]) for p in pools.values()})
    for k in sizes:
        ids = [p for p, row in pools.items() if len(row["candidates"]) == k]
        correct_of = {p: {c["candidate_id"]: bool(c["correct"])
                          for c in pools[p]["candidates"]} for p in ids}
        # Not 1/K: pools differ in how many correct images they hold, and 1/K
        # would be the expectation for a pool that does not exist.
        uniform = sum(sum(correct_of[p].values()) / k for p in ids) / len(ids)
        cells = []
        for criterion in CRITERIA:
            hits = sum(correct_of[p][pools[p]["selection"]["selected"][criterion]]
                       for p in ids)
            lo, hi = wilson(hits, len(ids))
            cells.append(f"{criterion} {hits / len(ids):5.1%} [{lo:.1%},{hi:.1%}]")
        print(f"  K={k}  n={len(ids):3d}  随机期望 {uniform:5.1%}   " + "   ".join(cells))


def counting_report(out_dir: Path) -> dict[str, dict[str, Any]]:
    """Paired count comparisons on facts known for the frozen question scope."""
    from selfsight.v4.factual_truth import (
        canonical_answer, factual_answer, load_verifications, question_scope,
    )

    found = load_verifications(out_dir)
    pools = {p["prompt_id"]: p for p in read_jsonl(out_dir / "pools.jsonl")}
    report: dict[str, dict[str, Any]] = {}
    for arm in ("naive", "rfo"):
        totals: collections.Counter[str] = collections.Counter()
        unknown: collections.Counter[str] = collections.Counter()
        seen: set[tuple[str, str]] = set()
        for row in read_jsonl(out_dir / f"observations.{arm}.jsonl"):
            key = (row["prompt_id"], row["candidate_id"])
            if key in seen:
                raise ValueError(f"Duplicate {arm} observation: {key}")
            seen.add(key)
            pool = pools[row["prompt_id"]]
            candidate = next(c for c in pool["candidates"]
                             if c["candidate_id"] == row["candidate_id"])
            questions = {q["question_id"]: q for q in pool["questions"]}
            answers = row["observation"]["answers"]
            ids = [a["question_id"] for a in answers]
            if len(ids) != len(set(ids)) or set(ids) != set(questions):
                raise ValueError(f"Incomplete or duplicate question set: {key}")
            for answer in answers:
                question = questions[answer["question_id"]]
                scope = question_scope(question)
                if scope is None or scope.kind != "count":
                    continue
                totals["total"] += 1
                truth = factual_answer(question, found.get(candidate["image_path"]))
                if not truth.known:
                    unknown[truth.reason] += 1
                    continue
                totals["known"] += 1
                observed = (None if answer.get("abstain") else
                            canonical_answer(question, answer["normalized_answer"]))
                totals["abstained"] += observed is None
                totals["off_recorded_target"] += observed != canonical_answer(
                    question, question["expected_answer"])
                totals["miscounted"] += observed != truth.answer
        expected = {(p["prompt_id"], c["candidate_id"])
                    for p in pools.values() for c in p["candidates"]}
        if seen != expected:
            raise ValueError(f"{arm}: observations do not cover the frozen candidate set")
        report[arm] = {**totals, "unknown_reasons": dict(unknown),
                       "unknown": sum(unknown.values())}
    return report


def counting_accuracy(out_dir: Path) -> None:
    print("\\n计数题：同一可判事实子集上的记录目标不一致率与图像计数错误率")
    for arm, row in counting_report(out_dir).items():
        n = row.get("known", 0)
        off = row.get("off_recorded_target", 0) / n if n else float("nan")
        wrong = row.get("miscounted", 0) / n if n else float("nan")
        print(f"  {arm:6s} 可判 {n}/{row.get('total', 0)}  记录目标不一致 {off:.1%}"
              f"  数错图 {wrong:.1%}  已知事实题弃答 {row.get('abstained', 0)}"
              f"  未知 {row['unknown']} {row['unknown_reasons']}")
    print("记录目标使用该运行冻结的 expected_answer；旧运行的目标口径缺陷仍须单独说明。")
    print("这些是可判定子集的描述统计；不同模型两臂之差不单独识别提示词效应。")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("outdir")
    args = parser.parse_args()

    new_pools = load_pools(Path(args.outdir))
    old_pools = load_pools(FROZEN)
    same = [p["prompt_id"] for p in new_pools.values()] == \
           [p["prompt_id"] for p in old_pools.values()]
    print(f"{args.outdir}: {len(new_pools)} pools, paired with the frozen set: {same}")
    if not same:
        raise SystemExit("not the same pools -- the prereg requires the same 232")

    print("\n" + "=" * 78)
    print("预注册判据（naive 臂，即预注册表里的「现行」那一列）")
    print("=" * 78)
    passed = verdict_table(measure(new_pools, "naive"), measure(old_pools, "naive"))
    movement(measure(new_pools, "naive"), measure(old_pools, "naive"))

    print("\n" + "-" * 78)
    print("盲观察者 rfo 臂（同样四个数，门槛未注册，不构成判定）")
    print("-" * 78)
    new_rfo, old_rfo = measure(new_pools, "rfo"), measure(old_pools, "rfo")
    for key, name, direction, threshold, _b, _a, _t in REGISTERED:
        print(f"  {name:22s} {direction}{threshold:9.0%} {old_rfo[key]:10.1%}"
              f" {new_rfo[key]:10.1%}")
    movement(new_rfo, old_rfo)

    selection_vs_random(new_pools)
    counting_accuracy(Path(args.outdir))

    print("\n" + "=" * 78)
    print(f"分辨力判据（naive 臂）：{'通过' if passed else '未通过'}")
    if not passed:
        print("按预注册第 3 节：记录未通过。不得在看过数之后调题目直到通过。")
    print("=" * 78)


if __name__ == "__main__":
    main()
