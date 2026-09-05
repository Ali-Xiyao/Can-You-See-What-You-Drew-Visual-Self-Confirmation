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


def counting_accuracy(out_dir: Path) -> None:
    """Two different numbers that were being read as one.

    "The answer differs from what the spec asked for" is not "the observer
    miscounted". A picture is allowed to differ from the request -- that
    divergence is the signal the probe exists to catch -- and on top of that the
    first open-counting run was scoring some correct counts as wrong (STATUS
    42). So report the deviation rate beside the accuracy measured against the
    detections in `verified.jsonl`, which is an independent record of what is in
    the image.

    The slug in `atom_id` says which phrase a question is about: `red-book` for
    the colour-qualified form, bare `book` for the category form the defective
    run used. Colours contain no hyphen and `canonical_noun` returns a single
    word, so the two cases separate cleanly.
    """
    from selfsight.v4.spec import canonical_noun

    runs = json.loads((out_dir / "runs.json").read_text(encoding="utf-8"))["runs"]
    detected: dict[str, collections.Counter] = {}
    for name in runs:
        for row in read_jsonl(Path(name) / "verified.jsonl"):
            if row.get("detections") is None:
                continue
            detected[row["image_path"]] = collections.Counter(
                (d.get("color"), canonical_noun(d["object"])) for d in row["detections"])

    image_of: dict[tuple[str, str], str] = {}
    asks: dict[tuple[str, str], tuple[str, str]] = {}
    for pool in read_jsonl(out_dir / "pools.jsonl"):
        for candidate in pool["candidates"]:
            image_of[(pool["prompt_id"], candidate["candidate_id"])] = candidate["image_path"]
        for question in pool["questions"]:
            if ":count:" in question["question_id"]:
                asks[(pool["prompt_id"], question["question_id"])] = (
                    question["atom_id"].split(":")[-1], question["expected_answer"])

    print("\n计数题：偏离请求 vs 数错图（后者以 verified.jsonl 的检出为准）")
    for arm in ("naive", "rfo"):
        n = off_spec = miscounted = skipped = 0
        for row in read_jsonl(out_dir / f"observations.{arm}.jsonl"):
            counts = detected.get(image_of[(row["prompt_id"], row["candidate_id"])])
            for answer in row["observation"]["answers"]:
                key = (row["prompt_id"], answer["question_id"])
                if key not in asks:
                    continue
                slug, wanted = asks[key]
                if counts is None:
                    skipped += 1
                    continue
                n += 1
                truth = sum(c for (colour, noun), c in counts.items()
                            if f"{colour}-{noun}" == slug or noun == slug)
                off_spec += answer["normalized_answer"] != wanted
                miscounted += answer["normalized_answer"] != str(truth)
        print(f"  {arm:6s} n={n:5d}  偏离请求 {off_spec / n:6.1%}"
              f"   数错图 {miscounted / n:6.1%}   不可判定 {skipped}")


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
