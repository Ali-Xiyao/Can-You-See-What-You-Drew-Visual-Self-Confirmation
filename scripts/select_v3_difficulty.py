"""Choose each family's object count from the calibration split, then freeze it.

REGISTERED BEFORE THE SWEEP RAN. The rule below is fixed; it must not be edited
after looking at `calibration_report.json`.

Why a rule at all: selection-based self-training can only gain where a K-pool
contains both correct and incorrect candidates. With per-candidate verifier
accuracy `p` and pool size K, a perfect selector scores `1-(1-p)^K` while taking
the first candidate scores `p`, so the headroom a mechanism could ever exploit is

    ceiling(p) = (1 - (1 - p)^K) - p

which peaks at p ~= 0.35 for K=4 and collapses at both ends. Gate A measured
p = 0.818 for existence at two objects (ceiling 0.181): the model is too accurate,
not too weak. Object count is the difficulty knob that moves p.

THE RULE
    1. For each family, prefer settings with p in [0.25, 0.55]; among those take
       the smallest |p - 0.35|.
    2. If no setting lands in the band, take the smallest |p - 0.35| overall and
       mark the family `out_of_band`.
    3. If the chosen setting is the largest swept and p is still above 0.55, mark
       the family `knob_exhausted`: object count alone cannot reach the band and
       the family needs a different difficulty axis, not more objects.

`p` MUST be computed over every generated candidate. `build_balanced_pool` drops
the candidate ids of a pool it judges non-informative, so counting only the ids
present in `natural_pools.jsonl` silently excludes every all-correct pool and
biases `p` downward -- on the existence pool-box run that error read 0.623
instead of 0.818, which would have picked the wrong difficulty. The counts are
therefore reconstructed from the pool `reason`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CANDIDATE_K = 4
TARGET_P = 0.35
BAND = (0.25, 0.55)


def selection_ceiling(p: float, k: int = CANDIDATE_K) -> float:
    """Headroom a perfect selector has over taking the first candidate."""

    return (1.0 - (1.0 - p) ** k) - p


def pool_counts(pool: dict) -> tuple[int, int, int]:
    """(correct, incorrect, abstained) for one pool, including dropped ids."""

    searched = int(pool.get("searched") or 0)
    abstained = int(pool.get("abstained") or 0)
    judged = max(searched - abstained, 0)
    reason = pool.get("reason")
    if reason == "no_verifier_incorrect_candidate":
        return judged, 0, abstained
    if reason == "no_verifier_correct_candidate":
        return 0, judged, abstained
    return (
        len(pool.get("correct_ids") or ()),
        len(pool.get("incorrect_ids") or ()),
        abstained,
    )


def measure(pool_files: list[Path]) -> dict[str, dict]:
    """Per-family candidate accuracy over the natural (unfiltered) pools."""

    stats: dict[str, dict] = {}
    for path in pool_files:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            pool = json.loads(line)
            family = str(pool.get("family"))
            entry = stats.setdefault(
                family,
                {"correct": 0, "incorrect": 0, "abstained": 0, "pools": 0, "informative": 0},
            )
            correct, incorrect, abstained = pool_counts(pool)
            entry["correct"] += correct
            entry["incorrect"] += incorrect
            entry["abstained"] += abstained
            entry["pools"] += 1
            entry["informative"] += int(bool(pool.get("informative")))
    for entry in stats.values():
        judged = entry["correct"] + entry["incorrect"]
        entry["candidates"] = judged + entry["abstained"]
        entry["p"] = entry["correct"] / judged if judged else None
        entry["ceiling"] = selection_ceiling(entry["p"]) if judged else None
        entry["informative_rate"] = entry["informative"] / entry["pools"] if entry["pools"] else None
    return stats


def choose(per_setting: dict[int, dict[str, dict]]) -> dict[str, dict]:
    """Apply the registered rule to each family."""

    settings = sorted(per_setting)
    families = sorted({family for stats in per_setting.values() for family in stats})
    chosen: dict[str, dict] = {}
    for family in families:
        options = [
            (setting, per_setting[setting][family])
            for setting in settings
            if family in per_setting[setting] and per_setting[setting][family]["p"] is not None
        ]
        if not options:
            chosen[family] = {"status": "no_measurement"}
            continue
        low, high = BAND
        in_band = [item for item in options if low <= item[1]["p"] <= high]
        pool = in_band or options
        setting, stats = min(pool, key=lambda item: abs(item[1]["p"] - TARGET_P))
        status = "in_band" if in_band else "out_of_band"
        if not in_band and setting == max(settings) and stats["p"] > high:
            status = "knob_exhausted"
        chosen[family] = {
            "objects_per_scene": setting,
            "status": status,
            "p": round(stats["p"], 4),
            "selection_ceiling": round(stats["ceiling"], 4),
            "informative_rate": round(stats["informative_rate"], 4),
            "candidates": stats["candidates"],
            "swept": {
                str(item[0]): {
                    "p": round(item[1]["p"], 4),
                    "ceiling": round(item[1]["ceiling"], 4),
                    "informative_rate": round(item[1]["informative_rate"], 4),
                }
                for item in options
            },
        }
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calib-root", required=True, help="runs/v3/calib")
    parser.add_argument("--setting", action="append", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.calib_root)
    per_setting = {
        setting: measure(sorted(root.glob(f"obj{setting}/*/natural_pools.jsonl")))
        for setting in args.setting
    }
    chosen = choose(per_setting)
    report = {
        "stage": "v3_difficulty_calibration",
        "rule": {
            "target_p": TARGET_P,
            "band": list(BAND),
            "candidate_k": CANDIDATE_K,
            "registered_before_sweep": True,
        },
        "swept_settings": sorted(per_setting),
        "per_setting": {
            str(setting): {
                family: {
                    key: (round(value, 4) if isinstance(value, float) else value)
                    for key, value in stats.items()
                }
                for family, stats in families.items()
            }
            for setting, families in per_setting.items()
        },
        "chosen": chosen,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{'family':<11} {'obj':>4} {'p':>7} {'ceiling':>8} {'infor':>7}  status")
    for family, item in sorted(chosen.items()):
        if item.get("status") == "no_measurement":
            print(f"{family:<11} {'-':>4} {'-':>7} {'-':>8} {'-':>7}  no_measurement")
            continue
        print(
            f"{family:<11} {item['objects_per_scene']:>4} {item['p']:>7.3f} "
            f"{item['selection_ceiling']:>8.3f} {item['informative_rate']:>7.3f}  {item['status']}"
        )
    print(f"\n-> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
