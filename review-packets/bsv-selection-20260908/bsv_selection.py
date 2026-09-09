"""Does blind self-verification pick better images than prompted self-verification?

The context ablation showed the backbone answers 0.654 blind and 0.336 told on
the trials where the picture and the description disagree. That is a perception
number. This asks the question a generation paper has to answer: does the extra
perception buy anything downstream, when the answers are used the way a
self-improvement loop uses them -- to choose which of K drawings to keep.

Nothing is generated or observed here. Both conditions were already run over the
same images with the same questions (v4_run_pipeline.py observe), and the
external correctness of every image is already adjudicated. This is arithmetic
over files on disk.

Two things to be honest about, both restated in RESULTS.md:

1.  The question set is detections-derived, so an option can name what was
    actually drawn. A real loop only has the description and would write a
    different distractor. The blind/prompted contrast is fair -- identical
    questions, identical images, only the context differs -- but the absolute
    selection accuracy is optimistic for both arms.
2.  Scores over roughly four binary questions tie constantly. Ties are resolved
    as the expectation under a uniform random tie-break, which is what an
    implementation that shuffles its pool gets on average; a first-index
    tie-break is reported beside it.
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ("main-2plus1", "main-1plus1plus1")
CONDITIONS = {"blind": "answers.jsonl", "prompted": "answers.prompted.jsonl"}

# gold_source "image" means the question is about something the description
# never asked for, so there is no description-implied answer and a selection
# loop could not have scored it.
USABLE_GOLD = ("spec_matches_image", "image_differs_from_spec")


def spec_agreement(row: dict) -> bool:
    """Did the model answer the way the description implies, whatever the pixels say?

    Every family here is a forced binary choice, and the pipeline already
    normalised the free-text answers into `correct` against the detections. So
    the model's choice is recoverable without re-parsing anything: it agreed
    with the detections iff `correct`, and the description agrees with the
    detections iff gold_source is spec_matches_image.

    This is exactly the quantity a self-selection loop computes. That loop never
    sees the detections; it asks what the description implies and counts
    matches.
    """
    if row["gold_source"] == "spec_matches_image":
        return bool(row["correct"])
    return not bool(row["correct"])


def load() -> dict:
    """One record per drawn image: its two scores, and whether it is actually right."""

    pools: dict = defaultdict(
        lambda: {"blind": [], "prompted": [], "image_correct": None})
    for run in RUNS:
        for condition, filename in CONDITIONS.items():
            with open(ROOT / "runs" / "v4" / run / filename, encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    if row["abstain"] or row["gold_source"] not in USABLE_GOLD:
                        continue
                    key = (run, row["spec_id"], row["candidate_index"])
                    record = pools[key]
                    conflict = row["gold_source"] == "image_differs_from_spec"
                    record[condition].append((spec_agreement(row), conflict))
                    if record["image_correct"] is None:
                        record["image_correct"] = bool(row["image_correct"])
                    elif record["image_correct"] != bool(row["image_correct"]):
                        raise ValueError(f"{key} is both correct and incorrect")
    return pools


def mean_agreement(answers: list, drop_conflicts: bool = False) -> float:
    kept = [hit for hit, conflict in answers if not (drop_conflicts and conflict)]
    return sum(kept) / len(kept) if kept else float("nan")


def by_spec(pools: dict) -> dict:
    grouped: dict = defaultdict(list)
    for (run, spec_id, index), record in pools.items():
        if not record["blind"] or not record["prompted"]:
            continue
        grouped[(run, spec_id)].append({
            "index": index,
            "blind": mean_agreement(record["blind"]),
            "prompted": mean_agreement(record["prompted"]),
            "blind_clean": mean_agreement(record["blind"], drop_conflicts=True),
            "prompted_clean": mean_agreement(record["prompted"], drop_conflicts=True),
            "correct": record["image_correct"],
            "asked": len(record["blind"]),
            "conflicts": sum(conflict for _, conflict in record["blind"]),
        })
    return {key: sorted(pool, key=lambda item: item["index"])
            for key, pool in grouped.items() if len(pool) >= 2}


def pick(pool: list, key: str, first_index: bool = False) -> float:
    """Expected correctness of the candidate this rule keeps."""

    best = max(item[key] for item in pool)
    tied = [item for item in pool if item[key] == best]
    if first_index:
        return float(tied[0]["correct"])
    return sum(item["correct"] for item in tied) / len(tied)


def rates(pools: list, first_index: bool = False) -> dict:
    n = len(pools)
    return {
        "blind": sum(pick(pool, "blind", first_index) for pool in pools) / n,
        "prompted": sum(pick(pool, "prompted", first_index) for pool in pools) / n,
        "random": sum(sum(item["correct"] for item in pool) / len(pool) for pool in pools) / n,
        "oracle": sum(float(any(item["correct"] for item in pool)) for pool in pools) / n,
    }


def paired_bootstrap(pools: list, draws: int = 20000, seed: int = 20260908):
    """Resample pools, not candidates: the pool is the unit a loop chooses within."""

    rng = random.Random(seed)
    n = len(pools)
    deltas = []
    for _ in range(draws):
        sample = [pools[rng.randrange(n)] for _ in range(n)]
        summary = rates(sample)
        deltas.append(summary["blind"] - summary["prompted"])
    deltas.sort()
    return deltas[int(0.025 * draws)], deltas[int(0.975 * draws)]


def sign_test(pools: list):
    """Exact two-sided sign test over the pools where the two rules differ."""

    up = sum(1 for pool in pools if pick(pool, "blind") > pick(pool, "prompted"))
    down = sum(1 for pool in pools if pick(pool, "blind") < pick(pool, "prompted"))
    n = up + down
    if n == 0:
        return up, down, 1.0
    tail = sum(math.comb(n, k) for k in range(min(up, down) + 1)) / 2 ** n
    return up, down, min(1.0, 2 * tail)


def report(title: str, pools: list) -> None:
    if len(pools) < 2:
        print(f"{title}: {len(pools)} pools, nothing to say")
        return
    summary = rates(pools)
    strict = rates(pools, first_index=True)
    low, high = paired_bootstrap(pools)
    up, down, p = sign_test(pools)
    headroom = summary["oracle"] - summary["random"]
    closed = (summary["blind"] - summary["prompted"]) / headroom if headroom > 0 else float("nan")
    print(f"{title}  ({len(pools)} pools, {sum(len(pool) for pool in pools)} images)")
    print(f"    random     {summary['random']:.3f}")
    print(f"    prompted   {summary['prompted']:.3f}"
          f"   (first-index tie-break {strict['prompted']:.3f})")
    print(f"    blind      {summary['blind']:.3f}"
          f"   (first-index tie-break {strict['blind']:.3f})")
    print(f"    oracle     {summary['oracle']:.3f}")
    print(f"    blind - prompted  {summary['blind'] - summary['prompted']:+.3f} "
          f"[{low:+.3f},{high:+.3f}]  sign test {up}:{down}, p={p:.3g}")
    print(f"    share of the random->oracle headroom closed: {closed:+.1%}")
    print()


def main() -> None:
    pools = by_spec(load())
    everything = list(pools.values())
    decidable = [pool for pool in everything
                 if 0 < sum(item["correct"] for item in pool) < len(pool)]
    conflicted = [pool for pool in decidable if any(item["conflicts"] for item in pool)]

    print("Selection benefit of blind over prompted self-verification")
    print("=" * 74)
    print()
    report("every pool with 2+ candidates", everything)
    report("pools where selection can matter (mixed correct/incorrect)", decidable)
    report("...and at least one description/pixel conflict among the questions", conflicted)

    blind_split = sum(1 for pool in everything if len({item["blind"] for item in pool}) > 1)
    told_split = sum(1 for pool in everything if len({item["prompted"] for item in pool}) > 1)
    print(f"pools where the score separates any two candidates: "
          f"blind {blind_split}/{len(everything)}, prompted {told_split}/{len(everything)}")
    print()

    print("-" * 74)
    print("replication: the two corpus runs are different training recipes")
    print("(2+1 and 1+1+1), drawn by different checkpoints. They are two")
    print("samples of the phenomenon, not one split in half.")
    print()
    for run in RUNS:
        report(f"{run}, selection can matter",
               [pool for key, pool in pools.items()
                if key[0] == run and 0 < sum(item["correct"] for item in pool) < len(pool)])

    print("-" * 74)
    print("control 1: is blind just breaking more ties?")
    print("Restricted to the pools where BOTH rules already order the candidates,")
    print("so neither wins by granularity. If the gap survives here it is the")
    print("answers that differ, not the resolution of the scale.")
    print()
    both_ordered = [pool for pool in decidable
                    if len({item["blind"] for item in pool}) > 1
                    and len({item["prompted"] for item in pool}) > 1]
    report("both rules discriminate, selection can matter", both_ordered)

    print("-" * 74)
    print("control 2: score on the non-conflict questions only.")
    print("On those trials the context ablation measured 0.979 blind against")
    print("0.990 told -- the description helps. So the mechanism predicts the")
    print("selection gap collapses here. If it does not, the gap was never")
    print("about the description overriding perception.")
    print()
    report("non-conflict questions only, selection can matter",
           non_conflict_pools(decidable))


def non_conflict_pools(pools: list) -> list:
    """The same pools rescored as if the conflicting questions had not been asked.

    A candidate whose questions are all conflicts has no score left, and a pool
    that loses a candidate stops being the same choice, so those pools drop out
    rather than quietly shrinking.
    """

    rebuilt = []
    for pool in pools:
        kept = [item for item in pool if item["asked"] > item["conflicts"]]
        if len(kept) != len(pool):
            continue
        rebuilt.append([{**item,
                         "blind": item["blind_clean"],
                         "prompted": item["prompted_clean"]} for item in kept])
    return rebuilt


if __name__ == "__main__":
    main()
