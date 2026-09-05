"""How accurate would the observer have to be for these questions to pass.

The ceiling script says perfect observation clears all four thresholds and the
real observer does not. That leaves the question a next pre-registration has to
answer before it can set a target: how much of the gap has to close.

Each answer is replaced by the truth from `detections` with probability p, over
many seeds, and the four criteria are recomputed. The reported accuracy is the
measured agreement with the detections at that p, not p itself, so the x-axis is
the quantity an actual observer change would move.

Two things this is not. It is not a prediction: it assumes the errors that get
fixed are a random subset, and a better observer would fix the easy ones first,
so the real curve is likely more favourable at low p and this understates how
far a modest improvement goes. And it is not a result -- the truth it mixes in
comes from the same detectors the verdict was built on, so everything here
inherits the circularity flagged in `v4_observation_ceiling.py`.

    envs/core/python.exe scripts/v4_accuracy_sensitivity.py runs/v4/gate-b-openct2
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v4_observation_ceiling import detections, truth_for  # noqa: E402
from v4_resolution_verdict import REGISTERED, load_pools  # noqa: E402
from v4_selector_resolution import Candidate, read_jsonl, resolution  # noqa: E402

SEEDS = 20
GRID = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("outdir")
    parser.add_argument("--arm", default="rfo", choices=["naive", "rfo"])
    args = parser.parse_args()
    out_dir = Path(args.outdir)

    pools = load_pools(out_dir)
    found = detections(out_dir)
    questions = {(p["prompt_id"], q["question_id"]): q
                 for p in pools.values() for q in p["questions"]}
    image_of = {(p["prompt_id"], c["candidate_id"]): c["image_path"]
                for p in pools.values() for c in p["candidates"]}

    # (prompt_id, candidate_id) -> [(said, truth, wanted), ...]
    answers: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
    for row in read_jsonl(out_dir / f"observations.{args.arm}.jsonl"):
        key = (row["prompt_id"], row["candidate_id"])
        counts = found[image_of[key]]
        answers[key] = [
            (a["normalized_answer"],
             truth_for(questions[(row["prompt_id"], a["question_id"])], counts),
             questions[(row["prompt_id"], a["question_id"])]["expected_answer"])
            for a in row["observation"]["answers"]]

    print(f"{args.outdir} [{args.arm}]：把回答按比例换成真值，判据怎么走"
          f"（{SEEDS} 个种子取平均）\n")
    header = "  ".join(f"{name.replace(' ', '')}" for _k, name, *_ in REGISTERED)
    print(f"  {'混入真值':>8s} {'实测数对图':>10s}   {header}")

    for p in GRID:
        totals = collections.defaultdict(float)
        accuracy = 0.0
        for seed in range(SEEDS if 0 < p < 1 else 1):
            rng = random.Random(f"{args.arm}:{p}:{seed}")
            scored: list[list[Candidate]] = []
            agree = total = 0
            for pool in pools.values():
                cells = []
                for candidate in pool["candidates"]:
                    rows = answers[(pool["prompt_id"], candidate["candidate_id"])]
                    said = [truth if rng.random() < p else got
                            for got, truth, _w in rows]
                    agree += sum(s == truth for s, (_g, truth, _w) in zip(said, rows))
                    total += len(rows)
                    hits = sum(s == wanted for s, (_g, _t, wanted) in zip(said, rows))
                    cells.append(Candidate(candidate["candidate_id"],
                                           hits / len(rows), bool(candidate["correct"])))
                scored.append(cells)
            got = resolution(scored)
            for key, *_ in REGISTERED:
                totals[key] += got[key]
            accuracy += agree / total
        runs = SEEDS if 0 < p < 1 else 1
        cells = []
        for key, _n, direction, threshold, *_ in REGISTERED:
            value = totals[key] / runs
            ok = value >= threshold if direction == ">=" else value <= threshold
            cells.append(f"{value:6.1%}{'*' if ok else ' '}")
        print(f"  {p:8.0%} {accuracy / runs:10.1%}   " + "  ".join(cells))

    print("""
  * = 达到门槛。「实测数对图」是与 detections 的一致率，即真实可移动的那个量。
  这不是预测：它假设被修好的错误是随机子集，而更强的观察者会先修容易的，
  所以真实曲线在低比例处大概率比这条更有利，此表低估了小幅改善的作用。
  也不是结果：混入的真值与裁定同源，继承 v4_observation_ceiling.py 里那条循环性保留。""")


if __name__ == "__main__":
    main()
