"""Does the training selector actually select? Measure it, on the frozen records.

STATUS 38 reports that it does not: the naive score puts 83.2% of candidates at
the ceiling, and on natural pools it picks the externally-correct image at the
rate uniform random would. Those numbers are load-bearing -- the whole next
round of the project is ordered by them -- so they live in a script that
recomputes them from `runs/v4/gate-b/` rather than in prose.

Three families of number, deliberately kept apart:

  selection    how often each criterion picks a correct candidate, against the
               uniform-random expectation for the same pools. Reported per K,
               because K=4 and K=6 are independent subsets and two subsets
               agreeing says more than one pooled number.

  resolution   whether the score can tell candidates apart at all: the ceiling
               rate, the all-tied rate, and -- the one that does not depend on
               the tie-break -- whether correct candidates outscore incorrect
               ones inside the same pool.

  composition  what the natural pools look like BEFORE any filtering. This has
               to come from each run's verified.jsonl, because pools.jsonl is
               already restricted to the mixed ones, and quoting a selection
               rate without it invites reading 39.1% as though every pool had
               something correct to find.

`--expect` re-checks the published figures; the run is only meaningful if the
frozen records still produce them.

    envs/core/python.exe scripts/v4_selector_resolution.py --expect
"""

from __future__ import annotations

import argparse
import collections
import json
from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Any, Iterable

GATE_B = Path("runs/v4/gate-b")
CRITERIA = ("naive", "rfo", "gold")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    score: float
    correct: bool


# ------------------------------------------------------------------ selection


def load_gate_b_pools() -> dict[str, dict[str, Any]]:
    pools = {row["prompt_id"]: row for row in read_jsonl(GATE_B / "pools.jsonl")}
    selection = {row["prompt_id"]: row for row in read_jsonl(GATE_B / "selection.jsonl")}
    disagree = set(pools) ^ set(selection)
    if disagree:
        raise SystemExit(f"pools.jsonl and selection.jsonl disagree on {len(disagree)} ids")
    for prompt_id, pool in pools.items():
        pool["selection"] = selection[prompt_id]
    return pools


def selection_table(pools: dict[str, dict[str, Any]], k: int) -> dict[str, Any]:
    ids = [p for p, row in pools.items() if len(row["candidates"]) == k]
    if not ids:
        return {}
    correct_of = {p: {c["candidate_id"]: bool(c["correct"]) for c in pools[p]["candidates"]}
                  for p in ids}
    out: dict[str, Any] = {
        "n_pools": len(ids),
        # The expectation for picking blind is the mean share of correct
        # candidates, not 1/K: pools differ in how many correct images they
        # hold, and 1/K would be the expectation for a pool that does not exist.
        "uniform": sum(sum(correct_of[p].values()) / k for p in ids) / len(ids),
    }
    for criterion in CRITERIA:
        picked = [correct_of[p][pools[p]["selection"]["selected"][criterion]] for p in ids]
        out[criterion] = sum(picked) / len(picked)
        out[criterion + "_hits"] = sum(picked)
    return out


# ----------------------------------------------------------------- resolution


def pool_candidates(pool: dict[str, Any], criterion: str = "naive") -> list[Candidate]:
    scores = pool["selection"]["scores"][criterion]
    return [Candidate(c["candidate_id"], float(scores[c["candidate_id"]]), bool(c["correct"]))
            for c in pool["candidates"]]


def resolution(pools: Iterable[list[Candidate]]) -> dict[str, Any]:
    pools = list(pools)
    n = len(pools)
    at_ceiling = total = all_tied = unique_top = 0
    higher = equal = lower = 0
    gaps: list[float] = []
    for candidates in pools:
        values = [c.score for c in candidates]
        top = max(values)
        at_ceiling += sum(1 for v in values if v == 1.0)
        total += len(values)
        all_tied += len(set(values)) == 1
        unique_top += sum(1 for v in values if v == top) == 1
        ok = [c.score for c in candidates if c.correct]
        bad = [c.score for c in candidates if not c.correct]
        if not ok or not bad:
            continue
        gap = sum(ok) / len(ok) - sum(bad) / len(bad)
        gaps.append(gap)
        higher += gap > 0
        equal += gap == 0
        lower += gap < 0
    return {
        "n_pools": n,
        "n_candidates": total,
        # The whole distribution, not just the ceiling: three distinct values
        # out of 1116 candidates is the finding, and it is invisible in a
        # single rate.
        "dist": collections.Counter(c.score for pool in pools for c in pool),
        "ceiling": at_ceiling / total,
        "all_tied": all_tied / n,
        "unique_top": unique_top / n,
        "sep_higher": higher / n,
        "sep_equal": equal / n,
        "sep_lower": lower / n,
        "mean_gap": sum(gaps) / len(gaps) if gaps else float("nan"),
    }


# ---------------------------------------------------------------- composition


def natural_composition(runs: list[str]) -> dict[str, Any]:
    """Every pool in the source runs, including the ones gate-b filtered out."""
    counts: collections.Counter = collections.Counter()
    by_k: collections.Counter = collections.Counter()
    for run in runs:
        verified = Path(run) / "verified.jsonl"
        if not verified.exists():
            raise SystemExit(f"{verified} is missing; composition needs every source run")
        pools: dict[str, list[bool]] = collections.defaultdict(list)
        for row in read_jsonl(verified):
            pools[row["spec_id"]].append(bool(row["image_correct"]))
        for oks in pools.values():
            kind = "all_correct" if all(oks) else "all_wrong" if not any(oks) else "mixed"
            counts[kind] += 1
            by_k[(len(oks), kind)] += 1
    return {"counts": dict(counts),
            "by_k": {f"{k}:{kind}": n for (k, kind), n in sorted(by_k.items())}}


# ---------------------------------------------------- prompted vs image-only
# STATUS 38.1. Exploratory and unregistered: a different question set from the
# one the selector uses, two of the four runs, and a tie-break chosen here. It
# is reported because its direction contradicts the proposal's mechanism, not
# because it settles anything.


def condition_pools(run: Path, filename: str) -> dict[str, list[Candidate]]:
    per: dict = collections.defaultdict(lambda: [0, 0])
    label: dict = {}
    for row in read_jsonl(run / filename):
        key = (row["spec_id"], row["candidate_index"])
        per[key][0] += bool(row["correct"])
        per[key][1] += 1
        label[key] = bool(row["image_correct"])
    pools: dict[str, list[Candidate]] = collections.defaultdict(list)
    for (spec_id, index), (ok, asked) in per.items():
        pools[f"{run.name}:{spec_id}"].append(
            Candidate(str(index), ok / asked if asked else 0.0, label[(spec_id, index)]))
    return pools


def mixed_only(pools: dict[str, list[Candidate]]) -> dict[str, list[Candidate]]:
    return {p: c for p, c in pools.items()
            if len(c) >= 2 and any(x.correct for x in c) and not all(x.correct for x in c)}


def pick(candidates: list[Candidate]) -> bool:
    top = max(c.score for c in candidates)
    # Fixed and stated rather than left to dict order: with 60% of pools fully
    # tied, the tie-break decides most picks, so it is part of the measurement.
    return min([c for c in candidates if c.score == top],
               key=lambda c: int(c.candidate_id)).correct


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact test on the discordant pairs; there is no scipy here."""
    m = b + c
    if m == 0:
        return 1.0
    tail = sum(comb(m, k) for k in range(0, min(b, c) + 1)) / 2 ** m
    return min(1.0, 2 * tail)


def prompt_contrast(runs: list[str]) -> dict[str, Any]:
    paired: list[tuple[bool, bool]] = []
    uniform_num = uniform_den = 0.0
    for name in runs:
        run = Path(name)
        if not (run / "answers.prompted.jsonl").exists():
            continue
        prompted = mixed_only(condition_pools(run, "answers.prompted.jsonl"))
        blind = condition_pools(run, "answers.jsonl")
        for pool_id, candidates in prompted.items():
            if pool_id not in blind:
                continue
            paired.append((pick(candidates), pick(blind[pool_id])))
            uniform_num += sum(c.correct for c in candidates) / len(candidates)
            uniform_den += 1
    b = sum(1 for x, y in paired if x and not y)
    c = sum(1 for x, y in paired if y and not x)
    n = len(paired)
    return {
        "n_pools": n,
        "uniform": uniform_num / uniform_den if uniform_den else float("nan"),
        "prompted": sum(x for x, _ in paired) / n if n else float("nan"),
        "image_only": sum(y for _, y in paired) / n if n else float("nan"),
        "only_prompted": b,
        "only_image_only": c,
        "p_two_sided": exact_mcnemar(b, c),
    }


# ------------------------------------------------------------------ reporting

PUBLISHED = {
    "k4": {"n_pools": 138, "uniform": 0.402, "naive": 0.391, "rfo": 0.551, "gold": 1.0},
    "k6": {"n_pools": 94, "uniform": 0.342, "naive": 0.340, "rfo": 0.511, "gold": 1.0},
    "resolution": {"n_candidates": 1116, "ceiling": 0.832, "all_tied": 0.603,
                   "unique_top": 0.034, "sep_higher": 0.349, "sep_equal": 0.612,
                   "mean_gap": 0.052},
    # Descending by score. Compared as a list rather than keyed by score,
    # because 0.667 in the table is 0.6666666666666666 on disk.
    "score_counts": [928, 171, 17],
    "composition": {"all_correct": 8, "mixed": 301, "all_wrong": 147},
    "contrast": {"n_pools": 134, "prompted": 0.590, "image_only": 0.493,
                 "p_two_sided": 0.0192},
}


def check(label: str, got: float, want: float, tolerance: float) -> None:
    if abs(got - want) > tolerance:
        raise SystemExit(
            f"refusing to report: {label} recomputes to {got:.4f}, STATUS 38 says "
            f"{want:.4f}. Either the frozen records moved or this script drifted "
            f"from the analysis that produced them.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--expect", action="store_true",
                        help="fail unless the published STATUS 38 figures come back")
    args = parser.parse_args()

    pools = load_gate_b_pools()
    runs = json.loads((GATE_B / "runs.json").read_text(encoding="utf-8"))["runs"]

    print("=== selection, natural mixed pools")
    tables = {}
    for k in (4, 6):
        table = selection_table(pools, k)
        tables["k" + str(k)] = table
        print(f"  K={k}  n={table['n_pools']}   uniform {table['uniform']:6.1%}   "
              + "   ".join(f"{c} {table[c]:6.1%} ({table[c + '_hits']}/{table['n_pools']})"
                           for c in CRITERIA))

    print("\n=== resolution of the naive score")
    res = resolution(pool_candidates(p) for p in pools.values())
    print(f"  {res['n_candidates']} candidates take {len(res['dist'])} distinct scores:")
    for score, n in sorted(res["dist"].items(), reverse=True):
        print(f"    {score:.3f}  {n:5d}  {n / res['n_candidates']:6.1%}")
    for key in ("ceiling", "all_tied", "unique_top", "sep_higher", "sep_equal", "sep_lower"):
        print(f"  {key:12s} {res[key]:6.1%}")
    print(f"  {'mean_gap':12s} {res['mean_gap']:+.4f}")

    print("\n=== natural pool composition, before any filtering")
    comp = natural_composition(runs)
    total = sum(comp["counts"].values())
    for kind, n in sorted(comp["counts"].items()):
        print(f"  {kind:12s} {n:4d}  {n/total:6.1%}")

    print("\n=== STATUS 38.1, exploratory: prompted vs image-only")
    contrast = prompt_contrast(runs)
    print(f"  n={contrast['n_pools']}   uniform {contrast['uniform']:6.1%}   "
          f"prompted {contrast['prompted']:6.1%}   image_only {contrast['image_only']:6.1%}")
    print(f"  discordant {contrast['only_prompted']}:{contrast['only_image_only']}   "
          f"exact McNemar two-sided p = {contrast['p_two_sided']:.4f}")

    if args.expect:
        for k in ("k4", "k6"):
            for key, want in PUBLISHED[k].items():
                check(f"{k}.{key}", tables[k][key], want,
                      0.5 if key == "n_pools" else 0.001)
        for key, want in PUBLISHED["resolution"].items():
            check(f"resolution.{key}", res[key], want,
                  0.5 if key == "n_candidates" else 0.001)
        counts = [n for _, n in sorted(res["dist"].items(), reverse=True)]
        if counts != PUBLISHED["score_counts"]:
            raise SystemExit(f"refusing to report: the naive score distribution is "
                             f"{counts}, STATUS 38 says {PUBLISHED['score_counts']}")
        for kind, want in PUBLISHED["composition"].items():
            check(f"composition.{kind}", comp["counts"][kind], want, 0.5)
        for key, want in PUBLISHED["contrast"].items():
            check(f"contrast.{key}", contrast[key], want,
                  0.5 if key == "n_pools" else 0.001)
        print("\nall STATUS 38 figures reproduced")


if __name__ == "__main__":
    main()
