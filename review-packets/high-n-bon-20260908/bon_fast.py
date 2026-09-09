"""Closed-form finite-pool BoN for large K, plus the regression that licenses it.

C(24,12) is 2.7 million subsets per pool, so the enumerating estimator used in
review-packets/bon-coupling-20260908 does not scale. Group the candidates by
distinct score in descending order; with m_j the size of group j and
M_j the running total,

    P(group j is the highest group present in a random N-subset)
        = [C(K - M_{j-1}, N) - C(K - M_j, N)] / C(K, N)
    E[correct @ N] = sum_j P_j * (mean gold within group j)

which is word-for-word the same rule as "uniform among the subset's top ties":
membership of group j in the subset is exchangeable, so the expected gold given
that group j is on top is the group mean regardless of how many of its members
were drawn. Exact, not sampled, and O(number of distinct scores).

PROTOCOL.md section 3 makes using this conditional on reproducing the validated
enumeration on the K=4/6 bank, which is what validate() does.
"""
from __future__ import annotations

import json
import math
import random
import sys
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

BOOTSTRAP = 5000
SEED = 20260908


def jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def bon_exact(scores, gold, n):
    """scores and gold are parallel sequences over one pool."""
    k = len(scores)
    if not 1 <= n <= k:
        raise ValueError(f"N={n} outside pool of {k}")
    order = sorted(range(k), key=lambda i: scores[i], reverse=True)
    groups, i = [], 0
    while i < k:
        j = i
        while j < k and scores[order[j]] == scores[order[i]]:
            j += 1
        members = order[i:j]
        groups.append((len(members), sum(bool(gold[m]) for m in members) / len(members)))
        i = j
    total = comb(k, n)
    uniform_top = top_size = 0.0
    cumulative = 0
    for size, mean_gold in groups:
        previous = cumulative
        cumulative += size
        weight = (comb(k - previous, n) - comb(k - cumulative, n)) / total
        uniform_top += weight * mean_gold
        top_size += size * comb(k - previous - 1, n - 1) / total
    correct = sum(bool(g) for g in gold)
    return {
        "uniform_top": uniform_top,
        "oracle_any_correct": 1.0 - comb(k - correct, n) / total,
        "top_size": top_size,
        "subsets": total,
    }


def aggregate(per_pool, ns, rng_seed=SEED):
    """per_pool: {pool_key: {n: bon_exact(...)}}. Pool-equal means, pool bootstrap."""
    rng = random.Random(rng_seed)
    keys = sorted(per_pool)
    out = {}
    for n in ns:
        present = [k for k in keys if n in per_pool[k]]
        if not present:
            continue
        gains = [per_pool[k][n]["uniform_top"] - per_pool[k][1]["uniform_top"] for k in present]
        size = len(gains)
        draws = sorted(
            sum(gains[rng.randrange(size)] for _ in range(size)) / size for _ in range(BOOTSTRAP)
        )
        out[n] = {
            "n_pools": size,
            "kl": math.log(n) - (n - 1) / n,
            "external": sum(per_pool[k][n]["uniform_top"] for k in present) / size,
            "gain": sum(gains) / size,
            "gain_ci": [draws[int(0.025 * BOOTSTRAP)], draws[int(0.975 * BOOTSTRAP) - 1]],
            "oracle": sum(per_pool[k][n]["oracle_any_correct"] for k in present) / size,
            "mean_top_size": sum(per_pool[k][n]["top_size"] for k in present) / size,
        }
    return out


def paired_delta(per_pool, n_a, n_b, rng_seed=SEED + 2):
    """gain(n_a) - gain(n_b), paired on pool. Used for the turnover rule."""
    rng = random.Random(rng_seed)
    keys = [k for k in sorted(per_pool) if n_a in per_pool[k] and n_b in per_pool[k]]
    diffs = [per_pool[k][n_a]["uniform_top"] - per_pool[k][n_b]["uniform_top"] for k in keys]
    size = len(diffs)
    draws = sorted(
        sum(diffs[rng.randrange(size)] for _ in range(size)) / size for _ in range(BOOTSTRAP)
    )
    return {
        "n_pools": size,
        "delta": sum(diffs) / size,
        "ci": [draws[int(0.025 * BOOTSTRAP)], draws[int(0.975 * BOOTSTRAP) - 1]],
    }


def validate():
    """Reproduce the enumerating estimator exactly on the K=4/6 openct2 bank."""
    sys.path.insert(0, str(ROOT / "review-packets/bon-coupling-20260908"))
    from bon_analysis import atomic_scores, cycle_scores, exact_bon  # noqa: E402

    bank = ROOT / "runs/v4/gate-b-openct2"
    pools = jsonl(bank / "pools.jsonl")
    scorers = {
        "atomic_prompt_on": atomic_scores(pools, jsonl(bank / "observations.naive.jsonl")),
        "cycle_logprob": cycle_scores(
            pools, ROOT / "review-packets/bon-coupling-20260908/cycle-scores.jsonl"
        ),
    }
    checked = 0
    worst = 0.0
    worst_where = None
    for name, by_pool in scorers.items():
        for pool in pools:
            ids = [c["candidate_id"] for c in pool["candidates"]]
            scores = [by_pool[pool["prompt_id"]][i] for i in ids]
            gold = [bool(c["correct"]) for c in pool["candidates"]]
            for n in range(1, len(ids) + 1):
                slow = exact_bon(pool["candidates"], by_pool[pool["prompt_id"]], n)
                fast = bon_exact(scores, gold, n)
                for key in ("uniform_top", "oracle_any_correct", "top_size", "subsets"):
                    delta = abs(float(slow[key]) - float(fast[key]))
                    if delta > worst:
                        worst, worst_where = delta, (name, pool["prompt_id"], n, key)
                    checked += 1
    return {
        "comparisons": checked,
        "max_abs_difference": worst,
        "worst_at": worst_where,
        "passes": worst < 1e-12,
        "rule": "PROTOCOL.md section 3: must match the validated enumeration exactly",
    }


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2, ensure_ascii=False))
