"""Exact finite-pool BoN, resolution diagnostics, pool bootstrap. CPU only.

Independent reimplementation: the estimator is re-derived here rather than
imported from the replay-ablation packet, so that reproducing that packet's
published 16-pool curve is a real cross-check and not a tautology.
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter
from fractions import Fraction
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BANK = ROOT / "runs/v4/gate-b-openct2"
OLD = ROOT / "review-packets/selector-measurement-next-20260906"

BOOTSTRAP = 5000
SEED = 20260908


def jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def exact_bon(pool, scores, n):
    """Average over every unordered n-subset; uniform among that subset's top ties."""
    ids = [c["candidate_id"] for c in pool]
    gold = {c["candidate_id"]: bool(c["correct"]) for c in pool}
    if not 1 <= n <= len(ids):
        raise ValueError("N=%d outside pool of %d" % (n, len(ids)))
    total = Counter()
    subsets = 0
    for subset in combinations(ids, n):
        best = max(scores[i] for i in subset)
        ties = [i for i in subset if scores[i] == best]
        total["uniform_top"] += Fraction(sum(gold[i] for i in ties), len(ties))
        total["oracle_any_correct"] += int(any(gold[i] for i in subset))
        total["top_size"] += len(ties)
        subsets += 1
    return {key: float(value / subsets) for key, value in total.items()} | {"subsets": subsets}


def curve(pools, scores_by_pool, ns):
    """Per-pool BoN values, one row per pool."""
    rows = []
    for pool in pools:
        key = pool["prompt_id"]
        scores = scores_by_pool[key]
        k = len(pool["candidates"])
        rows.append(
            {
                "prompt_id": key,
                "k": k,
                "by_n": {n: exact_bon(pool["candidates"], scores, n) for n in ns if n <= k},
            }
        )
    return rows


def aggregate(rows, ns, metric="uniform_top"):
    """Pool-equal means and paired gain over N=1, with pool bootstrap intervals."""
    rng = random.Random(SEED)
    usable = [r for r in rows if 1 in r["by_n"]]
    out = {}
    for n in ns:
        present = [r for r in usable if n in r["by_n"]]
        if not present:
            continue
        point = sum(r["by_n"][n][metric] for r in present) / len(present)
        gains = [r["by_n"][n][metric] - r["by_n"][1][metric] for r in present]
        gain = sum(gains) / len(gains)
        size = len(gains)
        draws = []
        for _ in range(BOOTSTRAP):
            draws.append(sum(gains[rng.randrange(size)] for _ in range(size)) / size)
        draws.sort()
        out[n] = {
            "n_pools": size,
            "point": point,
            "gain": gain,
            "gain_ci": [draws[int(0.025 * BOOTSTRAP)], draws[int(0.975 * BOOTSTRAP) - 1]],
            "kl": math.log(n) - (n - 1) / n,
            "mean_top_size": sum(r["by_n"][n]["top_size"] for r in present) / size,
            "oracle": sum(r["by_n"][n]["oracle_any_correct"] for r in present) / size,
        }
    return out


def paired_difference(rows_a, rows_b, n, metric="uniform_top"):
    """Scorer A minus scorer B at the same N, paired on prompt_id."""
    rng = random.Random(SEED + 1)
    index_b = {r["prompt_id"]: r for r in rows_b}
    diffs = [
        r["by_n"][n][metric] - index_b[r["prompt_id"]]["by_n"][n][metric]
        for r in rows_a
        if n in r["by_n"] and n in index_b[r["prompt_id"]]["by_n"]
    ]
    size = len(diffs)
    draws = sorted(
        sum(diffs[rng.randrange(size)] for _ in range(size)) / size for _ in range(BOOTSTRAP)
    )
    return {
        "n_pools": size,
        "difference": sum(diffs) / size,
        "ci": [draws[int(0.025 * BOOTSTRAP)], draws[int(0.975 * BOOTSTRAP) - 1]],
    }


def resolution(pools, scores_by_pool):
    """Saturation and tie diagnostics for a scorer on this population."""
    perfect = total = unique_top = 0
    tie_sizes = []
    for pool in pools:
        scores = scores_by_pool[pool["prompt_id"]]
        values = [scores[c["candidate_id"]] for c in pool["candidates"]]
        top = max(values)
        ties = sum(1 for v in values if v == top)
        tie_sizes.append(ties)
        unique_top += ties == 1
        perfect += sum(1 for v in values if v == 1)
        total += len(values)
    return {
        "n_pools": len(pools),
        "n_candidates": total,
        "perfect_score_candidates": perfect,
        "perfect_score_share": perfect / total,
        "pools_with_unique_top": unique_top,
        "unique_top_share": unique_top / len(pools),
        "mean_top_tie_size": sum(tie_sizes) / len(tie_sizes),
        "tie_size_histogram": dict(sorted(Counter(tie_sizes).items())),
    }


# ---------------------------------------------------------------- scorers


def atomic_scores(pools, observations):
    """Fraction of questions answered as expected. Exact rationals."""
    by_key = {(r["prompt_id"], r["candidate_id"]): r["observation"] for r in observations}
    out = {}
    for pool in pools:
        expected = {q["question_id"]: q["expected_answer"] for q in pool["questions"]}
        scores = {}
        for candidate in pool["candidates"]:
            answers = {
                a["question_id"]: a["normalized_answer"]
                for a in by_key[(pool["prompt_id"], candidate["candidate_id"])]["answers"]
            }
            hits = sum(answers.get(qid) == want for qid, want in expected.items())
            scores[candidate["candidate_id"]] = Fraction(hits, len(expected))
        out[pool["prompt_id"]] = scores
    return out


def cycle_scores(pools, path):
    """Continuous log p(prompt|image), one record per image."""
    by_key = {(r["prompt_id"], r["candidate_id"]): float(r["cycle_score"]) for r in jsonl(path)}
    return {
        pool["prompt_id"]: {
            c["candidate_id"]: by_key[(pool["prompt_id"], c["candidate_id"])]
            for c in pool["candidates"]
        }
        for pool in pools
    }


# ---------------------------------------------------------------- validation


def validate_against_published():
    """Reproduce the published 16-pool base/prompt_off curve: 40.63 / 60.21 / 72.40."""
    bank = json.loads((OLD / "fixed-bank.json").read_text(encoding="utf-8"))
    records = []
    for sub in ("observations", "matched-frame"):
        target = OLD / sub / "base-00000" / "answers.jsonl"
        if target.exists():
            records += jsonl(target)
    answers = {
        (r["condition"], r["prompt_id"], r["candidate_id"], a["question_id"]): a
        for r in records
        for a in r["observation"]["answers"]
    }
    pools, scores_by_pool = [], {}
    for pool in bank["pools"]:
        questions = pool["questions"][: pool["n_original_questions"]]
        scores = {}
        for candidate in pool["candidates"]:
            hits = sum(
                answers[
                    ("prompt_off", pool["prompt_id"], candidate["candidate_id"], q["question_id"])
                ]["normalized_answer"]
                == q["expected_answer"]
                for q in questions
            )
            scores[candidate["candidate_id"]] = Fraction(hits, len(questions))
        pools.append(pool)
        scores_by_pool[pool["prompt_id"]] = scores
    rows = curve(pools, scores_by_pool, (1, 2, 4))
    full = aggregate(rows, (1, 2, 4))
    primary = aggregate(
        [r for r, p in zip(rows, pools) if p["primary_scene_disjoint"]], (1, 2, 4)
    )
    published = {
        "full_16": {1: 0.4063, 2: 0.6021, 4: 0.7240},
        "primary_14": {1: 0.4107, 2: 0.6012, 4: 0.7202},
    }
    report = {}
    for name, got in (("full_16", full), ("primary_14", primary)):
        report[name] = {
            str(n): {
                "recomputed": round(got[n]["point"], 6),
                "published": published[name][n],
                "abs_diff": round(abs(got[n]["point"] - published[name][n]), 6),
                "matches": abs(got[n]["point"] - published[name][n]) < 5e-5,
            }
            for n in (1, 2, 4)
        }
        report[name]["n_pools"] = got[1]["n_pools"]
    return report


if __name__ == "__main__":
    print(json.dumps(validate_against_published(), indent=2, ensure_ascii=False))
