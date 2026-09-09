"""Apply the frozen protocol to the 232-pool bank and emit results.json.

Every decision rule here is the one written in PROTOCOL.md before any cycle
score existed. Nothing is chosen after seeing a number.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from bon_analysis import (
    BANK,
    aggregate,
    atomic_scores,
    curve,
    cycle_scores,
    jsonl,
    paired_difference,
    resolution,
)

HERE = Path(__file__).resolve().parent
NS = (1, 2, 3, 4, 5, 6)


def stratify(pools):
    return {
        "all_232": pools,
        "k4_138": [p for p in pools if len(p["candidates"]) == 4],
        "k6_94": [p for p in pools if len(p["candidates"]) == 6],
    }


def gold_composition(pools):
    out = {"pools": len(pools), "mixed": 0, "all_correct": 0, "all_wrong": 0,
           "candidates": 0, "correct_candidates": 0}
    for pool in pools:
        flags = [bool(c["correct"]) for c in pool["candidates"]]
        out["candidates"] += len(flags)
        out["correct_candidates"] += sum(flags)
        out["mixed" if any(flags) and not all(flags) else
            ("all_correct" if all(flags) else "all_wrong")] += 1
    out["candidate_base_rate"] = out["correct_candidates"] / out["candidates"]
    return out


def verdicts(res, agg, k_max):
    """The pre-registered decision rules, applied mechanically."""
    top = agg[k_max]
    return {
        "coupling_holds": top["gain_ci"][0] > 0,
        "coupling_rule": "95% lower bound of gain(N_max) > 0",
        "gain_at_n_max": top["gain"],
        "gain_ci_at_n_max": top["gain_ci"],
        "resolution_insufficient": (
            res["perfect_score_share"] > 0.50 or res["mean_top_tie_size"] > k_max / 2
        ),
        "resolution_rule": "perfect-score share > 50% or mean top tie > K/2",
        "unique_top_share": res["unique_top_share"],
        "resolution_repaired": res["unique_top_share"] >= 0.90,
        "repair_rule": "unique-top share >= 90%",
    }


def turnover(agg, ns):
    """Peak location and the post-peak change, per PROTOCOL section 4."""
    present = [n for n in ns if n in agg]
    peak = max(present, key=lambda n: agg[n]["gain"])
    last = present[-1]
    return {
        "peak_n": peak,
        "peak_kl": agg[peak]["kl"],
        "peak_gain": agg[peak]["gain"],
        "last_n": last,
        "post_peak_change": agg[last]["gain"] - agg[peak]["gain"],
        "turnover_detected": peak < last,
        "d_star_lower_bound_nats": agg[last]["kl"] if peak == last else None,
        "note": "K<=6 caps KL at 0.959 nats; no turnover is the pre-declared expected outcome",
    }


def main():
    pools = jsonl(BANK / "pools.jsonl")
    observations = jsonl(BANK / "observations.naive.jsonl")

    scorers = {
        "atomic_prompt_on": atomic_scores(pools, observations),
        "cycle_logprob": cycle_scores(pools, HERE / "cycle-scores.jsonl"),
    }

    report = {
        "protocol": "review-packets/bon-coupling-20260908/PROTOCOL.md",
        "population": {
            "source": "runs/v4/gate-b-openct2 (read-only)",
            "note": "candidate-availability filtered; every pool is mixed; NOT a natural population",
            "composition": {name: gold_composition(sub) for name, sub in stratify(pools).items()},
        },
        "kl_axis": {str(n): math.log(n) - (n - 1) / n for n in NS},
        "scorers": {},
    }

    rows_by_scorer = {}
    for name, scores in scorers.items():
        rows = curve(pools, scores, NS)
        rows_by_scorer[name] = rows
        entry = {}
        for stratum, sub in stratify(pools).items():
            keys = {p["prompt_id"] for p in sub}
            sub_rows = [r for r in rows if r["prompt_id"] in keys]
            k_max = min(len(p["candidates"]) for p in sub)
            agg = aggregate(sub_rows, NS)
            entry[stratum] = {
                "resolution": resolution(sub, scores),
                "by_n": {str(n): agg[n] for n in sorted(agg)},
                "verdicts": verdicts(resolution(sub, scores), agg, k_max),
                "turnover": turnover(agg, NS),
            }
        report["scorers"][name] = entry

    report["scorer_comparison"] = {
        f"n{n}": paired_difference(
            rows_by_scorer["cycle_logprob"], rows_by_scorer["atomic_prompt_on"], n
        )
        for n in (2, 4, 6)
    }
    report["scorer_comparison"]["rule"] = (
        "cycle minus atomic, paired on prompt_id; interval containing zero is "
        "reported as no difference, point estimates are not ranked"
    )

    (HERE / "results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    for name, entry in report["scorers"].items():
        print(f"\n===== {name} =====")
        for stratum, block in entry.items():
            res, ver = block["resolution"], block["verdicts"]
            print(f"  [{stratum}] unique-top {res['pools_with_unique_top']}/{res['n_pools']} "
                  f"({res['unique_top_share']:.1%})  mean tie {res['mean_top_tie_size']:.2f}  "
                  f"perfect {res['perfect_score_share']:.1%}")
            for n, block_n in block["by_n"].items():
                lo, hi = block_n["gain_ci"]
                print(f"      N={n}[{block_n['n_pools']}] KL={block_n['kl']:.3f}  ext={block_n['point']:.4f}  "
                      f"gain={block_n['gain']:+.4f} [{lo:+.4f},{hi:+.4f}]  "
                      f"tie={block_n['mean_top_size']:.2f}  oracle={block_n['oracle']:.4f}")
            print(f"      coupling={ver['coupling_holds']}  "
                  f"resolution_insufficient={ver['resolution_insufficient']}  "
                  f"repaired={ver['resolution_repaired']}  "
                  f"peak N={block['turnover']['peak_n']} turnover={block['turnover']['turnover_detected']}")
    print("\n===== cycle minus atomic =====")
    for key, value in report["scorer_comparison"].items():
        if isinstance(value, dict):
            print(f"  {key}: {value['difference']:+.4f} "
                  f"[{value['ci'][0]:+.4f},{value['ci'][1]:+.4f}]  n={value['n_pools']}")


if __name__ == "__main__":
    main()
