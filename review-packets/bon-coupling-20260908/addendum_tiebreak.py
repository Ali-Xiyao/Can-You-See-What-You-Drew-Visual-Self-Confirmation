"""Exploratory, post-hoc: is the atomic scorer's problem resolution or signal?

NOT in PROTOCOL.md. Registered here as exploratory and labelled as such in
RESULTS.md. The pre-registered comparison stands on its own; this only asks a
mechanistic follow-up question.

If the atomic ranking with cycle used purely as a tie-break reaches the pure
cycle curve, then the atomic ranking contributes nothing that cycle does not
already have, and the whole atomic deficit is resolution. If it lands between,
both the ranking and the resolution matter.
"""
from __future__ import annotations

import json
from pathlib import Path

from bon_analysis import (
    BANK,
    aggregate,
    atomic_scores,
    curve,
    cycle_scores,
    jsonl,
    paired_difference,
)

HERE = Path(__file__).resolve().parent
NS = (1, 2, 3, 4, 5, 6)


def main():
    pools = jsonl(BANK / "pools.jsonl")
    atomic = atomic_scores(pools, jsonl(BANK / "observations.naive.jsonl"))
    cycle = cycle_scores(pools, HERE / "cycle-scores.jsonl")
    hybrid = {
        pid: {cid: (atomic[pid][cid], cycle[pid][cid]) for cid in atomic[pid]}
        for pid in atomic
    }

    rows = {
        "atomic_prompt_on": curve(pools, atomic, NS),
        "cycle_logprob": curve(pools, cycle, NS),
        "atomic_then_cycle_tiebreak": curve(pools, hybrid, NS),
    }

    # Selection efficiency: share of the oracle-achievable gain that is realised.
    report = {"note": "exploratory post-hoc, not in PROTOCOL.md", "strata": {}}
    for stratum, sub in (
        ("all_232", pools),
        ("k4_138", [p for p in pools if len(p["candidates"]) == 4]),
        ("k6_94", [p for p in pools if len(p["candidates"]) == 6]),
    ):
        keys = {p["prompt_id"] for p in sub}
        block = {}
        for name, all_rows in rows.items():
            agg = aggregate([r for r in all_rows if r["prompt_id"] in keys], NS)
            block[name] = {
                str(n): {
                    "n_pools": agg[n]["n_pools"],
                    "external": agg[n]["point"],
                    "gain": agg[n]["gain"],
                    "gain_ci": agg[n]["gain_ci"],
                    "oracle": agg[n]["oracle"],
                    "selection_efficiency": (
                        agg[n]["gain"] / (agg[n]["oracle"] - agg[1]["point"])
                        if agg[n]["oracle"] > agg[1]["point"] else None
                    ),
                }
                for n in sorted(agg)
            }
        report["strata"][stratum] = block

    report["hybrid_minus_pure_cycle"] = {
        f"n{n}": paired_difference(rows["atomic_then_cycle_tiebreak"], rows["cycle_logprob"], n)
        for n in (2, 4, 6)
    }
    report["hybrid_minus_atomic"] = {
        f"n{n}": paired_difference(rows["atomic_then_cycle_tiebreak"], rows["atomic_prompt_on"], n)
        for n in (2, 4, 6)
    }

    (HERE / "addendum-tiebreak.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("selection efficiency = gain / (oracle - base); n_pools shown because")
    print("N>=5 exists only for K6 pools, so those rows are a different population.\n")
    for stratum, block in report["strata"].items():
        print(f"===== {stratum} =====")
        for name, by_n in block.items():
            cells = "  ".join(
                f"N{n}[{v['n_pools']}]={v['external']:.4f}"
                f"({v['selection_efficiency']:.0%})" if v["selection_efficiency"] else
                f"N{n}[{v['n_pools']}]={v['external']:.4f}"
                for n, v in by_n.items()
            )
            print(f"  {name:28s} {cells}")
        print()
    for label in ("hybrid_minus_pure_cycle", "hybrid_minus_atomic"):
        print(f"===== {label} =====")
        for key, value in report[label].items():
            print(f"  {key}: {value['difference']:+.4f} "
                  f"[{value['ci'][0]:+.4f},{value['ci'][1]:+.4f}]  n={value['n_pools']}")


if __name__ == "__main__":
    main()
