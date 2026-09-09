"""Apply the frozen high-N protocol and emit results.json.

Every rule here is in PROTOCOL.md, written before the cycle scores existed.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from bon_fast import aggregate, bon_exact, jsonl, paired_delta, validate

HERE = Path(__file__).resolve().parent
COHORTS = (14, 20, 24)


def composition(gold):
    if all(gold):
        return "all_correct"
    if not any(gold):
        return "all_wrong"
    return "mixed"


def build(workset, scores, k):
    """Nested prefix of the declared per-spec ordering; skip specs with fewer than k."""
    by_spec = defaultdict(list)
    for item in workset:
        by_spec[item["spec_id"]].append(item)
    pools = {}
    for spec_id, items in by_spec.items():
        items.sort(key=lambda c: c["order_index"])
        if len(items) < k:
            continue
        chosen = items[:k]
        pools[spec_id] = {
            "scores": [scores[c["image_path"]] for c in chosen],
            "gold": [c["correct"] for c in chosen],
            "batches": Counter(c["batch"].split("-")[0] for c in chosen),
            "composition": composition([c["correct"] for c in chosen]),
        }
    return pools


def per_pool_curves(pools, ns):
    return {
        spec_id: {n: bon_exact(pool["scores"], pool["gold"], n) for n in ns}
        for spec_id, pool in pools.items()
    }


def verdicts(agg, ns, curves):
    present = [n for n in ns if n in agg]
    n_max = present[-1]
    n_peak = max(present, key=lambda n: agg[n]["gain"])
    turn = paired_delta(curves, n_max, n_peak) if n_peak != n_max else None
    last_two = present[-2:]
    slope = (
        (agg[last_two[1]]["gain"] - agg[last_two[0]]["gain"])
        / (agg[last_two[1]]["kl"] - agg[last_two[0]]["kl"])
    )
    return {
        "coupling_holds": agg[n_max]["gain_ci"][0] > 0,
        "coupling_rule": "95% lower bound of gain(N_max) > 0",
        "n_max": n_max,
        "n_peak": n_peak,
        "peak_kl": agg[n_peak]["kl"],
        "turnover_delta": turn,
        "turnover_confirmed": bool(turn and turn["ci"][1] < 0),
        "turnover_rule": "paired 95% upper bound of gain(N_max) - gain(N_peak) < 0",
        "d_star_lower_bound_nats": agg[n_max]["kl"] if n_peak == n_max else None,
        "final_segment_pp_per_nat": slope * 100,
        "slope_note": "descriptive distance-to-turnover only; must not be extrapolated to D*",
    }


def main():
    check = validate()
    if not check["passes"]:
        raise SystemExit(f"estimator regression failed: {check}")

    workset = jsonl(HERE / "workset.jsonl")
    scores = {r["image_path"]: float(r["cycle_score"]) for r in jsonl(HERE / "cycle-scores.jsonl")}
    missing = [w["image_path"] for w in workset if w["image_path"] not in scores]
    if missing:
        raise SystemExit(f"{len(missing)} workset images have no cycle score")

    report = {
        "protocol": "review-packets/high-n-bon-20260908/PROTOCOL.md",
        "estimator_regression": check,
        "workset": {
            "candidates": len(workset),
            "specs": len({w["spec_id"] for w in workset}),
            "base_rate": sum(w["correct"] for w in workset) / len(workset),
        },
        "cohorts": {},
    }

    for k in COHORTS:
        pools = build(workset, scores, k)
        ns = tuple(range(1, k + 1))
        curves = per_pool_curves(pools, ns)
        agg = aggregate(curves, ns)
        comp = Counter(p["composition"] for p in pools.values())

        block = {
            "k": k,
            "n_pools": len(pools),
            "composition": dict(comp),
            "mixed_share": comp["mixed"] / len(pools),
            "by_n": {str(n): agg[n] for n in sorted(agg)},
            "verdicts": verdicts(agg, ns, curves),
            "strata": {},
            "batch_sensitivity": {},
        }

        for stratum in ("mixed", "all_wrong", "all_correct"):
            keys = {s for s, p in pools.items() if p["composition"] == stratum}
            if len(keys) < 2:
                block["strata"][stratum] = {"n_pools": len(keys), "note": "too few to report"}
                continue
            sub = {s: c for s, c in curves.items() if s in keys}
            sub_agg = aggregate(sub, ns)
            block["strata"][stratum] = {
                "n_pools": len(keys),
                "by_n": {str(n): sub_agg[n] for n in sorted(sub_agg)},
                "verdicts": verdicts(sub_agg, ns, sub),
            }

        # Batch sensitivity: pools whose K candidates are majority from one batch.
        for tag in ("main", "conf", "conf2"):
            keys = {s for s, p in pools.items() if p["batches"].most_common(1)[0][0] == tag}
            if len(keys) < 10:
                block["batch_sensitivity"][tag] = {"n_pools": len(keys), "note": "too few"}
                continue
            sub = {s: c for s, c in curves.items() if s in keys}
            sub_agg = aggregate(sub, ns)
            block["batch_sensitivity"][tag] = {
                "n_pools": len(keys),
                "gain_at_n_max": sub_agg[k]["gain"],
                "gain_ci": sub_agg[k]["gain_ci"],
                "peak_n": verdicts(sub_agg, ns, sub)["n_peak"],
            }

        report["cohorts"][f"k{k}"] = block

    (HERE / "results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    for name, block in report["cohorts"].items():
        v = block["verdicts"]
        print(f"\n===== {name}: {block['n_pools']} pools, composition {block['composition']} "
              f"(mixed {block['mixed_share']:.1%}) =====")
        for n, row in block["by_n"].items():
            lo, hi = row["gain_ci"]
            print(f"  N={n:>2} KL={row['kl']:.3f}  ext={row['external']:.4f}  "
                  f"gain={row['gain']:+.4f} [{lo:+.4f},{hi:+.4f}]  "
                  f"oracle={row['oracle']:.4f}  tie={row['mean_top_size']:.2f}")
        print(f"  coupling={v['coupling_holds']}  peak N={v['n_peak']} (KL {v['peak_kl']:.3f})  "
              f"turnover={v['turnover_confirmed']}  "
              f"final slope={v['final_segment_pp_per_nat']:+.1f} pp/nat")
        if v["turnover_delta"]:
            d = v["turnover_delta"]
            print(f"  gain(N_max)-gain(N_peak) = {d['delta']:+.4f} "
                  f"[{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}]")
        mixed = block["strata"].get("mixed", {})
        if "verdicts" in mixed:
            mv = mixed["verdicts"]
            print(f"  [mixed-only {mixed['n_pools']} pools] peak N={mv['n_peak']}  "
                  f"turnover={mv['turnover_confirmed']}  "
                  f"final slope={mv['final_segment_pp_per_nat']:+.1f} pp/nat")


if __name__ == "__main__":
    main()
