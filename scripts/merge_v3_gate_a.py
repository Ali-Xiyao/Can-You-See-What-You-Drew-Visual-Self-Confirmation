"""Merge sharded Gate A runs into one report. Shards are disjoint prompt ranges."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from selfsight.utils.jsonl import read_jsonl
from selfsight.v3.bank import BalancedPool, bank_supply_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-balanced-rate", type=float, default=0.40)
    parser.add_argument("--min-informative-pools", type=int, default=None)
    args = parser.parse_args()

    pools, shards, natural_hits, natural_total = [], [], 0, 0
    for shard in args.shard:
        root = Path(shard).resolve()
        report = json.loads((root / "gate_a.json").read_text(encoding="utf-8"))
        shards.append({"path": str(root), "prompts": report["prompts_searched"]})
        for row in read_jsonl(root / "pools.jsonl"):
            pools.append(
                BalancedPool(
                    prompt_id=row["prompt_id"],
                    family=row["family"],
                    candidate_ids=tuple(row["candidate_ids"]),
                    correct_ids=tuple(row["correct_ids"]),
                    incorrect_ids=tuple(row["incorrect_ids"]),
                    searched=int(row["searched"]),
                    abstained=int(row["abstained"]),
                    informative=bool(row["informative"]),
                    balanced=bool(row["balanced"]),
                    reason=row["reason"],
                )
            )
        natural = report.get("natural_informative_rate")
        if natural is not None:
            natural_hits += int(report["natural_informative_pools"])
            natural_total += int(report["natural_pools"])

    ids = [pool.prompt_id for pool in pools]
    if len(set(ids)) != len(ids):
        raise SystemExit("Shards overlap: duplicate prompt_id across shards")

    min_pools = args.min_informative_pools
    if min_pools is None:
        min_pools = round(args.min_balanced_rate * len(pools))
    merged = bank_supply_report(
        pools,
        families=["existence", "color", "spatial"],
        min_balanced_rate=args.min_balanced_rate,
        min_informative_pools=min_pools,
        natural_informative_rate=(natural_hits / natural_total) if natural_total else None,
    )
    merged["shards"] = shards
    merged["rate_measurement_only"] = True
    merged["absolute_supply_requirement_deferred"] = 128
    Path(args.output).write_text(json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8")

    print(f"prompts={merged['prompts_searched']}  passed={merged['passed']}")
    print(f"  balanced_rate    = {merged['balanced_rate']:.3f}  (threshold {args.min_balanced_rate})")
    print(f"  informative_rate = {merged['informative_rate']:.3f}")
    print(f"  natural_rate     = {merged['natural_informative_rate']:.3f}")
    print(f"  bank_lift        = {merged['bank_lift']:+.3f}")
    for family, stats in sorted(merged["by_family"].items()):
        flag = "PASS" if stats["rate"] >= args.min_balanced_rate else "FAIL"
        print(f"  {family:<10} {stats['balanced']:>2}/{stats['searched']:<3} rate={stats['rate']:.3f}  {flag}")
    print("  rejection reasons:", json.dumps(merged["rejection_reasons"], sort_keys=True))
    return 0 if merged["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
