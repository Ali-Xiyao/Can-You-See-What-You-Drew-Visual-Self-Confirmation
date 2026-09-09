"""Assemble the high-N candidate bank from images that already exist and are already adjudicated.

No generation, no detection. Every candidate here was drawn by the same frozen
generator for one of the 228 formal specs and already carries a gold verdict.
Pooling across the main/conf/conf2 batches is legitimate because the batches are
the same spec redrawn with different seeds: prompt text is identical per spec
(verified), and every (spec_id, seed) pair is distinct (verified).

The per-spec candidate ORDER is a fixed permutation derived from
sha256(spec_id + image_path). It cannot see the cycle score, the gold label or
the batch, so the fixed-K cohorts below are nested prefixes of one declared
ordering rather than a choice made after looking at anything.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = ROOT / "runs/v4"
BATCHES = [
    "main-2plus1", "main-1plus1plus1",
    "conf-2plus1", "conf-1plus1plus1",
    "conf2-2plus1", "conf2-1plus1plus1",
]


def order_key(spec_id, image_path):
    return hashlib.sha256(f"{spec_id}\x00{image_path}".encode("utf-8")).hexdigest()


def main():
    prompts, specs = {}, {}
    manifest = {}
    for batch in BATCHES:
        with open(RUNS / batch / "manifest.jsonl", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                manifest[row["image_path"]] = row
                seen = prompts.setdefault(row["spec_id"], row["prompt"])
                if seen != row["prompt"]:
                    raise SystemExit(f"prompt disagreement for {row['spec_id']}")
                specs.setdefault(row["spec_id"], row["spec"])

    by_spec = defaultdict(list)
    counts = Counter()
    for batch in BATCHES:
        with open(RUNS / batch / "verified.jsonl", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                counts["verified"] += 1
                # Frozen convention: pending_human is UNKNOWN regardless of the
                # stored image_correct, so it cannot enter a pool as a candidate.
                if row["resolution"] == "pending_human":
                    counts["excluded_pending_human"] += 1
                    continue
                by_spec[row["spec_id"]].append({
                    "spec_id": row["spec_id"],
                    "image_path": row["image_path"],
                    "prompt": prompts[row["spec_id"]],
                    "batch": batch,
                    "seed": row["seed"],
                    "candidate_index": row["candidate_index"],
                    "resolution": row["resolution"],
                    "correct": bool(row["image_correct"]),
                })
                counts["usable"] += 1

    for spec_id, items in by_spec.items():
        items.sort(key=lambda c: order_key(spec_id, c["image_path"]))
        for rank, item in enumerate(items):
            item["order_index"] = rank

    sizes = Counter(len(v) for v in by_spec.values())
    cohorts = {
        k: sorted(s for s, v in by_spec.items() if len(v) >= k) for k in (14, 20, 24)
    }

    with open(HERE / "workset.jsonl", "w", encoding="utf-8", newline="\n") as handle:
        for spec_id in sorted(by_spec):
            for item in by_spec[spec_id]:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "batches": BATCHES,
        "verified_rows": counts["verified"],
        "excluded_pending_human": counts["excluded_pending_human"],
        "usable_candidates": counts["usable"],
        "specs": len(by_spec),
        "candidates_per_spec": dict(sorted(sizes.items())),
        "base_rate": sum(c["correct"] for v in by_spec.values() for c in v) / counts["usable"],
        "cohorts": {
            f"k{k}": {
                "specs": len(v),
                "max_n": k,
                "kl_at_max_n": __import__("math").log(k) - (k - 1) / k,
            }
            for k, v in cohorts.items()
        },
        "ordering": "sha256(spec_id + image_path); blind to score, gold and batch",
    }
    (HERE / "workset-summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
