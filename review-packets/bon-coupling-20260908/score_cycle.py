"""Continuous cycle score log p(prompt|image) over the frozen 232-pool bank.

Read-only with respect to `runs/`: images and gold are inputs, every output goes
to this packet. Base weights only -- nothing here updates a parameter. Resume is
keyed on (prompt_id, candidate_id), which is safe because the score is a pure
function of the frozen image bytes and the frozen prompt.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

BANK = ROOT / "runs/v4/gate-b-openct2/pools.jsonl"
OUT = HERE / "cycle-scores.jsonl"
META = HERE / "cycle-scores-meta.json"


def load_bank():
    with open(BANK, encoding="utf-8") as handle:
        pools = [json.loads(line) for line in handle if line.strip()]
    work = []
    for pool in pools:
        for candidate in pool["candidates"]:
            work.append(
                {
                    "prompt_id": pool["prompt_id"],
                    "candidate_id": candidate["candidate_id"],
                    "prompt": pool["prompt"],
                    "image_path": candidate["image_path"],
                }
            )
    return pools, work


def done_keys():
    if not OUT.exists():
        return {}
    out = {}
    with open(OUT, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                out[(row["prompt_id"], row["candidate_id"])] = row["cycle_score"]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--recheck", type=int, default=32,
                        help="re-score this many already-scored images and require bit equality")
    parser.add_argument("--limit", type=int, default=0, help="smoke test: score at most this many")
    args = parser.parse_args()

    pools, work = load_bank()
    done = done_keys()
    todo = [w for w in work if (w["prompt_id"], w["candidate_id"]) not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"bank: {len(pools)} pools / {len(work)} images; already scored {len(done)}; "
          f"to score {len(todo)}", flush=True)

    from selfsight.backbones.showo2 import Showo2Adapter

    started = time.time()
    backbone = Showo2Adapter(device=args.device, lazy=False)
    print(f"backbone: {backbone.model_id} @ {backbone.revision} on {args.device} "
          f"(load {time.time() - started:.1f}s)", flush=True)

    t0 = time.time()
    with open(OUT, "a", encoding="utf-8") as handle:
        for index, item in enumerate(todo, 1):
            score = backbone.cycle_consistency_score(item["image_path"], item["prompt"])
            handle.write(json.dumps({
                "prompt_id": item["prompt_id"],
                "candidate_id": item["candidate_id"],
                "cycle_score": float(score),
                "image_path": item["image_path"],
            }, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 50 == 0 or index == len(todo):
                rate = (time.time() - t0) / index
                print(f"  {index}/{len(todo)}  {rate:.2f}s/img  "
                      f"eta {(len(todo) - index) * rate / 60:.1f}min", flush=True)

    # Determinism invariant: the score must be a pure function of the frozen bytes.
    scored = done_keys()
    recheck = [w for w in work if (w["prompt_id"], w["candidate_id"]) in scored][: args.recheck]
    mismatches = []
    for item in recheck:
        again = float(backbone.cycle_consistency_score(item["image_path"], item["prompt"]))
        before = scored[(item["prompt_id"], item["candidate_id"])]
        if again != before:
            mismatches.append({
                "prompt_id": item["prompt_id"],
                "candidate_id": item["candidate_id"],
                "first": before,
                "second": again,
            })

    meta = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_id": backbone.model_id,
        "revision": backbone.revision,
        "device": args.device,
        "n_pools": len(pools),
        "n_images": len(work),
        "n_scored": len(scored),
        "seconds_total": round(time.time() - started, 1),
        "determinism_recheck_n": len(recheck),
        "determinism_mismatches": mismatches,
        "determinism_ok": not mismatches,
    }
    META.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    if mismatches:
        raise SystemExit("determinism check failed; protocol says this round is void")
    if not args.limit and len(scored) != len(work):
        raise SystemExit(f"incomplete: {len(scored)}/{len(work)}")


if __name__ == "__main__":
    main()
