"""Cycle score log p(prompt|image) over the high-N workset. Base weights only.

Same scorer, same frozen revision, same determinism invariant as
review-packets/bon-coupling-20260908. Reads workset.jsonl, writes here, never
touches runs/.
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

WORKSET = HERE / "workset.jsonl"
OUT = HERE / "cycle-scores.jsonl"
META = HERE / "cycle-scores-meta.json"


def jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def done_scores():
    if not OUT.exists():
        return {}
    return {r["image_path"]: r["cycle_score"] for r in jsonl(OUT)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--recheck", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    work = jsonl(WORKSET)
    done = done_scores()
    todo = [w for w in work if w["image_path"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"workset {len(work)} images; scored {len(done)}; to score {len(todo)}", flush=True)

    from selfsight.backbones.showo2 import Showo2Adapter

    started = time.time()
    backbone = Showo2Adapter(device=args.device, lazy=False)
    print(f"backbone {backbone.model_id} @ {backbone.revision} on {args.device} "
          f"(load {time.time() - started:.1f}s)", flush=True)

    t0 = time.time()
    with open(OUT, "a", encoding="utf-8") as handle:
        for index, item in enumerate(todo, 1):
            score = backbone.cycle_consistency_score(item["image_path"], item["prompt"])
            handle.write(json.dumps({
                "spec_id": item["spec_id"],
                "image_path": item["image_path"],
                "cycle_score": float(score),
            }, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 200 == 0 or index == len(todo):
                rate = (time.time() - t0) / index
                print(f"  {index}/{len(todo)}  {rate:.2f}s/img  "
                      f"eta {(len(todo) - index) * rate / 60:.1f}min", flush=True)

    scored = done_scores()
    recheck = [w for w in work if w["image_path"] in scored][: args.recheck]
    mismatches = []
    for item in recheck:
        again = float(backbone.cycle_consistency_score(item["image_path"], item["prompt"]))
        if again != scored[item["image_path"]]:
            mismatches.append({"image_path": item["image_path"],
                               "first": scored[item["image_path"]], "second": again})

    meta = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_id": backbone.model_id,
        "revision": backbone.revision,
        "device": args.device,
        "n_workset": len(work),
        "n_scored": len(scored),
        "seconds_total": round(time.time() - started, 1),
        "determinism_recheck_n": len(recheck),
        "determinism_mismatches": mismatches,
        "determinism_ok": not mismatches,
    }
    META.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    if mismatches:
        raise SystemExit("determinism check failed")
    if not args.limit and len(scored) != len(work):
        raise SystemExit(f"incomplete: {len(scored)}/{len(work)}")


if __name__ == "__main__":
    main()
