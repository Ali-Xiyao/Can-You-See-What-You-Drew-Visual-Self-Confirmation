"""Compare observed and truthful scores on the same adjudicable pools.

This conditional diagnostic is neither an independent experiment nor a proof
that only observer accuracy matters. Unknown facts exclude the whole pool from
both actual and ideal columns; original denominators remain visible.
"""
from __future__ import annotations

import argparse
import collections
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from selfsight.v4.factual_truth import canonical_answer, factual_answer, load_verifications
from v4_resolution_verdict import REGISTERED, candidates, load_pools
from v4_selector_resolution import Candidate, resolution


def paired_diagnostic(pools: dict[str, dict[str, Any]],
                      found: dict[str, dict[str, Any]]) -> dict[str, Any]:
    arms: dict[str, list[list[Candidate]]] = {"naive": [], "rfo": [], "ideal": []}
    eligible: list[str] = []
    unknown: collections.Counter[str] = collections.Counter()
    total_questions = known_questions = total_candidates = unknown_candidates = 0
    unavailable_actual = 0
    for prompt_id, pool in pools.items():
        ideal: list[Candidate] = []
        usable = bool(pool["questions"])
        for candidate in pool["candidates"]:
            total_candidates += 1
            facts = [factual_answer(q, found.get(candidate["image_path"]))
                     for q in pool["questions"]]
            total_questions += len(facts)
            known_questions += sum(f.known for f in facts)
            unknown.update(f.reason for f in facts if not f.known)
            if not all(f.known for f in facts):
                unknown_candidates += 1
                usable = False
                continue
            # Legacy closed questions sometimes cannot express a truthful reply.
            representable = all(
                not q.get("choices") or f.answer in {
                    canonical_answer(q, choice) for choice in q["choices"]}
                for q, f in zip(pool["questions"], facts))
            if not representable:
                unknown["truth_not_in_choices"] += 1
                usable = False
                continue
            hits = sum(f.answer == canonical_answer(q, q["expected_answer"])
                       for q, f in zip(pool["questions"], facts))
            ideal.append(Candidate(candidate["candidate_id"], hits / len(facts),
                                   bool(candidate["correct"])))
        if not usable:
            continue
        if pool["selection"].get("dropped"):
            unavailable_actual += 1
            continue
        actual = {arm: candidates(pool, arm) for arm in ("naive", "rfo")}
        if any(not math.isfinite(c.score) for cs in actual.values() for c in cs):
            unavailable_actual += 1
            continue
        eligible.append(prompt_id)
        arms["ideal"].append(ideal)
        for arm in actual:
            arms[arm].append(actual[arm])
    return {"eligible_ids": eligible, "total_pools": len(pools),
            "total_candidates": total_candidates, "unknown_candidates": unknown_candidates,
            "total_questions": total_questions, "known_questions": known_questions,
            "unknown_reasons": dict(unknown), "unavailable_actual_pools": unavailable_actual,
            "metrics": {arm: resolution(rows) if rows else None for arm, rows in arms.items()}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("outdir")
    args = parser.parse_args()
    out_dir = Path(args.outdir)
    report = paired_diagnostic(load_pools(out_dir), load_verifications(out_dir))
    n = len(report["eligible_ids"])
    print(f"{out_dir}: 事实答案诊断；三列使用同一组 {n}/{report['total_pools']} 池")
    print(f"可判事实 {report['known_questions']}/{report['total_questions']}；"
          f"含未知事实候选 {report['unknown_candidates']}/{report['total_candidates']}；"
          f"实际分数不可用池 {report['unavailable_actual_pools']}")
    print(f"未知/不可表达原因：{report['unknown_reasons']}")
    if not n:
        print("没有可共同评估的完整池；不作理想分数推断。")
        return
    arms = report["metrics"]
    print(f"  {'判据':22s} {'参照门槛':>9s} {'naive':>8s} {'rfo':>8s} {'事实答案':>10s}")
    for key, name, direction, threshold, _b, _a, _t in REGISTERED:
        print(f"  {name:22s} {direction}{threshold:8.0%} {arms['naive'][key]:8.1%}"
              f" {arms['rfo'][key]:8.1%} {arms['ideal'][key]:10.1%}")
    print("这是可判定子集上的诊断，不改变原预注册实测判定。事实与整图裁定同源；")
    print("差距可能涉及观察、问题范围、计分及标签，不自动确定下一种方法，也不外推训练表现。")


if __name__ == "__main__":
    main()
