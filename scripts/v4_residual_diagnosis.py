"""Describe factual-answer separation in observed tied pools, without attribution.

All frozen questions, including colour-qualified legacy existence questions,
enter the answer vector. Unknown facts remain separate. A different vector is
not proof that the aggregate selector will rank correctly.
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from selfsight.v4.factual_truth import factual_answer, load_verifications
from v4_resolution_verdict import candidates, load_pools


def unresolved(pool: dict[str, Any], arm: str) -> bool:
    if pool["selection"].get("dropped"):
        return False
    scored = candidates(pool, arm)
    ok = [c.score for c in scored if c.correct]
    bad = [c.score for c in scored if not c.correct]
    return bool(ok and bad and sum(ok) / len(ok) == sum(bad) / len(bad))


def classify(pool: dict[str, Any], found: dict[str, dict[str, Any]]) -> tuple[str, dict[str, int]]:
    groups: dict[bool, list[tuple[str | None, ...]]] = {True: [], False: []}
    unknown: collections.Counter[str] = collections.Counter()
    for candidate in pool["candidates"]:
        facts = [factual_answer(q, found.get(candidate["image_path"]))
                 for q in pool["questions"]]
        unknown.update(f.reason for f in facts if not f.known)
        groups[bool(candidate["correct"])].append(tuple(f.answer for f in facts))
    if unknown:
        return "unknown", dict(unknown)
    pairs = [ok != bad for ok in groups[True] for bad in groups[False]]
    if not pairs:
        return "not_mixed", {}
    if all(pairs):
        return "all_pairs_distinct", {}
    if any(pairs):
        return "some_pairs_distinct", {}
    return "no_pairs_distinct", {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("outdir")
    args = parser.parse_args()
    out_dir = Path(args.outdir)
    pools, found = load_pools(out_dir), load_verifications(out_dir)
    print(f"{out_dir}: 实测均分相同的池；按完整题组的确定事实答案诊断")
    for arm in ("naive", "rfo"):
        counts: collections.Counter[str] = collections.Counter()
        unknown: collections.Counter[str] = collections.Counter()
        for pool in pools.values():
            if unresolved(pool, arm):
                category, reasons = classify(pool, found)
                counts[category] += 1
                unknown.update(reasons)
        print(f"  {arm}: 平分池 {sum(counts.values())}/{len(pools)}；"
              f"全部正确/错误对可区分 {counts['all_pairs_distinct']}；"
              f"部分可区分 {counts['some_pairs_distinct']}；"
              f"均不可区分 {counts['no_pairs_distinct']}；未知 {counts['unknown']}")
        print(f"    未知事实原因：{dict(unknown)}")
    print("可区分只描述事实答案向量，不能证明当前计分会排对，也不能把差距全部归因于观察者。")
    print("不可区分提示检查题目覆盖与裁定；未知事实需要保留为未知，不能视为物体不存在。")


if __name__ == "__main__":
    main()
