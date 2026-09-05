"""Where the resolution that is still missing actually went.

After the scoring defect in STATUS 42 is fixed, some pools will still be tied.
The useful question is not "which question should we change next" but "is there
anything left for a spec-side question to see". Those are different, and only
one of them is answerable by writing better questions.

For every pool that is still unresolved, this splits the candidates by the
adjudicated verdict and compares what the detectors actually found in the
images, as a (colour, canonical noun) multiset -- the same rule the gold verdict
uses and the same vocabulary any spec-side question can address:

  in vocabulary   the images differ on a phrase the questions already name.
              The question was asked, about the right thing, and the answers
              still did not separate the pool. Nothing about the wording is at
              fault; this is the observer or the scoring.

  out of vocabulary   the images differ, but only in objects the spec never
              mentions, so no spec-determined question can name them. That is
              coverage gap 3 in STATUS 40, left open on purpose, and closing it
              means letting the image choose the question.

  invisible   they do not differ in that multiset at all. The verdict split them
              on something else -- the adjudication ladder resolves by crop and
              by human review, not by multiset alone -- so no question phrased
              over objects, colours and counts can reach it, however it is
              worded. This is what the image-gold half of the selector is for.

The three call for different work, which is the point of separating them: only
the second and third are arguments for changing what gets asked.

Reported for whichever run directory is given, so the corrected run and the
defective one can be compared on the same footing.

    envs/core/python.exe scripts/v4_residual_diagnosis.py runs/v4/gate-b-openct2
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from selfsight.v4.spec import canonical_noun  # noqa: E402
from v4_resolution_verdict import candidates, load_pools  # noqa: E402
from v4_selector_resolution import read_jsonl  # noqa: E402


def detections(out_dir: Path) -> dict[str, collections.Counter]:
    runs = json.loads((out_dir / "runs.json").read_text(encoding="utf-8"))["runs"]
    found: dict[str, collections.Counter] = {}
    for name in runs:
        for row in read_jsonl(Path(name) / "verified.jsonl"):
            if row.get("detections") is None:
                continue
            found[row["image_path"]] = collections.Counter(
                (d.get("color"), canonical_noun(d["object"])) for d in row["detections"])
    return found


def unresolved(pool: dict[str, Any], arm: str) -> bool:
    """No separation to speak of: correct and incorrect average to the same score."""
    scored = candidates(pool, arm)
    ok = [c.score for c in scored if c.correct]
    bad = [c.score for c in scored if not c.correct]
    if not ok or not bad:
        return False
    return sum(ok) / len(ok) == sum(bad) / len(bad)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("outdir")
    args = parser.parse_args()
    out_dir = Path(args.outdir)

    pools = load_pools(out_dir)
    found = detections(out_dir)
    image_of = {(p["prompt_id"], c["candidate_id"]): c["image_path"]
                for p in pools.values() for c in p["candidates"]}

    print(f"{args.outdir}: 仍不可分的池，按「还有没有东西可看」拆开\n")
    print(f"  {'臂':6s} {'仍平分':>8s} {'题内可达':>9s} {'题外':>7s}"
          f" {'不可见':>8s} {'缺检出':>8s}")
    for arm in ("naive", "rfo"):
        still = reachable = out_of_vocab = invisible = missing = 0
        examples: list[str] = []
        for prompt_id, pool in pools.items():
            if not unresolved(pool, arm):
                continue
            still += 1
            groups: dict[bool, list[collections.Counter]] = {True: [], False: []}
            incomplete = False
            for candidate in pool["candidates"]:
                counts = found.get(image_of[(prompt_id, candidate["candidate_id"])])
                if counts is None:
                    incomplete = True
                    break
                groups[bool(candidate["correct"])].append(counts)
            if incomplete:
                missing += 1
                continue
            # Two levels, because they call for different work. A spec-side
            # question can only name phrases the spec mentions, so a difference
            # that lives entirely in objects nobody asked for is out of reach of
            # *this* question set however it is worded -- that is coverage gap 3
            # in STATUS 40, and it was left open on purpose.
            asked = {q["atom_id"].split(":")[-1] for q in pool["questions"]
                     if ":count:" in q["question_id"]}

            def project(counts: collections.Counter) -> tuple:
                kept: dict[str, int] = {}
                for (colour, noun), value in counts.items():
                    for slug in asked:
                        if f"{colour}-{noun}" == slug or noun == slug:
                            kept[slug] = kept.get(slug, 0) + value
                return tuple(sorted(kept.items()))

            differs = any(ok != bad for ok in groups[True] for bad in groups[False])
            in_vocab = any(project(ok) != project(bad)
                           for ok in groups[True] for bad in groups[False])
            reachable += in_vocab
            out_of_vocab += differs and not in_vocab
            invisible += not differs
            if in_vocab and len(examples) < 3:
                examples.append(prompt_id)
        print(f"  {arm:6s} {still:8d} {reachable:8d} {out_of_vocab:8d}"
              f" {invisible:8d} {missing:8d}")
        if examples:
            print(f"         题内可达的例子: {', '.join(examples)}")

    print("""
  题内可达 = 差异落在题目实际问到的短语上：题问了，却没分开。
         这是观察或计分的问题，不是词表的问题。
  题外 = 两组图确实不同，但差异全在 spec 从未提及的物体上。现有题面命名不到它
         （STATUS 40 的覆盖缺口 3，当初有意留下）。要够到它，题必须由图而不是
         由 spec 决定——那已经是 image-gold 一侧。
  不可见 = 两组图在 (颜色, 规范名) 上完全一致，裁定是靠裁剪复核或人工判的。
         任何用物体/颜色/数量措辞的题都够不到，无论怎么改。这一部分只能由
         image-gold 一侧回答（预注册 §1 第 2 部分），不是把现有题写得更狠。""")


if __name__ == "__main__":
    main()
