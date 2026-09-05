"""What the questions would be worth to an observer that never miscounts.

The residual diagnosis says the pools that stay tied are almost all "in
vocabulary": the images differ, the question names the thing they differ on, and
the answers still do not separate them. That points at the observer rather than
at the wording, but pointing is not measuring. This measures it.

Every observer answer is replaced by the truth about the image, taken from the
`detections` in `verified.jsonl`, and the four registered criteria are recomputed
with the same questions, the same spec golds and the same scoring rule. A
perfect observer says "yes" when the phrase is in the picture and reports the
number it actually counts; it is still graded against what the *prompt* asked
for, so the score stays a cycle-consistency score.

Reading the result:

  criteria pass under a perfect observer   the questions can do the job and the
      shortfall is observation accuracy. Effort belongs on the observer -- a
      stronger one, repeated sampling, agreement between two -- and not on
      rewriting questions.

  criteria still fail under a perfect observer   the questions themselves cannot
      express the verdict, and no amount of observation accuracy will fix it.

This is a diagnostic. It is not a run, it cannot pass the prereg, and the number
it produces must never be quoted as a result of the experiment: an observer that
reads the detector's answer sheet is not measuring self-confirmation. The
detections also come from the same detectors the verdict was built on, so this
is an upper bound on what a spec-side scorer could ever recover from that
verdict, not an estimate of what a real observer would get.

    envs/core/python.exe scripts/v4_observation_ceiling.py runs/v4/gate-b-openct2
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
from v4_resolution_verdict import REGISTERED, load_pools, measure  # noqa: E402
from v4_selector_resolution import Candidate, read_jsonl, resolution  # noqa: E402


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


def truth_for(question: dict[str, Any], counts: collections.Counter) -> str:
    """What an observer with perfect sight says about this picture.

    The slug in `atom_id` is the phrase the question names: `red-book` for the
    colour-qualified form, bare `book` for the category form. Colours hold no
    hyphen and `canonical_noun` returns one word, so the two cases separate.
    """
    slug = question["atom_id"].split(":")[-1]
    seen = sum(value for (colour, noun), value in counts.items()
               if f"{colour}-{noun}" == slug or noun == slug)
    if ":count:" in question["question_id"]:
        return str(seen)
    return "yes" if seen else "no"


def perfect_pools(pools: dict[str, dict[str, Any]],
                  found: dict[str, collections.Counter]) -> list[list[Candidate]]:
    out: list[list[Candidate]] = []
    for prompt_id, pool in pools.items():
        scored: list[Candidate] = []
        usable = True
        for candidate in pool["candidates"]:
            counts = found.get(candidate["image_path"])
            if counts is None:
                usable = False
                break
            hits = sum(truth_for(q, counts) == q["expected_answer"]
                       for q in pool["questions"])
            scored.append(Candidate(candidate["candidate_id"],
                                    hits / len(pool["questions"]),
                                    bool(candidate["correct"])))
        if usable:
            out.append(scored)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("outdir")
    args = parser.parse_args()
    out_dir = Path(args.outdir)

    pools = load_pools(out_dir)
    found = detections(out_dir)
    ceiling = resolution(perfect_pools(pools, found))
    arms = {arm: measure(pools, arm) for arm in ("naive", "rfo")}

    print(f"{args.outdir}: 若观察 100% 准确，同一套题能做到什么\n")
    print(f"  {'判据':22s} {'门槛':>9s} {'naive':>8s} {'rfo':>8s} {'完美观察':>10s} {'':4s}")
    verdict = True
    for key, name, direction, threshold, _b, _a, _t in REGISTERED:
        value = ceiling[key]
        ok = value >= threshold if direction == ">=" else value <= threshold
        verdict &= ok
        print(f"  {name:22s} {direction}{threshold:8.0%} {arms['naive'][key]:8.1%}"
              f" {arms['rfo'][key]:8.1%} {value:10.1%}  {'PASS' if ok else 'FAIL'}")
    print(f"\n  完美观察下 {ceiling['n_pools']} 池，四条判据："
          f"{'全部通过' if verdict else '仍未全部通过'}")
    print(f"  同分池 {ceiling['all_tied']:.1%}   满分候选 {ceiling['ceiling']:.1%}"
          f"   均分差 {ceiling['mean_gap']:+.4f}")

    if verdict:
        print("""
  结论：题目能表达这个裁定，短板是观察准确率。
  下一步的功夫应当花在观察者上（更强的观察者、重复采样、双观察者一致性），
  不是改题。改题在这里买不到东西——题已经问对了地方。""")
    else:
        print("""
  结论：即使观察 100% 准确，这套题仍达不到门槛。
  短板在题目能表达什么，不只在观察准确率。改观察者买不到全部差距。""")
    print("""
  这是诊断，不是实验结果。读检测器答案的观察者不在测自我确认，
  而且 detections 与裁定同源，所以这是「spec 侧评分器最多能从该裁定里
  恢复多少」的上界，不是真实观察者的预期值。""")


if __name__ == "__main__":
    main()
