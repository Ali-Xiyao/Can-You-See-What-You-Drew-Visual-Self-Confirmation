"""Does the count answer follow the picture, or the request?

`v4_accuracy_sensitivity.py` prices the remaining gap at about 1.2 points of
counting accuracy, by replacing answers with the truth at some rate and watching
the criteria move. That is only a fair model of a better observer if the errors
it removes are noise. This checks whether they are.

Every counting question is split by something the observer does not control:
whether the picture happens to agree with what the prompt asked for. Both halves
use the same model, the same questions and the same scoring, so the contrast
inside one arm is not confounded by anything -- unlike the gap *between* the
arms, where the model and the condition change together and which therefore is
not a leakage measurement.

  picture == request   the requested answer and the true answer coincide, so the
        cell says what the arm can do when nothing is at stake. Both arms should
        be high; if one is not, it cannot count and nothing else here means much.

  picture != request   the two answers come apart, and the arm has to choose. An
        observer that is merely noisy scatters: its errors land on both sides of
        the truth and only sometimes on the requested number. An observer reading
        the prompt concentrates on the requested number.

The second cell is the one that decides whether observer-side work -- a stronger
model, repeated sampling, higher resolution -- can buy the missing resolution.
Errors that are a policy do not come out with a better observer; they come out by
removing what produces them, and for the prompted arm that is the prompt itself.

    envs/core/python.exe scripts/v4_recitation_split.py runs/v4/gate-b-openct2
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v4_observation_ceiling import detections  # noqa: E402
from v4_selector_resolution import read_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("outdir")
    args = parser.parse_args()
    out_dir = Path(args.outdir)

    found = detections(out_dir)
    image_of: dict[tuple[str, str], str] = {}
    asks: dict[tuple[str, str], tuple[str, str]] = {}
    for pool in read_jsonl(out_dir / "pools.jsonl"):
        for candidate in pool["candidates"]:
            image_of[(pool["prompt_id"], candidate["candidate_id"])] = candidate["image_path"]
        for question in pool["questions"]:
            if ":count:" in question["question_id"]:
                asks[(pool["prompt_id"], question["question_id"])] = (
                    question["atom_id"].split(":")[-1], question["expected_answer"])

    print(f"{args.outdir}: 计数题的回答跟着图走，还是跟着请求走\n")
    print(f"  {'臂':6s} {'图 == 请求':>26s}   {'图 != 请求':>26s}")
    print(f"  {'':6s} {'n':>6s} {'说请求值':>10s} {'数对图':>8s}   "
          f"{'n':>6s} {'说请求值':>10s} {'数对图':>8s}")
    for arm in ("naive", "rfo"):
        cells = {True: [0, 0, 0], False: [0, 0, 0]}
        deltas: collections.Counter = collections.Counter()
        for row in read_jsonl(out_dir / f"observations.{arm}.jsonl"):
            counts = found.get(image_of[(row["prompt_id"], row["candidate_id"])])
            if counts is None:
                continue
            for answer in row["observation"]["answers"]:
                key = (row["prompt_id"], answer["question_id"])
                if key not in asks:
                    continue
                slug, wanted = asks[key]
                truth = str(sum(v for (colour, noun), v in counts.items()
                                if f"{colour}-{noun}" == slug or noun == slug))
                said = answer["normalized_answer"]
                cell = cells[truth == wanted]
                cell[0] += 1
                cell[1] += said == wanted
                cell[2] += said == truth
                if truth != wanted and said != truth:
                    try:
                        deltas[int(said) - int(truth)] += 1
                    except ValueError:
                        deltas[None] += 1
        agree, apart = cells[True], cells[False]
        print(f"  {arm:6s} {agree[0]:6d} {agree[1] / agree[0]:10.1%} {agree[2] / agree[0]:8.1%}   "
              f"{apart[0]:6d} {apart[1] / apart[0]:10.1%} {apart[2] / apart[0]:8.1%}")
        print(f"         图 != 请求 时答错了多少（说的减真实）: "
              f"{dict(sorted(deltas.items(), key=lambda kv: (kv[0] is None, kv[0])))}")

    print("""
  只有「图 != 请求」那半边有判别力，因为另一半两个答案重合，答对不说明看了。
  同一臂内两半边的对比是干净的：同一个模型、同一套题，唯一的差别是图碰巧是否
  符合请求。跨臂比较不干净——naive 是 Show-o2-1.5B（看得到提示词）、rfo 是
  Qwen2-VL-2B（盲），模型和条件一起变了，所以两臂之差不是泄漏量的度量。

  怎么读「说请求值」这一栏：噪声会散开（错误落在真值两侧，只是偶尔撞上请求值），
  读提示词会聚在请求值上。这决定了观察者侧的改动（更强的模型、重复采样、
  更高分辨率）能不能买到缺的那点分辨力。""")


if __name__ == "__main__":
    main()
