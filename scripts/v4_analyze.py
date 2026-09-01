"""Turn v4 answers into the numbers the paper reports.

Three things are measured here and they are not the same thing:

    accuracy         can the model read its own picture at all
    self-confirmation  when it drew the wrong thing, does it answer with the
                     prompt instead of the pixels
    selection ceiling  is there anything for a selection experiment to select

The second is the claim. It is only measurable on the trials where the drawn
image and the requested scene disagree, which is why those trials are tagged at
construction time and counted separately here. On the trials where the image
matches the spec, answering from the prompt and answering from the pixels give
the same reply, and the two hypotheses are indistinguishable -- reporting overall
accuracy alone would let a pure prompt-reciter look like a competent observer.

Abstentions are reported, never dropped. An unparseable reply is a fact about the
model; folding it into "wrong" inflates the error rate and folding it into the
denominator-free accuracy hides it.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
from typing import Any, Iterable


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson interval. Normal approximation breaks at the rates seen here.

    Several cells sit at or near 0 or 1, where the Wald interval runs outside
    [0, 1] and reports a width of zero at exactly 0 successes -- which would read
    as certainty from no evidence.
    """
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    answered = [r for r in rows if r["correct"] is not None]
    correct = sum(1 for r in answered if r["correct"])
    low, high = wilson(correct, len(answered))
    return {
        "n_trials": len(rows),
        "n_answered": len(answered),
        "abstain_rate": round(1 - len(answered) / len(rows), 4) if rows else 0.0,
        "accuracy": round(correct / len(answered), 4) if answered else None,
        "ci95": [round(low, 4), round(high, 4)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in (args.run / "answers.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit("no answers to analyse")

    report: dict[str, Any] = {"overall": rate(rows)}

    report["by_family"] = {
        family: rate(group)
        for family, group in _group(rows, lambda r: r["family"]).items()
    }

    # The trials that carry the claim, against the ones that cannot.
    differs = [r for r in rows if r["gold_source"] == "image_differs_from_spec"]
    matches = [r for r in rows if r["gold_source"] != "image_differs_from_spec"]
    report["gold_source"] = {
        "image_differs_from_spec": rate(differs),
        "not_diagnostic": rate(matches),
        "by_tag": {
            tag: rate(group)
            for tag, group in _group(rows, lambda r: r["gold_source"]).items()
        },
        "note": (
            "Only image_differs_from_spec is diagnostic: there the correct answer "
            "contradicts the prompt, so a prompt-reciter is wrong and a "
            "picture-reader is right. spec_matches_image means the two give the "
            "same answer. The bare `image` tag is the middle case -- the spec was "
            "not met, but both options appear in the prompt, so a prompt-reciter "
            "is at chance rather than wrong; it is reported but not counted as "
            "diagnostic."
        ),
    }
    if differs:
        report["gold_source"]["by_family"] = {
            family: rate(group)
            for family, group in _group(differs, lambda r: r["family"]).items()
        }

    # Self-confirmation: the model drew something the spec did not ask for, was
    # asked about the picture, and answered with the spec.
    answered_differs = [r for r in differs if r["correct"] is not None]
    report["self_confirmation"] = {
        "n": len(answered_differs),
        "rate": (
            round(sum(1 for r in answered_differs if not r["correct"])
                  / len(answered_differs), 4)
            if answered_differs else None
        ),
        "definition": (
            "share of image_differs_from_spec trials answered with the requested "
            "scene rather than the drawn one; chance is 0.5 under 2AFC"
        ),
    }

    # Does reading the picture correctly track having drawn it correctly? If
    # accuracy is the same either way, the model is not reading the image.
    report["by_image_correct"] = {
        str(key): rate(group)
        for key, group in _group(rows, lambda r: bool(r["image_correct"])).items()
    }

    # Rows the verifier could not settle are counted, not dropped. They stay in
    # every table above, because the images two detectors disagree about are the
    # occluded and ambiguous ones and removing them resamples the corpus toward
    # easy scenes. This block is the caveat the paper reports alongside them.
    pending = [r for r in rows if r["resolution"] == "pending_human"]
    report["pending_human"] = {
        "n_trials": len(pending),
        "share": round(len(pending) / len(rows), 4),
        "n_images": len({r["image_path"] for r in pending}),
        "included_in_tables_above": True,
        "note": "awaiting human adjudication; their gold may change",
    }
    if pending:
        settled = [r for r in rows if r["resolution"] != "pending_human"]
        report["pending_human"]["overall_excluding_pending"] = rate(settled)

    # Position bias: a model answering "A" regardless would score at chance
    # without looking, and the gold letter is balanced by construction.
    letters = collections.Counter(
        (r["gold"] if r["correct"] else ("B" if r["gold"] == "A" else "A"))
        for r in rows if r["correct"] is not None
    )
    total = sum(letters.values()) or 1
    report["answer_letter_share"] = {
        "A": round(letters["A"] / total, 4),
        "B": round(letters["B"] / total, 4),
        "gold_A_share": round(
            sum(1 for r in rows if r["gold"] == "A") / len(rows), 4
        ),
    }

    text = json.dumps(report, indent=2, ensure_ascii=False)
    (args.out or args.run / "analysis.json").write_text(text, encoding="utf-8")
    print(text)


def _group(rows, key):
    out: dict[Any, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        out[key(row)].append(row)
    return dict(out)


if __name__ == "__main__":
    main()
