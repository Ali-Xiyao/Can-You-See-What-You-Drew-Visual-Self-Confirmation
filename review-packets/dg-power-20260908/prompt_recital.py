"""Is the model reading its own image, or reciting the prompt back?

`s_select` asks the model yes/no questions about the image it just generated,
and scores the answer against `expected_answer`, which is derived from the
SPEC -- what should be in the picture, not what is. A model that ignores the
image and answers straight from the prompt scores 1.0 on every item.

So the aggregate 94.7% cannot distinguish perception from recitation. What can:
condition on whether the image was externally correct. If the model is looking,
`s_select` must fall sharply on the images that are wrong. If it is reciting,
`s_select` stays flat, because the prompt did not change.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RUN = Path(__file__).resolve().parents[2] / "runs" / "v4" / "decoupling-pilot-20260906"


def load(arm: str, step: int) -> tuple[dict[str, float], dict[str, bool], dict[str, dict]]:
    directory = RUN / "evaluations" / arm / f"step-{step:05d}"
    select, report = {}, {}
    with open(directory / "s_select.jsonl", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            select[row["prompt_id"]] = float(row["s_select"])
    correct = {}
    with open(directory / "verified.jsonl", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            correct[row["spec_id"]] = bool(row["image_correct"])
            report[row["spec_id"]] = row["report"]
    return select, correct, report


def summarise(values: np.ndarray) -> str:
    if len(values) == 0:
        return "     n/a      "
    sem = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
    return f"{values.mean():.3f} +- {sem:.3f}"


def main() -> None:
    print("s_select conditioned on whether the image was actually right\n")
    print(f"{'arm':9s}{'step':>5}{'right n':>9}{'s_select | right':>20}"
          f"{'wrong n':>9}{'s_select | wrong':>20}{'gap':>9}")
    gaps = []
    for arm in ("naive", "rfo_gold"):
        for step in (0, 8, 16, 24, 32):
            select, correct, _ = load(arm, step)
            shared = sorted(set(select) & set(correct))
            right = np.array([select[k] for k in shared if correct[k]])
            wrong = np.array([select[k] for k in shared if not correct[k]])
            gap = right.mean() - wrong.mean()
            gaps.append(gap)
            print(f"{arm:9s}{step:>5}{len(right):>9}{summarise(right):>20}"
                  f"{len(wrong):>9}{summarise(wrong):>20}{gap:>+9.3f}")

    print(f"\nmedian gap across all ten checkpoints: {np.median(gaps):+.3f}")
    print("a model that reads the image would show a large positive gap;")
    print("a model reciting the prompt shows ~0.\n")

    # Where does the residual discrimination come from? Split by failure mode.
    select, correct, report = load("naive", 32)
    shared = sorted(set(select) & set(correct))
    buckets: dict[str, list[float]] = {}
    for key in shared:
        entry = report[key]
        if entry["objects_and_color_ok"]:
            name = "fully correct"
        elif entry["missing"]:
            name = "object missing entirely"
        else:
            name = "object present, attribute wrong"
        buckets.setdefault(name, []).append(select[key])
    print("naive/step-32, s_select by failure mode:")
    for name, values in sorted(buckets.items()):
        array = np.array(values)
        print(f"  {name:34s} n={len(array):>3}   {summarise(array)}")
    print("\nan object that is not in the picture at all is the easiest possible")
    print("thing to notice; s_select staying high there is the strongest evidence")
    print("that the answer is coming from the prompt and not from the pixels.")


if __name__ == "__main__":
    main()
