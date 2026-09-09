"""Does having the prompt in context make the model stop reading its own image?

`v4_run_pipeline.py observe` already ran both conditions on the corpus runs:

    image_only  the model sees the picture and the question, nothing else
    prompted    the same question, with the generating prompt prepended

and `v4/questions.py` grades against the DETECTIONS, never against the spec, and
records `gold_source` per trial:

    spec_matches_image       the spec and the pixels agree; both hypotheses
                             ("reading" and "reciting") predict the same answer,
                             so the trial cannot separate them
    image_differs_from_spec  the spec says one thing and the pixels say another.
                             A model reading the picture is right, a model
                             reciting the prompt is wrong. This is the trial that
                             decides it.

Nothing here is new measurement. It is a read of files already on disk.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ("main-2plus1", "main-1plus1plus1")
CONDITIONS = {"image_only": "answers.jsonl", "prompted": "answers.prompted.jsonl"}


def wilson(hits: int, total: int) -> tuple[float, float, float]:
    """Wilson interval: the accuracies here run close to 1.0, where Wald lies."""
    if total == 0:
        return float("nan"), float("nan"), float("nan")
    point = hits / total
    z = 1.96
    denominator = 1 + z**2 / total
    centre = (point + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(point * (1 - point) / total + z**2 / (4 * total**2)) / denominator
    return point, centre - margin, centre + margin


def load() -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {name: [] for name in CONDITIONS}
    for run in RUNS:
        for condition, filename in CONDITIONS.items():
            path = ROOT / "runs" / "v4" / run / filename
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    if row["abstain"]:
                        continue
                    row["run"] = run
                    rows[condition].append(row)
    return rows


def cell(rows: list[dict]) -> str:
    hits = sum(bool(row["correct"]) for row in rows)
    point, low, high = wilson(hits, len(rows))
    return f"{point:.3f} [{low:.3f},{high:.3f}] n={len(rows):<5}"


def by(rows: list[dict], *keys: str) -> dict[tuple, list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    return grouped


def mcnemar(pairs: list[tuple[bool, bool]]) -> tuple[int, int, float]:
    """Exact two-sided McNemar on the discordant pairs."""
    only_first = sum(1 for a, b in pairs if a and not b)
    only_second = sum(1 for a, b in pairs if b and not a)
    n = only_first + only_second
    if n == 0:
        return only_first, only_second, 1.0
    k = min(only_first, only_second)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return only_first, only_second, min(1.0, 2 * tail)


def main() -> None:
    rows = load()
    print("accuracy against what is IN THE PICTURE (Wilson 95%)\n")
    print(f"{'gold_source':26s}{'image_only':32s}{'prompted':32s}{'delta':>8}")
    for source in ("spec_matches_image", "image_differs_from_spec", "image"):
        cells = {}
        for condition in CONDITIONS:
            selected = [row for row in rows[condition] if row["gold_source"] == source]
            cells[condition] = selected
        blind = sum(bool(r["correct"]) for r in cells["image_only"]) / max(1, len(cells["image_only"]))
        told = sum(bool(r["correct"]) for r in cells["prompted"]) / max(1, len(cells["prompted"]))
        print(f"{source:26s}{cell(cells['image_only']):32s}{cell(cells['prompted']):32s}"
              f"{told - blind:>+8.3f}")

    print("\n\nthe decisive subset, split by family "
          "(gold_source = image_differs_from_spec)\n")
    print(f"{'family':14s}{'image_only':32s}{'prompted':32s}{'delta':>8}")
    for family in ("existence", "counting", "binding", "absence", "spatial"):
        cells = {}
        for condition in CONDITIONS:
            cells[condition] = [row for row in rows[condition]
                                if row["gold_source"] == "image_differs_from_spec"
                                and row["family"] == family]
        if not cells["image_only"] and not cells["prompted"]:
            continue
        blind = sum(bool(r["correct"]) for r in cells["image_only"]) / max(1, len(cells["image_only"]))
        told = sum(bool(r["correct"]) for r in cells["prompted"]) / max(1, len(cells["prompted"]))
        print(f"{family:14s}{cell(cells['image_only']):32s}{cell(cells['prompted']):32s}"
              f"{told - blind:>+8.3f}")

    # Paired: the same trial under both conditions.
    print("\n\npaired on (image, question), exact McNemar\n")
    print(f"{'gold_source':26s}{'pairs':>7}{'blind only':>12}{'told only':>11}{'p':>10}")
    for source in ("spec_matches_image", "image_differs_from_spec", "image"):
        index = {}
        for condition in CONDITIONS:
            for row in rows[condition]:
                if row["gold_source"] != source:
                    continue
                index.setdefault((row["image_path"], row["question"]), {})[condition] = row
        pairs = [(bool(v["image_only"]["correct"]), bool(v["prompted"]["correct"]))
                 for v in index.values() if len(v) == 2]
        blind_only, told_only, p_value = mcnemar(pairs)
        print(f"{source:26s}{len(pairs):>7}{blind_only:>12}{told_only:>11}{p_value:>10.2e}")
    print("\n'blind only' = trials the model got right without the prompt and wrong with it.")


if __name__ == "__main__":
    main()
