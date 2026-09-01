"""Pair the two observation conditions trial by trial.

`v4_analyze.py` scores one answers file at a time. The claim is not about either
file's level -- it is about the difference between them on the *same* trial, and
about that difference having opposite signs on the two kinds of trial:

    gold_source == spec_matches_image      the picture shows what was requested
    gold_source == image_differs_from_spec the picture contradicts the request

Only the second kind can separate "reads the pixels" from "recites the prompt".
The first kind is the control that rules out the boring explanation: if adding
the generation description merely lengthened and cluttered the context, both
rows would fall. If deference to the description is what changed, the rows move
in opposite directions.

Paired, because the two conditions run over the identical (image, question)
grid. An unpaired comparison of two accuracies throws away the pairing and, with
per-trial difficulty varying as much as it does here, inflates the variance
several fold. McNemar uses only the discordant pairs, which is the whole of the
evidence about a within-trial change.

Trials are matched on (image_path, family, question). Question text is part of
the key because one image carries several questions from the same family, and
matching without it would pair a counting question about mugs with one about
plates and report the difference as an effect of the condition.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (row["image_path"], row["family"], row["question"])


def mcnemar(b: int, c: int) -> dict[str, Any]:
    """Exact-ish McNemar on the discordant counts.

    b = correct under the first condition and wrong under the second, c = the
    reverse. The continuity-corrected chi-square is reported with its two-sided
    p; with b + c in the hundreds here the correction barely matters, but it is
    kept because several per-family cells have b + c under 20.
    """
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "chi2": None, "p": None}
    chi2 = (abs(b - c) - 1) ** 2 / n
    # Two-sided p from the chi-square with one degree of freedom, which is
    # erfc(sqrt(chi2 / 2)) -- no scipy in the observer env.
    p = math.erfc(math.sqrt(chi2 / 2))
    return {"b": b, "c": c, "chi2": round(chi2, 3), "p": p}


def cell(pairs: list[tuple[dict, dict]], first: str, second: str) -> dict[str, Any]:
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    a_correct = sum(1 for a, _ in pairs if a["correct"])
    b_correct = sum(1 for _, b in pairs if b["correct"])
    flipped_out = sum(1 for a, b in pairs if a["correct"] and not b["correct"])
    flipped_in = sum(1 for a, b in pairs if not a["correct"] and b["correct"])
    test = mcnemar(flipped_out, flipped_in)
    return {
        "n": n,
        first: round(a_correct / n, 4),
        second: round(b_correct / n, 4),
        "delta": round((b_correct - a_correct) / n, 4),
        f"{first}_to_wrong": flipped_out,
        f"{second}_to_right": flipped_in,
        "chi2": test["chi2"],
        "p": test["p"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path, action="append",
                        help="repeat to pool halves; keys carry the run so two "
                             "runs cannot collide on an image path")
    parser.add_argument("--first", default="answers.jsonl")
    parser.add_argument("--second", default="answers.prompted.jsonl")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--exclude-pending", action="store_true",
                        help="drop images still awaiting human adjudication")
    args = parser.parse_args()

    first_rows: dict[tuple[str, ...], dict[str, Any]] = {}
    second_rows: dict[tuple[str, ...], dict[str, Any]] = {}
    for run in args.run:
        for row in read_jsonl(run / args.first):
            first_rows[(str(run),) + key(row)] = row
        for row in read_jsonl(run / args.second):
            second_rows[(str(run),) + key(row)] = row
    first_name = first_rows and next(iter(first_rows.values())).get(
        "condition", "first") or "first"
    second_name = second_rows and next(iter(second_rows.values())).get(
        "condition", "second") or "second"

    shared = sorted(set(first_rows) & set(second_rows))
    pairs = [(first_rows[k], second_rows[k]) for k in shared]
    if args.exclude_pending:
        pairs = [(a, b) for a, b in pairs if a.get("resolution") != "pending_human"]
    # Abstentions are dropped only from the paired set, and only when they occur,
    # because a pair needs both halves scored. The count is reported so a
    # condition that abstains its way out of being wrong cannot hide here.
    scored = [(a, b) for a, b in pairs
              if a["correct"] is not None and b["correct"] is not None]

    by_source: dict[str, list] = collections.defaultdict(list)
    for a, b in scored:
        by_source[a.get("gold_source", "unknown")].append((a, b))
    diagnostic = by_source.get("image_differs_from_spec", [])
    by_family: dict[str, list] = collections.defaultdict(list)
    for a, b in diagnostic:
        by_family[a["family"]].append((a, b))

    def confirmation(index: int) -> float | None:
        """Share of diagnostic trials answered with the requested scene.

        Under 2AFC with the distractor taken from the unmet spec claim, a wrong
        answer *is* the requested scene, so this is 1 - accuracy on exactly
        these trials. Chance is 0.5, not 0.
        """
        if not diagnostic:
            return None
        wrong = sum(1 for pair in diagnostic if not pair[index]["correct"])
        return round(wrong / len(diagnostic), 4)

    report = {
        "run": [str(run) for run in args.run],
        "conditions": [first_name, second_name],
        "n_trials_paired": len(pairs),
        "n_trials_scored": len(scored),
        "n_unmatched": {
            args.first: len(set(first_rows) - set(second_rows)),
            args.second: len(set(second_rows) - set(first_rows)),
        },
        "excluded_pending_human": bool(args.exclude_pending),
        "overall": cell(scored, first_name, second_name),
        "by_gold_source": {
            source: cell(rows, first_name, second_name)
            for source, rows in sorted(by_source.items())
        },
        "diagnostic_by_family": {
            family: cell(rows, first_name, second_name)
            for family, rows in sorted(by_family.items())
        },
        "self_confirmation": {
            first_name: confirmation(0),
            second_name: confirmation(1),
            "n": len(diagnostic),
            "definition": "share of image_differs_from_spec trials answered "
                          "with the requested scene rather than the drawn one; "
                          "chance is 0.5 under 2AFC",
        },
        "abstain": {
            first_name: sum(1 for a, _ in pairs if a["correct"] is None),
            second_name: sum(1 for _, b in pairs if b["correct"] is None),
        },
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
