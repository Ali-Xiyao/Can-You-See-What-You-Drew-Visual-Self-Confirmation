"""Read one Tier B arm's answers and print the table the write-up needs.

Kept separate from `v4_analyze.py`, which reports the natural pool. The two
pools do not share a unit: the natural pool's diagnostic trials are the ones
where the generator happened to draw something other than what was asked, and
the truth came from a detector; here every pair is a manufactured difference and
the truth came from the edit. Reporting them through the same function would
make it easy to pool them by accident, and they must not be pooled.

    python scripts/v4_tierb_analyze.py --outdir runs/v4/tierb-delete

Three numbers per edit and condition:

  original  the unedited image, which agrees with its prompt. A model reciting
            the prompt scores 1.0 here for the wrong reason, so this row is a
            floor, not evidence.
  edited    the image that contradicts the prompt on exactly one atom. This is
            the diagnostic row: the gap between conditions is deference.
  sham      the artifact control. Neither member contradicts anything, so a gap
            here is the filler's fingerprint rather than deference, and any
            drop on the edited row has to clear it.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mcnemar(before: dict[str, bool], after: dict[str, bool]) -> tuple[int, int, float]:
    """Paired test over the trials answered in both conditions.

    Continuity-corrected, and computed with `erfc` because the observer
    environment has no scipy. Same formula the natural pool uses, so the two are
    comparable even though the pools are not.
    """
    b = sum(1 for k in before if k in after and before[k] and not after[k])
    c = sum(1 for k in before if k in after and not before[k] and after[k])
    if b + c == 0:
        return b, c, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return b, c, math.erfc(math.sqrt(chi2 / 2))


def rate(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "     --"
    hits = sum(1 for r in rows if r["correct"])
    return f"{hits / len(rows):.3f} ({hits}/{len(rows)})"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()

    answers = {
        "image_only": read_jsonl(args.outdir / "answers.jsonl"),
        "prompted": read_jsonl(args.outdir / "answers.prompted.jsonl"),
    }
    edits = sorted({r["metadata"]["edit"] for rows in answers.values()
                    for r in rows})

    for edit in edits:
        print(f"\n=== {edit}")
        print(f"{'':<10}{'image_only':>18}{'prompted':>18}")
        for member in ("original", "edited"):
            cells = []
            for condition in ("image_only", "prompted"):
                cells.append(rate([r for r in answers[condition]
                                   if r["metadata"]["edit"] == edit
                                   and r["metadata"]["member"] == member]))
            print(f"{member:<10}{cells[0]:>18}{cells[1]:>18}")

        for member in ("original", "edited"):
            keyed = {
                condition: {r["question_id"]: r["correct"]
                            for r in answers[condition]
                            if r["metadata"]["edit"] == edit
                            and r["metadata"]["member"] == member}
                for condition in ("image_only", "prompted")
            }
            b, c, p = mcnemar(keyed["image_only"], keyed["prompted"])
            print(f"  {member}: image_only-only right {b}, prompted-only "
                  f"right {c}, McNemar p={p:.3g}")

        for condition in ("image_only", "prompted"):
            rows = [r for r in answers[condition]
                    if r["metadata"]["edit"] == edit]
            abstained = sum(1 for r in rows if r.get("abstained"))
            print(f"  {condition}: {abstained}/{len(rows)} abstained")

    # Pair-level, the unit the design is built on: a model that tracks pixels
    # answers the two members differently, and one reciting the prompt cannot.
    for edit in edits:
        for condition in ("image_only", "prompted"):
            pairs: dict[str, dict[str, bool]] = collections.defaultdict(dict)
            for r in answers[condition]:
                if r["metadata"]["edit"] == edit:
                    pairs[r["metadata"]["pair_id"]][r["metadata"]["member"]] = \
                        r["correct"]
            full = [v for v in pairs.values() if len(v) == 2]
            both = sum(1 for v in full if all(v.values()))
            neither = sum(1 for v in full if not any(v.values()))
            print(f"\n{edit} {condition}: {len(full)} pairs, both right "
                  f"{both / max(1, len(full)):.3f}, one right "
                  f"{(len(full) - both - neither) / max(1, len(full)):.3f}, "
                  f"neither {neither / max(1, len(full)):.3f}")


if __name__ == "__main__":
    main()
