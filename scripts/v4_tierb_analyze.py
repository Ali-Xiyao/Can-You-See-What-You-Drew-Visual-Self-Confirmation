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
    origin_block(args.outdir, answers)

    for edit in edits:
        if edit == "origin":
            continue
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


def origin_block(outdir: Path, answers: dict[str, list[dict[str, Any]]]) -> None:
    """The contrast the origin arm exists for: same absence, different author.

    `original` is the generator's own omission and `edited` is one we made. Both
    members are missing a requested object, so the interesting number is not
    either member's accuracy -- it is how far each falls when the description is
    prepended, and whether those two falls differ.

    Also reported paired within the condition, because the two members are
    matched on spec, noun and question: on the same prompt, does the model hold
    the line better on the absence it did not cause?
    """
    accepted = {r["pair_id"]: r for r in read_jsonl(outdir / "accepted.jsonl")}
    if not any(r.get("edit") == "origin" for r in accepted.values()):
        return
    strict = {k for k, r in accepted.items() if r.get("other_mismatches", 0) == 0}

    for label, keep in (("all pairs", set(accepted)),
                        (f"only-this-mistake ({len(strict)})", strict)):
        print()
        print(f"=== origin, {label}")
        drops = {}
        for member, name in (("original", "the generator's own omission"),
                             ("edited", "one we deleted ourselves")):
            keyed = {
                condition: {r["metadata"]["pair_id"]: r["correct"]
                            for r in answers[condition]
                            if r["metadata"]["edit"] == "origin"
                            and r["metadata"]["member"] == member
                            and r["metadata"]["pair_id"] in keep}
                for condition in ("image_only", "prompted")
            }
            io, pr = keyed["image_only"], keyed["prompted"]
            if not io:
                continue
            a_rate = sum(io.values()) / len(io)
            b_rate = sum(pr.values()) / len(pr)
            drops[member] = a_rate - b_rate
            b, c, p = mcnemar(io, pr)
            print(f"  {name:<32} image_only {a_rate:.3f}  prompted {b_rate:.3f}  "
                  f"drop {a_rate - b_rate:+.3f}  ({b}->{c}, p={p:.3g})")
        if len(drops) == 2:
            print(f"  difference of drops (own minus ours): "
                  f"{drops['original'] - drops['edited']:+.3f}")

        for condition in ("image_only", "prompted"):
            per = {}
            for r in answers[condition]:
                if (r["metadata"]["edit"] == "origin"
                        and r["metadata"]["pair_id"] in keep):
                    per.setdefault(r["metadata"]["pair_id"], {})[
                        r["metadata"]["member"]] = r["correct"]
            own = {k: v["original"] for k, v in per.items() if len(v) == 2}
            ours = {k: v["edited"] for k, v in per.items() if len(v) == 2}
            b, c, p = mcnemar(own, ours)
            print(f"  {condition}: paired within cell, own-only right {b}, "
                  f"ours-only right {c}, p={p:.3g}")


if __name__ == "__main__":
    main()
