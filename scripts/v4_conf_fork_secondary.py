"""The two pre-declared secondary analyses of 27 that the fork script omits.

27 declared, before the third corpus existed, two secondaries beyond the ones
25 already carried:

  1. cell-level clustering. One pair per (spec_id, noun) cell rather than per
     image. K went 6 -> 14, so one cell now yields more images and the image
     unit still admits pseudo-replication across them. The cell unit is the
     stricter one, which is exactly why it cannot rescue anything: it can only
     shrink n.
  2. fixed-effect pooling of batch 2 and batch 3. Both ran under the same
     pre-registration, so pooling them is standard -- and 27 wrote it down in
     advance precisely so it could not be remembered only after a null.

Neither can change the verdict. 27 is explicit that if the primary is not
significant and the pooled estimate is, that is written as "significant after
pooling two replications" and never as "confirmed".

This lives apart from v4_conf_fork.py so that the script which produced the
frozen primary record is not edited after the fact. The cost of that choice is
a duplicated unit-building block, and the guard against it drifting is the
--expect assertion: the image-clustered table recomputed here must reproduce
the primary numbers exactly, or this script refuses to print anything.
"""

from __future__ import annotations

import argparse
import collections
import math
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v4_conf_fork as fork  # noqa: E402
from selfsight.v4.spec import canonical_noun  # noqa: E402


def build_units(outdir: Path, labels: dict[tuple[str, str], bool]) -> list[dict[str, Any]]:
    """One record per usable deletion pair. Mirrors v4_conf_fork.main."""
    accepted = {r["pair_id"]: r for r in fork.read_jsonl(outdir / "accepted.jsonl")}
    answers = {c: fork.read_jsonl(outdir / name) for c, name in
               (("image_only", "answers.jsonl"),
                ("prompted", "answers.prompted.jsonl"))}
    edited = {
        c: {r["metadata"]["pair_id"]: r["correct"] for r in rows
            if r["metadata"]["member"] == "edited"}
        for c, rows in answers.items()
    }
    units = []
    for pair_id, row in accepted.items():
        if row.get("edit") != "delete":
            continue
        if pair_id not in edited["image_only"] or pair_id not in edited["prompted"]:
            continue
        key = (row["spec"]["spec_id"], canonical_noun(row["noun"]))
        label = labels.get(key)
        if label is None:
            continue
        units.append({
            "pair_id": pair_id,
            "image": pair_id.split(":")[0],
            "cell": key[0] + "|" + key[1],
            "stratum": "prone" if label else "never",
            "flip": edited["image_only"][pair_id] and not edited["prompted"][pair_id],
        })
    return units


def cluster(units: list[dict[str, Any]], unit_key: str) -> dict[str, tuple[int, int]]:
    """Draw one pair per (unit_key, stratum) with the pre-declared seed."""
    rng = random.Random(fork.CLUSTER_SEED)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for unit in units:
        grouped[(unit[unit_key], unit["stratum"])].append(unit)
    drawn = [rng.choice(sorted(v, key=lambda u: u["pair_id"]))
             for _, v in sorted(grouped.items())]
    return {s: (sum(1 for u in drawn if u["stratum"] == s and u["flip"]),
                sum(1 for u in drawn if u["stratum"] == s))
            for s in ("prone", "never")}


def show(title: str, table: dict[str, tuple[int, int]]) -> tuple[float, float]:
    (a, n1), (c, n2) = table["prone"], table["never"]
    one, two = fork.fisher(a, n1 - a, c, n2 - c)
    print("=== " + title)
    print(f"{'':<8}{'n':>6}{'flips':>8}{'rate':>9}")
    for stratum, (hits, n) in table.items():
        print(f"{stratum:<8}{n:>6}{hits:>8}{hits / max(1, n):>9.3f}")
    print(f"  Fisher exact: one-sided p={one:.4g}, two-sided p={two:.4g}")
    return one, two


def mantel_haenszel(tables: list[dict[str, tuple[int, int]]]) -> tuple[float, float, float]:
    """Fixed-effect MH odds ratio and its one-sided p, over 2x2 strata."""
    num = den = obs = exp = var = 0.0
    for t in tables:
        (a, n1), (c, n2) = t["prone"], t["never"]
        b, d = n1 - a, n2 - c
        n = a + b + c + d
        if n == 0:
            continue
        num += a * d / n
        den += b * c / n
        obs += a
        exp += (a + b) * (a + c) / n
        if n > 1:
            var += (a + b) * (c + d) * (a + c) * (b + d) / (n * n * (n - 1))
    orat = float("inf") if den == 0 else num / den
    if var == 0:
        return orat, 1.0, 1.0
    chi2 = (abs(obs - exp) - 0.5) ** 2 / var
    two = math.erfc(math.sqrt(chi2 / 2))
    one = two / 2 if obs > exp else 1.0 - two / 2
    return orat, one, two


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--pool-outdir", type=Path)
    p.add_argument("--expect", nargs=4, type=int, metavar=("A", "N1", "C", "N2"),
                   help="primary table to reproduce: prone flips, prone n, never flips, never n")
    args = p.parse_args()

    labels = fork.proneness(tuple(fork.OLD_RUNS))
    units = build_units(args.outdir, labels)
    by_image = cluster(units, "image")

    if args.expect:
        a, n1, c, n2 = args.expect
        got = (by_image["prone"], by_image["never"])
        if got != ((a, n1), (c, n2)):
            raise SystemExit(
                f"refusing to report: the image-clustered table recomputed here is "
                f"{got}, the primary record says {((a, n1), (c, n2))}. The duplicated "
                f"unit-building block has drifted from v4_conf_fork.py.")
        print(f"[ok] reproduces the primary image-clustered table {got}")
        print()

    print("These are pre-declared secondaries. Neither can change the 27 verdict,")
    print("which is: not confirmed, the fork stays exploratory.")
    print()
    show("secondary 1: clustered by (spec, noun) cell -- the stricter unit",
         cluster(units, "cell"))

    if args.pool_outdir:
        print()
        pool_units = build_units(args.pool_outdir, labels)
        b2 = cluster(pool_units, "image")
        show("batch 2 recomputed here, clustered by image", b2)
        print()
        orat, one, two = mantel_haenszel([b2, by_image])
        print("=== secondary 2: fixed-effect pooling of batch 2 and batch 3")
        print(f"  Mantel-Haenszel OR {orat:.3f}, one-sided p={one:.4g}, two-sided p={two:.4g}")
        pa = b2["prone"][0] + by_image["prone"][0]
        pn1 = b2["prone"][1] + by_image["prone"][1]
        pc = b2["never"][0] + by_image["never"][0]
        pn2 = b2["never"][1] + by_image["never"][1]
        print(f"  pooled counts prone {pa}/{pn1} = {pa / max(1, pn1):.3f}, "
              f"never {pc}/{pn2} = {pc / max(1, pn2):.3f}")
        if one < 0.05:
            print("  NOTE: significant after pooling two replications. Per 27 this is")
            print("  written exactly that way and never as 'confirmed'.")
        else:
            print("  not significant after pooling either.")


if __name__ == "__main__":
    main()
