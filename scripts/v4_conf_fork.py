"""The pre-registered confirmatory test of the fork (STATUS 24, plan in 25).

Exploratory 24 split the deletion pairs by whether the generator had ever
omitted that (spec, noun) on some other seed, and found the prompted drop
concentrated in the omission-prone half. The split variable was chosen after
seeing the paired arm, so that finding cannot confirm itself. This script runs
the test that was written down before the confirmatory corpus existed.

Everything here follows 25 and nothing here may be tuned to the outcome:

  labels     computed from the OLD corpus only (runs/v4/main-*), never from the
             images being tested. A (spec_id, singleton noun) cell is prone if
             that noun is entirely absent from any adjudicated old image of that
             spec, and never otherwise. Reproduces 429 cells, 242 / 187.
  unit       one image per stratum. Several pairs can come from one image and
             they are not independent, so within a stratum one pair per image is
             drawn at random with the pre-declared seed 20260902.
  outcome    on the edited member: image_only correct and prompted wrong. That
             is the deference event the whole design is built to catch.
  test       2x2 Fisher exact, one-sided in the direction 24 reported, with the
             two-sided p alongside. Confirmed iff one-sided p < 0.05 and the
             direction matches.
  failure    not significant, or reversed, or the strata differ by more than
             0.05 in image_only accuracy, or by more than 5 points in exclusion
             rate. Any of those demotes 24 to an exploratory observation. The
             secondary analyses below cannot overturn that.

Exclusion rate is measured on candidates, not on accepted pairs: every
(new-corpus image, singleton noun of its spec) is a candidate, and a candidate
is excluded when its image never reached a verdict (pending_human or an
unnameable object). Measuring it on accepted pairs instead would hide exactly
the bias it is there to detect, since an image that never got adjudicated can
never produce a pair.

    python scripts/v4_conf_fork.py --outdir runs/v4/tierb-conf
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
from pathlib import Path
from typing import Any

from selfsight.v4.spec import SceneSpec, canonical_noun

OLD_RUNS = ("runs/v4/main-2plus1", "runs/v4/main-1plus1plus1")
NEW_RUNS = ("runs/v4/conf-2plus1", "runs/v4/conf-1plus1plus1")
CLUSTER_SEED = "20260902"
UNADJUDICATED = {"pending_human", "unnameable"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def singleton_nouns(spec: SceneSpec) -> list[str]:
    return [canonical_noun(o.object) for o in spec.objects if o.count == 1]


def proneness(runs: tuple[str, ...]) -> dict[tuple[str, str], bool]:
    """True where the generator has been seen to leave this noun out entirely."""
    cells: dict[tuple[str, str], bool] = {}
    for name in runs:
        run = Path(name)
        specs = {row["image_path"]: SceneSpec.from_dict(row["spec"])
                 for row in read_jsonl(run / "manifest.jsonl")}
        for row in read_jsonl(run / "verified.jsonl"):
            if row["resolution"] in UNADJUDICATED:
                continue
            spec = specs.get(row["image_path"])
            if spec is None:
                continue
            present = {canonical_noun(d["object"]) for d in row["detections"]}
            for noun in singleton_nouns(spec):
                key = (spec.spec_id, noun)
                cells[key] = cells.get(key, False) or noun not in present
    return cells


def exclusions(runs: tuple[str, ...],
               labels: dict[tuple[str, str], bool]) -> dict[str, tuple[int, int]]:
    """Per stratum: how many candidates the adjudication budget threw away."""
    counts: dict[str, list[int]] = {"prone": [0, 0], "never": [0, 0]}
    for name in runs:
        run = Path(name)
        specs = {row["image_path"]: SceneSpec.from_dict(row["spec"])
                 for row in read_jsonl(run / "manifest.jsonl")}
        for row in read_jsonl(run / "verified.jsonl"):
            spec = specs.get(row["image_path"])
            if spec is None:
                continue
            for noun in singleton_nouns(spec):
                label = labels.get((spec.spec_id, noun))
                if label is None:
                    continue
                cell = counts["prone" if label else "never"]
                cell[1] += 1
                if row["resolution"] in UNADJUDICATED:
                    cell[0] += 1
    return {k: (v[0], v[1]) for k, v in counts.items()}


def log_choose(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    """Exact 2x2 test. One-sided p for a being large, plus the two-sided p.

    Written out because the observer environment has no scipy, and built on the
    same hypergeometric the pre-registration names.
    """
    row1, row2 = a + b, c + d
    col1, total = a + c, a + b + c + d
    denom = log_choose(total, col1)

    def prob(x: int) -> float:
        return math.exp(log_choose(row1, x) + log_choose(row2, col1 - x) - denom)

    lo = max(0, col1 - row2)
    hi = min(row1, col1)
    observed = prob(a)
    greater = sum(prob(x) for x in range(a, hi + 1))
    both = sum(prob(x) for x in range(lo, hi + 1)
               if prob(x) <= observed * (1 + 1e-9))
    return min(1.0, greater), min(1.0, both)


def mcnemar(pairs: list[tuple[bool, bool]]) -> tuple[int, int, float]:
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if y and not x)
    if b + c == 0:
        return b, c, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return b, c, math.erfc(math.sqrt(chi2 / 2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--old", nargs="*", default=list(OLD_RUNS))
    parser.add_argument("--new", nargs="*", default=list(NEW_RUNS))
    args = parser.parse_args()

    labels = proneness(tuple(args.old))
    n_prone = sum(1 for v in labels.values() if v)
    print(f"labels from the old corpus: {len(labels)} cells, "
          f"{n_prone} prone, {len(labels) - n_prone} never")

    accepted = {r["pair_id"]: r for r in read_jsonl(args.outdir / "accepted.jsonl")}
    answers = {c: read_jsonl(args.outdir / name) for c, name in
               (("image_only", "answers.jsonl"),
                ("prompted", "answers.prompted.jsonl"))}
    edited = {
        c: {r["metadata"]["pair_id"]: r["correct"] for r in rows
            if r["metadata"]["member"] == "edited"}
        for c, rows in answers.items()
    }

    # One record per usable pair, before clustering.
    units: list[dict[str, Any]] = []
    unlabelled = 0
    for pair_id, row in accepted.items():
        if row.get("edit") != "delete":
            continue
        if pair_id not in edited["image_only"] or pair_id not in edited["prompted"]:
            continue
        key = (row["spec"]["spec_id"], canonical_noun(row["noun"]))
        label = labels.get(key)
        if label is None:
            unlabelled += 1
            continue
        units.append({
            "pair_id": pair_id,
            "image": pair_id.split(":")[0],
            "stratum": "prone" if label else "never",
            "noun": key[1],
            "flip": edited["image_only"][pair_id] and not edited["prompted"][pair_id],
            "image_only": edited["image_only"][pair_id],
            "prompted": edited["prompted"][pair_id],
            "hole_share": row["hole_share"],
            "centre_coverage": row["centre_coverage"],
            "strict": row["gate"] == "strict",
        })
    print(f"{len(units)} answered deletion pairs carry a label, "
          f"{unlabelled} had no old-corpus cell")

    # Pre-declared clustering: one pair per image per stratum, drawn at random.
    rng = random.Random(CLUSTER_SEED)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for unit in units:
        grouped[(unit["image"], unit["stratum"])].append(unit)
    drawn = [rng.choice(sorted(v, key=lambda u: u["pair_id"]))
             for _, v in sorted(grouped.items())]

    table = {}
    for stratum in ("prone", "never"):
        rows = [u for u in drawn if u["stratum"] == stratum]
        table[stratum] = (sum(1 for u in rows if u["flip"]), len(rows))
    (a, n1), (c, n2) = table["prone"], table["never"]
    b, d = n1 - a, n2 - c
    one, two = fisher(a, b, c, d)

    print()
    print("=== primary test, clustered by image")
    print(f"{'':<8}{'n':>6}{'flips':>8}{'rate':>9}")
    for stratum, (hits, n) in table.items():
        print(f"{stratum:<8}{n:>6}{hits:>8}{(hits / max(1, n)):>9.3f}")
    print(f"  Fisher exact: one-sided (prone higher) p={one:.4g}, "
          f"two-sided p={two:.4g}")

    checks = []
    direction = (a / max(1, n1)) > (c / max(1, n2))
    checks.append(("direction matches the exploratory split", direction))
    checks.append(("one-sided p < 0.05", one < 0.05))

    print()
    print("=== pre-declared failure conditions")
    io = {}
    for stratum in ("prone", "never"):
        rows = [u for u in drawn if u["stratum"] == stratum]
        io[stratum] = sum(1 for u in rows if u["image_only"]) / max(1, len(rows))
        pr = sum(1 for u in rows if u["prompted"]) / max(1, len(rows))
        print(f"  {stratum:<6} image_only {io[stratum]:.3f}  prompted {pr:.3f}  "
              f"drop {io[stratum] - pr:+.3f}")
    gap = abs(io["prone"] - io["never"])
    checks.append(("image_only strata differ by <= 0.05", gap <= 0.05))
    print(f"  image_only gap {gap:.3f}")

    excl = exclusions(tuple(args.new), labels)
    rates = {}
    for stratum, (lost, total) in excl.items():
        rates[stratum] = lost / max(1, total)
        print(f"  {stratum:<6} excluded {lost}/{total} = {rates[stratum]:.3f}")
    egap = abs(rates["prone"] - rates["never"])
    checks.append(("exclusion rates differ by <= 5 points", egap <= 0.05))
    print(f"  exclusion gap {egap:.3f}")

    for name, ok in checks:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    print()
    print("VERDICT: " + ("confirmed" if all(ok for _, ok in checks)
                         else "not confirmed, the fork stays exploratory"))

    print()
    print("=== secondary (cannot change the verdict)")
    for stratum in ("prone", "never"):
        rows = [u for u in drawn if u["stratum"] == stratum]
        if not rows:
            continue
        print(f"  {stratum:<6} hole_share "
              f"{sum(u['hole_share'] for u in rows) / len(rows):.3f}  "
              f"centre {sum(u['centre_coverage'] for u in rows) / len(rows):.3f}  "
              f"strict {sum(1 for u in rows if u['strict']) / len(rows):.3f}")

    by_image: dict[str, dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    for unit in drawn:
        by_image[unit["image"]][unit["stratum"]] = unit
    both_strata = [v for v in by_image.values() if len(v) == 2]
    b_, c_, p_ = mcnemar([(v["prone"]["flip"], v["never"]["flip"])
                          for v in both_strata])
    print(f"  same-image pairing: {len(both_strata)} images carry both strata, "
          f"prone-only flip {b_}, never-only flip {c_}, McNemar p={p_:.3g}")

    shared = ({u["noun"] for u in drawn if u["stratum"] == "prone"}
              & {u["noun"] for u in drawn if u["stratum"] == "never"})
    for stratum in ("prone", "never"):
        rows = [u for u in drawn if u["stratum"] == stratum and u["noun"] in shared]
        hits = sum(1 for u in rows if u["flip"])
        print(f"  shared-noun {stratum:<6} {hits}/{len(rows)} = "
              f"{hits / max(1, len(rows)):.3f}")


if __name__ == "__main__":
    main()
