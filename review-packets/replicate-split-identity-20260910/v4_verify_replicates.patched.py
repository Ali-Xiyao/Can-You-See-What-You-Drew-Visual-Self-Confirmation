"""Are these five runs five seeds, or one seed five times, or the wrong split?

The failure this exists for is silent. `training.seed` in a config that
`v4_train.py` predates is not an error, it is ignored, and the run directory
names, the configs and the logs then all agree on a claim that is false. The
only place the truth survives is the artefacts: what each round recorded as
its initialization seed, and what each run recorded as its split.

Three conditions, and all three are required. Distinct training seeds without
an identical split means five experiments on five partitions, which do not
pair. An identical split without distinct training seeds means one run copied
five times. And five replicates that agree with each other but not with the
main run pair perfectly among themselves while comparing against nothing --
deviation 7.2's guard catches that one at analysis time, which is after the
400 GPU-hours have been spent.

The split is compared by *identity* (`selfsight.analysis.drift.split_digest`,
everything the file records except the wall-clock stamp), not by the `digest`
field the file carries. Measured 2026-09-10: `v4_train.split_digest` is a pure
function of `sorted(runs)`, `config["seed"]`, `data.local_outcome` and
`data.local_probe`, and computing it from config text alone gives one value,
9bab14d4..., for the main run and all five replicate configs -- before any
split stage runs. Comparing that field across the five replicates is a
condition that cannot fail, and it stays true for a replicate whose corpus
gained a prompt and whose held-out set therefore differs. Identity covers the
prompt lists the recipe produced, so it separates that case; the recipe digest
is still reported, because recipe-same/identity-different says the corpus
moved while recipe-different says the config did.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from selfsight.analysis.drift import split_digest

VERSION = "v4_verify_replicates 2"


def initialization_seeds(run: Path) -> set[int]:
    """Every initialization seed the run's finished rounds recorded."""

    seeds = set()
    for done in sorted(run.glob("rounds/round-*/done.json")):
        payload = json.loads(done.read_text(encoding="utf-8"))
        if "initialization_seed" in payload:
            seeds.add(int(payload["initialization_seed"]))
    return seeds


def recipe_digest(run: Path) -> str:
    """The `digest` field the split stage wrote: a hash of the recipe only.

    Reported, not compared. See the module docstring for why.
    """

    return str(json.loads((run / "split.json").read_text(encoding="utf-8"))["digest"])


def verify(runs: list[Path], main: Path | None = None) -> dict[str, Any]:
    if len(runs) < 2:
        raise ValueError(f"verifying replicates needs at least two runs; got {len(runs)}")
    per_run = []
    for run in runs:
        seeds = initialization_seeds(run)
        if not seeds:
            raise ValueError(f"{run} has no finished round recording an initialization seed; "
                             f"nothing here can tell you which seed it trained on")
        if len(seeds) > 1:
            raise ValueError(f"{run} recorded more than one initialization seed {sorted(seeds)}; "
                             f"a replicate that changed seed mid-run is not a replicate")
        per_run.append({"run": run.name, "training_seed": seeds.pop(),
                        "split_identity": split_digest(run),
                        "recipe_digest": recipe_digest(run),
                        "rounds": len(list(run.glob("rounds/round-*/done.json")))})

    training = [row["training_seed"] for row in per_run]
    splits = {row["split_identity"] for row in per_run}
    distinct = len(set(training)) == len(training)
    shared = len(splits) == 1
    main_identity = split_digest(main) if main is not None else None
    matches_main = main_identity is None or splits == {main_identity}
    good = distinct and shared and matches_main
    return {
        "version": VERSION,
        "runs": per_run,
        "main_run": None if main is None else main.name,
        "main_split_identity": main_identity,
        "checked_against_main": main is not None,
        "training_seeds_distinct": distinct,
        "split_shared": shared,
        "split_matches_main": matches_main,
        "recipe_digests_shared": len({row["recipe_digest"] for row in per_run}) == 1,
        "verdict": "five seeds" if good else "NOT independent seeds",
        "reason": ("" if good else
                   ("; ".join(filter(None, [
                       "" if distinct else f"training seeds repeat: {sorted(training)}",
                       "" if shared else f"runs do not share a split: {sorted(splits)}",
                       "" if matches_main else
                       (f"the replicates' split is not the main run's: {sorted(splits)} "
                        f"!= {main_identity}")])))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--main", type=Path, required=True,
                        help="the main run whose split the replicates must share; required, "
                             "because five replicates that agree only with each other are the "
                             "case this refuses to be silent about")
    args = parser.parse_args()
    report = verify(list(args.runs), main=args.main)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["verdict"] != "five seeds":
        sys.exit(f"{VERSION}: {report['reason']}")


if __name__ == "__main__":
    main()
