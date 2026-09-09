"""Are these five runs five seeds, or one seed five times?

The failure this exists for is silent. `training.seed` in a config that
`v4_train.py` predates is not an error, it is ignored, and the run directory
names, the configs and the logs then all agree on a claim that is false. The
only place the truth survives is the artefacts: what each round recorded as
its initialization seed, and what each run recorded as its split.

Two conditions, and both are required. Distinct training seeds without an
identical split means five experiments on five partitions, which do not pair.
An identical split without distinct training seeds means one run copied five
times.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

VERSION = "v4_verify_replicates 1"


def initialization_seeds(run: Path) -> set[int]:
    """Every initialization seed the run's finished rounds recorded."""

    seeds = set()
    for done in sorted(run.glob("rounds/round-*/done.json")):
        payload = json.loads(done.read_text(encoding="utf-8"))
        if "initialization_seed" in payload:
            seeds.add(int(payload["initialization_seed"]))
    return seeds


def split_fingerprint(run: Path) -> str:
    """The digest the run's split recorded, which replicates must share."""

    return str(json.loads((run / "split.json").read_text(encoding="utf-8"))["digest"])


def verify(runs: list[Path]) -> dict[str, Any]:
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
                        "split_digest": split_fingerprint(run),
                        "rounds": len(list(run.glob("rounds/round-*/done.json")))})

    training = [row["training_seed"] for row in per_run]
    splits = {row["split_digest"] for row in per_run}
    distinct = len(set(training)) == len(training)
    shared = len(splits) == 1
    return {
        "version": VERSION,
        "runs": per_run,
        "training_seeds_distinct": distinct,
        "split_shared": shared,
        "verdict": "five seeds" if distinct and shared else "NOT independent seeds",
        "reason": ("" if distinct and shared else
                   ("; ".join(filter(None, [
                       "" if distinct else f"training seeds repeat: {sorted(training)}",
                       "" if shared else f"runs do not share a split: {sorted(splits)}"])))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    report = verify(list(args.runs))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["verdict"] != "five seeds":
        sys.exit(f"{VERSION}: {report['reason']}")


if __name__ == "__main__":
    main()
