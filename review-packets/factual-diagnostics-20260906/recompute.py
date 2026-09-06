"""Read frozen runs, write corrected diagnostics only in this review packet."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from selfsight.v4.factual_truth import load_verifications
from v4_observation_ceiling import paired_diagnostic
from v4_resolution_verdict import counting_report, load_pools, measure


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    inputs: set[Path] = set()
    dirs = [ROOT / "runs/v4" / name for name in
            ("gate-b", "gate-b-openct", "gate-b-openct2")]
    for directory in dirs:
        inputs.update(directory / name for name in (
            "runs.json", "pools.jsonl", "selection.jsonl",
            "observations.naive.jsonl", "observations.rfo.jsonl"))
        names = json.loads((directory / "runs.json").read_text(encoding="utf-8"))["runs"]
        inputs.update(ROOT / name / "verified.jsonl" for name in names)
    before = {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p)
              for p in sorted(inputs)}
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "head_before_repair_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "interpretation": "Detector-derived known facts only; no causal or population claim.",
        "runs": {},
    }
    for directory in dirs:
        pools = load_pools(directory)
        report["runs"][directory.name] = {
            "counting": counting_report(directory),
            "ceiling": paired_diagnostic(pools, load_verifications(directory)),
            "registered_actual_metrics": {arm: measure(pools, arm)
                                           for arm in ("naive", "rfo")},
        }
    after = {name: digest(ROOT / name) for name in before}
    if before != after:
        raise RuntimeError("Frozen input changed during read-only recomputation")
    report["input_sha256"] = before
    report["frozen_inputs_unchanged"] = True
    target = OUT / "counting-and-ceiling.json"
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                 allow_nan=False) + "\n", encoding="utf-8")
    for name, row in report["runs"].items():
        print(name, json.dumps(row["counting"], ensure_ascii=False))
        print("eligible", len(row["ceiling"]["eligible_ids"]),
              "facts", row["ceiling"]["known_questions"], "/",
              row["ceiling"]["total_questions"])
        print(json.dumps(row["ceiling"]["metrics"], ensure_ascii=False))
    print("input_files", len(before), "unchanged", before == after)


if __name__ == "__main__":
    main()
