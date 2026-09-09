"""Does the run record what each E3 endpoint needs? Checked, not assumed.

Endpoint 1 does. Endpoints 2 and 3 do not, and the reason is the same in both
cases: the per-checkpoint evaluation records one observation condition
(prompted) of one candidate (index 0) against one question set (spec-derived).
Endpoint 2 wants the other condition. Endpoint 3 wants the other condition,
the other three candidates, and the other question set.

Both gaps are recoverable offline -- the images and the per-round adapters
are saved and the backbone is the 1.5B -- so closing them is a re-observation
pass, not a retrain. This script measures the cost of that pass from
latencies already on disk rather than guessing it.

    envs/core/python.exe review-packets/e3-data-availability-20260909/availability.py

Reads saved artifacts only. No GPU, and nothing is written into a run.
"""

from __future__ import annotations

import collections
import datetime as dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "runs/v4/decoupling-main-20260908"
CHECKPOINT = MAIN / "evaluations/naive/step-00016"
# The only two-condition observation on disk: a pipeline run that ran
# stage_observe twice. It is what fixes questions-per-image for the
# detector-informed set, which is the set endpoint 3 needs.
TWO_CONDITION = ROOT / "runs/v4/main-1plus1plus1/answers.jsonl"


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line
            in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def what_the_run_records() -> dict:
    select = rows(CHECKPOINT / "s_select.jsonl")
    verified = rows(CHECKPOINT / "verified.jsonl")
    manifest = rows(CHECKPOINT / "manifest.jsonl")
    evaluate = (ROOT / "src/selfsight/v4/evaluate.py").read_text(encoding="utf-8")
    candidates = {int(row["candidate_index"]) for row in verified}
    scored = {Path(row["image_path"]).name for row in select}
    scored_candidates = {int(row["candidate_index"]) for row in manifest
                         if Path(row["image_path"]).name in scored}
    seconds = questions = 0.0
    for row in select:
        observation = row["observation"]
        start = dt.datetime.fromisoformat(observation["started_at"])
        end = dt.datetime.fromisoformat(observation["finished_at"])
        seconds += (end - start).total_seconds()
        questions += len(observation["answers"])
    return {
        "images_drawn": len(manifest),
        "candidates_drawn": sorted(candidates),
        "candidates_self_reported": sorted(scored_candidates),
        "self_reports": len(select),
        "condition": ("prompted" if "observe_naive(" in evaluate else "unknown")
                      + (" only" if "observe_rfo(" not in evaluate else " and blind"),
        "spec_questions_per_image": questions / len(select),
        "seconds_per_question": seconds / questions,
    }


def detector_questions_per_image() -> float:
    counts = collections.Counter(row["image_path"] for row in rows(TWO_CONDITION))
    return sum(counts.values()) / len(counts)


def main() -> None:
    print(__doc__.split("    envs/core")[0].strip())
    print()
    facts = what_the_run_records()
    print("=" * 78)
    print("What one (arm, checkpoint) of the main run actually holds")
    print("=" * 78)
    for key, value in facts.items():
        print(f"  {key:28s} {value}")

    per_question = facts["seconds_per_question"]
    spec_q = facts["spec_questions_per_image"]
    detector_q = detector_questions_per_image()
    print(f"  {'detector_questions_per_image':28s} {detector_q:.1f}"
          f"   (from {TWO_CONDITION.parent.name})")

    print()
    print("=" * 78)
    print("What each endpoint needs, and what it costs to get it")
    print("=" * 78)
    # (label, images, conditions, questions per image)
    passes = [
        ("endpoint 2  blind gap, candidate 0, spec questions", 64, 1, spec_q),
        ("endpoint 3  selection + conflict, 4 candidates, both conditions",
         256, 2, detector_q),
    ]
    print(f"{'pass':64s}{'min/unit':>10}{'main 26':>9}{'x5 130':>9}")
    total = 0.0
    for label, images, conditions, per_image in passes:
        unit = images * conditions * per_image * per_question / 60
        total += unit
        print(f"{label:64s}{unit:>10.1f}{unit * 26 / 60:>8.1f}h{unit * 130 / 60:>8.1f}h")
    print(f"{'both, one adapter load':64s}{total:>10.1f}"
          f"{total * 26 / 60:>8.1f}h{total * 130 / 60:>8.1f}h")
    print()
    print("  A unit is one (arm, checkpoint). 26 = 13 checkpoints x 2 arms for the")
    print("  main run; 130 = the same for five replicates. Adapter loads are on top,")
    print("  13 per arm, and the backbone stays resident across them.")
    print()
    print("  Against ~400 GPU-hours for the five replicates this is about 6%, and")
    print("  it is a 1.5B observer on one card -- so unlike the replicates it does")
    print("  not need both. Whether it fits beside the adjudication ladder on card 0")
    print("  is a VRAM question that has to be measured, not assumed.")


if __name__ == "__main__":
    main()
