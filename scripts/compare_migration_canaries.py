"""Apply the pre-registered Windows-to-A800 migration acceptance thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from selfsight.utils.hashing import sha256_file
from selfsight.utils.jsonl import atomic_write_json, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--a800", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    local_rows = {row["scene_id"]: row for row in read_jsonl(args.local / "canary_rows.jsonl")}
    a800_rows = {row["scene_id"]: row for row in read_jsonl(args.a800 / "canary_rows.jsonl")}
    if set(local_rows) != set(a800_rows) or len(local_rows) != 32:
        raise RuntimeError("Canary manifests do not contain the same fixed 32 scene IDs")
    ids = sorted(local_rows)
    answer_agreement = sum(
        local_rows[item]["observer_answer"] == a800_rows[item]["observer_answer"] for item in ids
    ) / len(ids)
    verifier_agreement = sum(
        local_rows[item]["verifier_answer"] == a800_rows[item]["verifier_answer"] for item in ids
    ) / len(ids)
    local_summary = json.loads((args.local / "canary_summary.json").read_text(encoding="utf-8"))
    a800_summary = json.loads((args.a800 / "canary_summary.json").read_text(encoding="utf-8"))
    metric_differences = {
        key: abs(float(local_summary[key]) - float(a800_summary[key]))
        for key in ("observer_accuracy", "verifier_accuracy", "verifier_coverage")
    }
    identity_keys = (
        "model_id",
        "revision",
        "source_revision",
        "dependency_revisions",
        "eligible_families",
    )
    conditions = {
        "observer_answer_agreement_at_least_95pct": answer_agreement >= 0.95,
        "verifier_label_agreement_at_least_95pct": verifier_agreement >= 0.95,
        "all_metric_differences_at_most_1pt": max(metric_differences.values()) <= 0.01,
        "backbone_identity_identical": all(
            local_summary.get(key) == a800_summary.get(key) for key in identity_keys
        ),
    }
    report = {
        "schema_version": 1,
        "gate": "a800_migration",
        "passed": all(conditions.values()),
        "conditions": conditions,
        "observer_answer_agreement": answer_agreement,
        "verifier_label_agreement": verifier_agreement,
        "metric_absolute_differences": metric_differences,
        "local_summary": local_summary,
        "a800_summary": a800_summary,
        "evidence": {
            "local_rows": {
                "path": str((args.local / "canary_rows.jsonl").resolve()),
                "sha256": sha256_file(args.local / "canary_rows.jsonl"),
            },
            "local_summary": {
                "path": str((args.local / "canary_summary.json").resolve()),
                "sha256": sha256_file(args.local / "canary_summary.json"),
            },
            "a800_rows": {
                "path": str((args.a800 / "canary_rows.jsonl").resolve()),
                "sha256": sha256_file(args.a800 / "canary_rows.jsonl"),
            },
            "a800_summary": {
                "path": str((args.a800 / "canary_summary.json").resolve()),
                "sha256": sha256_file(args.a800 / "canary_summary.json"),
            },
        },
        "next_action": (
            "Formal three-seed E2 is authorized."
            if all(conditions.values())
            else "Stop formal runs and resolve the platform discrepancy before scaling."
        ),
    }
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
