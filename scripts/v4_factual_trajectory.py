"""Describe three atomic agreement trajectories on a fixed factual cohort.

For each arm separately, retain the same spec/question only if its image fact
is known at every requested checkpoint. Average questions within each image,
then give images equal weight. This is not a new event test or an estimate of
whole-image correctness. No models are loaded and no run outputs are changed.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from selfsight.v4.factual_truth import canonical_answer, factual_answer, question_scope

METRICS = ("response_matches_request", "fact_matches_request", "response_matches_fact")
GROUPS = ("all", "count", "existence")
QUESTION_FIELDS = ("question_id", "text", "expected_answer", "choices", "family",
                   "question_format")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _index(rows: Sequence[Mapping[str, Any]], field: str, label: str) -> dict[str, Any]:
    result = {}
    for row in rows:
        key = str(row[field])
        if key in result:
            raise ValueError(f"Duplicate {label}: {key}")
        result[key] = row
    return result


def _answer_status(row: Mapping[str, Any] | None,
                   answer: Mapping[str, Any] | None) -> str:
    if row is None:
        return "missing_observation"
    observation = row.get("observation")
    if row.get("error") is not None or (
            isinstance(observation, Mapping) and observation.get("error") is not None):
        return "observation_error"
    if not isinstance(observation, Mapping):
        return "missing_observation"
    if answer is None:
        return "missing_answer"
    if answer.get("error") is not None:
        return "answer_error"
    if answer.get("abstain"):
        return "abstain"
    if answer.get("normalized_answer") is None:
        return "null_answer"
    if not str(answer["normalized_answer"]).strip():
        return "empty_answer"
    return "available"


def _question_table(questions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = _index(questions, "question_id", "question")
    for question in result.values():
        if any(key not in question for key in QUESTION_FIELDS[:4]):
            raise ValueError("Question lacks its actual text, expected answer or choices")
        scope = question_scope(question)
        if scope is None or scope.kind not in GROUPS[1:]:
            raise ValueError(f"Unsupported question: {question['question_id']}")
        if question.get("family", scope.kind) != scope.kind:
            raise ValueError(f"Question family disagrees with actual text: {question['question_id']}")
        if not canonical_answer(question, question["expected_answer"]):
            raise ValueError(f"Missing expected answer: {question['question_id']}")
    return result


def _signature(questions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {qid: {field: q.get(field) for field in QUESTION_FIELDS}
            for qid, q in questions.items()}


def arm_report(checkpoints: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """Pure report; each checkpoint contains manifest, observations, verifications.

    Missing observation rows use the stable question schedule recorded at other
    checkpoints, preserving their expected denominator. If no actual schedule
    exists for a spec at any checkpoint, refuse rather than regenerate it.
    """
    steps = sorted(checkpoints)
    if not steps:
        raise ValueError("At least one checkpoint is required")
    manifests, observed, verified, answers = {}, {}, {}, {}
    questions: dict[str, dict[str, Any]] = {}
    for step in steps:
        data = checkpoints[step]
        manifests[step] = _index(data["manifest"], "spec_id", "manifest spec")
        observed[step] = _index(data["observations"], "prompt_id", "observation spec")
        verified[step] = _index(data["verifications"], "image_path", "verification image")
        if step != steps[0] and set(manifests[step]) != set(manifests[steps[0]]):
            raise ValueError("Manifest spec coverage changed between checkpoints")
        if not set(observed[step]) <= set(manifests[step]):
            raise ValueError("Unexpected observation spec")
        image_paths = [str(row["image_path"]) for row in manifests[step].values()]
        if len(set(image_paths)) != len(image_paths):
            raise ValueError("Duplicate manifest image")
        if not set(verified[step]) <= set(image_paths):
            raise ValueError("Unexpected verification image")
        answers[step] = {}
        for sid, row in observed[step].items():
            if str(row["image_path"]) != str(manifests[step][sid]["image_path"]):
                raise ValueError(f"Observation image does not match manifest: {step}/{sid}")
            if "questions" in row:
                current = _question_table(row["questions"])
                if not current:
                    raise ValueError(f"Empty question schedule: {step}/{sid}")
                if sid in questions and _signature(current) != _signature(questions[sid]):
                    raise ValueError(f"Question meaning or ID changed: {step}/{sid}")
                questions[sid] = current
            elif _answer_status(row, None) != "observation_error":
                raise ValueError(f"Non-error observation lacks actual questions: {step}/{sid}")
            observation = row.get("observation")
            observation = observation if isinstance(observation, Mapping) else {}
            table = _index(observation.get("answers") or [], "question_id", "answer")
            answers[step][sid] = table
    specs = sorted(manifests[steps[0]])
    if set(questions) != set(specs):
        raise ValueError("No recorded actual question schedule for a manifest spec")
    for step in steps:
        for sid, table in answers[step].items():
            if not set(table) <= set(questions[sid]):
                raise ValueError(f"Unexpected answer question: {step}/{sid}")

    records = []
    for sid in specs:
        for qid, question in sorted(questions[sid].items()):
            requested = canonical_answer(question, question["expected_answer"])
            record = {"spec_id": sid, "question_id": qid, "question": question,
                      "family": question_scope(question).kind, "requested_answer": requested,
                      "checkpoints": {}}
            for step in steps:
                image_path = str(manifests[step][sid]["image_path"])
                verification = verified[step].get(image_path)
                if verification is not None and str(verification.get("spec_id", sid)) != sid:
                    raise ValueError(f"Verification spec does not match image: {step}/{sid}")
                fact = factual_answer(question, verification)
                truth = canonical_answer(question, fact.answer) if fact.known else None
                answer = answers[step].get(sid, {}).get(qid)
                status = _answer_status(observed[step].get(sid), answer)
                said = (canonical_answer(question, answer["normalized_answer"])
                        if status == "available" else None)
                record["checkpoints"][str(step)] = {
                    "image_path": image_path, "fact_known": fact.known,
                    "fact_reason": fact.reason, "fact_answer": truth,
                    "answer_status": status, "response": said,
                    "response_matches_request": status == "available" and said == requested,
                    "fact_matches_request": truth == requested if fact.known else None,
                    "response_matches_fact": (status == "available" and said == truth)
                    if fact.known else None,
                }
            record["retained_in_common_factual_cohort"] = all(
                point["fact_known"] for point in record["checkpoints"].values())
            records.append(record)

    groups = {}
    for family in GROUPS:
        selected = [r for r in records if family == "all" or r["family"] == family]
        retained = [r for r in selected if r["retained_in_common_factual_cohort"]]
        selected_specs = {r["spec_id"] for r in selected}
        retained_specs = {r["spec_id"] for r in retained}
        retained_counts = collections.Counter(r["spec_id"] for r in retained)
        total_counts = collections.Counter(r["spec_id"] for r in selected)
        unknown_union = collections.Counter()
        for record in selected:
            unknown_union.update({point["fact_reason"] for point in record["checkpoints"].values()
                                  if not point["fact_known"]})
        coverage, points = {}, {}
        for step in steps:
            key = str(step)
            all_points = [r["checkpoints"][key] for r in selected]
            unknown = [p for p in all_points if not p["fact_known"]]
            coverage[key] = {
                "requested_questions": len(selected), "known_questions": len(selected) - len(unknown),
                "unknown_questions": len(unknown),
                "known_fraction": (len(selected) - len(unknown)) / len(selected) if selected else None,
                "unknown_reasons": dict(collections.Counter(p["fact_reason"] for p in unknown)),
                "answer_status_all_requested": dict(collections.Counter(p["answer_status"] for p in all_points)),
                "images_with_any_known_question": len({r["spec_id"] for r in selected
                                                       if r["checkpoints"][key]["fact_known"]}),
            }
            images = []
            for sid in sorted(retained_specs):
                rows = [r["checkpoints"][key] for r in retained if r["spec_id"] == sid]
                counts = {metric: sum(p[metric] for p in rows) for metric in METRICS}
                images.append({"spec_id": sid, "image_path": str(manifests[step][sid]["image_path"]),
                               "n_questions": len(rows), "numerators": counts,
                               "means": {metric: count / len(rows) for metric, count in counts.items()},
                               "answer_status": dict(collections.Counter(p["answer_status"] for p in rows))})
            points[key] = {"n_images": len(images), "n_questions": len(retained),
                           "means": {metric: _mean([row["means"][metric] for row in images])
                                     for metric in METRICS},
                           "answer_status": dict(collections.Counter(
                               r["checkpoints"][key]["answer_status"] for r in retained)),
                           "per_image": images}
        groups[family] = {
            "cohort": {"requested_questions": len(selected), "retained_questions": len(retained),
                       "excluded_questions": len(selected) - len(retained),
                       "requested_image_slots": len(selected_specs),
                       "retained_image_slots": len(retained_specs),
                       "excluded_image_slots": sorted(selected_specs - retained_specs),
                       "partially_retained_image_slots": sorted(
                           sid for sid in retained_specs if retained_counts[sid] < total_counts[sid]),
                       "excluded_question_reason_union": dict(unknown_union),
                       "reason_union_note": "Unique questions per reason across checkpoints; reasons may overlap."},
            "checkpoint_coverage_only": coverage, "fixed_cohort_trajectories": points,
        }
    return {"steps": steps, "question_stability_validated": True,
            "question_fields_checked": list(QUESTION_FIELDS), "groups": groups, "records": records}


def factual_trajectory_report(arms: Mapping[str, Mapping[int, Mapping[str, Any]]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scope": "Descriptive atomic agreement on each arm's common factual question cohort",
        "metric_definitions": {
            "response_matches_request": "Response equals requested target (self-score on retained questions).",
            "fact_matches_request": "Known image fact equals requested target (external atomic agreement).",
            "response_matches_fact": "Response equals the known image fact (observation accuracy).",
        },
        "aggregation": "Within each family, average retained questions per image, then average images equally.",
        "limitations": [
            "Each arm uses its own all-checkpoint known-question intersection; arm means are not a common-cohort comparison.",
            "Facts are detector/crop-derived operational labels, not independent human truth.",
            "The intersection is a selected subset; checkpoint coverage is reported without comparing changing factual denominators.",
            "Missing, errored and abstained responses remain in the fixed denominator and count as nonmatches.",
            "Unknown facts are excluded by the common intersection and never filled with zero.",
            "These atomic quantities do not replace whole-image correctness, prove prompt leakage or establish D* or a lead.",
            "No confidence intervals or independent-question inference are computed.",
        ],
        "arms": {arm: arm_report(checkpoints) for arm, checkpoints in arms.items()},
    }


def build_report(run_dir: str | Path, *, arms: Sequence[str] = ("naive", "rfo_gold"),
                 steps: Sequence[int] = (0, 8, 16, 24)) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    paths = { (arm, step, name): run_dir / "evaluations" / arm / f"step-{step:05d}" / filename
              for arm in arms for step in steps for name, filename in
              (("manifest", "manifest.jsonl"), ("observations", "s_select.jsonl"),
               ("verifications", "verified.jsonl")) }
    split_path = run_dir / "split.json"

    def hashes() -> dict[str, str | None]:
        return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
                for path in [split_path, *paths.values()]}

    before = hashes()
    outcome = json.loads(split_path.read_text(encoding="utf-8"))["outcome"]
    if not outcome or len(set(outcome)) != len(outcome):
        raise ValueError("Frozen outcome IDs must be nonempty and unique")
    data = {arm: {step: {} for step in steps} for arm in arms}
    for (arm, step, name), path in paths.items():
        if name == "manifest" or path.exists():
            data[arm][step][name] = _read_jsonl(path)
        else:
            data[arm][step][name] = []
    for arm, checkpoints in data.items():
        for step, checkpoint in checkpoints.items():
            if {row["spec_id"] for row in checkpoint["manifest"]} != set(outcome):
                raise ValueError(f"Manifest does not match frozen outcome IDs: {arm}/{step}")
    report = factual_trajectory_report(data)
    if hashes() != before:
        raise ValueError("Input files changed while building the diagnostic")
    report["input_run"] = str(run_dir)
    report["frozen_outcome_ids"] = outcome
    report["input_sha256"] = before
    report["input_hashes_unchanged_after_analysis"] = True
    report["code_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in (Path(__file__).resolve(),
                                          PROJECT / "src/selfsight/v4/factual_truth.py")}
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, help="New JSON file outside all run directories")
    args = parser.parse_args(argv)
    if args.output is not None:
        output = args.output.resolve()
        if any(output.is_relative_to(path) for path in (args.run_dir.resolve(), PROJECT / "runs")):
            parser.error("--output must be outside run directories")
        if output.exists():
            parser.error("--output must be a new file")
    report = build_report(args.run_dir)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
