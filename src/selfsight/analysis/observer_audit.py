"""Gate -1 capability, option-order, and yes-bias audits."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from selfsight.data.subsets import stable_stratified_sample
from selfsight.observers.client import ObserverServiceClient
from selfsight.rfo.isolation import make_blind_request
from selfsight.schemas import AtomicQuestion, QuestionFormat
from selfsight.utils.hashing import sha256_file
from selfsight.utils.jsonl import atomic_write_json, read_jsonl


def audit_observer_manifest(
    client: ObserverServiceClient,
    manifest_path: str | Path,
    *,
    limit: int | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    rows = []
    family_open: dict[str, list[bool]] = defaultdict(list)
    forced_binary_expected = []
    forced_binary_predicted = []
    order_pairs = []
    records = list(read_jsonl(manifest_path))
    if limit is not None:
        records = stable_stratified_sample(
            records,
            limit,
            stratum=lambda record: str(record["atom"]["family"]),
            item_id=lambda record: str(record["scene"]["scene_id"]),
            seed=20260827,
        )
    for index, record in enumerate(records):
        questions = tuple(AtomicQuestion.from_dict(item) for item in record["questions"])
        request = make_blind_request(record["reference_image"], questions, f"gate-1-{index:06d}")
        result = client.observe(request)
        by_id = {answer.question_id: answer for answer in result.answers}
        normalized_forced = []
        for question in questions:
            answer = by_id[question.question_id]
            correct = answer.normalized_answer == question.expected_answer
            rows.append(
                {
                    "scene_id": record["scene"]["scene_id"],
                    "family": question.family.value,
                    "format": question.question_format.value,
                    "choice_order_seed": question.choice_order_seed,
                    "expected": question.expected_answer,
                    "predicted": answer.normalized_answer,
                    "abstain": answer.abstain,
                    "correct": correct,
                }
            )
            if question.question_format == QuestionFormat.OPEN:
                family_open[question.family.value].append(correct)
            elif question.question_format == QuestionFormat.FORCED_CHOICE:
                normalized_forced.append(answer.normalized_answer)
                if question.expected_answer in {"yes", "no"}:
                    forced_binary_expected.append(question.expected_answer)
                    forced_binary_predicted.append(answer.normalized_answer)
        if len(normalized_forced) == 2:
            order_pairs.append(normalized_forced[0] == normalized_forced[1])
    if not rows:
        raise ValueError("Observer audit received no records")
    family_accuracy = {key: float(np.mean(values)) for key, values in family_open.items()}
    macro = float(np.mean(list(family_accuracy.values())))
    expected_yes_rate = (
        float(np.mean([item == "yes" for item in forced_binary_expected]))
        if forced_binary_expected
        else float("nan")
    )
    predicted_yes_rate = (
        float(np.mean([item == "yes" for item in forced_binary_predicted]))
        if forced_binary_predicted
        else float("nan")
    )
    passing_families = [key for key, value in family_accuracy.items() if value >= 0.80]
    abstain_rate = float(np.mean([row["abstain"] for row in rows]))
    report = {
        "schema_version": 1,
        "observer_id": result.observer_id,
        "observer_revision": result.observer_revision,
        "records": len(rows),
        "images": len(rows) // 3,
        "family_open_accuracy": family_accuracy,
        "macro_open_accuracy": macro,
        "passing_families_at_80pct": passing_families,
        "abstain_rate": abstain_rate,
        "expected_yes_rate": expected_yes_rate,
        "predicted_yes_rate": predicted_yes_rate,
        "absolute_yes_bias": abs(predicted_yes_rate - expected_yes_rate),
        "forced_choice_order_agreement": float(np.mean(order_pairs)) if order_pairs else float("nan"),
        "gate_minus_1_capability_pass": len(passing_families) >= 4,
        "gate_minus_1_bias_pass": abs(predicted_yes_rate - expected_yes_rate) <= 0.10,
        "rows": rows,
    }
    report["gate_minus_1_pass"] = bool(
        report["gate_minus_1_capability_pass"] and report["gate_minus_1_bias_pass"]
    )
    if output_path is not None:
        atomic_write_json(output_path, report)
    return report


def choose_matched_observer(
    showo_macro_accuracy: float,
    candidate_reports: Mapping[str, Mapping[str, Any]],
    *,
    delta_max: float = 0.03,
) -> dict[str, Any]:
    eligible = []
    for observer_id, report in candidate_reports.items():
        delta = abs(float(report["macro_open_accuracy"]) - showo_macro_accuracy)
        if bool(report.get("gate_minus_1_capability_pass")) and delta <= delta_max:
            eligible.append((delta, observer_id, report))
    if not eligible:
        return {
            "matched": False,
            "observer_id": None,
            "reason": f"No capable observer lies within {delta_max:.3f} macro accuracy",
        }
    delta, observer_id, report = min(eligible, key=lambda item: (item[0], item[1]))
    return {
        "matched": True,
        "observer_id": observer_id,
        "delta": delta,
        "observer_revision": report["observer_revision"],
    }


def finalize_gate_minus_1(
    *,
    reference_audit_path: str | Path,
    showo_report_path: str | Path,
    candidate_report_paths: list[str | Path],
    output_path: str | Path | None = None,
    delta_max: float = 0.03,
) -> dict[str, Any]:
    reference_path = Path(reference_audit_path).resolve()
    showo_path = Path(showo_report_path).resolve()
    candidate_paths = [Path(path).resolve() for path in candidate_report_paths]
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    showo = json.loads(showo_path.read_text(encoding="utf-8"))
    candidates = {}
    candidate_evidence = []
    for path in candidate_paths:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        observer_id = str(report["observer_id"])
        if observer_id in candidates:
            raise ValueError(f"Duplicate Gate -1 candidate report: {observer_id}")
        candidates[observer_id] = report
        candidate_evidence.append(
            {"path": str(path), "sha256": sha256_file(path), "observer_id": observer_id}
        )
    match = choose_matched_observer(
        float(showo["macro_open_accuracy"]), candidates, delta_max=delta_max
    )
    conditions = {
        "reference_verifier": bool(reference.get("gate_reference_pass")),
        "showo_capability_floor": bool(showo.get("gate_minus_1_capability_pass")),
        "showo_bias_control": bool(showo.get("gate_minus_1_bias_pass")),
        "matched_heterogeneous_observer": bool(match.get("matched")),
    }
    if match.get("matched"):
        selected = candidates[str(match["observer_id"])]
        conditions["matched_observer_capability_floor"] = bool(
            selected.get("gate_minus_1_capability_pass")
        )
        conditions["matched_observer_bias_control"] = bool(
            selected.get("gate_minus_1_bias_pass")
        )
    else:
        conditions["matched_observer_capability_floor"] = False
        conditions["matched_observer_bias_control"] = False
    report = {
        "schema_version": 1,
        "gate": "minus_1_pre_e1",
        "passed": all(conditions.values()),
        "conditions": conditions,
        "showo_observer_id": showo.get("observer_id"),
        "showo_revision": showo.get("observer_revision"),
        "showo_macro_open_accuracy": showo.get("macro_open_accuracy"),
        "matched_observer": match,
        "candidate_macro_accuracies": {
            key: value.get("macro_open_accuracy") for key, value in candidates.items()
        },
        "criteria": {"matched_observer_delta_max": delta_max},
        "evidence_reports": {
            "reference": {"path": str(reference_path), "sha256": sha256_file(reference_path)},
            "showo": {"path": str(showo_path), "sha256": sha256_file(showo_path)},
            "candidates": candidate_evidence,
        },
        "next_action": (
            "Run E1 and Gate -1b before the local E2 pilot."
            if all(conditions.values())
            else "Stop phenomenon experiments and follow the registered capability-floor fallback."
        ),
    }
    if output_path is not None:
        atomic_write_json(output_path, report)
    return report
