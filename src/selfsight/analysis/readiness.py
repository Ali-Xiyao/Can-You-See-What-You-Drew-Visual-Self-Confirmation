"""Fail-closed Gate -2 Joint Generate–Observe Readiness decisions."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from selfsight.utils.hashing import sha256_file
from selfsight.utils.jsonl import atomic_write_json


def _read_json(path: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON: {resolved}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object: {resolved}")
    return resolved, value


def _read_yaml(path: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).resolve()
    value = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a YAML mapping: {resolved}")
    return resolved, value


def _fraction(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must be numeric") from error
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{label} must lie in [0, 1], got {parsed}")
    return parsed


def _points(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must be numeric") from error
    if not 0.0 <= parsed <= 100.0:
        raise ValueError(f"{label} must lie in [0, 100], got {parsed}")
    return parsed


def _mapping(report: Mapping[str, Any], key: str, label: str) -> Mapping[str, Any]:
    value = report.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} is missing mapping {key!r}")
    return value


def _require_identity(report: Mapping[str, Any], *, model_id: str, revision: str, label: str) -> None:
    if report.get("model_id") != model_id or report.get("revision") != revision:
        raise RuntimeError(
            f"{label} identity mismatch: expected {model_id}@{revision}, got "
            f"{report.get('model_id')}@{report.get('revision')}"
        )


def _require_runtime_lock(
    report: Mapping[str, Any],
    *,
    source_revision: str,
    dependency_revisions: Mapping[str, str],
    label: str,
) -> None:
    if report.get("source_revision") != source_revision:
        raise RuntimeError(
            f"{label} source revision mismatch: expected {source_revision}, "
            f"got {report.get('source_revision')}"
        )
    actual = report.get("dependency_revisions")
    if not isinstance(actual, Mapping) or dict(actual) != dict(dependency_revisions):
        raise RuntimeError(f"{label} dependency revision mismatch")


def _evidence(path: Path, label: str) -> dict[str, Any]:
    return {"label": label, "path": str(path), "sha256": sha256_file(path)}


def _validate_predecessor(
    backbone: Mapping[str, Any], predecessor_path: str | Path | None
) -> dict[str, Any] | None:
    rank = int(backbone["candidate_rank"])
    if rank == 1:
        if predecessor_path is not None:
            raise ValueError("Candidate rank 1 must not provide a predecessor decision")
        return None
    if predecessor_path is None:
        raise RuntimeError("Fallback candidates require a hashed predecessor decision")
    path, predecessor = _read_json(predecessor_path, "predecessor decision")
    if predecessor.get("gate") != "minus_2_joint_readiness":
        raise RuntimeError("Fallback predecessor is not a Gate -2 decision")
    if bool(predecessor.get("passed")):
        raise RuntimeError("A passed predecessor does not authorize a fallback candidate")
    expected = _mapping(predecessor, "fallback", "predecessor decision").get("next_model_id")
    if expected != backbone["backbone_id"]:
        raise RuntimeError(
            f"Predecessor authorizes {expected!r}, not {backbone['backbone_id']!r}"
        )
    return {**_evidence(path, "predecessor"), "model_id": predecessor.get("model_id")}


def finalize_joint_readiness(
    *,
    backbone_config_path: str | Path,
    readiness_config_path: str | Path,
    canary_report_path: str | Path,
    reference_report_path: str | Path,
    generated_report_path: str | Path,
    human_report_path: str | Path,
    lora_report_path: str | Path,
    output_path: str | Path | None = None,
    predecessor_path: str | Path | None = None,
) -> dict[str, Any]:
    """Finalize Gate -2 and bind every threshold, identity, input report, and family."""

    backbone_path, backbone = _read_yaml(backbone_config_path, "backbone config")
    readiness_path, readiness = _read_yaml(readiness_config_path, "readiness config")
    canary_path, canary = _read_json(canary_report_path, "A1 canary report")
    reference_path, reference = _read_json(reference_report_path, "A2 reference report")
    generated_path, generated = _read_json(generated_report_path, "A3 generated report")
    human_path, human = _read_json(human_report_path, "A3 blind-human report")
    lora_path, lora = _read_json(lora_report_path, "A4 LoRA report")
    model_id = str(backbone["backbone_id"])
    revision = str(backbone["revision"])
    source_revision = str(backbone["source"]["revision"])
    dependency_revisions = {
        str(key): str(value) for key, value in backbone["dependencies"].items()
    }
    for label, report in (
        ("A1 canary", canary),
        ("A2 reference", reference),
        ("A3 generated", generated),
        ("A3 blind-human", human),
        ("A4 LoRA", lora),
    ):
        _require_identity(report, model_id=model_id, revision=revision, label=label)
        _require_runtime_lock(
            report,
            source_revision=source_revision,
            dependency_revisions=dependency_revisions,
            label=label,
        )

    main_families = tuple(str(item) for item in readiness["main_families"])
    thresholds = readiness["thresholds"]
    reference_thresholds = thresholds["reference"]
    generated_thresholds = thresholds["generated"]
    joint_thresholds = thresholds["joint"]

    reference_accuracy = _mapping(reference, "family_open_accuracy", "A2 reference")
    missing_reference = sorted(set(main_families).difference(reference_accuracy))
    if missing_reference:
        raise RuntimeError(f"A2 reference report is missing families: {missing_reference}")
    reference_min = _fraction(
        reference_thresholds["family_open_accuracy_min"], "reference family minimum"
    )
    observation_families = tuple(
        family
        for family in main_families
        if _fraction(reference_accuracy[family], f"A2 {family} accuracy") >= reference_min
    )
    yes_bias_points = _points(reference["absolute_yes_bias_points"], "A2 yes bias")
    repeat_agreement = _fraction(reference["repeat_agreement"], "A2 repeat agreement")
    abstain_rate = _fraction(reference["abstain_rate"], "A2 abstain rate")
    gate_b_checks = {
        "family_open_accuracy": len(observation_families)
        >= int(reference_thresholds["families_passing_min"]),
        "yes_bias": yes_bias_points
        <= float(reference_thresholds["yes_bias_points_max"]),
        "repeat_agreement": repeat_agreement
        >= float(reference_thresholds["repeat_agreement_min"]),
        "abstain_rate": abstain_rate <= float(reference_thresholds["abstain_rate_max"]),
    }

    family_coverage = _mapping(generated, "family_coverage", "A3 generated")
    family_oracle = _mapping(generated, "family_oracle_at_4", "A3 generated")
    family_precision = _mapping(human, "family_precision", "A3 blind-human")
    for label, values in (
        ("coverage", family_coverage),
        ("Oracle@4", family_oracle),
        ("precision", family_precision),
    ):
        missing = sorted(set(main_families).difference(values))
        if missing:
            raise RuntimeError(f"A3 {label} report is missing families: {missing}")
    overall_coverage = _fraction(generated["overall_coverage"], "A3 overall coverage")
    overall_oracle = _fraction(generated["overall_oracle_at_4"], "A3 overall Oracle@4")
    coverage_swing = _points(
        generated["fixed_seed_coverage_swing_points"], "A3 fixed-seed coverage swing"
    )
    overall_precision = _fraction(human["overall_precision"], "A3 overall precision")
    generated_retained = tuple(
        family
        for family in main_families
        if _fraction(family_coverage[family], f"A3 {family} coverage")
        >= float(generated_thresholds["family_coverage_min"])
    )
    gate_c_checks = {
        "blind_verifier_precision": overall_precision
        >= float(generated_thresholds["verifier_precision_min"]),
        "overall_primary_answer_coverage": overall_coverage
        >= float(generated_thresholds["overall_coverage_min"]),
        "retained_family_coverage": all(
            family in generated_retained for family in observation_families
        ),
        "oracle_at_4": overall_oracle >= float(generated_thresholds["oracle_at_4_min"]),
        "fixed_seed_coverage_swing": coverage_swing
        <= float(generated_thresholds["fixed_seed_coverage_swing_points_max"]),
    }

    joint_families = tuple(
        family
        for family in main_families
        if _fraction(reference_accuracy[family], f"joint {family} observation") >= reference_min
        and _fraction(family_coverage[family], f"joint {family} coverage")
        >= float(generated_thresholds["family_coverage_min"])
        and _fraction(family_precision[family], f"joint {family} precision")
        >= float(generated_thresholds["verifier_precision_min"])
        and _fraction(family_oracle[family], f"joint {family} Oracle@4")
        >= float(generated_thresholds["oracle_at_4_min"])
    )
    gate_a_checks = {
        "same_checkpoint_generate_and_observe": bool(canary.get("passed")),
        "lora_backward_and_resume": bool(lora.get("passed")),
        "frozen_step0_supported": bool(lora.get("frozen_step0_supported")),
    }
    gate_d_checks = {
        "joint_eligible_families": len(joint_families)
        >= int(joint_thresholds["families_passing_min"])
    }
    gate_checks = {
        "minus_2a_unified_functionality": all(gate_a_checks.values()),
        "minus_2b_reference_observation": all(gate_b_checks.values()),
        "minus_2c_generated_measurability": all(gate_c_checks.values()),
        "minus_2d_joint_families": all(gate_d_checks.values()),
    }
    passed = all(gate_checks.values())
    predecessor = _validate_predecessor(backbone, predecessor_path)
    fallback_model = backbone.get("fallback", {}).get("on_failure")
    report = {
        "schema_version": 2,
        "gate": "minus_2_joint_readiness",
        "model_id": model_id,
        "revision": revision,
        "candidate_rank": int(backbone["candidate_rank"]),
        "native_resolution": int(backbone["official_profile"]["resolution"]),
        "source": dict(backbone["source"]),
        "dependency_revisions": dependency_revisions,
        "passed": passed,
        "checks": gate_checks,
        "subchecks": {
            "minus_2a": gate_a_checks,
            "minus_2b": gate_b_checks,
            "minus_2c": gate_c_checks,
            "minus_2d": gate_d_checks,
        },
        "observation_eligible_families": list(observation_families),
        "generated_retained_families": list(generated_retained),
        "selected_eligible_families": list(joint_families),
        "metrics": {
            "family_open_accuracy": dict(reference_accuracy),
            "absolute_yes_bias_points": yes_bias_points,
            "repeat_agreement": repeat_agreement,
            "abstain_rate": abstain_rate,
            "overall_coverage": overall_coverage,
            "family_coverage": dict(family_coverage),
            "overall_oracle_at_4": overall_oracle,
            "family_oracle_at_4": dict(family_oracle),
            "overall_precision": overall_precision,
            "family_precision": dict(family_precision),
            "fixed_seed_coverage_swing_points": coverage_swing,
        },
        "thresholds": thresholds,
        "evidence": {
            "backbone_config": _evidence(backbone_path, "backbone_config"),
            "readiness_config": _evidence(readiness_path, "readiness_config"),
            "canary": _evidence(canary_path, "canary"),
            "reference": _evidence(reference_path, "reference"),
            "generated": _evidence(generated_path, "generated"),
            "human": _evidence(human_path, "human"),
            "lora": _evidence(lora_path, "lora"),
            "predecessor": predecessor,
        },
        "fallback": {
            "next_model_id": None if passed else fallback_model,
            "action": (
                "Run family-restricted E1 and E2."
                if passed
                else (
                    f"Stop phenomenon work and audit fallback {fallback_model}."
                    if fallback_model
                    else "Stop phenomenon work and report the conditional negative result."
                )
            ),
        },
    }
    if output_path is not None:
        if Path(output_path).exists():
            raise FileExistsError(f"Refusing to overwrite Gate -2 decision: {output_path}")
        atomic_write_json(output_path, report)
    return report


def require_joint_readiness(path: str | Path) -> dict[str, Any]:
    """Validate a green Gate -2 decision and all of its hashed local evidence."""

    decision_path, report = _read_json(path, "Gate -2 decision")
    del decision_path
    if report.get("gate") != "minus_2_joint_readiness":
        raise RuntimeError("The supplied report is not a Gate -2 decision")
    checks = _mapping(report, "checks", "Gate -2 decision")
    calculated = bool(checks) and all(bool(value) for value in checks.values())
    if bool(report.get("passed")) != calculated:
        raise RuntimeError("Gate -2 decision is internally inconsistent")
    eligible = report.get("selected_eligible_families")
    if not isinstance(eligible, list) or len(set(eligible)) < 4:
        raise RuntimeError("Gate -2 has fewer than four unique eligible families")
    if not calculated:
        raise RuntimeError("Gate -2 is red; E1, Gate -1b, and E2 are forbidden")
    evidence = _mapping(report, "evidence", "Gate -2 decision")
    for label, record in evidence.items():
        if label == "predecessor" and record is None:
            continue
        if not isinstance(record, Mapping):
            raise TypeError(f"Gate -2 evidence {label!r} is malformed")
        evidence_path = Path(str(record.get("path", ""))).resolve()
        expected = record.get("sha256")
        if not evidence_path.is_file() or not isinstance(expected, str):
            raise RuntimeError(f"Gate -2 evidence {label!r} is unavailable")
        if sha256_file(evidence_path) != expected:
            raise RuntimeError(f"Gate -2 evidence SHA-256 mismatch: {label}")
    return report
