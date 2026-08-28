"""Fail-closed Gate -2 Joint Generate–Observe Readiness decisions."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from selfsight.data.readiness_precision import score_generated_precision_audit
from selfsight.utils.hashing import rgb_sha256, sha256_file
from selfsight.utils.jsonl import atomic_write_json, read_jsonl


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


def _require_identity(
    report: Mapping[str, Any], *, model_id: str, revision: str, label: str
) -> None:
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


def validate_human_precision_artifacts(human: Mapping[str, Any]) -> None:
    """Recursively validate the exact blind-review sheet and hidden answer key."""

    for field, hash_field, label in (
        ("review_csv", "review_csv_sha256", "blind review CSV"),
        ("answer_key", "answer_key_sha256", "blind review answer key"),
    ):
        path = Path(str(human.get(field, ""))).resolve()
        expected = human.get(hash_field)
        if not path.is_file() or not isinstance(expected, str):
            raise RuntimeError(f"{label} evidence is unavailable")
        if sha256_file(path) != expected:
            raise RuntimeError(f"{label} SHA-256 mismatch")


def validate_human_precision_report(
    human: Mapping[str, Any], *, families: list[str], threshold: float
) -> None:
    """Re-score immutable blind annotations and require an exact report match."""

    validate_human_precision_artifacts(human)
    recomputed = score_generated_precision_audit(
        str(human["review_csv"]),
        str(human["answer_key"]),
        families=families,
        threshold=threshold,
    )
    if dict(human) != recomputed:
        raise RuntimeError("A3 human report differs from a fresh score of its bound evidence")


def validate_generated_artifacts(generated: Mapping[str, Any]) -> dict[str, int]:
    """Recompute A3 row/image integrity instead of trusting summary counters."""

    rows_path = Path(str(generated.get("rows", ""))).resolve()
    expected_rows_hash = generated.get("rows_sha256")
    if not rows_path.is_file() or not isinstance(expected_rows_hash, str):
        raise RuntimeError("A3 generated rows evidence is unavailable")
    if sha256_file(rows_path) != expected_rows_hash:
        raise RuntimeError("A3 generated rows SHA-256 mismatch")
    rows = list(read_jsonl(rows_path))
    candidates = int(generated.get("candidates", -1))
    candidate_ids = {str(row.get("candidate_id", "")) for row in rows}
    image_paths = {str(Path(str(row.get("image_path", ""))).resolve()) for row in rows}
    if (
        candidates <= 0
        or len(rows) != candidates
        or len(candidate_ids) != candidates
        or len(image_paths) != candidates
        or "" in candidate_ids
    ):
        raise RuntimeError("A3 decision requires collision-free candidate artifacts")
    for row in rows:
        image_path = Path(str(row.get("image_path", ""))).resolve()
        expected_rgb_hash = row.get("rgb_sha256")
        if not image_path.is_file() or not isinstance(expected_rgb_hash, str):
            raise RuntimeError(f"A3 candidate image is unavailable: {image_path}")
        if rgb_sha256(image_path) != expected_rgb_hash:
            raise RuntimeError(f"A3 candidate RGB SHA-256 mismatch: {image_path}")
    recorded_ids = int(generated.get("unique_candidate_ids", -1))
    recorded_paths = int(generated.get("unique_image_paths", -1))
    if recorded_ids != len(candidate_ids) or recorded_paths != len(image_paths):
        raise RuntimeError(
            "A3 recorded artifact counts are inconsistent with collision-free candidates"
        )
    return {
        "candidates": candidates,
        "unique_candidate_ids": len(candidate_ids),
        "unique_image_paths": len(image_paths),
    }


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
    checks = _mapping(predecessor, "checks", "predecessor decision")
    if not checks or all(bool(value) for value in checks.values()):
        raise RuntimeError("Fallback predecessor red status is internally inconsistent")
    if int(predecessor.get("candidate_rank", -1)) != rank - 1:
        raise RuntimeError("Fallback predecessor candidate rank is not immediately prior")
    expected = _mapping(predecessor, "fallback", "predecessor decision").get("next_model_id")
    if expected != backbone["backbone_id"]:
        raise RuntimeError(f"Predecessor authorizes {expected!r}, not {backbone['backbone_id']!r}")
    evidence = _mapping(predecessor, "evidence", "predecessor decision")
    required_evidence = {
        "backbone_config",
        "readiness_config",
        "canary",
        "reference",
        "generated",
        "human",
        "lora",
        "predecessor",
    }
    if set(evidence) != required_evidence:
        raise RuntimeError("Fallback predecessor evidence set is incomplete")
    for label, record in evidence.items():
        if record is None:
            if label not in {"human", "lora", "predecessor"}:
                raise RuntimeError(f"Unexpected missing predecessor evidence: {label}")
            continue
        if not isinstance(record, Mapping):
            raise TypeError(f"Malformed predecessor evidence: {label}")
        evidence_path = Path(str(record.get("path", ""))).resolve()
        evidence_hash = record.get("sha256")
        if not evidence_path.is_file() or not isinstance(evidence_hash, str):
            raise RuntimeError(f"Unavailable predecessor evidence: {label}")
        if sha256_file(evidence_path) != evidence_hash:
            raise RuntimeError(f"Predecessor evidence SHA-256 mismatch: {label}")
    decision_mode = predecessor.get("decision_mode")
    if decision_mode == "upstream_stop_before_human_and_a4":
        skipped = set(predecessor.get("skipped_by_stop_rule", ()))
        if skipped != {"blind_human_precision", "a4_lora_backward_resume"}:
            raise RuntimeError("Upstream-stop predecessor has an invalid skipped-evidence contract")
        if evidence.get("human") is not None or evidence.get("lora") is not None:
            raise RuntimeError("Upstream-stop predecessor must not contain human/A4 evidence")
    elif decision_mode == "stop_after_human_before_a4":
        skipped = set(predecessor.get("skipped_by_stop_rule", ()))
        if skipped != {"a4_lora_backward_resume"}:
            raise RuntimeError("Human-stop predecessor has an invalid skipped-evidence contract")
        if evidence.get("human") is None or evidence.get("lora") is not None:
            raise RuntimeError("Human-stop predecessor must contain human but not A4 evidence")
    elif evidence.get("human") is None or evidence.get("lora") is None:
        raise RuntimeError("Completed predecessor is missing human/A4 evidence")
    if evidence.get("human") is not None:
        _, human = _read_json(str(evidence["human"]["path"]), "predecessor human report")
        if decision_mode == "stop_after_human_before_a4" or "review_csv" in human:
            _, predecessor_readiness = _read_yaml(
                str(evidence["readiness_config"]["path"]),
                "predecessor readiness config",
            )
            validate_human_precision_report(
                human,
                families=[str(item) for item in predecessor_readiness["main_families"]],
                threshold=float(
                    predecessor_readiness["thresholds"]["generated"]["verifier_precision_min"]
                ),
            )
    return {**_evidence(path, "predecessor"), "model_id": predecessor.get("model_id")}


def finalize_joint_readiness_stop(
    *,
    backbone_config_path: str | Path,
    readiness_config_path: str | Path,
    canary_report_path: str | Path,
    reference_report_path: str | Path,
    generated_report_path: str | Path,
    human_report_path: str | Path | None = None,
    output_path: str | Path | None = None,
    predecessor_path: str | Path | None = None,
) -> dict[str, Any]:
    """Freeze an A3 failure without fabricating evidence after the failed check."""

    backbone_path, backbone = _read_yaml(backbone_config_path, "backbone config")
    readiness_path, readiness = _read_yaml(readiness_config_path, "readiness config")
    canary_path, canary = _read_json(canary_report_path, "A1 canary report")
    reference_path, reference = _read_json(reference_report_path, "A2 reference report")
    generated_path, generated = _read_json(generated_report_path, "A3 generated report")
    human_path: Path | None = None
    human: dict[str, Any] | None = None
    if human_report_path is not None:
        human_path, human = _read_json(human_report_path, "A3 blind-human report")
    model_id = str(backbone["backbone_id"])
    revision = str(backbone["revision"])
    source_revision = str(backbone["source"]["revision"])
    dependencies = {str(key): str(value) for key, value in backbone["dependencies"].items()}
    identity_reports = [
        ("A1 canary", canary),
        ("A2 reference", reference),
        ("A3 generated", generated),
    ]
    if human is not None:
        identity_reports.append(("A3 blind-human", human))
    for label, report in identity_reports:
        _require_identity(report, model_id=model_id, revision=revision, label=label)
        _require_runtime_lock(
            report,
            source_revision=source_revision,
            dependency_revisions=dependencies,
            label=label,
        )
    artifact_counts = validate_generated_artifacts(generated)
    main_families = tuple(str(item) for item in readiness["main_families"])
    thresholds = readiness["thresholds"]
    reference_thresholds = thresholds["reference"]
    generated_thresholds = thresholds["generated"]
    reference_accuracy = _mapping(reference, "family_open_accuracy", "A2 reference")
    family_coverage = _mapping(generated, "family_coverage", "A3 generated")
    family_oracle = _mapping(generated, "family_oracle_at_4", "A3 generated")
    for label, values in (
        ("reference accuracy", reference_accuracy),
        ("generated coverage", family_coverage),
        ("generated Oracle@4", family_oracle),
    ):
        missing = sorted(set(main_families).difference(values))
        if missing:
            raise RuntimeError(f"{label} is missing families: {missing}")
    reference_min = float(reference_thresholds["family_open_accuracy_min"])
    observation_families = tuple(
        family
        for family in main_families
        if _fraction(reference_accuracy[family], f"A2 {family} accuracy") >= reference_min
    )
    gate_b = {
        "family_open_accuracy": len(observation_families)
        >= int(reference_thresholds["families_passing_min"]),
        "yes_bias": _points(reference["absolute_yes_bias_points"], "A2 yes bias")
        <= float(reference_thresholds["yes_bias_points_max"]),
        "repeat_agreement": _fraction(reference["repeat_agreement"], "A2 repeat")
        >= float(reference_thresholds["repeat_agreement_min"]),
        "abstain_rate": _fraction(reference["abstain_rate"], "A2 abstain")
        <= float(reference_thresholds["abstain_rate_max"]),
    }
    overall_coverage = _fraction(generated["overall_coverage"], "A3 overall coverage")
    overall_oracle = _fraction(generated["overall_oracle_at_4"], "A3 overall Oracle@4")
    swing = _points(generated["fixed_seed_coverage_swing_points"], "A3 fixed-seed coverage swing")
    generated_retained = tuple(
        family
        for family in main_families
        if _fraction(family_coverage[family], f"A3 {family} coverage")
        >= float(generated_thresholds["family_coverage_min"])
    )
    automatic_c = {
        "overall_primary_answer_coverage": overall_coverage
        >= float(generated_thresholds["overall_coverage_min"]),
        "retained_family_coverage": all(
            family in generated_retained for family in observation_families
        ),
        "oracle_at_4": overall_oracle >= float(generated_thresholds["oracle_at_4_min"]),
        "fixed_seed_coverage_swing": swing
        <= float(generated_thresholds["fixed_seed_coverage_swing_points_max"]),
    }
    recorded_checks = _mapping(generated, "checks", "A3 generated")
    if any(bool(recorded_checks.get(key)) != value for key, value in automatic_c.items()):
        raise RuntimeError("A3 automatic checks are internally inconsistent")
    automatic_pass = all(automatic_c.values())
    if bool(generated.get("passed_without_human_precision")) != automatic_pass:
        raise RuntimeError("A3 automatic pass flag is internally inconsistent")
    family_precision: Mapping[str, Any] | None = None
    overall_precision: float | None = None
    if human is None:
        if automatic_pass:
            raise RuntimeError("A3 automatic Gate is green; blind-human audit and A4 are required")
        decision_mode = "upstream_stop_before_human_and_a4"
        skipped = ["blind_human_precision", "a4_lora_backward_resume"]
    else:
        if not automatic_pass:
            raise RuntimeError("A3 automatic Gate is red; use the upstream stop path")
        validate_human_precision_report(
            human,
            families=list(main_families),
            threshold=float(generated_thresholds["verifier_precision_min"]),
        )
        if human.get("blind") is not True:
            raise RuntimeError("A3 human report is not marked as blinded")
        required = int(human.get("required_annotations", -1))
        complete = int(human.get("complete_annotations", -1))
        incomplete = human.get("incomplete_ids")
        if required <= 0 or complete != required or incomplete != []:
            raise RuntimeError("A3 human report is incomplete")
        recorded_threshold = _fraction(human.get("threshold"), "A3 human threshold")
        precision_threshold = float(generated_thresholds["verifier_precision_min"])
        if recorded_threshold != precision_threshold:
            raise RuntimeError("A3 human precision threshold differs from the readiness config")
        family_precision = _mapping(human, "family_precision", "A3 blind-human")
        audited_counts = _mapping(human, "family_audited_counts", "A3 blind-human")
        missing_precision = sorted(set(main_families).difference(family_precision))
        missing_counts = sorted(set(main_families).difference(audited_counts))
        if missing_precision or missing_counts:
            raise RuntimeError("A3 human report is missing registered family evidence")
        if any(int(audited_counts[family]) <= 0 for family in main_families):
            raise RuntimeError("A3 human report has an unaudited family")
        overall_precision = _fraction(human["overall_precision"], "A3 overall precision")
        calculated_human_pass = overall_precision >= precision_threshold
        if bool(human.get("passed")) != calculated_human_pass:
            raise RuntimeError("A3 human pass flag is internally inconsistent")
        if calculated_human_pass:
            raise RuntimeError("A3 human Gate is green; A4 is required")
        decision_mode = "stop_after_human_before_a4"
        skipped = ["a4_lora_backward_resume"]
    predecessor = _validate_predecessor(backbone, predecessor_path)
    fallback_model = backbone.get("fallback", {}).get("on_failure")
    report = {
        "schema_version": 2,
        "gate": "minus_2_joint_readiness",
        "decision_mode": decision_mode,
        "model_id": model_id,
        "revision": revision,
        "candidate_rank": int(backbone["candidate_rank"]),
        "native_resolution": int(backbone["official_profile"]["resolution"]),
        "source": dict(backbone["source"]),
        "dependency_revisions": dependencies,
        "passed": False,
        "checks": {
            "minus_2a_unified_functionality": False,
            "minus_2b_reference_observation": all(gate_b.values()),
            "minus_2c_generated_measurability": False,
            "minus_2d_joint_families": False,
        },
        "subchecks": {
            "minus_2a": {
                "same_checkpoint_generate_and_observe": bool(canary.get("passed")),
                "lora_backward_and_resume": False,
                "frozen_step0_supported": False,
            },
            "minus_2b": gate_b,
            "minus_2c": {"blind_verifier_precision": False, **automatic_c},
            "minus_2d": {"joint_eligible_families": False},
        },
        "skipped_by_stop_rule": skipped,
        "observation_eligible_families": list(observation_families),
        "generated_retained_families": list(generated_retained),
        "selected_eligible_families": [],
        "metrics": {
            "family_open_accuracy": dict(reference_accuracy),
            "family_coverage": dict(family_coverage),
            "family_oracle_at_4": dict(family_oracle),
            "overall_coverage": overall_coverage,
            "overall_oracle_at_4": overall_oracle,
            "fixed_seed_coverage_swing_points": swing,
            "overall_precision": overall_precision,
            "family_precision": None if family_precision is None else dict(family_precision),
            "artifact_counts": artifact_counts,
        },
        "thresholds": thresholds,
        "evidence": {
            "backbone_config": _evidence(backbone_path, "backbone_config"),
            "readiness_config": _evidence(readiness_path, "readiness_config"),
            "canary": _evidence(canary_path, "canary"),
            "reference": _evidence(reference_path, "reference"),
            "generated": _evidence(generated_path, "generated"),
            "human": None if human_path is None else _evidence(human_path, "human"),
            "lora": None,
            "predecessor": predecessor,
        },
        "fallback": {
            "next_model_id": fallback_model,
            "action": (
                f"Stop phenomenon work and audit fallback {fallback_model}."
                if fallback_model
                else "Stop phenomenon work and report the conditional negative result."
            ),
        },
    }
    if output_path is not None:
        if Path(output_path).exists():
            raise FileExistsError(f"Refusing to overwrite Gate -2 decision: {output_path}")
        atomic_write_json(output_path, report)
    return report


def finalize_joint_readiness_stop_after_human(
    *,
    backbone_config_path: str | Path,
    readiness_config_path: str | Path,
    canary_report_path: str | Path,
    reference_report_path: str | Path,
    generated_report_path: str | Path,
    human_report_path: str | Path,
    output_path: str | Path | None = None,
    predecessor_path: str | Path | None = None,
) -> dict[str, Any]:
    """Freeze a complete blind-human A3 failure and skip the now-irrelevant A4."""

    return finalize_joint_readiness_stop(
        backbone_config_path=backbone_config_path,
        readiness_config_path=readiness_config_path,
        canary_report_path=canary_report_path,
        reference_report_path=reference_report_path,
        generated_report_path=generated_report_path,
        human_report_path=human_report_path,
        output_path=output_path,
        predecessor_path=predecessor_path,
    )


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
    dependency_revisions = {str(key): str(value) for key, value in backbone["dependencies"].items()}
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
    artifact_counts = validate_generated_artifacts(generated)

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
        "yes_bias": yes_bias_points <= float(reference_thresholds["yes_bias_points_max"]),
        "repeat_agreement": repeat_agreement >= float(reference_thresholds["repeat_agreement_min"]),
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
            "artifact_counts": artifact_counts,
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
