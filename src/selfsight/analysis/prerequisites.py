"""Fail-closed validation for preregistered phenomenon-experiment Gates."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from selfsight.utils.hashing import sha256_file


def read_report(path: str | Path, *, label: str) -> dict[str, Any]:
    """Read a JSON object and reject missing or malformed evidence reports."""

    report_path = Path(path).resolve()
    if not report_path.is_file():
        raise RuntimeError(f"{label} report does not exist: {report_path}")
    try:
        value = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read {label} report: {report_path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{label} report must be a JSON object: {report_path}")
    return value


def require_gate_minus_one(path: str | Path) -> dict[str, Any]:
    """Require a complete, internally consistent, green Gate -1 decision."""

    report = read_report(path, label="Gate -1")
    if report.get("gate") != "minus_1_pre_e1":
        raise RuntimeError("The supplied report is not a completed Gate -1 decision")
    conditions = report.get("conditions")
    if not isinstance(conditions, Mapping) or not conditions:
        raise RuntimeError("Gate -1 report has no auditable conditions")
    calculated_pass = all(bool(value) for value in conditions.values())
    if bool(report.get("passed")) != calculated_pass:
        raise RuntimeError("Gate -1 report is internally inconsistent")
    if not calculated_pass:
        raise RuntimeError(
            "Gate -1 is not green; the registered stop rule forbids E1, Gate -1b, and E2"
        )
    return report


def require_selected_detector(
    gate_report: Mapping[str, Any],
    *,
    model_id: str,
    revision: str,
) -> Mapping[str, Any]:
    """Bind a detector invocation to the exact observer selected by Gate -1."""

    matched = gate_report.get("matched_observer")
    if not isinstance(matched, Mapping) or not bool(matched.get("matched")):
        raise RuntimeError("Gate -1 does not contain a selected heterogeneous observer")
    expected_id = matched.get("observer_id")
    expected_revision = matched.get("observer_revision")
    if not isinstance(expected_id, str) or not isinstance(expected_revision, str):
        raise TypeError("Gate -1 selected-observer identity is incomplete")
    if model_id != expected_id or revision != expected_revision:
        raise RuntimeError(
            "Detector identity does not match Gate -1 selection: "
            f"requested {model_id}@{revision}, expected {expected_id}@{expected_revision}"
        )
    return matched


def require_selected_detector_audit(
    gate_report: Mapping[str, Any],
    audit_path: str | Path,
    *,
    model_id: str,
    revision: str,
) -> dict[str, Any]:
    """Require the exact hashed audit used to select the detector."""

    require_selected_detector(gate_report, model_id=model_id, revision=revision)
    audit = read_report(audit_path, label="selected detector audit")
    if audit.get("observer_id") != model_id or audit.get("observer_revision") != revision:
        raise RuntimeError("Detector audit identity does not match the requested detector")
    if not bool(audit.get("gate_minus_1_capability_pass")):
        raise RuntimeError("Selected detector audit does not pass the capability floor")
    if not bool(audit.get("gate_minus_1_bias_pass")):
        raise RuntimeError("Selected detector audit does not pass the bias control")
    evidence = gate_report.get("evidence_reports")
    candidates = evidence.get("candidates") if isinstance(evidence, Mapping) else None
    if not isinstance(candidates, list):
        raise TypeError("Gate -1 has no candidate evidence index")
    records = [
        item
        for item in candidates
        if isinstance(item, Mapping) and item.get("observer_id") == model_id
    ]
    if len(records) != 1 or not isinstance(records[0].get("sha256"), str):
        raise RuntimeError("Gate -1 does not identify one hashed audit for the selected detector")
    actual = sha256_file(Path(audit_path).resolve())
    if actual != records[0]["sha256"]:
        raise RuntimeError("Selected detector audit SHA-256 does not match Gate -1 evidence")
    return audit


def require_generated_domain(
    path: str | Path, *, configured_coverage_min: float
) -> dict[str, Any]:
    """Require the locked generated-RGB coverage definition and threshold."""

    report = read_report(path, label="generated-domain Gate")
    if report.get("parseability_gate_basis") != "primary_answer_coverage":
        raise RuntimeError("Generated-domain Gate must use primary answer coverage")
    overall = report.get("overall")
    if not isinstance(overall, Mapping):
        raise TypeError("Generated-domain Gate report has no overall metrics")
    try:
        coverage = float(overall["primary_answer_coverage"])
        samples = int(overall["samples"])
        report_threshold = float(report["parseability_min"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("Generated-domain Gate report has malformed coverage fields") from error
    if not 0.0 <= coverage <= 1.0 or samples <= 0:
        raise RuntimeError("Generated-domain Gate report has an invalid sample basis")
    if not 0.0 <= report_threshold <= 1.0:
        raise RuntimeError("Generated-domain Gate report has an invalid threshold")
    calculated_report_pass = samples > 0 and coverage >= report_threshold
    if bool(report.get("gate_generated_domain_pass")) != calculated_report_pass:
        raise RuntimeError("Generated-domain Gate report is internally inconsistent")
    if report_threshold < configured_coverage_min:
        raise RuntimeError(
            "Generated-domain Gate report lowered the configured coverage threshold "
            f"({report_threshold:.3f} < {configured_coverage_min:.3f})"
        )
    if not calculated_report_pass or coverage < configured_coverage_min:
        raise RuntimeError(
            "Generated RGB coverage is below the registered threshold; E1, Gate -1b, and E2 "
            f"are forbidden ({coverage:.3f} < {configured_coverage_min:.3f})"
        )
    return report
