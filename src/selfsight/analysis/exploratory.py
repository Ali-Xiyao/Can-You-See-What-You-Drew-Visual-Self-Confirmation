"""Fail-loud authorization for explicitly non-formal post-Gate diagnostics."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path
from typing import Any

import yaml

from selfsight.analysis.readiness import (
    validate_generated_artifacts,
    validate_human_precision_report,
)
from selfsight.utils.hashing import sha256_file, sha256_json
from selfsight.utils.jsonl import atomic_write_json

STAGE = "exploratory_post_gate_authorization"
AUTHORIZED_STAGES = (
    "a4_lora_backward_resume",
    "e1_observation_contexts",
    "gradient_gate",
    "paired_local_e2",
)


def _read_json(path: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object: {resolved}")
    return resolved, value


def _read_yaml(path: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    value = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a YAML mapping: {resolved}")
    return resolved, value


def _evidence(path: Path, label: str) -> dict[str, str]:
    return {"label": label, "path": str(path), "sha256": sha256_file(path)}


def _mapping(report: Mapping[str, Any], key: str, label: str) -> Mapping[str, Any]:
    value = report.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} is missing mapping {key!r}")
    return value


def _require_bound_evidence(
    decision: Mapping[str, Any], *, label: str, path: Path
) -> None:
    evidence = _mapping(decision, "evidence", "frozen decision").get(label)
    if not isinstance(evidence, Mapping):
        raise TypeError(f"Frozen decision does not bind {label!r}")
    if Path(str(evidence.get("path", ""))).resolve() != path:
        raise RuntimeError(f"Frozen decision {label} path differs from supplied evidence")
    if evidence.get("sha256") != sha256_file(path):
        raise RuntimeError(f"Frozen decision {label} SHA-256 differs from supplied evidence")


def _identity(backbone: Mapping[str, Any], reports: Sequence[Mapping[str, Any]]) -> None:
    model_id = str(backbone["backbone_id"])
    revision = str(backbone["revision"])
    source_revision = str(_mapping(backbone, "source", "backbone")["revision"])
    dependencies = {
        str(key): str(value)
        for key, value in _mapping(backbone, "dependencies", "backbone").items()
    }
    for report in reports:
        if report.get("model_id") != model_id or report.get("revision") != revision:
            raise RuntimeError("Exploratory evidence model identity mismatch")
        if report.get("source_revision") != source_revision:
            raise RuntimeError("Exploratory evidence source revision mismatch")
        if report.get("dependency_revisions") != dependencies:
            raise RuntimeError("Exploratory evidence dependency revision mismatch")


def _project_root(backbone_path: Path) -> Path:
    # Registered backbone configs live under <project>/configs/backbones.
    return backbone_path.parent.parent.parent.resolve()


def _validate_output_root(backbone_path: Path, output_root: Path) -> None:
    expected_parent = (_project_root(backbone_path) / "runs" / "exploratory-post-gate").resolve()
    try:
        output_root.relative_to(expected_parent)
    except ValueError as error:
        raise RuntimeError(
            f"Exploratory outputs must stay below {expected_parent}, got {output_root}"
        ) from error


def build_exploratory_authorization(
    *,
    backbone_config_path: str | Path,
    readiness_config_path: str | Path,
    decision_path: str | Path,
    canary_report_path: str | Path,
    reference_report_path: str | Path,
    generated_report_path: str | Path,
    human_report_path: str | Path,
    families: Sequence[str],
    output_root: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Bind a red registered decision to a narrow, non-formal diagnostic route."""

    backbone_path, backbone = _read_yaml(backbone_config_path, "backbone config")
    readiness_path, readiness = _read_yaml(readiness_config_path, "readiness config")
    decision_file, decision = _read_json(decision_path, "frozen readiness decision")
    canary_path, canary = _read_json(canary_report_path, "A1 canary")
    reference_path, reference = _read_json(reference_report_path, "A2 reference")
    generated_path, generated = _read_json(generated_report_path, "A3 generated")
    human_path, human = _read_json(human_report_path, "A3 blind-human")
    root = Path(output_root).resolve()
    _validate_output_root(backbone_path, root)

    if decision.get("gate") != "minus_2_joint_readiness" or decision.get("passed") is not False:
        raise RuntimeError("Exploratory continuation requires an immutable red Gate -2 decision")
    if decision.get("decision_mode") != "stop_after_human_before_a4":
        raise RuntimeError("Exploratory A4 requires the registered stop-after-human decision mode")
    if decision.get("skipped_by_stop_rule") != ["a4_lora_backward_resume"]:
        raise RuntimeError("Frozen decision does not record A4 as the sole skipped check")
    for label, path in (
        ("backbone_config", backbone_path),
        ("readiness_config", readiness_path),
        ("canary", canary_path),
        ("reference", reference_path),
        ("generated", generated_path),
        ("human", human_path),
    ):
        _require_bound_evidence(decision, label=label, path=path)

    _identity(backbone, (canary, reference, generated, human))
    if not bool(canary.get("passed")):
        raise RuntimeError("A1 engineering canary is red")
    validate_generated_artifacts(generated)
    main_families = [str(item) for item in readiness["main_families"]]
    verifier_threshold = float(readiness["thresholds"]["generated"]["verifier_precision_min"])
    validate_human_precision_report(
        human, families=main_families, threshold=verifier_threshold
    )

    selected = tuple(dict.fromkeys(str(item) for item in families))
    if not selected or len(selected) != len(families):
        raise ValueError("Exploratory families must be a non-empty unique sequence")
    if any(family not in main_families for family in selected):
        raise ValueError("Exploratory families must belong to the registered main family set")
    reference_accuracy = _mapping(reference, "family_open_accuracy", "A2 reference")
    coverage = _mapping(generated, "family_coverage", "A3 generated")
    oracle = _mapping(generated, "family_oracle_at_4", "A3 generated")
    precision = _mapping(human, "family_precision", "A3 blind-human")
    thresholds = readiness["thresholds"]
    minima = {
        "reference_accuracy": float(thresholds["reference"]["family_open_accuracy_min"]),
        "generated_coverage": float(thresholds["generated"]["family_coverage_min"]),
        "oracle_at_4": float(thresholds["generated"]["oracle_at_4_min"]),
        "human_precision": verifier_threshold,
    }
    metrics: dict[str, dict[str, float]] = {}
    for family in selected:
        family_metrics = {
            "reference_accuracy": float(reference_accuracy[family]),
            "generated_coverage": float(coverage[family]),
            "oracle_at_4": float(oracle[family]),
            "human_precision": float(precision[family]),
        }
        failures = [key for key, value in family_metrics.items() if value < minima[key]]
        if failures:
            raise RuntimeError(f"Exploratory family {family!r} fails evidence checks: {failures}")
        metrics[family] = family_metrics

    report: dict[str, Any] = {
        "schema_version": 1,
        "stage": STAGE,
        "non_formal": True,
        "registered_decision_unchanged": True,
        "model_downloads_allowed": False,
        "model_id": str(backbone["backbone_id"]),
        "revision": str(backbone["revision"]),
        "source_revision": str(backbone["source"]["revision"]),
        "dependency_revisions": dict(backbone["dependencies"]),
        "families": list(selected),
        "authorized_stages": list(AUTHORIZED_STAGES),
        "output_root": str(root),
        "thresholds": minima,
        "metrics": metrics,
        "evidence": {
            "backbone_config": _evidence(backbone_path, "backbone_config"),
            "readiness_config": _evidence(readiness_path, "readiness_config"),
            "frozen_decision": _evidence(decision_file, "frozen_decision"),
            "canary": _evidence(canary_path, "canary"),
            "reference": _evidence(reference_path, "reference"),
            "generated": _evidence(generated_path, "generated"),
            "human": _evidence(human_path, "human"),
        },
    }
    report["authorization_digest"] = sha256_json(report)
    if output_path is not None:
        destination = Path(output_path).resolve()
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite exploratory authorization: {destination}")
        try:
            destination.relative_to(root)
        except ValueError as error:
            raise RuntimeError("Authorization artifact must be written below its output root") from error
        atomic_write_json(destination, report)
    return report


def validate_exploratory_authorization(
    path: str | Path,
    *,
    stage: str,
    backbone_config_path: str | Path,
    decision_path: str | Path,
    canary_report_path: str | Path,
    reference_report_path: str | Path,
    generated_report_path: str | Path,
    human_report_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Rebuild and compare an authorization before an exploratory stage runs."""

    authorization_path, recorded = _read_json(path, "exploratory authorization")
    if stage not in AUTHORIZED_STAGES or stage not in recorded.get("authorized_stages", ()):
        raise RuntimeError(f"Exploratory stage {stage!r} is not authorized")
    if recorded.get("stage") != STAGE or recorded.get("non_formal") is not True:
        raise RuntimeError("Invalid exploratory authorization marker")
    if recorded.get("registered_decision_unchanged") is not True:
        raise RuntimeError("Exploratory authorization does not preserve the registered decision")
    if recorded.get("model_downloads_allowed") is not False:
        raise RuntimeError("Exploratory authorization unexpectedly permits model downloads")
    digest = recorded.get("authorization_digest")
    unsigned = dict(recorded)
    unsigned.pop("authorization_digest", None)
    if digest != sha256_json(unsigned):
        raise RuntimeError("Exploratory authorization digest mismatch")
    rebuilt = build_exploratory_authorization(
        backbone_config_path=backbone_config_path,
        readiness_config_path=recorded["evidence"]["readiness_config"]["path"],
        decision_path=decision_path,
        canary_report_path=canary_report_path,
        reference_report_path=reference_report_path,
        generated_report_path=generated_report_path,
        human_report_path=human_report_path,
        families=recorded["families"],
        output_root=recorded["output_root"],
    )
    if rebuilt != recorded:
        raise RuntimeError("Exploratory authorization differs from freshly validated evidence")
    if output_path is not None:
        destination = Path(output_path).resolve()
        root = Path(str(recorded["output_root"])).resolve()
        try:
            destination.relative_to(root)
        except ValueError as error:
            raise RuntimeError(f"Exploratory output escapes authorized root: {destination}") from error
        if destination == authorization_path:
            raise RuntimeError("Exploratory output cannot overwrite its authorization artifact")
    return recorded


def validate_bound_exploratory_authorization(
    path: str | Path,
    *,
    stage: str,
    backbone_config_path: str | Path,
    decision_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate an authorization using the evidence paths sealed inside it."""

    _authorization_path, recorded = _read_json(path, "exploratory authorization")
    evidence = _mapping(recorded, "evidence", "exploratory authorization")
    required = ("canary", "reference", "generated", "human")
    missing = [label for label in required if not isinstance(evidence.get(label), Mapping)]
    if missing:
        raise RuntimeError(f"Exploratory authorization is missing bound evidence: {missing}")
    return validate_exploratory_authorization(
        path,
        stage=stage,
        backbone_config_path=backbone_config_path,
        decision_path=decision_path,
        canary_report_path=evidence["canary"]["path"],
        reference_report_path=evidence["reference"]["path"],
        generated_report_path=evidence["generated"]["path"],
        human_report_path=evidence["human"]["path"],
        output_path=output_path,
    )


def validate_exploratory_observer_audit(
    audit_path: str | Path,
    *,
    model_id: str,
    revision: str,
    eligible_families: Sequence[str],
    family_accuracy_min: float,
    yes_bias_max: float,
    abstain_rate_max: float,
) -> dict[str, Any]:
    """Validate a frozen observer on a diagnostic family subset without a four-family claim."""

    _path, audit = _read_json(audit_path, "exploratory public-observer audit")
    if audit.get("observer_id") != model_id or audit.get("observer_revision") != revision:
        raise RuntimeError("Exploratory public-observer audit identity mismatch")
    selected = tuple(dict.fromkeys(str(item) for item in eligible_families))
    if not selected:
        raise ValueError("Exploratory public-observer audit requires at least one family")
    family = _mapping(audit, "family_open_accuracy", "public-observer audit")
    missing = sorted(set(selected).difference(family))
    if missing:
        raise RuntimeError(f"Public-observer audit is missing diagnostic families: {missing}")
    values = {name: float(family[name]) for name in selected}
    bias = float(audit.get("absolute_yes_bias", float("inf")))
    abstain = float(audit.get("abstain_rate", float("inf")))
    invalid = {
        name: value for name, value in values.items() if not isfinite(value) or not 0 <= value <= 1
    }
    if invalid or not isfinite(bias) or not isfinite(abstain):
        raise RuntimeError("Exploratory public-observer audit contains invalid metrics")
    below = {name: value for name, value in values.items() if value < family_accuracy_min}
    if below:
        raise RuntimeError(f"Public observer is below the diagnostic-family floor: {below}")
    if not 0 <= bias <= yes_bias_max:
        raise RuntimeError(f"Public observer yes-bias exceeds the diagnostic limit: {bias:.3f}")
    if not 0 <= abstain <= abstain_rate_max:
        raise RuntimeError(
            f"Public observer abstention exceeds the diagnostic limit: {abstain:.3f}"
        )
    return audit
