"""Hash-bound v2.3 calibration and local-only authorization."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from selfsight.data.questions import normalize_answer
from selfsight.schemas import AtomicQuestion
from selfsight.utils.hashing import sha256_file, sha256_json
from selfsight.utils.jsonl import atomic_write_json
from selfsight.v23.data import ASPECT_RATIO_RANGE, DISPLAY_NAME, PRIMARY_FAMILIES, display_text

V23_BASE_COMMIT = "509e774631fadcb1acb8a9820327d697e312dc32"
AUTHORIZED_STAGES = ("gradient_survival_gate", "three_seed_training", "checkpoint_evaluation")


def _project_root() -> Path:
    configured = os.environ.get("SELFSIGHT_PROJECT_ROOT")
    return Path(configured).resolve() if configured else Path(__file__).resolve().parents[3]


def _json(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {resolved}")
    return resolved, value


def _evidence(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def _v23_run_root(path: str | Path) -> Path:
    root = Path(path).resolve()
    project = _project_root()
    expected = (project / "runs" / "v2.3-rfo-gold").resolve()
    try:
        root.relative_to(expected)
    except ValueError as error:
        raise RuntimeError(f"v2.3 outputs must stay below {expected}, got {root}") from error
    return root


def build_v23_calibration(
    *,
    human_report_path: str | Path,
    answer_key_path: str | Path,
    review_csv_path: str | Path,
    output_path: str | Path,
    threshold: float = 0.95,
) -> dict[str, Any]:
    """Reuse the user's blind labels under their explicitly stated box interpretation."""

    human_path, human = _json(human_report_path)
    key_path, key = _json(answer_key_path)
    review_path = Path(review_csv_path).resolve()
    output = _v23_run_root(output_path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite v2.3 calibration: {output}")
    if human.get("review_csv_sha256") != sha256_file(review_path):
        raise RuntimeError("Frozen human report does not bind the supplied review CSV")
    if human.get("answer_key_sha256") != sha256_file(key_path):
        raise RuntimeError("Frozen human report does not bind the supplied answer key")
    review = {}
    with review_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            review[str(row["audit_id"])] = row
    rows = []
    counts: dict[str, int] = defaultdict(int)
    matches: dict[str, int] = defaultdict(int)
    for keyed in key["rows"]:
        family = str(keyed["family"])
        if family not in PRIMARY_FAMILIES:
            continue
        audit_id = str(keyed["audit_id"])
        if audit_id not in review:
            raise RuntimeError(f"Review CSV is missing v2.3 calibration row {audit_id}")
        question_value = dict(keyed["primary_question"])
        question_value["text"] = display_text(str(question_value["text"]))
        question = AtomicQuestion.from_dict(question_value)
        raw = str(review[audit_id].get("human_answer", ""))
        parseable = str(review[audit_id].get("parseable_yes_no", "")).strip().lower() == "yes"
        normalized = normalize_answer(raw, question) if parseable else None
        verifier = str(keyed["verifier_answer"])
        matched = normalized == verifier
        counts[family] += 1
        matches[family] += int(matched)
        rows.append(
            {
                "audit_id": audit_id,
                "family": family,
                "image_rgb_sha256": keyed["image_rgb_sha256"],
                "question": question_value,
                "human_normalized": normalized,
                "verifier_answer": verifier,
                "matched": matched,
            }
        )
    total = sum(counts.values())
    total_matches = sum(matches.values())
    family_agreement = {family: matches[family] / counts[family] for family in PRIMARY_FAMILIES}
    agreement = total_matches / total if total else 0.0
    passed = bool(
        total == 28
        and agreement >= threshold
        and all(family_agreement[family] >= threshold for family in PRIMARY_FAMILIES)
    )
    report = {
        "schema_version": 1,
        "benchmark_version": "2.3",
        "stage": "v23_box_human_verifier_calibration",
        "non_formal": True,
        "reused_blind_labels": True,
        "new_annotation_round": False,
        "user_attested_annotation_policy": (
            "Rectangle-like outputs were counted as square; v2.3 names this tolerant category box."
        ),
        "vocabulary": {
            "internal_shape": "square",
            "display_shape": DISPLAY_NAME,
            "quadrilateral_aspect_ratio_range": list(ASPECT_RATIO_RANGE),
        },
        "threshold": threshold,
        "rows": rows,
        "samples": total,
        "family_counts": dict(counts),
        "agreement": agreement,
        "family_agreement": family_agreement,
        "passed": passed,
        "evidence": {
            "frozen_human_report": _evidence(human_path),
            "frozen_answer_key": _evidence(key_path),
            "frozen_review_csv": _evidence(review_path),
        },
    }
    report["calibration_digest"] = sha256_json(report)
    atomic_write_json(output, report)
    return report


def build_v23_authorization(
    *,
    config_path: str | Path,
    backbone_config_path: str | Path,
    data_registry_path: str | Path,
    calibration_path: str | Path,
    a4_report_path: str | Path,
    frozen_v22_decision_path: str | Path,
    output_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    backbone_file = Path(backbone_config_path).resolve()
    registry_path, registry = _json(data_registry_path)
    calibration_file, calibration = _json(calibration_path)
    a4_path, a4 = _json(a4_report_path)
    decision_path, decision = _json(frozen_v22_decision_path)
    root = _v23_run_root(output_root)
    destination = _v23_run_root(output_path)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite v2.3 authorization: {destination}")
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    backbone = yaml.safe_load(backbone_file.read_text(encoding="utf-8"))
    if config.get("benchmark_version") != "2.3" or config.get("profile") != "v2p3_local_rfo_gold":
        raise RuntimeError("Not a locked v2.3 local config")
    if config["training"]["arms"] != ["naive", "rfo_self", "rfo_gold"]:
        raise RuntimeError("v2.3 requires exactly the three registered trainable arms")
    if int(config["training"]["candidate_k"]) != 4:
        raise RuntimeError("v2.3 requires K=4")
    if registry.get("benchmark_version") != "2.3" or registry.get("registry_digest") is None:
        raise RuntimeError("Invalid v2.3 data registry")
    unsigned_registry = dict(registry)
    registry_digest = unsigned_registry.pop("registry_digest")
    if registry_digest != sha256_json(unsigned_registry):
        raise RuntimeError("v2.3 data registry digest mismatch")
    if calibration.get("passed") is not True or float(calibration["threshold"]) != 0.95:
        raise RuntimeError("v2.3 box calibration has not cleared 95%")
    if a4.get("passed") is not True or a4.get("non_formal") is not True:
        raise RuntimeError("v2.3 requires the completed local HQ A4 canary")
    if decision.get("passed") is not False or decision.get("gate") != "minus_2_joint_readiness":
        raise RuntimeError("v2.3 must bind the frozen red v2.2 decision")
    if backbone.get("backbone_id") != config["model"]["trainable_id"]:
        raise RuntimeError("v2.3 config/backbone identity mismatch")
    report = {
        "schema_version": 1,
        "benchmark_version": "2.3",
        "stage": "v23_local_mechanism_authorization",
        "non_formal": True,
        "formal_claims_allowed": False,
        "a800_allowed": False,
        "model_downloads_allowed": False,
        "v22_registered_evidence_unchanged": True,
        "implementation_base_commit": V23_BASE_COMMIT,
        "model_id": backbone["backbone_id"],
        "revision": backbone["revision"],
        "seeds": list(config["seeds"]),
        "families": list(config["data"]["primary_families"]),
        "arms": list(config["training"]["arms"]),
        "authorized_stages": list(AUTHORIZED_STAGES),
        "output_root": str(root),
        "evidence": {
            "config": _evidence(config_file),
            "backbone": _evidence(backbone_file),
            "data_registry": _evidence(registry_path),
            "box_calibration": _evidence(calibration_file),
            "a4": _evidence(a4_path),
            "frozen_v22_decision": _evidence(decision_path),
        },
    }
    report["authorization_digest"] = sha256_json(report)
    atomic_write_json(destination, report)
    return report


def validate_v23_authorization(
    path: str | Path,
    *,
    stage: str,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    authorization_path, report = _json(path)
    if stage not in AUTHORIZED_STAGES or stage not in report.get("authorized_stages", ()):
        raise RuntimeError(f"v2.3 stage is not authorized: {stage}")
    if report.get("non_formal") is not True or report.get("model_downloads_allowed") is not False:
        raise RuntimeError("Invalid v2.3 authorization safety markers")
    unsigned = dict(report)
    digest = unsigned.pop("authorization_digest", None)
    if digest != sha256_json(unsigned):
        raise RuntimeError("v2.3 authorization digest mismatch")
    for evidence in report["evidence"].values():
        evidence_path = Path(str(evidence["path"])).resolve()
        if sha256_file(evidence_path) != evidence["sha256"]:
            raise RuntimeError(f"v2.3 bound evidence changed: {evidence_path}")
    if output_path is not None:
        output = _v23_run_root(output_path)
        root = Path(str(report["output_root"])).resolve()
        try:
            output.relative_to(root)
        except ValueError as error:
            raise RuntimeError("v2.3 stage output escapes its authorization root") from error
        if output == authorization_path:
            raise RuntimeError("v2.3 stage cannot overwrite its authorization")
    return report
