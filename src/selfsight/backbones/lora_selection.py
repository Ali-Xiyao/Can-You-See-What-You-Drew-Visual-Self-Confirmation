"""Hash-bound LoRA target selection derived from an actual A1 Show-o2 module-tree audit."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from selfsight.utils.hashing import sha256_file, sha256_json


def build_lora_target_selection(
    canary_report: str | Path,
    *,
    suffixes: Sequence[str],
) -> dict[str, Any]:
    report_path = Path(canary_report).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    tree_path = Path(str(report["lora_module_tree"])).resolve()
    if sha256_file(tree_path) != report["lora_module_tree_sha256"]:
        raise RuntimeError("A1 LoRA module-tree SHA-256 mismatch")
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    if tree.get("model_id") != report.get("model_id") or tree.get("revision") != report.get(
        "revision"
    ):
        raise RuntimeError("A1 module-tree identity mismatch")
    normalized_suffixes = tuple(dict.fromkeys(str(item).strip() for item in suffixes if str(item).strip()))
    if not normalized_suffixes:
        raise ValueError("At least one explicit audited module suffix is required")
    available = tuple(str(item) for item in tree["shared_transformer_candidates"])
    targets = tuple(
        name for name in available if name.rsplit(".", 1)[-1] in normalized_suffixes
    )
    missing = sorted(
        suffix for suffix in normalized_suffixes if not any(name.endswith(f".{suffix}") for name in targets)
    )
    if missing:
        raise ValueError(f"Requested suffixes are absent from the audited shared transformer: {missing}")
    selection_payload = {
        "suffixes": list(normalized_suffixes),
        "target_modules": list(targets),
    }
    return {
        "schema_version": 2,
        "stage": "showo2_lora_target_selection",
        "model_id": report["model_id"],
        "revision": report["revision"],
        "source_revision": report["source_revision"],
        "dependency_revisions": report["dependency_revisions"],
        "canary_report": str(report_path),
        "canary_report_sha256": sha256_file(report_path),
        "module_tree": str(tree_path),
        "module_tree_sha256": sha256_file(tree_path),
        **selection_payload,
        "selection_digest": sha256_json(selection_payload),
        "forbidden_modules_selected": [],
    }


def validate_lora_target_selection(
    selection_path: str | Path,
    *,
    canary_report: str | Path,
) -> dict[str, Any]:
    path = Path(selection_path).resolve()
    selection = json.loads(path.read_text(encoding="utf-8"))
    report_path = Path(canary_report).resolve()
    if Path(str(selection["canary_report"])).resolve() != report_path:
        raise RuntimeError("LoRA target selection points to a different A1 canary")
    if sha256_file(report_path) != selection["canary_report_sha256"]:
        raise RuntimeError("LoRA target selection A1 canary SHA-256 mismatch")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    tree_path = Path(str(selection["module_tree"])).resolve()
    if sha256_file(tree_path) != selection["module_tree_sha256"]:
        raise RuntimeError("LoRA target selection module-tree SHA-256 mismatch")
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    payload = {
        "suffixes": selection["suffixes"],
        "target_modules": selection["target_modules"],
    }
    if sha256_json(payload) != selection["selection_digest"]:
        raise RuntimeError("LoRA target selection digest mismatch")
    if selection.get("model_id") != report.get("model_id") or selection.get(
        "revision"
    ) != report.get("revision"):
        raise RuntimeError("LoRA target selection identity mismatch")
    available = {str(item) for item in tree["shared_transformer_candidates"]}
    targets = tuple(str(item) for item in selection["target_modules"])
    if not targets or not set(targets).issubset(available):
        raise RuntimeError("LoRA selection contains an unaudited or non-shared module")
    if selection.get("forbidden_modules_selected"):
        raise RuntimeError("LoRA selection records forbidden modules")
    return selection
