import json
from pathlib import Path

import pytest

from selfsight.backbones.lora_selection import (
    build_lora_target_selection,
    validate_lora_target_selection,
)
from selfsight.utils.hashing import sha256_file
from selfsight.utils.jsonl import atomic_write_json


def _audit(tmp_path: Path) -> tuple[Path, Path]:
    tree_path = atomic_write_json(
        tmp_path / "tree.json",
        {
            "model_id": "model",
            "revision": "revision",
            "shared_transformer_candidates": [
                "showo.model.layers.0.self_attn.q_proj",
                "showo.model.layers.0.self_attn.k_proj",
                "showo.model.layers.0.mlp.down_proj",
            ],
        },
    )
    report_path = atomic_write_json(
        tmp_path / "canary.json",
        {
            "model_id": "model",
            "revision": "revision",
            "source_revision": "source",
            "dependency_revisions": {"dep": "revision"},
            "lora_module_tree": str(tree_path),
            "lora_module_tree_sha256": sha256_file(tree_path),
        },
    )
    return report_path, tree_path


def test_lora_selection_is_derived_from_hashed_module_tree(tmp_path: Path) -> None:
    canary, _ = _audit(tmp_path)
    selection = build_lora_target_selection(canary, suffixes=("q_proj", "down_proj"))
    selection_path = atomic_write_json(tmp_path / "selection.json", selection)
    validated = validate_lora_target_selection(selection_path, canary_report=canary)
    assert validated["suffixes"] == ["q_proj", "down_proj"]
    assert len(validated["target_modules"]) == 2


def test_lora_selection_rejects_tampered_module_tree(tmp_path: Path) -> None:
    canary, tree = _audit(tmp_path)
    selection = build_lora_target_selection(canary, suffixes=("q_proj",))
    selection_path = atomic_write_json(tmp_path / "selection.json", selection)
    tree.write_text(json.dumps({"shared_transformer_candidates": []}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="module-tree SHA-256 mismatch"):
        validate_lora_target_selection(selection_path, canary_report=canary)
