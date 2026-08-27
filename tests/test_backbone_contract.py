from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from selfsight.backbones.base import UnifiedBackbone
from selfsight.backbones.showo2 import Showo2Adapter
from selfsight.models import load_model_lock


def test_showo2_lazy_identity_matches_lock_and_protocol():
    adapter = Showo2Adapter(lazy=True)
    lock = load_model_lock()
    model = next(item for item in lock["models"] if item["id"] == adapter.model_id)
    repository = next(
        item for item in lock["repositories"] if item["id"] == adapter.identity.source_repository
    )
    assert isinstance(adapter, UnifiedBackbone)
    assert adapter.revision == model["revision"]
    assert adapter.identity.source_revision == repository["revision"]
    assert adapter.identity.native_resolution == 432
    assert adapter.capabilities.unified_functionality
    assert adapter.resource_report().loaded is False


def test_showo2_dependency_set_is_locked_and_gate_ordered():
    adapter = Showo2Adapter(lazy=True)
    revisions = adapter.dependency_revisions()
    assert set(revisions) == {
        "showlab/show-o2-1.5B",
        "Wan-AI/Wan2.1-T2V-14B",
        "google/siglip-so400m-patch14-384",
        "Qwen/Qwen2.5-1.5B-Instruct",
    }
    lock = load_model_lock()
    first = {item["id"] for item in lock["models"] if item["group"] == "readiness_candidate_1"}
    fallbacks = {
        item["id"]
        for item in lock["models"]
        if item["group"] in {"readiness_fallback_hq", "readiness_fallback_7b"}
    }
    assert set(revisions) == first
    assert first.isdisjoint(fallbacks)


def test_showo2_config_revision_mismatch_fails_before_model_load(tmp_path: Path):
    source = yaml.safe_load(Path("configs/backbones/showo2_1p5b.yaml").read_text(encoding="utf-8"))
    source["revision"] = "0" * 40
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(source), encoding="utf-8")
    with pytest.raises(ValueError, match="config/lock revision mismatch"):
        Showo2Adapter(backbone_config=path, lazy=True)

