from __future__ import annotations

from dataclasses import replace

import pytest

from selfsight.data.counterfactuals import build_tier_b
from selfsight.data.manifest import _scene_record
from selfsight.data.portability import rebase_scene_records, rebase_tier_b_records
from selfsight.data.renderer import render_scene
from selfsight.schemas import as_serializable
from selfsight.utils.hashing import rgb_sha256


def test_scene_manifest_rebase_verifies_hashes(tmp_path, registered_splits) -> None:
    scene = registered_splits["tier_a_probe"][0]
    image = tmp_path / "reference_images" / scene.split / f"{scene.scene_id}.png"
    render_scene(scene, image)
    original = _scene_record(scene, image)
    original["reference_image"] = "H:\\obsolete\\reference.png"
    rebased = rebase_scene_records([original], tmp_path)
    assert rebased[0]["reference_image"] == str(image.resolve())
    assert original["reference_image"] == "H:\\obsolete\\reference.png"
    original["reference_rgb_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="RGB SHA-256"):
        rebase_scene_records([original], tmp_path)


def test_tier_b_rebase_preserves_complete_pair(tmp_path, registered_splits) -> None:
    pair = build_tier_b(registered_splits["tier_a_outcome"])[0]
    image_root = tmp_path / "reference_images" / "tier_b"
    source_path = image_root / f"{pair.source.scene_id}.png"
    counterfactual_path = image_root / f"{pair.counterfactual.scene_id}.png"
    render_scene(pair.source, source_path)
    render_scene(pair.counterfactual, counterfactual_path)
    record = {
        "schema_version": 1,
        "pair": as_serializable(pair),
        "source_image": "H:\\obsolete\\source.png",
        "counterfactual_image": "H:\\obsolete\\counterfactual.png",
        "source_rgb_sha256": rgb_sha256(source_path),
        "counterfactual_rgb_sha256": rgb_sha256(counterfactual_path),
    }
    rebased = rebase_tier_b_records([record], tmp_path)
    assert rebased[0]["source_image"] == str(source_path.resolve())
    assert rebased[0]["counterfactual_image"] == str(counterfactual_path.resolve())
    broken = replace(pair.source, scene_id="missing")
    record["pair"]["source"] = as_serializable(broken)
    with pytest.raises(FileNotFoundError, match="incomplete"):
        rebase_tier_b_records([record], tmp_path)
