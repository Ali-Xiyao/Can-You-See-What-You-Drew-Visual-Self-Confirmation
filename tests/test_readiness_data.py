from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from selfsight.data.questions import build_primary_atom
from selfsight.data.readiness import (
    DATA_NAMESPACE,
    MAIN_FAMILIES,
    SPLIT_COUNTS,
    build_minimal_scenes,
    materialize_readiness_dataset,
)
from selfsight.data.verifier import verify_image
from selfsight.schemas import Color, QuestionFamily, SceneSpec
from selfsight.utils.hashing import sha256_file
from selfsight.utils.jsonl import read_jsonl


def test_minimal_readiness_splits_are_balanced_and_disjoint():
    splits = {
        name: build_minimal_scenes(name, per_family=count)
        for name, count in SPLIT_COUNTS.items()
    }
    for name, scenes in splits.items():
        assert len(scenes) == SPLIT_COUNTS[name] * len(MAIN_FAMILIES)
        assert Counter(scene.family for scene in scenes) == {
            family: SPLIT_COUNTS[name] for family in MAIN_FAMILIES
        }
        assert all(scene.metadata["benchmark_version"] == "2.2" for scene in scenes)
    signatures = {name: {scene.signature for scene in scenes} for name, scenes in splits.items()}
    assert signatures["canary"].isdisjoint(signatures["reference"])
    assert signatures["canary"].isdisjoint(signatures["generated"])
    assert signatures["reference"].isdisjoint(signatures["generated"])


def test_spatial_minimal_prompts_never_use_relative_size():
    scenes = build_minimal_scenes("reference", per_family=20)
    spatial = [scene for scene in scenes if scene.family == QuestionFamily.SPATIAL]
    assert {str(scene.metadata["relation"]) for scene in spatial} <= {"left_of", "above"}
    assert all("larger" not in scene.prompt.lower() for scene in spatial)


def test_minimal_prompts_do_not_add_irrelevant_constraints():
    scenes = build_minimal_scenes("generated", per_family=10)
    colors = [scene for scene in scenes if scene.family == QuestionFamily.COLOR]
    sizes = [scene for scene in scenes if scene.family == QuestionFamily.SIZE]
    bindings = [scene for scene in scenes if scene.family == QuestionFamily.BINDING]
    assert all("small" not in scene.prompt and "large" not in scene.prompt for scene in colors)
    assert all(not any(color.value in scene.prompt for color in Color) for scene in sizes)
    assert all(len(scene.objects) == 2 for scene in bindings)


def test_materialized_readiness_references_verify_and_are_idempotent(tmp_path: Path):
    root = tmp_path / DATA_NAMESPACE
    first = materialize_readiness_dataset(root)
    first_hashes = {name: sha256_file(path) for name, path in first["manifests"].items()}
    second = materialize_readiness_dataset(root)
    second_hashes = {name: sha256_file(path) for name, path in second["manifests"].items()}
    assert first_hashes == second_hashes
    assert not any(first["split_signature_overlap"].values())

    for manifest in first["manifests"].values():
        for record in read_jsonl(manifest):
            scene = SceneSpec.from_dict(record["scene"])
            atom = build_primary_atom(scene)
            result = verify_image(record["reference_image"], (atom,))
            assert result.answers[atom.atom_id] == atom.answer


def test_readiness_materialization_refuses_v1_namespace(tmp_path: Path):
    with pytest.raises(ValueError, match=DATA_NAMESPACE):
        materialize_readiness_dataset(tmp_path / "selfsight-v1")
