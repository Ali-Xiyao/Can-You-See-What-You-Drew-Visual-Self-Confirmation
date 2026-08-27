from __future__ import annotations

from pathlib import Path

from selfsight.config import load_config
from selfsight.data.counterfactuals import build_tier_b
from selfsight.data.manifest import select_tier_d_sources
from selfsight.schemas import as_serializable
from selfsight.training.paired import assert_same_schedule, build_paired_schedule


def test_registered_split_counts_and_zero_overlap(registered_splits):
    assert {key: len(value) for key, value in registered_splits.items()} == {
        "train": 2400,
        "tier_a_probe": 200,
        "tier_a_outcome": 600,
    }
    signatures = {key: {scene.signature for scene in scenes} for key, scenes in registered_splits.items()}
    assert len(set().union(*signatures.values())) == 3200
    assert signatures["train"].isdisjoint(signatures["tier_a_probe"])
    assert signatures["train"].isdisjoint(signatures["tier_a_outcome"])
    assert signatures["tier_a_probe"].isdisjoint(signatures["tier_a_outcome"])
    template_ids = {key: {scene.template_id for scene in scenes} for key, scenes in registered_splits.items()}
    assert template_ids["train"].isdisjoint(template_ids["tier_a_probe"])
    assert template_ids["train"].isdisjoint(template_ids["tier_a_outcome"])


def test_local_and_a800_configs_preserve_experiment_definition():
    local = load_config(Path("configs/local_3090.yaml"))
    a800 = load_config(Path("configs/a800_80g.yaml"))
    assert local.values["model"]["trainable_id"] == a800.values["model"]["trainable_id"]
    assert local.values["model"]["image_resolution"] == a800.values["model"]["image_resolution"] == 512
    assert local.values["model"]["generation_timesteps"] == a800.values["model"]["generation_timesteps"] == 25
    assert local.values["training"]["lora"] == a800.values["training"]["lora"]
    assert local.values["training"]["arms"] == a800.values["training"]["arms"]
    assert local.values["hardware"]["distributed"] is False
    assert a800.values["hardware"]["distributed"] is False


def test_showo2_local_and_a800_configs_preserve_joint_backbone_definition():
    local = load_config(Path("configs/local_3090_showo2.yaml"))
    a800 = load_config(Path("configs/a800_80g_showo2.yaml"))
    assert local.values["model"] == a800.values["model"]
    assert local.values["model"]["trainable_id"] == "showlab/show-o2-1.5B"
    assert local.values["model"]["image_resolution"] == 432
    assert local.values["model"]["generation_timesteps"] == 50
    assert local.values["training"]["lora"] == a800.values["training"]["lora"]
    assert local.values["training"]["lora"]["target_selection_required"] is True
    assert local.values["hardware"]["observer_device"] == "cuda:1"
    assert a800.values["hardware"]["observer_device"] == "cuda:0"


def test_paired_schedule_is_deterministic(registered_splits):
    prompt_ids = [scene.scene_id for scene in registered_splits["train"]]
    left = build_paired_schedule(prompt_ids, rounds=10, prompts_per_round=64, candidate_k=2, seed=7)
    right = build_paired_schedule(prompt_ids, rounds=10, prompts_per_round=64, candidate_k=2, seed=7)
    assert_same_schedule(left, right)
    assert len(left) == 640
    assert all(len(item.candidate_seeds) == 2 for item in left)


def test_tier_d_selection_is_deterministic_and_balanced(registered_splits):
    tier_a = [{"scene": as_serializable(scene)} for scene in registered_splits["tier_a_outcome"]]
    tier_b = [{"pair": pair.to_dict()} for pair in build_tier_b(registered_splits["tier_a_outcome"])]
    first_a, first_b = select_tier_d_sources(tier_a, tier_b)
    second_a, second_b = select_tier_d_sources(tier_a, tier_b)
    assert [row["scene"]["scene_id"] for row in first_a] == [
        row["scene"]["scene_id"] for row in second_a
    ]
    assert [row["pair"]["pair_id"] for row in first_b] == [
        row["pair"]["pair_id"] for row in second_b
    ]
    assert len(first_a) == 300
    assert len(first_b) == 150
    assert {
        family: sum(row["scene"]["family"] == family for row in first_a)
        for family in ("existence", "count", "color", "size", "spatial", "binding")
    } == {family: 50 for family in ("existence", "count", "color", "size", "spatial", "binding")}
    assert {
        category: sum(row["pair"]["category"] == category for row in first_b)
        for category in (
            "count_delete",
            "color_change",
            "relation_left_right",
            "relation_size",
            "binding_swap",
        )
    } == {
        "count_delete": 30,
        "color_change": 30,
        "relation_left_right": 30,
        "relation_size": 30,
        "binding_swap": 30,
    }
