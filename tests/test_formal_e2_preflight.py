from __future__ import annotations

import pytest

from scripts.run_formal_e2 import _require_eligible_manifest_capacity, _seed_invariant
from selfsight.utils.jsonl import atomic_write_jsonl


def test_formal_seed_invariant_ignores_only_registered_seed_fields() -> None:
    first = {
        "profile": "a800_80g_showo2",
        "seed": 1,
        "model": {"trainable_id": "model"},
        "training": {"seeds": [1, 2, 3], "rounds": 20},
    }
    second = {
        "profile": "a800_80g_showo2_seed_2",
        "seed": 2,
        "model": {"trainable_id": "model"},
        "training": {"seeds": [2], "rounds": 20},
    }
    assert _seed_invariant(first) == _seed_invariant(second)
    second["training"]["rounds"] = 19
    assert _seed_invariant(first) != _seed_invariant(second)


def test_formal_manifest_capacity_is_family_conditioned(tmp_path) -> None:
    manifest = atomic_write_jsonl(
        tmp_path / "train.jsonl",
        (
            {
                "scene": {
                    "scene_id": f"{family}-{index}",
                    "family": family,
                }
            }
            for index, family in enumerate(("existence", "binding", "size"))
        ),
    )
    _require_eligible_manifest_capacity(
        manifest,
        families=("existence", "binding"),
        required=2,
        label="training",
    )
    with pytest.raises(RuntimeError, match="2 unique eligible cases; 3 required"):
        _require_eligible_manifest_capacity(
            manifest,
            families=("existence", "binding"),
            required=3,
            label="training",
        )
