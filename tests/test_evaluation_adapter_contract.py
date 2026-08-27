from __future__ import annotations

import pytest

from selfsight.pilot.evaluate import _records
from selfsight.utils.jsonl import atomic_write_jsonl


def test_evaluation_records_filter_before_deterministic_sampling(tmp_path) -> None:
    manifest = tmp_path / "outcome.jsonl"
    rows = [
        {
            "scene": {"scene_id": f"scene-{family}", "family": family},
            "atom": {"family": family},
        }
        for family in ("existence", "size", "binding")
    ]
    atomic_write_jsonl(manifest, rows)
    selected = _records(
        manifest,
        limit=2,
        seed=17,
        eligible_families=("existence", "binding"),
    )
    assert {row["scene"]["family"] for row in selected} == {"existence", "binding"}
    with pytest.raises(RuntimeError, match="only 2 eligible records"):
        _records(
            manifest,
            limit=3,
            seed=17,
            eligible_families=("existence", "binding"),
        )
