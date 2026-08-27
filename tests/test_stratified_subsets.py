from __future__ import annotations

from collections import Counter

import pytest

from selfsight.data.subsets import stable_stratified_sample


def _records(per_family: int = 100) -> list[dict[str, str]]:
    return [
        {"id": f"{family}-{index:03d}", "family": family}
        for family in ("binding", "color", "count", "existence", "size", "spatial")
        for index in range(per_family)
    ]


@pytest.mark.parametrize("size", [32, 120, 600])
def test_stable_stratified_sample_is_balanced_and_deterministic(size: int) -> None:
    records = _records()
    first = stable_stratified_sample(
        records,
        size,
        stratum=lambda record: record["family"],
        item_id=lambda record: record["id"],
        seed=20260827,
    )
    second = stable_stratified_sample(
        list(reversed(records)),
        size,
        stratum=lambda record: record["family"],
        item_id=lambda record: record["id"],
        seed=20260827,
    )
    assert [record["id"] for record in first] == [record["id"] for record in second]
    counts = Counter(record["family"] for record in first)
    assert max(counts.values()) - min(counts.values()) <= 1
    assert len({record["id"] for record in first}) == size
