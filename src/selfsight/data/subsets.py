"""Deterministic balanced subsets for manifests whose on-disk order is family-blocked."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import TypeVar

from selfsight.utils.hashing import sha256_json

T = TypeVar("T")


def stable_stratified_sample(
    values: Sequence[T],
    size: int,
    *,
    stratum: Callable[[T], str],
    item_id: Callable[[T], str],
    seed: int,
) -> list[T]:
    """Return an exact, deterministic subset with stratum counts differing by at most one."""

    if size <= 0:
        raise ValueError("Stratified subset size must be positive")
    if size > len(values):
        raise ValueError(f"Cannot select {size} records from only {len(values)}")
    groups: dict[str, list[T]] = defaultdict(list)
    seen: set[str] = set()
    for value in values:
        identifier = item_id(value)
        if identifier in seen:
            raise ValueError(f"Duplicate item ID in stratified source: {identifier}")
        seen.add(identifier)
        groups[stratum(value)].append(value)
    names = sorted(groups)
    if not names:
        raise ValueError("Cannot stratify an empty sequence")
    base, remainder = divmod(size, len(names))
    selected: list[T] = []
    for index, name in enumerate(names):
        quota = base + int(index < remainder)
        candidates = sorted(
            groups[name],
            key=lambda value: sha256_json([seed, "within", name, item_id(value)]),
        )
        if len(candidates) < quota:
            raise ValueError(
                f"Stratum {name!r} has {len(candidates)} records but requires {quota}"
            )
        selected.extend(candidates[:quota])
    return sorted(
        selected,
        key=lambda value: sha256_json([seed, "across", item_id(value)]),
    )
