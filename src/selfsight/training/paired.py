"""Deterministic prompt/candidate schedules shared by Naive and RFO arms."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptScheduleEntry:
    round_index: int
    prompt_id: str
    candidate_seeds: tuple[int, ...]


def _seed_from_parts(*parts: object) -> int:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFF_FFFF


def build_paired_schedule(
    prompt_ids: Sequence[str],
    *,
    rounds: int,
    prompts_per_round: int,
    candidate_k: int,
    seed: int,
) -> list[PromptScheduleEntry]:
    required = rounds * prompts_per_round
    if len(set(prompt_ids)) < required:
        raise ValueError(f"Need {required} unique prompt IDs, got {len(set(prompt_ids))}")
    shuffled = list(dict.fromkeys(prompt_ids))
    random.Random(seed).shuffle(shuffled)
    entries = []
    for flat_index, prompt_id in enumerate(shuffled[:required]):
        round_index = flat_index // prompts_per_round
        candidate_seeds = tuple(
            _seed_from_parts(seed, round_index, prompt_id, candidate_index)
            for candidate_index in range(candidate_k)
        )
        entries.append(PromptScheduleEntry(round_index, prompt_id, candidate_seeds))
    return entries


def assert_same_schedule(left: Sequence[PromptScheduleEntry], right: Sequence[PromptScheduleEntry]) -> None:
    if list(left) != list(right):
        raise ValueError("Training arms do not share the same prompt/candidate schedule")
