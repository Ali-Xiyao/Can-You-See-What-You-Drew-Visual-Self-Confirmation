"""Bounded candidate-bank search and Gate A supply accounting.

Measured on Show-o2-1.5B-HQ, random K=4 sampling produced an informative pool
(one verifier-correct and one verifier-incorrect candidate) for only 9 of 42
prompts -- 21%, split existence 6/16, color 2/13, spatial 1/13. At that rate a
selection-based self-training experiment has almost no variance to exploit and
the registered probe floor is unreachable. See `docs/EVIDENCE_LOG.md` section 4.

v3.0 therefore searches a bounded bank of ``M`` candidates per prompt and exposes
a balanced K=4 subset to every selector. Two invariants make this legitimate
rather than a way of manufacturing an effect:

* the bank is a *calibration stimulus for the gradient probe only*. Training and
  the external-correctness curve use natural K=4 sampling, and the two must be
  reported separately (proposal section 6.1);
* search seeds are drawn from a domain disjoint from the gradient and dropout
  seeds, and the bank cap is fixed before any gradient output is inspected.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

SEARCH_SEED_BASE = 700_000_000
GRADIENT_SEED_BASE = 900_000_000
DEFAULT_BANK_SIZE = 16
DEFAULT_K = 4
DEFAULT_MIN_PER_SIDE = 2

GATE_A_MIN_BALANCED_RATE = 0.40
GATE_A_MIN_INFORMATIVE_POOLS = 128


@dataclass(frozen=True)
class BankCandidate:
    """One searched candidate scored by the deterministic generated-image verifier."""

    candidate_id: str
    sampling_seed: int
    gold_score: float
    abstained: bool = False

    @property
    def correct(self) -> bool:
        return not self.abstained and math.isfinite(self.gold_score) and self.gold_score >= 1.0

    @property
    def incorrect(self) -> bool:
        return not self.abstained and math.isfinite(self.gold_score) and self.gold_score < 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "sampling_seed": self.sampling_seed,
            "gold_score": self.gold_score,
            "abstained": self.abstained,
        }


@dataclass(frozen=True)
class BalancedPool:
    """A K-candidate pool presented unchanged to every selection criterion."""

    prompt_id: str
    family: str
    candidate_ids: tuple[str, ...]
    correct_ids: tuple[str, ...]
    incorrect_ids: tuple[str, ...]
    searched: int
    abstained: int
    informative: bool
    balanced: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "family": self.family,
            "candidate_ids": list(self.candidate_ids),
            "correct_ids": list(self.correct_ids),
            "incorrect_ids": list(self.incorrect_ids),
            "searched": self.searched,
            "abstained": self.abstained,
            "informative": self.informative,
            "balanced": self.balanced,
            "reason": self.reason,
        }


def assert_disjoint_seed_domains(
    search_seeds: Iterable[int],
    reserved_seeds: Iterable[int],
) -> None:
    """Search seeds must never collide with gradient/dropout seeds.

    A collision would make the bank search share randomness with the measurement,
    which is the cheapest way to fabricate an apparent effect.
    """

    search = {int(value) for value in search_seeds}
    reserved = {int(value) for value in reserved_seeds}
    overlap = sorted(search.intersection(reserved))
    if overlap:
        raise ValueError(f"Search seeds collide with reserved gradient/dropout seeds: {overlap[:8]}")


def search_seeds(prompt_key: str | int, bank_size: int = DEFAULT_BANK_SIZE) -> tuple[int, ...]:
    """Deterministic search-seed block for one prompt, disjoint from the gradient domain.

    Keyed by a stable hash of the prompt/scene identifier rather than by position,
    so that sharding the probe across GPUs, resuming it, or reordering the
    manifest cannot make two prompts share a seed block.
    """

    if bank_size <= 0:
        raise ValueError("bank_size must be positive")
    span = GRADIENT_SEED_BASE - SEARCH_SEED_BASE
    if bank_size > span:
        raise ValueError("bank_size exceeds the reserved search-seed domain")
    digest = hashlib.sha256(str(prompt_key).encode("utf-8")).digest()
    block = int.from_bytes(digest[:8], "big") % (span // bank_size)
    start = SEARCH_SEED_BASE + block * bank_size
    return tuple(start + offset for offset in range(bank_size))


def _ordered(candidates: Sequence[BankCandidate]) -> list[BankCandidate]:
    return sorted(candidates, key=lambda item: (item.sampling_seed, item.candidate_id))


def build_balanced_pool(
    prompt_id: str,
    family: str,
    candidates: Sequence[BankCandidate],
    *,
    k: int = DEFAULT_K,
    min_per_side: int = DEFAULT_MIN_PER_SIDE,
) -> BalancedPool:
    """Take a balanced K-subset out of a searched bank, deterministically.

    Preference order is `min_per_side` correct plus `min_per_side` incorrect. When
    one side is short the pool is topped up from the other side so that K is
    always met; the pool is then still `informative` (it can distinguish
    selectors) but not `balanced`. A pool with zero candidates on either side is
    neither, and the prompt must be dropped from every arm symmetrically.
    """

    if k <= 1:
        raise ValueError("k must be at least 2")
    if min_per_side < 1 or 2 * min_per_side > k:
        raise ValueError("min_per_side must be >= 1 and at most k/2")
    ordered = _ordered(candidates)
    abstained = sum(1 for item in ordered if item.abstained)
    correct = [item for item in ordered if item.correct]
    incorrect = [item for item in ordered if item.incorrect]

    def empty(reason: str) -> BalancedPool:
        return BalancedPool(
            prompt_id=prompt_id,
            family=family,
            candidate_ids=(),
            correct_ids=(),
            incorrect_ids=(),
            searched=len(ordered),
            abstained=abstained,
            informative=False,
            balanced=False,
            reason=reason,
        )

    if len(ordered) < k:
        return empty("bank_smaller_than_k")
    if not correct:
        return empty("no_verifier_correct_candidate")
    if not incorrect:
        return empty("no_verifier_incorrect_candidate")
    if len(correct) + len(incorrect) < k:
        return empty("too_many_abstentions")

    chosen_correct = correct[:min_per_side]
    chosen_incorrect = incorrect[:min_per_side]
    selected = [*chosen_correct, *chosen_incorrect]
    remaining = [
        item
        for item in ordered
        if item not in selected and (item.correct or item.incorrect)
    ]
    for item in remaining:
        if len(selected) >= k:
            break
        selected.append(item)
    selected = _ordered(selected[:k])
    final_correct = tuple(item.candidate_id for item in selected if item.correct)
    final_incorrect = tuple(item.candidate_id for item in selected if item.incorrect)
    balanced = len(final_correct) >= min_per_side and len(final_incorrect) >= min_per_side
    return BalancedPool(
        prompt_id=prompt_id,
        family=family,
        candidate_ids=tuple(item.candidate_id for item in selected),
        correct_ids=final_correct,
        incorrect_ids=final_incorrect,
        searched=len(ordered),
        abstained=abstained,
        informative=bool(final_correct and final_incorrect),
        balanced=balanced,
        reason="balanced" if balanced else "informative_but_unbalanced",
    )


def bank_supply_report(
    pools: Sequence[BalancedPool],
    *,
    families: Sequence[str],
    min_balanced_rate: float = GATE_A_MIN_BALANCED_RATE,
    min_informative_pools: int = GATE_A_MIN_INFORMATIVE_POOLS,
    natural_informative_rate: float | None = None,
) -> dict[str, Any]:
    """Gate A: can bounded bank search supply the probe at all?

    `natural_informative_rate` is the measured rate for plain K=4 sampling on the
    same prompts. Reporting both makes the cost of the bank explicit instead of
    hiding it: the bank is only justified if it materially beats natural
    sampling.
    """

    if not pools:
        raise ValueError("Gate A requires at least one searched pool")
    total = len(pools)
    informative = [pool for pool in pools if pool.informative]
    balanced = [pool for pool in pools if pool.balanced]
    by_family: dict[str, dict[str, Any]] = {}
    for family in families:
        subset = [pool for pool in pools if pool.family == family]
        if not subset:
            by_family[family] = {"searched": 0, "informative": 0, "balanced": 0, "rate": None}
            continue
        by_family[family] = {
            "searched": len(subset),
            "informative": sum(1 for pool in subset if pool.informative),
            "balanced": sum(1 for pool in subset if pool.balanced),
            "rate": sum(1 for pool in subset if pool.balanced) / len(subset),
        }
    reasons: dict[str, int] = {}
    for pool in pools:
        reasons[pool.reason] = reasons.get(pool.reason, 0) + 1
    balanced_rate = len(balanced) / total
    searched_candidates = sum(pool.searched for pool in pools)
    rate_ok = balanced_rate >= min_balanced_rate
    supply_ok = len(informative) >= min_informative_pools
    report = {
        "gate": "a_candidate_supply",
        "prompts_searched": total,
        "candidates_searched": searched_candidates,
        "candidates_per_pool": searched_candidates / total,
        "informative_pools": len(informative),
        "balanced_pools": len(balanced),
        "balanced_rate": balanced_rate,
        "informative_rate": len(informative) / total,
        "min_balanced_rate": min_balanced_rate,
        "min_informative_pools": min_informative_pools,
        "balanced_rate_ok": bool(rate_ok),
        "informative_supply_ok": bool(supply_ok),
        "by_family": by_family,
        "rejection_reasons": dict(sorted(reasons.items())),
        "passed": bool(rate_ok and supply_ok),
        "action_if_failed": (
            "Selection-based self-training has no exploitable variance on this backbone. "
            "Stop the mainline and report the capability-floor result; do not raise the "
            "bank cap after inspecting gradients."
        ),
    }
    if natural_informative_rate is not None:
        report["natural_informative_rate"] = float(natural_informative_rate)
        report["bank_lift"] = balanced_rate - float(natural_informative_rate)
    return report


def paired_pool_assignment(
    pools: Sequence[BalancedPool],
    arms: Sequence[str],
) -> dict[str, list[str]]:
    """Every arm receives the identical accepted prompt set, in identical order.

    Any prompt that is not informative is removed from *all* arms, so no arm can
    gain a different prompt count, candidate pool or ordering through abstention.
    """

    if not arms:
        raise ValueError("At least one arm is required")
    accepted = [pool.prompt_id for pool in pools if pool.informative]
    if len(set(accepted)) != len(accepted):
        raise ValueError("Duplicate prompt in accepted pool set")
    return {arm: list(accepted) for arm in arms}


def assert_pools_identical_across_arms(
    presented: Mapping[str, Sequence[Sequence[str]]],
) -> None:
    """Fail loudly if any arm saw a different candidate pool."""

    arms = list(presented)
    if len(arms) < 2:
        return
    reference = [tuple(pool) for pool in presented[arms[0]]]
    for arm in arms[1:]:
        current = [tuple(pool) for pool in presented[arm]]
        if current != reference:
            mismatched = [
                index
                for index, (left, right) in enumerate(zip(reference, current, strict=False))
                if left != right
            ]
            raise ValueError(
                f"Arm {arm} was shown different candidate pools at indices {mismatched[:5]}"
            )
