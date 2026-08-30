"""Selection headroom: what any selector could buy on this backbone, before training.

Gate C asks whether a perfect selector (RFO-Gold) beats Naive on external
correctness after training. The only mechanism by which it can is that on some
pools Gold feeds a verifier-correct image where Naive feeds a wrong one. That
per-step difference is measurable at step 0, with no training and no optimizer:

    Oracle@K  - the perfect selector's score, i.e. "at least one of the K is correct"
    Naive@K   - the score of whatever the naive criterion picks
    natural   - the score of an unselected draw

`Oracle@K - Naive@K` is the driving signal Gate C would have to amplify. Training
can dilute or compound it, so this is not a strict upper bound, but a signal of
zero has nothing to compound. Measuring it costs hours instead of the 200-320
A800 GPU-hours of Phase 5.

The same numbers are the capability-floor row the paper needs regardless
(proposal Figure 2), so this is not a detour around the registered gate.

Gold and Oracle@K are the same quantity here by construction: the gold selector
picks a verifier-correct candidate whenever the pool contains one.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

GO_THRESHOLD = 0.05
STOP_THRESHOLD = 0.03
"""Registered before any headroom number was read.

    > GO_THRESHOLD    Gate C has a real mechanism to amplify; run Phase 3/4/5.
    < STOP_THRESHOLD  the driving signal is under the paired noise Gate C would
                      have to resolve; diagnose the objective before spending
                      A800 hours.
    between           enlarge the prompt set and re-measure.
"""


@dataclass(frozen=True)
class PromptHeadroom:
    """One prompt's natural K-draw, scored by the programmatic verifier."""

    prompt_id: str
    family: str
    gold_scores: tuple[float, ...]
    candidate_ids: tuple[str, ...]
    selected_candidate_id: str | None = None

    @property
    def scored(self) -> tuple[float, ...]:
        return tuple(value for value in self.gold_scores if math.isfinite(value))

    @property
    def usable(self) -> bool:
        return len(self.scored) > 0

    @property
    def natural(self) -> float:
        """Expected score of one unselected draw from this pool."""

        return sum(self.scored) / len(self.scored)

    @property
    def first_candidate(self) -> float:
        return self.gold_scores[0]

    @property
    def oracle(self) -> float:
        return 1.0 if any(value > 0.0 for value in self.scored) else 0.0

    @property
    def informative(self) -> bool:
        return any(value > 0.0 for value in self.scored) and any(
            value == 0.0 for value in self.scored
        )

    @property
    def selected(self) -> float | None:
        if self.selected_candidate_id is None:
            return None
        index = self.candidate_ids.index(self.selected_candidate_id)
        value = self.gold_scores[index]
        return value if math.isfinite(value) else None


def load_natural_pools(
    packet_dirs: Sequence[str | Path],
    *,
    candidate_k: int,
) -> list[PromptHeadroom]:
    """Read the natural K-draw of every bank packet.

    The bank probe writes M=16 candidates per prompt and treats the first
    `candidate_k` as the unfiltered natural draw (`v3.supply.run_bank_probe`).
    Using the same slice keeps this measurement on the same pools as Gate A
    rather than on a differently-sampled set.
    """

    rows: list[PromptHeadroom] = []
    seen: set[str] = set()
    for directory in packet_dirs:
        for path in sorted(Path(directory).glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            prompt_id = str(payload["scene_id"])
            if prompt_id in seen:
                raise ValueError(f"Duplicate prompt across packet dirs: {prompt_id}")
            seen.add(prompt_id)
            scored = payload["scored"][:candidate_k]
            if len(scored) < candidate_k:
                raise ValueError(f"{path.name} has fewer than {candidate_k} candidates")
            rows.append(
                PromptHeadroom(
                    prompt_id=prompt_id,
                    family=str(payload["family"]),
                    gold_scores=tuple(float(item["gold_score"]) for item in scored),
                    candidate_ids=tuple(str(item["candidate_id"]) for item in scored),
                )
            )
    return rows


def attach_selections(
    rows: Sequence[PromptHeadroom],
    selections: Mapping[str, str | None],
) -> list[PromptHeadroom]:
    """Return rows carrying the candidate a criterion picked, keyed by prompt."""

    return [
        replace(row, selected_candidate_id=selections.get(row.prompt_id))
        for row in rows
    ]


def _mean(values: Iterable[float]) -> float | None:
    collected = [value for value in values if value is not None]
    return sum(collected) / len(collected) if collected else None


def _family_block(rows: Sequence[PromptHeadroom]) -> dict[str, Any]:
    usable = [row for row in rows if row.usable]
    selected = [row.selected for row in usable if row.selected is not None]
    natural = _mean(row.natural for row in usable)
    oracle = _mean(row.oracle for row in usable)
    naive = _mean(selected) if selected else None
    block: dict[str, Any] = {
        "prompts": len(rows),
        "usable_prompts": len(usable),
        "informative_pools": sum(row.informative for row in usable),
        "informative_rate": (
            sum(row.informative for row in usable) / len(usable) if usable else None
        ),
        "first_candidate_rate": _mean(row.first_candidate for row in usable),
        "natural_rate": natural,
        "oracle_at_k": oracle,
        "naive_cycle_rate": naive,
        "naive_scored_prompts": len(selected),
        "selection_ceiling": (
            oracle - natural if oracle is not None and natural is not None else None
        ),
        "headroom_over_naive": (
            oracle - naive if oracle is not None and naive is not None else None
        ),
    }
    return block


def _verdict(headroom: float | None) -> str:
    if headroom is None:
        return "pending_naive_selection"
    if headroom > GO_THRESHOLD:
        return "go"
    if headroom < STOP_THRESHOLD:
        return "stop"
    return "enlarge"


def headroom_report(
    rows: Sequence[PromptHeadroom],
    *,
    candidate_k: int,
) -> dict[str, Any]:
    """Per-family and overall selection headroom, with the registered verdict."""

    if not rows:
        raise ValueError("No prompts to summarize")
    families = sorted({row.family for row in rows})
    overall = _family_block(rows)
    return {
        "schema_version": 1,
        "benchmark_version": "3.0",
        "stage": "v3_selection_headroom",
        "candidate_k": candidate_k,
        "decision_rule": {
            "go_threshold": GO_THRESHOLD,
            "stop_threshold": STOP_THRESHOLD,
            "quantity": "oracle_at_k - naive_cycle_rate",
            "registered_before_measurement": True,
        },
        "overall": overall,
        "by_family": {
            family: _family_block([row for row in rows if row.family == family])
            for family in families
        },
        "verdict": _verdict(overall["headroom_over_naive"]),
        "gold_equals_oracle_note": (
            "RFO-Gold picks a verifier-correct candidate whenever one exists, so "
            "oracle_at_k is exactly the gold selector's score on these pools."
        ),
    }
