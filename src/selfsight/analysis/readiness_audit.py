"""Pure summaries used by the GPU-backed Gate -2 audit runners."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def summarize_reference_rows(
    rows: Sequence[Mapping[str, Any]], thresholds: Mapping[str, Any]
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Reference readiness audit has no rows")
    family_correct: dict[str, list[bool]] = defaultdict(list)
    repeat_pairs = []
    forced_expected = []
    forced_predicted = []
    abstentions = []
    for row in rows:
        family = str(row["family"])
        open_first = row.get("open_prediction")
        open_repeat = row.get("open_repeat_prediction")
        family_correct[family].append(open_first == row.get("expected"))
        repeat_pairs.append(open_first == open_repeat)
        abstentions.append(open_first is None)
        for prediction, expected in zip(
            row.get("forced_predictions", ()), row.get("forced_expected", ())
        ):
            abstentions.append(prediction is None)
            if expected in {"yes", "no"}:
                forced_expected.append(expected)
                forced_predicted.append(prediction)
    family_accuracy = {
        family: float(np.mean(values)) for family, values in sorted(family_correct.items())
    }
    passing = [
        family
        for family, accuracy in family_accuracy.items()
        if accuracy >= float(thresholds["family_open_accuracy_min"])
    ]
    expected_yes = (
        float(np.mean([value == "yes" for value in forced_expected]))
        if forced_expected
        else 0.0
    )
    predicted_yes = (
        float(np.mean([value == "yes" for value in forced_predicted]))
        if forced_predicted
        else 0.0
    )
    yes_bias_points = abs(predicted_yes - expected_yes) * 100.0
    repeat_agreement = float(np.mean(repeat_pairs))
    abstain_rate = float(np.mean(abstentions))
    checks = {
        "families_passing": len(passing) >= int(thresholds["families_passing_min"]),
        "yes_bias": yes_bias_points <= float(thresholds["yes_bias_points_max"]),
        "repeat_agreement": repeat_agreement >= float(thresholds["repeat_agreement_min"]),
        "abstain_rate": abstain_rate <= float(thresholds["abstain_rate_max"]),
    }
    return {
        "images": len(rows),
        "family_open_accuracy": family_accuracy,
        "passing_families": passing,
        "macro_open_accuracy": float(np.mean(list(family_accuracy.values()))),
        "expected_yes_rate": expected_yes,
        "predicted_yes_rate": predicted_yes,
        "absolute_yes_bias_points": yes_bias_points,
        "repeat_agreement": repeat_agreement,
        "abstain_rate": abstain_rate,
        "checks": checks,
        "passed": all(checks.values()),
    }


def summarize_generated_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    families: Sequence[str],
    oracle_families: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Summarize deterministic verifier coverage/correctness without treating it as human truth."""

    if not rows:
        raise ValueError("Generated readiness audit has no rows")
    by_prompt: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_prompt[str(row["scene_id"])].append(row)
    family_prompts: dict[str, list[list[Mapping[str, Any]]]] = defaultdict(list)
    for candidates in by_prompt.values():
        family_prompts[str(candidates[0]["family"])].append(candidates)

    family_coverage = {}
    family_oracle = {}
    oracle_family_set = set(oracle_families or families)
    missing_oracle = sorted(oracle_family_set.difference(families))
    if missing_oracle:
        raise ValueError(f"Oracle families are not registered main families: {missing_oracle}")
    seed_coverages: dict[int, list[bool]] = defaultdict(list)
    for family in families:
        prompts = family_prompts.get(family, [])
        if not prompts:
            raise RuntimeError(f"Generated readiness rows are missing family: {family}")
        first = [min(items, key=lambda row: int(row["candidate_index"])) for items in prompts]
        family_coverage[family] = float(
            np.mean([bool(row["primary_answer_covered"]) for row in first])
        )
        family_oracle[family] = float(
            np.mean([any(bool(row["primary_correct"]) for row in items) for items in prompts])
        )
        if family in oracle_family_set:
            for items in prompts:
                for row in items:
                    seed_coverages[int(row["candidate_index"])].append(
                        bool(row["primary_answer_covered"])
                    )
    all_prompts = list(by_prompt.values())
    first_rows = [
        min(items, key=lambda row: int(row["candidate_index"])) for items in all_prompts
    ]
    seed_rates = {
        str(index): float(np.mean(values)) for index, values in sorted(seed_coverages.items())
    }
    swing = (max(seed_rates.values()) - min(seed_rates.values())) * 100.0 if seed_rates else 0.0
    oracle_prompts = [
        items for items in all_prompts if str(items[0]["family"]) in oracle_family_set
    ]
    if not oracle_prompts:
        raise RuntimeError("No generated prompts belong to the Oracle@K evaluation families")
    expected_indices = set(range(max(seed_coverages) + 1))
    if set(seed_coverages) != expected_indices:
        raise RuntimeError("Oracle families have incomplete candidate indices")
    if any(len(seed_coverages[index]) != len(oracle_prompts) for index in expected_indices):
        raise RuntimeError("Oracle-family fixed-seed coverage has unequal denominators")
    return {
        "prompts": len(all_prompts),
        "candidates": len(rows),
        "overall_coverage": float(
            np.mean([bool(row["primary_answer_covered"]) for row in first_rows])
        ),
        "family_coverage": family_coverage,
        "overall_oracle_at_4": float(
            np.mean(
                [any(bool(row["primary_correct"]) for row in items) for items in oracle_prompts]
            )
        ),
        "family_oracle_at_4": family_oracle,
        "oracle_evaluation_families": sorted(oracle_family_set),
        "oracle_evaluation_prompts": len(oracle_prompts),
        "coverage_by_candidate_index": seed_rates,
        "fixed_seed_coverage_swing_points": swing,
    }
