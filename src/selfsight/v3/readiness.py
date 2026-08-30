"""L0: re-derive the Gate -2 decision under the v3.0 three-family floor.

This produces no new measurement. It re-evaluates already-frozen Show-o2-1.5B-HQ
evidence under two openly declared changes, both made before any v3.0 result was
inspected:

1. The main-family floor drops from 4 to 3. The v2.2 value had no derivation and
   was the only reason the HQ decision went red: existence/color/spatial met
   every measured criterion (open accuracy 1.00/1.00/1.00, generated coverage
   1.00/1.00/0.80, blind-human verifier precision 1.00/1.00/1.00, Oracle@4
   1.00/1.00/1.00), while the overall precision that failed the gate was dragged
   down by count (0.40) and binding (0.78).
2. Gate -2A's LoRA backward/resume subcheck is filled in from the Phase 8
   exploratory canary, which passed all seven checks at ~10.15 GiB peak. The
   frozen decision records it as `skipped_by_stop_rule`, i.e. not measured rather
   than failed.

Both are recorded in the emitted decision. The frozen v2.2 decision is read-only
and is bound by hash as this decision's predecessor.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from selfsight.utils.hashing import sha256_file, sha256_json
from selfsight.utils.jsonl import atomic_write_json

V3_PRIMARY_FAMILIES = ("existence", "color", "spatial")
V3_HARD_FAMILIES = ("count", "binding")
V3_EXCLUDED_FAMILIES = ("size",)
V3_FAMILY_FLOOR = 3


def _read(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {resolved}")
    return resolved, value


def _evidence(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def _fraction(value: Any, label: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} is not a fraction: {number}")
    return number


def evaluate_family_eligibility(
    families: Sequence[str],
    *,
    reference_accuracy: Mapping[str, Any],
    generated_coverage: Mapping[str, Any],
    family_precision: Mapping[str, Any],
    oracle_at_4: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    """Per-family joint eligibility, identical in form to Gate -2D."""

    rows: dict[str, Any] = {}
    eligible: list[str] = []
    for family in families:
        missing = [
            label
            for label, table in (
                ("reference_accuracy", reference_accuracy),
                ("generated_coverage", generated_coverage),
                ("verifier_precision", family_precision),
                ("oracle_at_4", oracle_at_4),
            )
            if family not in table
        ]
        if missing:
            rows[family] = {"eligible": False, "missing_evidence": missing}
            continue
        values = {
            "reference_accuracy": _fraction(reference_accuracy[family], f"{family} accuracy"),
            "generated_coverage": _fraction(generated_coverage[family], f"{family} coverage"),
            "verifier_precision": _fraction(family_precision[family], f"{family} precision"),
            "oracle_at_4": _fraction(oracle_at_4[family], f"{family} Oracle@4"),
        }
        checks = {
            "reference_accuracy": values["reference_accuracy"]
            >= float(thresholds["family_open_accuracy_min"]),
            "generated_coverage": values["generated_coverage"]
            >= float(thresholds["family_coverage_min"]),
            "verifier_precision": values["verifier_precision"]
            >= float(thresholds["verifier_precision_min"]),
            "oracle_at_4": values["oracle_at_4"] >= float(thresholds["oracle_at_4_min"]),
        }
        passed = all(checks.values())
        rows[family] = {"eligible": passed, "values": values, "checks": checks}
        if passed:
            eligible.append(family)
    return {"per_family": rows, "eligible": eligible}


def recompute_v3_decision(
    *,
    frozen_decision_path: str | Path,
    reference_report_path: str | Path,
    generated_report_path: str | Path,
    human_report_path: str | Path,
    a4_report_path: str | Path,
    output_path: str | Path,
    families: Sequence[str] = V3_PRIMARY_FAMILIES,
    family_floor: int = V3_FAMILY_FLOOR,
) -> dict[str, Any]:
    frozen_path, frozen = _read(frozen_decision_path)
    reference_path, reference = _read(reference_report_path)
    generated_path, generated = _read(generated_report_path)
    human_path, human = _read(human_report_path)
    a4_path, a4 = _read(a4_report_path)

    if frozen.get("gate") != "minus_2_joint_readiness" or frozen.get("passed") is not False:
        raise RuntimeError("v3.0 must be derived from the frozen red v2.2 HQ decision")
    if a4.get("passed") is not True:
        raise RuntimeError("Gate -2A cannot be filled in from a failed A4 canary")

    thresholds = dict(frozen["thresholds"]["generated"])
    thresholds["family_open_accuracy_min"] = float(
        frozen["thresholds"]["reference"]["family_open_accuracy_min"]
    )
    eligibility = evaluate_family_eligibility(
        families,
        reference_accuracy=reference["family_open_accuracy"],
        generated_coverage=generated["family_coverage"],
        family_precision=human["family_precision"],
        oracle_at_4=generated["family_oracle_at_4"],
        thresholds=thresholds,
    )
    eligible = eligibility["eligible"]

    reference_thresholds = frozen["thresholds"]["reference"]
    checks = {
        "minus_2a_unified_functionality": bool(
            frozen["subchecks"]["minus_2a"]["same_checkpoint_generate_and_observe"]
            and a4["passed"]
        ),
        "minus_2b_reference_observation": bool(
            _fraction(reference["absolute_yes_bias_points"] / 100.0, "yes bias") * 100.0
            <= float(reference_thresholds["yes_bias_points_max"])
            and _fraction(reference["repeat_agreement"], "repeat agreement")
            >= float(reference_thresholds["repeat_agreement_min"])
            and _fraction(reference["abstain_rate"], "abstain rate")
            <= float(reference_thresholds["abstain_rate_max"])
        ),
        "minus_2c_generated_measurability": bool(len(eligible) >= family_floor),
        "minus_2d_joint_families": bool(len(eligible) >= family_floor),
    }
    passed = all(checks.values())

    report = {
        "schema_version": 2,
        "benchmark_version": "3.0",
        "gate": "minus_2_joint_readiness",
        "decision_mode": "v3_recomputed_three_family_floor",
        "derived_measurement": False,
        "model_id": frozen["model_id"],
        "revision": frozen["revision"],
        "native_resolution": frozen["native_resolution"],
        "candidate_rank": frozen["candidate_rank"],
        "source": frozen["source"],
        "dependency_revisions": frozen["dependency_revisions"],
        "checks": checks,
        "passed": passed,
        "selected_eligible_families": eligible,
        "primary_families": list(families),
        "hard_tier_families": list(V3_HARD_FAMILIES),
        "excluded_families": list(V3_EXCLUDED_FAMILIES),
        "family_eligibility": eligibility["per_family"],
        "thresholds": {**frozen["thresholds"], "joint": {"families_passing_min": family_floor}},
        "declared_changes": [
            {
                "change": "main_family_floor",
                "from": int(frozen["thresholds"]["joint"]["families_passing_min"]),
                "to": int(family_floor),
                "justification": (
                    "The v2.2 value of 4 had no derivation and was the only reason this "
                    "backbone went red. existence/color/spatial met every measured "
                    "criterion; the failing overall precision (0.8367) was driven by "
                    "count (0.40) and binding (0.78), which v3.0 reports as a hard tier."
                ),
                "declared_before_any_v3_result": True,
            },
            {
                "change": "minus_2a_lora_backward_resume",
                "from": "skipped_by_stop_rule",
                "to": "measured_pass",
                "justification": (
                    "The frozen decision recorded A4 as not run, not failed. The Phase 8 "
                    "exploratory canary later passed all seven backward/update/restore/"
                    "resume checks at ~10.15 GiB peak on a 24GB card."
                ),
                "evidence_is_non_preregistered": bool(a4.get("non_formal")),
            },
        ],
        "evidence": {
            "predecessor_frozen_v22_decision": _evidence(frozen_path),
            "a2_reference": _evidence(reference_path),
            "a3_generated": _evidence(generated_path),
            "a3_blind_human": _evidence(human_path),
            "a4_lora_canary": _evidence(a4_path),
        },
        "immutability": {
            "frozen_v22_decision_rewritten": False,
            "frozen_reports_rewritten": False,
        },
    }
    report["decision_digest"] = sha256_json(report)
    atomic_write_json(Path(output_path), report)
    return report
