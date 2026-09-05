"""Describe a dynamic pilot without forcing a breakpoint or confirming a lead.

Reads evaluations/{arm}/step-*/{manifest,verified,s_select}.jsonl and
gradient-probes/**/report.json (also accepts probes/). The windows and thresholds are
fixed before this pilot; every look remains exploratory and unadjusted for
multiple looks. Missing image verdicts are unknown, never negative labels.

    envs/core/python.exe scripts/v4_decoupling_report.py --outdir RUN
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml


RULES = {
    "window_checkpoints": 3,
    "internal_delta_min": 0.02,
    "external_delta_max": 0.02,
    "gradient_drop_min": 0.01,
    "gradient_consecutive_checkpoints": 2,
    "gradient_paired_ci_min_common": 4,
    "confidence_level": 0.95,
}
LIMITATIONS = [
    "exploratory: this report does not confirm D* or predictive utility",
    "multiple looks and arms are not multiplicity-adjusted",
    "observed lead is descriptive; forecasting improvement requires separate validation",
    "complete paired specs may differ from the full outcome population; inspect coverage",
]
NUMERIC_TOLERANCE = 1e-12
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/v4_decoupling_pilot.yaml"
DEFAULT_PROTOCOL = ROOT / "docs/prereg/2026-09-06-decoupling-pilot.md"


def configured_rules(config: dict[str, Any]) -> dict[str, Any]:
    rules = dict(RULES)
    pilot = config.get("pilot", {})
    mapping = {"window_checkpoints": "effect_window_checkpoints",
               "internal_delta_min": "internal_change_margin",
               "external_delta_max": "external_improvement_margin",
               "gradient_drop_min": "gradient_drop_margin",
               "gradient_consecutive_checkpoints": "gradient_confirmation_checkpoints"}
    for key, source in mapping.items():
        if source in pilot:
            rules[key] = pilot[source]
    rules["gradient_paired_ci_min_common"] = config.get("gradient_probe", {}).get("minimum_size", 4)
    for key in ("window_checkpoints", "gradient_consecutive_checkpoints",
                "gradient_paired_ci_min_common"):
        rules[key] = int(rules[key])
        if rules[key] < 2:
            raise ValueError(f"{key} must be at least two")
    for key in ("internal_delta_min", "external_delta_max", "gradient_drop_min"):
        rules[key] = float(rules[key])
        if not 0 <= rules[key] <= 1:
            raise ValueError(f"{key} must be in [0, 1]")
    return rules


def file_identity(path: Path | None) -> dict[str, str] | None:
    return ({"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            if path is not None else None)


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _image_index(rows: Sequence[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row["image_path"])
        if key in result:
            raise ValueError(f"Duplicate image_path in {label}: {key}")
        result[key] = row
    return result


def read_outcome(directory: Path) -> dict[str, Any]:
    """Keep a spec only when all its outcome images have both measurements."""
    step = int(directory.name.removeprefix("step-"))
    manifest_path = directory / "manifest.jsonl"
    verified_path = directory / "verified.jsonl"
    selection_path = directory / "s_select.jsonl"
    manifest = _jsonl(manifest_path)
    _image_index(manifest, str(manifest_path))
    verified = _image_index(_jsonl(verified_path), str(verified_path))
    selection = _image_index(_jsonl(selection_path), str(selection_path))
    grouped: dict[str, list[tuple[float | None, float | None]]] = defaultdict(list)
    counts = {"images": len(manifest), "known_score_images": 0,
              "known_verdict_images": 0, "joint_images": 0,
              "answer_available": 0, "answer_total": 0}
    for row in manifest:
        image_path = str(row["image_path"])
        spec_id = str(row["spec_id"])
        score_row = selection.get(image_path, {})
        recorded_id = score_row.get("prompt_id", score_row.get("spec_id"))
        if recorded_id is not None and str(recorded_id) != spec_id:
            raise ValueError(f"s_select spec mismatch for {image_path}")
        score = _finite(score_row.get("s_select"))
        if score is not None and not 0 <= score <= 1:
            raise ValueError(f"s_select outside [0, 1] for {image_path}")
        verdict_row = verified.get(image_path, {})
        verdict: float | None = None
        # Human unnameable images have a known image-level failure verdict.
        # Their per-question visual facts are unknown, a different quantity.
        if (verdict_row.get("resolution") != "pending_human"
                and isinstance(verdict_row.get("image_correct"), bool)):
            verdict = float(verdict_row["image_correct"])
        counts["known_score_images"] += score is not None
        counts["known_verdict_images"] += verdict is not None
        counts["joint_images"] += score is not None and verdict is not None
        counts["answer_available"] += int(score_row.get("available", 0))
        counts["answer_total"] += int(score_row.get("total", 0))
        grouped[spec_id].append((score, verdict))
    complete = {}
    for spec_id, values in grouped.items():
        if all(score is not None and verdict is not None for score, verdict in values):
            complete[spec_id] = {
                "s_select": sum(score for score, _ in values) / len(values),
                "external": sum(verdict for _, verdict in values) / len(values),
                "n_images": len(values),
            }
    counts["specs"] = len(grouped)
    counts["complete_specs"] = len(complete)
    counts["incomplete_specs"] = len(grouped) - len(complete)
    counts["answer_coverage"] = (counts["answer_available"] / counts["answer_total"]
                                if counts["answer_total"] else None)
    return {"step": step, "arm": directory.parent.name, "directory": str(directory),
            "coverage": counts, "complete_specs": complete,
            "missing_files": [p.name for p in (manifest_path, verified_path, selection_path)
                              if not p.exists()]}


def paired_changes(start: dict[str, Any], end: dict[str, Any], *,
                   bootstrap_samples: int = 2000, seed: int = 20260905) -> dict[str, Any]:
    """Bootstrap paired spec-level changes using the same resamples for both metrics."""
    ids = sorted(set(start["complete_specs"]) & set(end["complete_specs"]))
    n = len(ids)
    out: dict[str, Any] = {"n_paired_specs": n, "paired_spec_ids": ids,
                           "n_start_complete": len(start["complete_specs"]),
                           "n_end_complete": len(end["complete_specs"]),
                           "ci_method": "paired_spec_percentile_bootstrap",
                           "bootstrap_samples": bootstrap_samples}
    delta = np.asarray([[end["complete_specs"][sid][key]
                         - start["complete_specs"][sid][key]
                         for key in ("s_select", "external")] for sid in ids], dtype=float)
    intervals: np.ndarray | None = None
    if n >= 2:
        rng = np.random.default_rng(seed)
        samples = rng.integers(0, n, size=(bootstrap_samples, n))
        intervals = np.quantile(delta[samples].mean(axis=1), [0.025, 0.975], axis=0)
    for index, key in enumerate(("s_select", "external")):
        out[key] = {"delta": float(delta[:, index].mean()) if n else None,
                    "ci_low": float(intervals[0, index]) if intervals is not None else None,
                    "ci_high": float(intervals[1, index]) if intervals is not None else None}
    out["ci_status"] = "available" if n >= 2 else "insufficient_paired_specs"
    return out


def behavior_windows(outcomes: Sequence[dict[str, Any]], *, bootstrap_samples: int = 2000,
                     seed: int = 20260905, rules: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    rules = rules or RULES
    ordered = sorted(outcomes, key=lambda row: row["step"])
    result = []
    width = int(rules["window_checkpoints"])
    for index in range(width - 1, len(ordered)):
        window = ordered[index - width + 1:index + 1]
        changes = paired_changes(window[0], window[-1], bootstrap_samples=bootstrap_samples,
                                 seed=seed + int(window[-1]["step"]))
        internal, external = changes["s_select"], changes["external"]
        candidate = (internal["delta"] is not None and external["delta"] is not None
                     and internal["delta"] >= rules["internal_delta_min"] - NUMERIC_TOLERANCE
                     and external["delta"] <= rules["external_delta_max"] + NUMERIC_TOLERANCE)
        supported = bool(candidate and internal["ci_low"] is not None
                         and internal["ci_low"] > NUMERIC_TOLERANCE
                         and external["ci_high"] <= rules["external_delta_max"] + NUMERIC_TOLERANCE)
        result.append({"window_steps": [row["step"] for row in window],
                       "available_at_step": window[-1]["step"], **changes,
                       "candidate": bool(candidate), "interval_supported": supported,
                       "exploratory": True, "multiple_looks_adjusted": False})
    return result


def gradient_trajectory(reports: Sequence[dict[str, Any]], steps: Sequence[int], *,
                        rules: dict[str, Any] | None = None) -> dict[str, Any]:
    rules = rules or RULES
    by_step: dict[int, dict[str, Any]] = {}
    for row in reports:
        step = int(row["checkpoint"]["step"])
        if step in by_step:
            raise ValueError(f"Duplicate gradient checkpoint {step}")
        by_step[step] = row
    baseline = _finite(by_step.get(0, {}).get("gda_free", {}).get("cosine"))
    all_steps = sorted(set(steps) | set(by_step))
    points = []
    candidate_streak = supported_streak = 0
    candidate_alarm = supported_alarm = None
    for step in all_steps:
        row = by_step.get(step, {})
        cosine = _finite(row.get("gda_free", {}).get("cosine"))
        point_delta = cosine - baseline if cosine is not None and baseline is not None else None
        marginal_delta = point_delta
        delta = row.get("delta_from_reference", {}).get("gda_free", {})
        low, high = _finite(delta.get("ci_low")), _finite(delta.get("ci_high"))
        paired_point = _finite(delta.get("point_difference"))
        paired_ci = (delta.get("reference_step") == 0
                     and "paired" in str(delta.get("method", ""))
                     and int(delta.get("n_common", 0)) >= rules["gradient_paired_ci_min_common"]
                     and paired_point is not None and low is not None and high is not None
                     and low <= high)
        if paired_ci:
            point_delta = paired_point
        point_flag = bool(step > 0 and point_delta is not None
                          and point_delta <= -rules["gradient_drop_min"] + NUMERIC_TOLERANCE)
        supported = bool(point_flag and paired_ci and high < -NUMERIC_TOLERANCE)
        candidate_streak = candidate_streak + 1 if point_flag else 0
        supported_streak = supported_streak + 1 if supported else 0
        # Timestamp is the second confirming checkpoint, never the earlier one.
        if candidate_alarm is None and candidate_streak >= rules["gradient_consecutive_checkpoints"]:
            candidate_alarm = step
        if supported_alarm is None and supported_streak >= rules["gradient_consecutive_checkpoints"]:
            supported_alarm = step
        points.append({"step": step, "cosine": cosine, "delta_from_step0": point_delta,
                       "marginal_point_delta": marginal_delta,
                       "paired_ci_low": low if paired_ci else None,
                       "paired_ci_high": high if paired_ci else None,
                       "n_common": int(delta.get("n_common", 0)),
                       "point_flag": point_flag, "interval_supported_flag": supported,
                       "support_status": ("paired_difference_ci" if paired_ci
                                          else "unsupported_no_paired_difference_ci"),
                       "missing_report": step not in by_step})
    return {"baseline_cosine": baseline, "points": points,
            "candidate_alarm_step": candidate_alarm,
            "interval_supported_alarm_step": supported_alarm,
            "alarm_timestamp": "final_checkpoint_of_required_consecutive_flags",
            "warning": "marginal cosine intervals are never subtracted to obtain a difference CI"}


def classify_lead(event_step: int | None, alarm_step: int | None) -> dict[str, Any]:
    if event_step is None:
        status = "no_event"
    elif alarm_step is None:
        status = "missed"
    elif alarm_step < event_step:
        status = "earlier"
    elif alarm_step == event_step:
        status = "simultaneous"
    else:
        status = "late"
    return {"status": status, "event_step": event_step, "alarm_step": alarm_step,
            "lead_steps": event_step - alarm_step
            if event_step is not None and alarm_step is not None else None,
            "interpretation": "descriptive as of the currently available checkpoints"}


def analyze_arm(outcomes: Sequence[dict[str, Any]], gradients: Sequence[dict[str, Any]], *,
                bootstrap_samples: int = 2000, seed: int = 20260905,
                rules: dict[str, Any] | None = None) -> dict[str, Any]:
    rules = rules or RULES
    windows = behavior_windows(outcomes, bootstrap_samples=bootstrap_samples, seed=seed, rules=rules)
    candidate = next((w["available_at_step"] for w in windows if w["candidate"]), None)
    supported = next((w["available_at_step"] for w in windows if w["interval_supported"]), None)
    trajectory = gradient_trajectory(gradients, [row["step"] for row in outcomes], rules=rules)
    return {"status": "insufficient_checkpoints"
            if len(outcomes) < rules["window_checkpoints"] else "descriptive",
            "checkpoints": [{k: v for k, v in row.items() if k != "complete_specs"}
                            for row in sorted(outcomes, key=lambda row: row["step"])],
            "windows": windows, "first_candidate_step": candidate,
            "first_interval_supported_step": supported, "gradient": trajectory,
            "lead": classify_lead(supported, trajectory["interval_supported_alarm_step"]),
            "lead_reference": "interval_supported_behavior_and_gradient_candidates",
            "point_only_lead": classify_lead(candidate, trajectory["candidate_alarm_step"]),
            "point_only_lead_is_supported": False}


def build_report(outdir: Path, *, bootstrap_samples: int = 2000, seed: int = 20260905,
                 config_path: Path | None = None, protocol_path: Path | None = None) -> dict[str, Any]:
    if bootstrap_samples < 100:
        raise ValueError("At least 100 bootstrap samples are required")
    outcomes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    gradients: dict[str, list[dict[str, Any]]] = defaultdict(list)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path else {}
    rules = configured_rules(config)
    provenance = {"config": file_identity(config_path), "protocol": file_identity(protocol_path)}
    prior_path = outdir / "decoupling_report.json"
    if prior_path.exists():
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        for key in ("config", "protocol"):
            old = prior.get("provenance", {}).get(key)
            new = provenance[key]
            if old and (not new or old["sha256"] != new["sha256"]):
                raise ValueError(f"Frozen {key} changed; use a new run version")
        if prior.get("rules", rules) != rules:
            raise ValueError("Frozen reporting rules changed; use a new run version")
    for directory in sorted((outdir / "evaluations").glob("*/step-*")):
        if directory.is_dir():
            outcome = read_outcome(directory)
            outcomes[outcome["arm"]].append(outcome)
    probe_paths = sorted({path for name in ("gradient-probes", "probes")
                          for path in (outdir / name).glob("**/report.json")})
    for path in probe_paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        if "checkpoint" not in row or "gda_free" not in row:
            continue
        arm = row["checkpoint"].get("arm")
        if arm is None:
            raise ValueError(f"Gradient report has no checkpoint.arm: {path}")
        gradients[str(arm)].append(row)
    shared_base = gradients.pop("base", [])
    if len(shared_base) > 1 or any(int(row["checkpoint"]["step"]) != 0 for row in shared_base):
        raise ValueError("The shared base must be one step-0 gradient report")
    arms = sorted(set(outcomes) | set(gradients))
    for arm in arms:
        if shared_base and not any(int(row["checkpoint"]["step"]) == 0 for row in gradients[arm]):
            gradients[arm] = [*shared_base, *gradients[arm]]
    return {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            "outdir": str(outdir), "exploratory": True, "multiple_looks_adjusted": False,
            "rules": rules, "provenance": provenance,
            "numeric_comparison_tolerance": NUMERIC_TOLERANCE,
            "bootstrap_seed": seed, "limitations": LIMITATIONS,
            "arms": {arm: analyze_arm(outcomes[arm], gradients[arm],
                                       bootstrap_samples=bootstrap_samples, seed=seed, rules=rules)
                     for arm in arms}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    if not args.outdir.is_dir():
        parser.error("--outdir must be an existing pilot directory")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report = build_report(args.outdir,
                          bootstrap_samples=int(config.get("gradient_probe", {}).get("resamples", 2000)),
                          seed=int(config["seed"]), config_path=args.config, protocol_path=args.protocol)
    output = args.outdir / "decoupling_report.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(output)
    print(f"wrote {output}; exploratory, multiple looks not adjusted")
    for arm, row in report["arms"].items():
        print(f"{arm}: behavioral candidate={row['first_candidate_step']}, "
              f"interval-supported={row['first_interval_supported_step']}, "
              f"gradient alarm={row['gradient']['interval_supported_alarm_step']}, "
              f"descriptive lead={row['lead']['status']}")


if __name__ == "__main__":
    main()
