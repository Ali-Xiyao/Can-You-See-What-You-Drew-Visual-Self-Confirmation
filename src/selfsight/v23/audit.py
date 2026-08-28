"""Evidence-integrity audit for a completed or fail-closed v2.3 gradient gate."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from selfsight.schemas import BlindObservationRequest, CandidateRecord, SelectionDecision
from selfsight.utils.hashing import rgb_sha256, sha256_file
from selfsight.utils.jsonl import atomic_write_json, read_jsonl
from selfsight.v23.selection import finite_score_span


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _decision(value: dict[str, Any]) -> SelectionDecision:
    return SelectionDecision(
        prompt_id=str(value["prompt_id"]),
        arm=str(value["arm"]),
        candidate_pool_ids=tuple(str(item) for item in value["candidate_pool_ids"]),
        selected_candidate_id=(
            str(value["selected_candidate_id"])
            if value.get("selected_candidate_id") is not None
            else None
        ),
        scores={str(key): float(score) for key, score in value["scores"].items()},
        selector_id=str(value["selector_id"]),
        observer_revision=str(value["observer_revision"]),
        abstain=bool(value.get("abstain", False)),
        reason=str(value.get("reason", "")),
    )


def audit_v23_gradient_gate(
    run_dir: str | Path,
    destination: str | Path,
) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    gate_path = root / "gradient_gate.json"
    gate = _json(gate_path)
    if gate.get("stage") != "v23_gradient_survival_gate":
        raise RuntimeError("Not a v2.3 gradient survival report")
    repeat = gate["repeats"][0]
    repeat_dir = root / "repeat-01"
    packet_paths = sorted((repeat_dir / "prepared").glob("*.json"))
    packets = [_json(path) for path in packet_paths]
    candidates = [
        CandidateRecord.from_dict(value)
        for value in read_jsonl(repeat_dir / "candidate_manifest.jsonl")
    ]
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    image_paths = [str(Path(candidate.image_path).resolve()) for candidate in candidates]
    rgb_valid = all(
        Path(candidate.image_path).is_file()
        and rgb_sha256(candidate.image_path) == candidate.rgb_sha256
        for candidate in candidates
    )
    processed = int(gate["early_stop"]["processed_prompts"])
    expected_candidates = processed * 4

    selections = {
        arm: list(read_jsonl(repeat_dir / f"selection_{arm}.jsonl"))
        for arm in ("naive", "rfo_self", "rfo_gold")
    }
    packet_by_prompt = {str(packet["scene_id"]): packet for packet in packets}
    informative_ids = list(_json(repeat_dir / "informative_filter.json")["accepted_prompt_ids"])
    advantage_values = []
    for prompt_id in informative_ids:
        packet = packet_by_prompt[prompt_id]
        naive = _decision(packet["decisions"]["naive"])
        gold = _decision(packet["decisions"]["rfo_gold"])
        if naive.selected_candidate_id is None or gold.selected_candidate_id is None:
            continue
        naive_score = float(gold.scores[naive.selected_candidate_id])
        gold_score = float(gold.scores[gold.selected_candidate_id])
        if math.isfinite(naive_score) and math.isfinite(gold_score):
            advantage_values.append(gold_score - naive_score)

    family_screened: Counter[str] = Counter()
    family_informative: Counter[str] = Counter()
    box_vocabulary_clean = True
    for packet in packets:
        family = str(packet["atom"]["family"])
        family_screened[family] += 1
        if finite_score_span(_decision(packet["decisions"]["rfo_gold"])) == 1.0:
            family_informative[family] += 1
        visible = [str(packet["question"]["text"])]
        visible.extend(str(value["metadata"].get("prompt", "")) for value in packet["candidates"])
        box_vocabulary_clean = box_vocabulary_clean and all(
            "square" not in text.lower() for text in visible
        )

    wire_path = root / "frozen_observer_wire.jsonl"
    wire_rows = list(read_jsonl(wire_path))
    requests = [BlindObservationRequest.from_wire(value) for value in wire_rows]
    wire_paths_inside = True
    wire_rgb_valid = True
    for request in requests:
        image = Path(request.image_path).resolve()
        try:
            image.relative_to(root)
        except ValueError:
            wire_paths_inside = False
        wire_rgb_valid = wire_rgb_valid and image.is_file() and rgb_sha256(image) == request.rgb_sha256
    request_counts = Counter(request.request_id for request in requests)
    retry_rows = sum(count - 1 for count in request_counts.values() if count > 1)

    noise = repeat["noise_floor"]
    gold_cosine = float(repeat["naive_gold"]["cosine"])
    checks = {
        "gate_is_fail_closed": gate.get("passed") is False and gate.get("stop_rule") is not None,
        "no_model_download_or_a800": gate.get("model_downloads_used") is False
        and gate.get("a800_used") is False,
        "packet_count_matches_early_stop": len(packets) == processed,
        "candidate_count_is_k4": len(candidates) == expected_candidates,
        "candidate_ids_unique": len(candidate_ids) == len(set(candidate_ids)),
        "candidate_paths_unique": len(image_paths) == len(set(image_paths)),
        "candidate_rgb_hashes_valid": rgb_valid,
        "three_selection_logs_aligned": all(len(values) == processed for values in selections.values()),
        "box_vocabulary_has_no_square_wording": box_vocabulary_clean,
        "blind_wire_schema_valid": len(requests) == len(wire_rows),
        "blind_wire_paths_inside_run": wire_paths_inside,
        "blind_wire_rgb_hashes_valid": wire_rgb_valid,
        "identical_control_passed": repeat["conditions"]["identical_cosine"] is True,
    }
    report = {
        "schema_version": 1,
        "benchmark_version": "2.3",
        "stage": "v23_gradient_evidence_audit",
        "passed": all(checks.values()),
        "checks": checks,
        "counts": {
            "processed_prompts": processed,
            "scheduled_prompts": int(repeat["probe_prompts_scheduled"]),
            "candidate_records": len(candidates),
            "informative_prompts": len(informative_ids),
            "wire_rows": len(wire_rows),
            "wire_unique_requests": len(request_counts),
            "wire_retry_rows": retry_rows,
        },
        "family_screened": dict(sorted(family_screened.items())),
        "family_informative": dict(sorted(family_informative.items())),
        "gold_score_advantage": {
            "mean": sum(advantage_values) / len(advantage_values),
            "finite_denominator": len(advantage_values),
            "informative_pool_denominator": len(informative_ids),
        },
        "gradient_interpretation": {
            "identical_cosine": repeat["identical"]["cosine"],
            "naive_self_cosine": repeat["naive_self"]["cosine"],
            "naive_gold_cosine": gold_cosine,
            "noise_interval": noise,
            "naive_gold_inside_split_half_interval": noise["low"] <= gold_cosine <= noise["high"],
        },
        "evidence": {
            "gradient_gate": {"path": str(gate_path), "sha256": sha256_file(gate_path)},
            "candidate_manifest": {
                "path": str((repeat_dir / "candidate_manifest.jsonl").resolve()),
                "sha256": sha256_file(repeat_dir / "candidate_manifest.jsonl"),
            },
            "informative_filter": {
                "path": str((repeat_dir / "informative_filter.json").resolve()),
                "sha256": sha256_file(repeat_dir / "informative_filter.json"),
            },
            "blind_wire": {"path": str(wire_path), "sha256": sha256_file(wire_path)},
        },
    }
    atomic_write_json(destination, report)
    return report
