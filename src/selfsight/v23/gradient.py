"""Resumable v2.3 three-repeat RFO-Gold gradient survival gate."""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from selfsight.backbones.showo2 import Showo2Adapter, Showo2GenerationBatch
from selfsight.config import load_config, write_config_snapshot
from selfsight.data.candidates import CandidateManifest
from selfsight.observers.client import ObserverServiceClient
from selfsight.rfo.isolation import make_blind_request
from selfsight.rfo.selection import select_candidate
from selfsight.schemas import (
    Atom,
    AtomicQuestion,
    CandidateRecord,
    QuestionFormat,
    SceneSpec,
    SelectionDecision,
    as_serializable,
)
from selfsight.training.gradients import compare_gradients, noise_interval
from selfsight.utils.cuda import cuda_device_index
from selfsight.utils.evidence import write_host_manifest
from selfsight.utils.hashing import rgb_sha256, sha256_file, sha256_json
from selfsight.utils.jsonl import atomic_write_json, atomic_write_jsonl, read_jsonl
from selfsight.v23.protocol import validate_v23_authorization
from selfsight.v23.selection import (
    V23_ARMS,
    finite_score_span,
    gold_observation,
    gold_selection_advantage,
    select_common_informative,
)


def _stable_seed(*parts: object) -> int:
    return int(sha256_json(list(parts))[:8], 16) & 0x7FFF_FFFF


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _open_question(record: Mapping[str, Any]) -> AtomicQuestion:
    questions = tuple(AtomicQuestion.from_dict(value) for value in record["questions"])
    for question in questions:
        if question.question_format == QuestionFormat.OPEN:
            return question
    raise RuntimeError(f"v2.3 record has no open question: {record['scene']['scene_id']}")


def _decision_from_dict(value: Mapping[str, Any]) -> SelectionDecision:
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


def _validate_prepared_packet(
    packet: Mapping[str, Any],
    *,
    scene_id: str,
    repeat_index: int,
) -> tuple[list[CandidateRecord], dict[str, SelectionDecision]]:
    if packet.get("scene_id") != scene_id or int(packet.get("repeat_index", -1)) != repeat_index:
        raise RuntimeError(f"Prepared v2.3 packet identity mismatch: {scene_id}")
    candidates = [CandidateRecord.from_dict(value) for value in packet["candidates"]]
    if len(candidates) != 4:
        raise RuntimeError(f"Prepared v2.3 packet is not K=4: {scene_id}")
    for candidate in candidates:
        image = Path(candidate.image_path)
        if not image.is_file() or rgb_sha256(image) != candidate.rgb_sha256:
            raise RuntimeError(f"Prepared v2.3 candidate RGB changed: {candidate.candidate_id}")
    decisions = {
        arm: _decision_from_dict(packet["decisions"][arm])
        for arm in V23_ARMS
    }
    return candidates, decisions


def _prepare_repeat(
    *,
    repeat_index: int,
    records: Sequence[Mapping[str, Any]],
    adapter: Showo2Adapter,
    frozen_observer: ObserverServiceClient,
    repeat_dir: Path,
    seed: int,
    candidate_k: int,
    minimum_gold_gap: float,
    minimum_common_informative: int,
) -> tuple[
    list[CandidateRecord],
    dict[str, list[SelectionDecision]],
    dict[str, Any],
    dict[str, SceneSpec],
]:
    if candidate_k != 4:
        raise RuntimeError("v2.3 RFO-Gold protocol is locked to K=4")
    packet_root = repeat_dir / "prepared"
    packet_root.mkdir(parents=True, exist_ok=True)
    all_candidates: list[CandidateRecord] = []
    decisions: dict[str, list[SelectionDecision]] = {arm: [] for arm in V23_ARMS}
    scenes: dict[str, SceneSpec] = {}
    processed_order: list[str] = []
    informative_so_far = 0
    early_stop: dict[str, Any] | None = None
    for record_index, record in enumerate(records):
        scene = SceneSpec.from_dict(dict(record["scene"]))
        atom = Atom.from_dict(dict(record["atom"]))
        question = _open_question(record)
        scenes[scene.scene_id] = scene
        packet_path = packet_root / f"{record_index:03d}-{scene.scene_id}.json"
        if packet_path.is_file():
            candidates, prompt_decisions = _validate_prepared_packet(
                _read_json(packet_path), scene_id=scene.scene_id, repeat_index=repeat_index
            )
        else:
            seeds = tuple(
                _stable_seed("v2.3-gradient-candidate", seed, repeat_index, scene.scene_id, index)
                for index in range(candidate_k)
            )
            generated = adapter.generate_images(
                (scene.prompt,) * candidate_k,
                seeds,
                repeat_dir / "candidates",
                f"v23-gradient-repeat-{repeat_index:02d}",
            )
            candidates = [
                replace(candidate, prompt_id=scene.scene_id, scene_id=scene.scene_id)
                for candidate in generated
            ]
            observations: dict[str, dict[str, Any]] = {arm: {} for arm in V23_ARMS}
            for candidate in candidates:
                observations["naive"][candidate.candidate_id] = adapter.observe_atoms(
                    candidate.image_path, (question,)
                )
                observations["rfo_self"][candidate.candidate_id] = frozen_observer.observe(
                    make_blind_request(
                        candidate.image_path,
                        (question,),
                        f"v23-self-{repeat_index}-{candidate.candidate_id}",
                    )
                )
                observations["rfo_gold"][candidate.candidate_id] = gold_observation(
                    candidate, question, atom
                )
            prompt_decisions = {}
            for arm in V23_ARMS:
                first = next(iter(observations[arm].values()))
                prompt_decisions[arm] = select_candidate(
                    prompt_id=scene.scene_id,
                    arm=arm,
                    candidates=candidates,
                    observations=observations[arm],
                    questions=(question,),
                    selector_id=first.observer_id,
                    observer_revision=first.observer_revision,
                )
            atomic_write_json(
                packet_path,
                {
                    "schema_version": 1,
                    "benchmark_version": "2.3",
                    "repeat_index": repeat_index,
                    "scene_id": scene.scene_id,
                    "question": as_serializable(question),
                    "atom": as_serializable(atom),
                    "candidates": [as_serializable(candidate) for candidate in candidates],
                    "decisions": {
                        arm: as_serializable(prompt_decisions[arm]) for arm in V23_ARMS
                    },
                },
            )
        all_candidates.extend(candidates)
        for arm in V23_ARMS:
            decisions[arm].append(prompt_decisions[arm])
        processed_order.append(scene.scene_id)
        gold_gap = finite_score_span(prompt_decisions["rfo_gold"])
        common_available = all(
            not prompt_decisions[arm].abstain
            and prompt_decisions[arm].selected_candidate_id is not None
            for arm in V23_ARMS
        )
        informative_so_far += int(
            common_available and gold_gap is not None and gold_gap >= minimum_gold_gap
        )
        print(
            f"[v2.3 gradient] repeat={repeat_index + 1} "
            f"prepared={record_index + 1}/{len(records)} scene={scene.scene_id}",
            flush=True,
        )
        remaining = len(records) - len(processed_order)
        maximum_possible = informative_so_far + remaining
        if maximum_possible < minimum_common_informative:
            early_stop = {
                "triggered": True,
                "reason": "minimum_common_informative_is_mathematically_unreachable",
                "processed_prompts": len(processed_order),
                "informative_so_far": informative_so_far,
                "remaining_prompts": remaining,
                "maximum_possible_informative": maximum_possible,
                "required_common_informative": minimum_common_informative,
            }
            print(
                f"[v2.3 gradient] repeat={repeat_index + 1} early-stop="
                f"{informative_so_far}+{remaining}<{minimum_common_informative}",
                flush=True,
            )
            break

    CandidateManifest(repeat_dir / "candidate_manifest.jsonl").write(all_candidates)
    for arm in V23_ARMS:
        atomic_write_jsonl(
            repeat_dir / f"selection_{arm}.jsonl",
            (as_serializable(decision) for decision in decisions[arm]),
        )
    paired, informative = select_common_informative(
        processed_order,
        decisions,
        minimum_gold_gap=minimum_gold_gap,
    )
    informative["scheduled_prompt_count"] = len(records)
    informative["early_stop"] = early_stop
    atomic_write_json(repeat_dir / "informative_filter.json", informative)
    return all_candidates, paired, informative, scenes


def _selected_map(
    decisions: Sequence[SelectionDecision],
    candidates: Sequence[CandidateRecord],
) -> dict[str, CandidateRecord]:
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    return {
        decision.prompt_id: by_id[str(decision.selected_candidate_id)]
        for decision in decisions
        if decision.selected_candidate_id is not None and not decision.abstain
    }


def _batches(
    prompt_ids: Sequence[str],
    selected: Mapping[str, CandidateRecord],
    scenes: Mapping[str, SceneSpec],
    *,
    micro_size: int,
    seed: int,
) -> list[Showo2GenerationBatch]:
    output: list[Showo2GenerationBatch] = []
    for start in range(0, len(prompt_ids), micro_size):
        ids = tuple(prompt_ids[start : start + micro_size])
        output.append(
            Showo2GenerationBatch(
                prompts=tuple(scenes[prompt_id].prompt for prompt_id in ids),
                images=tuple(selected[prompt_id].image_path for prompt_id in ids),
                sample_ids=ids,
                latent_seed=_stable_seed("v2.3-gradient-loss", seed, ids),
            )
        )
    return output


def _selection_agreement(
    left: Sequence[SelectionDecision], right: Sequence[SelectionDecision]
) -> float:
    right_map = {decision.prompt_id: decision for decision in right}
    values = [
        decision.selected_candidate_id == right_map[decision.prompt_id].selected_candidate_id
        for decision in left
        if decision.prompt_id in right_map
    ]
    return sum(values) / len(values) if values else float("nan")


def _gradient_repeat(
    *,
    repeat_index: int,
    adapter: Showo2Adapter,
    candidates: Sequence[CandidateRecord],
    paired: Mapping[str, Sequence[SelectionDecision]],
    informative: Mapping[str, Any],
    scenes: Mapping[str, SceneSpec],
    family_by_prompt: Mapping[str, str],
    repeat_dir: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gradient_config = config["gradient_probe"]
    minimum = int(gradient_config["minimum_common_informative"])
    prompt_ids = list(informative["accepted_prompt_ids"])
    family_counts = Counter(family_by_prompt[prompt_id] for prompt_id in prompt_ids)
    report: dict[str, Any] = {
        "schema_version": 1,
        "benchmark_version": "2.3",
        "stage": "v23_gradient_repeat",
        "repeat_index": repeat_index,
        "probe_prompts": int(informative["prompt_count"]),
        "probe_prompts_scheduled": int(informative["scheduled_prompt_count"]),
        "early_stop": informative.get("early_stop"),
        "common_informative": len(prompt_ids),
        "minimum_common_informative": minimum,
        "family_counts": dict(sorted(family_counts.items())),
        "selection_agreement": {
            "naive_self": _selection_agreement(paired["naive"], paired["rfo_self"]),
            "naive_gold": _selection_agreement(paired["naive"], paired["rfo_gold"]),
            "self_gold": _selection_agreement(paired["rfo_self"], paired["rfo_gold"]),
        },
        "gold_selected_score_advantage": gold_selection_advantage(
            paired["naive"], paired["rfo_gold"]
        ),
    }
    if len(prompt_ids) < 8:
        report.update(
            {
                "passed": False,
                "reason": "Fewer than 8 informative common pools; gradients are not diagnostic",
                "conditions": {"minimum_common_informative": False},
            }
        )
        atomic_write_json(repeat_dir / "gradient_repeat.json", report)
        return report

    selected = {arm: _selected_map(paired[arm], candidates) for arm in V23_ARMS}
    micro_size = int(config["training"]["micro_batch_size"])
    repeat_seed = _stable_seed(config["seed"], "v2.3-gradient-repeat", repeat_index)
    snapshots = {}
    for arm in V23_ARMS:
        snapshots[arm] = adapter.compute_lora_gradient_accumulated(
            _batches(
                prompt_ids,
                selected[arm],
                scenes,
                micro_size=micro_size,
                seed=repeat_seed,
            ),
            f"repeat_{repeat_index}_{arm}",
        )
        print(f"[v2.3 gradient] repeat={repeat_index + 1} gradient={arm}", flush=True)
    identical = adapter.compute_lora_gradient_accumulated(
        _batches(
            prompt_ids,
            selected["naive"],
            scenes,
            micro_size=micro_size,
            seed=repeat_seed,
        ),
        f"repeat_{repeat_index}_naive_identical",
    )
    identical_comparison = compare_gradients(snapshots["naive"], identical)
    naive_self = compare_gradients(snapshots["naive"], snapshots["rfo_self"])
    naive_gold = compare_gradients(snapshots["naive"], snapshots["rfo_gold"])

    noise_cosines = []
    rng = random.Random(_stable_seed(repeat_seed, "split-half"))
    for split_index in range(int(gradient_config["noise_floor_splits"])):
        shuffled = list(prompt_ids)
        rng.shuffle(shuffled)
        midpoint = len(shuffled) // 2
        halves = (shuffled[:midpoint], shuffled[midpoint:])
        half_snapshots = []
        for half_index, half in enumerate(halves):
            half_snapshots.append(
                adapter.compute_lora_gradient_accumulated(
                    _batches(
                        half,
                        selected["naive"],
                        scenes,
                        micro_size=micro_size,
                        seed=_stable_seed(repeat_seed, "split", split_index, half_index),
                    ),
                    f"repeat_{repeat_index}_noise_{split_index}_{half_index}",
                )
            )
        noise_cosines.append(compare_gradients(*half_snapshots).cosine)
    noise = noise_interval(noise_cosines)
    conditions = {
        "minimum_common_informative": len(prompt_ids) >= minimum,
        "identical_cosine": identical_comparison.cosine
        >= float(gradient_config["identical_cosine_min"]),
        "naive_gold_separated": naive_gold.cosine
        <= float(gradient_config["naive_gold_cosine_max"]),
        "gold_score_advantage": report["gold_selected_score_advantage"]
        >= float(gradient_config["gold_score_advantage_min"]),
    }
    report.update(
        {
            "passed": all(conditions.values()),
            "conditions": conditions,
            "identical": as_serializable(identical_comparison),
            "naive_self": as_serializable(naive_self),
            "naive_gold": as_serializable(naive_gold),
            "noise_floor": noise,
            "noise_cosines": noise_cosines,
        }
    )
    atomic_write_json(repeat_dir / "gradient_repeat.json", report)
    return report


def run_v23_gradient_gate(
    *,
    config_path: str | Path,
    authorization_path: str | Path,
    probe_manifest: str | Path,
    backbone_config: str | Path,
    frozen_observer_python: str | Path,
    output_dir: str | Path,
    lora_target_modules: Sequence[str],
) -> dict[str, Any]:
    """Run or resume the locked 96 x 3 local gradient survival experiment."""

    import torch

    output = Path(output_dir).resolve()
    authorization = validate_v23_authorization(
        authorization_path, stage="gradient_survival_gate", output_path=output
    )
    config = load_config(config_path)
    values = config.values
    records = list(read_jsonl(probe_manifest))
    expected_probe = int(values["gradient_probe"]["size"])
    if len(records) != expected_probe:
        raise RuntimeError(f"v2.3 gradient probe must contain exactly {expected_probe} rows")
    registry = _read_json(authorization["evidence"]["data_registry"]["path"])
    registered_probe = Path(str(registry["manifests"]["gradient_probe"])).resolve()
    if Path(probe_manifest).resolve() != registered_probe:
        raise RuntimeError("v2.3 gradient probe path differs from the authorized registry")
    if sha256_file(probe_manifest) != registry["manifest_sha256"]["gradient_probe"]:
        raise RuntimeError("v2.3 gradient probe SHA-256 mismatch")
    if output.exists() and (output / "gradient_gate.json").is_file():
        return _read_json(output / "gradient_gate.json")
    output.mkdir(parents=True, exist_ok=True)
    write_config_snapshot(config, output / "resolved_config.json")
    if not (output / "host_manifest.json").is_file():
        write_host_manifest(output / "host_manifest.json")
    adapter = Showo2Adapter(
        backbone_config=Path(backbone_config).resolve(),
        device=str(values["hardware"]["generator_device"]),
        lazy=False,
    )
    if adapter.model_id != authorization["model_id"] or adapter.revision != authorization["revision"]:
        raise RuntimeError("v2.3 gradient backbone differs from authorization")
    lora = values["training"]["lora"]
    lora_summary = adapter.attach_lora(
        rank=int(lora["rank"]),
        alpha=int(lora["alpha"]),
        dropout=float(lora["dropout"]),
        target_modules=tuple(lora_target_modules),
        gradient_checkpointing=bool(values["training"]["gradient_checkpointing"]),
    )
    command = [
        str(Path(frozen_observer_python).resolve()),
        "-m",
        "selfsight.observers.service",
        "--backend",
        "showo2",
        "--model-id",
        adapter.model_id,
        "--revision",
        adapter.revision,
        "--device",
        str(values["hardware"]["observer_device"]),
        "--backbone-config",
        str(Path(backbone_config).resolve()),
        "--ready-report",
        str(output / "frozen_observer_ready.json"),
    ]
    family_by_prompt = {
        str(record["scene"]["scene_id"]): str(record["scene"]["family"])
        for record in records
    }
    repeat_reports = []
    with ObserverServiceClient(command, output / "frozen_observer_wire.jsonl") as frozen_observer:
        for repeat_index in range(int(values["gradient_probe"]["repeats"])):
            repeat_dir = output / f"repeat-{repeat_index + 1:02d}"
            repeat_dir.mkdir(parents=True, exist_ok=True)
            candidates, paired, informative, scenes = _prepare_repeat(
                repeat_index=repeat_index,
                records=records,
                adapter=adapter,
                frozen_observer=frozen_observer,
                repeat_dir=repeat_dir,
                seed=int(values["seed"]),
                candidate_k=int(values["gradient_probe"]["candidate_k"]),
                minimum_gold_gap=float(values["training"]["informative_gold_gap_min"]),
                minimum_common_informative=int(
                    values["gradient_probe"]["minimum_common_informative"]
                ),
            )
            repeat_reports.append(
                _gradient_repeat(
                    repeat_index=repeat_index,
                    adapter=adapter,
                    candidates=candidates,
                    paired=paired,
                    informative=informative,
                    scenes=scenes,
                    family_by_prompt=family_by_prompt,
                    repeat_dir=repeat_dir,
                    config=values,
                )
            )
            if informative.get("early_stop") is not None:
                break

    identical_passes = sum(
        bool(report.get("conditions", {}).get("identical_cosine")) for report in repeat_reports
    )
    separation_passes = sum(
        bool(report.get("conditions", {}).get("naive_gold_separated"))
        and bool(report.get("conditions", {}).get("gold_score_advantage"))
        for report in repeat_reports
    )
    availability_passes = sum(
        bool(report.get("conditions", {}).get("minimum_common_informative"))
        for report in repeat_reports
    )
    required = int(values["gradient_probe"]["required_passing_repeats"])
    conditions = {
        "identical_all_repeats": identical_passes == len(repeat_reports),
        "minimum_common_all_repeats": availability_passes == len(repeat_reports),
        "gold_treatment_at_least_required_repeats": separation_passes >= required,
    }
    passed = all(conditions.values())
    report = {
        "schema_version": 1,
        "benchmark_version": "2.3",
        "stage": "v23_gradient_survival_gate",
        "non_formal": True,
        "formal_claims_allowed": False,
        "model_downloads_used": False,
        "a800_used": False,
        "model_id": adapter.model_id,
        "revision": adapter.revision,
        "passed": passed,
        "conditions": conditions,
        "repeat_pass_counts": {
            "identical": identical_passes,
            "minimum_common_informative": availability_passes,
            "gold_treatment": separation_passes,
            "required_gold_treatment": required,
        },
        "repeats_planned": int(values["gradient_probe"]["repeats"]),
        "repeats_completed": len(repeat_reports),
        "early_stop": (
            repeat_reports[-1].get("early_stop")
            if repeat_reports and repeat_reports[-1].get("early_stop") is not None
            else None
        ),
        "repeats": repeat_reports,
        "lora_trainable_parameters": int(
            lora_summary["trainable_parameters"]
            if isinstance(lora_summary, Mapping)
            else lora_summary.trainable_parameters
        ),
        "lora_target_modules": list(lora_target_modules),
        "peak_gpu_bytes": int(
            torch.cuda.max_memory_allocated(cuda_device_index(adapter.device))
        ),
        "authorization": {
            "path": str(Path(authorization_path).resolve()),
            "sha256": sha256_file(authorization_path),
            "digest": authorization["authorization_digest"],
        },
        "stop_rule": (
            None
            if passed
            else "Do not launch v2.3 three-seed training; diagnose correction loss/probe first."
        ),
    }
    atomic_write_json(output / "gradient_gate.json", report)
    return report
