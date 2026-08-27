"""Gate -1b: real LoRA-gradient signal, noise-floor, and detector separation audit."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from selfsight.backbones.showo2 import Showo2Adapter, Showo2GenerationBatch
from selfsight.config import load_config, write_config_snapshot
from selfsight.data.candidates import CandidateManifest
from selfsight.data.questions import build_primary_atom, build_question
from selfsight.data.subsets import stable_stratified_sample
from selfsight.data.verifier import verify_image
from selfsight.observers.client import ObserverServiceClient
from selfsight.rfo.isolation import make_blind_request
from selfsight.rfo.selection import select_candidate
from selfsight.schemas import (
    AtomicObservation,
    CandidateRecord,
    ObservationResult,
    SceneSpec,
    SelectionDecision,
    as_serializable,
)
from selfsight.showo_adapter import ShowoAdapter, ShowoSFTBatch
from selfsight.training.gradients import compare_gradients, noise_interval
from selfsight.utils.cuda import cuda_device_index
from selfsight.utils.evidence import write_host_manifest
from selfsight.utils.hashing import sha256_json
from selfsight.utils.jsonl import atomic_write_json, atomic_write_jsonl, read_jsonl


def _stable_seed(*parts: object) -> int:
    return int(sha256_json(list(parts))[:8], 16) & 0x7FFF_FFFF


def _gold_observation(candidate: CandidateRecord, question: Any, atom: Any) -> ObservationResult:
    answer = verify_image(candidate.image_path, [atom]).answers[atom.atom_id]
    return ObservationResult(
        request_id=f"gold-{candidate.candidate_id}",
        observer_id="selfsight/program-verifier",
        observer_revision="deterministic-v1",
        rgb_sha256=candidate.rgb_sha256,
        answers=(
            AtomicObservation(
                question_id=question.question_id,
                raw_answer=answer or "unparsed",
                normalized_answer=answer,
                abstain=answer is None,
            ),
        ),
    )


def _batches(
    prompt_ids: list[str],
    selected: dict[str, CandidateRecord],
    scenes: dict[str, SceneSpec],
    *,
    adapter: Any,
    micro_size: int,
    seed: int,
) -> list[Any]:
    output = []
    for start in range(0, len(prompt_ids), micro_size):
        ids = prompt_ids[start : start + micro_size]
        common = {
            "prompts": tuple(scenes[prompt_id].prompt for prompt_id in ids),
            "images": tuple(selected[prompt_id].image_path for prompt_id in ids),
            "sample_ids": tuple(ids),
        }
        batch_seed = _stable_seed(seed, tuple(ids))
        if isinstance(adapter, Showo2Adapter):
            output.append(
                Showo2GenerationBatch(
                    **common,
                    latent_seed=batch_seed,
                )
            )
        else:
            output.append(
                ShowoSFTBatch(
                    **common,
                    mask_seed=batch_seed,
                )
            )
    return output


def _selected_map(
    decisions: list[SelectionDecision], candidates: list[CandidateRecord]
) -> dict[str, CandidateRecord]:
    candidate_map = {candidate.candidate_id: candidate for candidate in candidates}
    return {
        decision.prompt_id: candidate_map[str(decision.selected_candidate_id)]
        for decision in decisions
        if not decision.abstain and decision.selected_candidate_id is not None
    }


def run_gradient_gate(
    *,
    config_path: str | Path,
    probe_manifest: str | Path,
    detector_command: list[str],
    output_dir: str | Path,
    adapter: Any | None = None,
    lora_target_modules: Sequence[str] | None = None,
    eligible_families: Sequence[str] | None = None,
    evidence_bindings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    import torch

    config = load_config(config_path)
    probe_size = int(config.values["gradient_probe"]["size"])
    source_records = list(read_jsonl(probe_manifest))
    family_filter = tuple(dict.fromkeys(str(item) for item in eligible_families or ()))
    family_set = set(family_filter)
    if family_set:
        source_records = [
            record
            for record in source_records
            if str(record["scene"]["family"]) in family_set
        ]
    if len(source_records) < probe_size:
        raise RuntimeError(
            f"Gradient probe has only {len(source_records)} eligible records; {probe_size} required"
        )
    records = stable_stratified_sample(
        source_records,
        probe_size,
        stratum=lambda record: str(record["atom"]["family"]),
        item_id=lambda record: str(record["scene"]["scene_id"]),
        seed=int(config.values["seed"]),
    )
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_config_snapshot(config, output / "resolved_config.json")
    write_host_manifest(output / "host_manifest.json")
    if adapter is None:
        adapter = ShowoAdapter(
            device=str(config.values["hardware"]["generator_device"]),
            trainable=True,
            generation_timesteps=int(config.values["model"]["generation_timesteps"]),
            guidance_scale=float(config.values["model"]["guidance_scale"]),
            temperature=float(config.values["model"]["temperature"]),
        )
    lora = config.values["training"]["lora"]
    targets = tuple(lora_target_modules or lora["target_modules"])
    lora_summary = adapter.attach_lora(
        rank=int(lora["rank"]),
        alpha=int(lora["alpha"]),
        dropout=float(lora["dropout"]),
        target_modules=targets,
        gradient_checkpointing=bool(config.values["training"]["gradient_checkpointing"]),
    )
    k = int(config.values["training"]["candidate_k"])
    scenes: dict[str, SceneSpec] = {}
    for record in records:
        scene = SceneSpec.from_dict(record["scene"])
        scenes[scene.scene_id] = scene

    all_candidates: list[CandidateRecord] = []
    decisions: dict[str, list[SelectionDecision]] = {key: [] for key in ("naive", "rfo", "gold")}
    detector_identity: tuple[str, str] | None = None
    detector_ready = output / "detector_ready.json"
    command = [*detector_command, "--ready-report", str(detector_ready)]
    with ObserverServiceClient(command, output / "detector_wire.jsonl") as detector:
        for index, record in enumerate(records):
            scene = scenes[record["scene"]["scene_id"]]
            atom = build_primary_atom(scene)
            question = build_question(atom)
            seeds = tuple(_stable_seed(config.values["seed"], scene.scene_id, item) for item in range(k))
            generated = adapter.generate_images(
                [scene.prompt] * k,
                seeds,
                output / "candidates",
                "step-0-gradient-probe",
            )
            candidates = [
                replace(candidate, prompt_id=scene.scene_id, scene_id=scene.scene_id)
                for candidate in generated
            ]
            observations: dict[str, dict[str, ObservationResult]] = {
                key: {} for key in ("naive", "rfo", "gold")
            }
            for candidate in candidates:
                observations["naive"][candidate.candidate_id] = adapter.observe_atoms(
                    candidate.image_path, (question,)
                )
                observations["rfo"][candidate.candidate_id] = detector.observe(
                    make_blind_request(candidate.image_path, (question,), f"g-rfo-{candidate.candidate_id}")
                )
                detector_identity = (
                    observations["rfo"][candidate.candidate_id].observer_id,
                    observations["rfo"][candidate.candidate_id].observer_revision,
                )
                observations["gold"][candidate.candidate_id] = _gold_observation(
                    candidate, question, atom
                )
            for criterion in ("naive", "rfo", "gold"):
                first_observation = next(iter(observations[criterion].values()))
                decisions[criterion].append(
                    select_candidate(
                        prompt_id=scene.scene_id,
                        arm=criterion,
                        candidates=candidates,
                        observations=observations[criterion],
                        questions=(question,),
                        selector_id=first_observation.observer_id,
                        observer_revision=first_observation.observer_revision,
                    )
                )
            all_candidates.extend(candidates)
    CandidateManifest(output / "candidate_manifest.jsonl").write(all_candidates)
    for criterion, values in decisions.items():
        atomic_write_jsonl(
            output / f"selection_{criterion}.jsonl",
            (as_serializable(decision) for decision in values),
        )
    selected = {criterion: _selected_map(values, all_candidates) for criterion, values in decisions.items()}
    common = [
        scene.scene_id
        for scene in scenes.values()
        if all(scene.scene_id in selected[criterion] for criterion in ("naive", "rfo", "gold"))
    ]
    availability = len(common) / probe_size
    if len(common) < 8:
        report = {
            "schema_version": 1,
            "gate": "minus_1b",
            "model_id": adapter.model_id,
            "revision": adapter.revision,
            "passed": False,
            "reason": "Fewer than 8 probe prompts had non-abstaining selections under all criteria",
            "common_samples": len(common),
            "availability": availability,
            "eligible_families": list(family_filter),
            "evidence_bindings": dict(evidence_bindings or {}),
            "fallback": "Do not report GDA; continue E2 with entropy/public-view baselines only.",
        }
        atomic_write_json(output / "gate_minus_1b.json", report)
        return report

    micro_size = int(config.values["training"]["micro_batch_size"])
    snapshots = {}
    for criterion in ("naive", "rfo", "gold"):
        snapshots[criterion] = adapter.compute_lora_gradient_accumulated(
            _batches(
                common,
                selected[criterion],
                scenes,
                adapter=adapter,
                micro_size=micro_size,
                seed=int(config.values["seed"]),
            ),
            criterion,
        )
    identical = adapter.compute_lora_gradient_accumulated(
        _batches(
            common,
            selected["naive"],
            scenes,
            adapter=adapter,
            micro_size=micro_size,
            seed=int(config.values["seed"]),
        ),
        "naive_identical_repeat",
    )
    identical_comparison = compare_gradients(snapshots["naive"], identical)
    free = compare_gradients(snapshots["naive"], snapshots["rfo"])
    gold = compare_gradients(snapshots["naive"], snapshots["gold"])

    noise_values = []
    rng = random.Random(int(config.values["seed"]) + 17_003)
    for split_index in range(int(config.values["gradient_probe"]["noise_floor_splits"])):
        shuffled = list(common)
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
                        adapter=adapter,
                        micro_size=micro_size,
                        seed=_stable_seed(config.values["seed"], split_index, half_index),
                    ),
                    f"naive_noise_{split_index}_{half_index}",
                )
            )
        noise_values.append(compare_gradients(*half_snapshots).cosine)
    noise = noise_interval(noise_values)
    expected_effect = float(config.values["gradient_probe"]["expected_gda_effect"])
    conditions = {
        "availability_at_least_75pct": availability >= 0.75,
        "identical_selection_cosine": identical_comparison.cosine
        >= float(config.values["gradient_probe"]["identical_cosine_min"]),
        "noise_interval_narrow": (noise["high"] - noise["low"]) < expected_effect / 2.0,
        "early_gda_inside_noise_floor": noise["low"] <= free.cosine <= noise["high"],
        "lora_projection_non_degenerate": bool(np.isfinite(free.cosine) and abs(free.cosine) >= 0.05),
        "detector_trainer_separated": bool(
            detector_identity is not None and detector_identity[0] != adapter.model_id
        ),
    }
    report = {
        "schema_version": 1,
        "gate": "minus_1b",
        "model_id": adapter.model_id,
        "revision": adapter.revision,
        "passed": all(conditions.values()),
        "conditions": conditions,
        "common_samples": len(common),
        "availability": availability,
        "lora_trainable_parameters": int(
            lora_summary["trainable_parameters"]
            if isinstance(lora_summary, Mapping)
            else lora_summary.trainable_parameters
        ),
        "lora_target_modules": list(targets),
        "eligible_families": list(family_filter),
        "evidence_bindings": dict(evidence_bindings or {}),
        "gda_free": as_serializable(free),
        "gda_gold": as_serializable(gold),
        "identical_control": as_serializable(identical_comparison),
        "noise_floor": noise,
        "noise_cosines": noise_values,
        "detector_training_split": {
            "training_selector": "frozen step-0 Show-o (not used in this detector audit)",
            "g_rfo_detector": detector_identity[0] if detector_identity else None,
            "g_rfo_detector_revision": detector_identity[1] if detector_identity else None,
            "same_object": False,
        },
        "peak_gpu_bytes": int(
            torch.cuda.max_memory_allocated(cuda_device_index(adapter.device))
        ),
        "fallback": (
            None
            if all(conditions.values())
            else "Do not report GDA; continue E2 with entropy/public-view baselines only."
        ),
    }
    atomic_write_json(output / "gate_minus_1b.json", report)
    return report
