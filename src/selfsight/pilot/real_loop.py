"""Resumable paired Naive/RFO rejection-sampling SFT loop for Show-o."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

from selfsight.analysis.prerequisites import require_gate_minus_one, require_generated_domain
from selfsight.analysis.readiness import require_joint_readiness
from selfsight.backbones.showo2 import (
    Showo2Adapter,
    Showo2GenerationBatch,
    Showo2ReplayBatch,
)
from selfsight.config import load_config, write_config_snapshot
from selfsight.data.candidates import CandidateManifest
from selfsight.data.questions import build_primary_atom, build_question
from selfsight.data.subsets import stable_stratified_sample
from selfsight.observers.client import ObserverServiceClient
from selfsight.rfo.isolation import make_blind_request
from selfsight.rfo.selection import balance_paired_decisions, select_candidate
from selfsight.schemas import CandidateRecord, SceneSpec, SelectionDecision, as_serializable
from selfsight.showo_adapter import ShowoAdapter, ShowoReplayBatch, ShowoSFTBatch
from selfsight.training.checkpoint import load_checkpoint, save_checkpoint
from selfsight.training.paired import PromptScheduleEntry, build_paired_schedule
from selfsight.utils.evidence import write_host_manifest
from selfsight.utils.hashing import rgb_sha256, sha256_file, sha256_json
from selfsight.utils.jsonl import atomic_write_json, atomic_write_jsonl, read_jsonl

ARMS = ("naive", "rfo_self")


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected an object in {path}")
    return value


def _assert_prerequisites(
    gate_report: str | Path | None,
    gradient_gate_report: str | Path,
    generated_domain_report: str | Path,
    *,
    generated_coverage_min: float,
) -> bool:
    require_gate_minus_one(gate_report)
    gradient = _read_json(gradient_gate_report)
    if gradient.get("gate") != "minus_1b":
        raise RuntimeError("The supplied gradient report is not a completed Gate -1b audit")
    require_generated_domain(
        generated_domain_report, configured_coverage_min=generated_coverage_min
    )
    return bool(gradient.get("passed"))


def _assert_joint_prerequisites(
    decision_path: str | Path,
    gradient_gate_report: str | Path,
) -> tuple[bool, tuple[str, ...]]:
    """Bind E2 to one green Gate -2 and its exact Gate -1b derivative."""

    decision_path = Path(decision_path).resolve()
    decision = require_joint_readiness(decision_path)
    gradient_path = Path(gradient_gate_report).resolve()
    gradient = _read_json(gradient_path)
    if gradient.get("gate") != "minus_1b":
        raise RuntimeError("The supplied gradient report is not a completed Gate -1b audit")
    if (
        gradient.get("model_id") != decision.get("model_id")
        or gradient.get("revision") != decision.get("revision")
    ):
        raise RuntimeError("Gate -1b backbone identity does not match Gate -2")
    eligible = tuple(str(item) for item in decision["selected_eligible_families"])
    if tuple(str(item) for item in gradient.get("eligible_families", ())) != eligible:
        raise RuntimeError("Gate -1b eligible families do not match Gate -2")
    bindings = gradient.get("evidence_bindings")
    joint = bindings.get("joint_readiness_decision") if isinstance(bindings, Mapping) else None
    if not isinstance(joint, Mapping) or joint.get("sha256") != sha256_file(decision_path):
        raise RuntimeError("Gate -1b is not hash-bound to the supplied Gate -2 decision")
    return bool(gradient.get("passed")), eligible


def _training_records(manifest_path: str | Path) -> tuple[dict[str, Any], list[str]]:
    records: dict[str, Any] = {}
    order = []
    for record in read_jsonl(manifest_path):
        scene = SceneSpec.from_dict(record["scene"])
        records[scene.scene_id] = {
            "scene": scene,
            "reference_image": str(Path(record["reference_image"]).resolve()),
            "question": build_question(build_primary_atom(scene)),
        }
        order.append(scene.scene_id)
    return records, order


def _stable_seed(*parts: object) -> int:
    return int(sha256_json(list(parts))[:8], 16) & 0x7FFF_FFFF


def _optimizer_and_scheduler(adapter: Any, config: Any):
    import torch

    training = config.values["training"]
    parameters = [parameter for parameter in adapter.model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    total_steps = int(training["rounds"]) * int(training["optimizer_steps_per_round"])
    warmup_steps = max(1, round(total_steps * float(training["warmup_ratio"])))

    def multiplier(step: int) -> float:
        return min(1.0, float(step + 1) / warmup_steps)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)
    return optimizer, scheduler


def _completed_rounds(run_root: Path) -> list[int]:
    output = []
    for path in (run_root / "rounds").glob("round-*"):
        if (path / "DONE.json").is_file():
            try:
                output.append(int(path.name.split("-")[-1]))
            except ValueError:
                continue
    return sorted(output)


def _abandon_incomplete(path: Path) -> None:
    if not path.exists():
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = path.with_name(f"{path.name}.abandoned-{stamp}")
    os.replace(path, destination)


def _initialize_round_zero(
    run_root: Path,
    adapter: Any,
    optimizer: Any,
    scheduler: Any,
    config: Any,
    checkpoint_config_digest: str,
    checkpoint_config_values: dict[str, Any],
) -> None:
    final = run_root / "rounds" / "round-00"
    if (final / "DONE.json").is_file():
        return
    temporary = run_root / ".round-00.inprogress"
    _abandon_incomplete(temporary)
    temporary.mkdir(parents=True)
    for arm in ARMS:
        save_checkpoint(
            temporary / "arms" / arm,
            model=adapter.model,
            optimizer=optimizer,
            scheduler=scheduler,
            config_digest=checkpoint_config_digest,
            config_values=checkpoint_config_values,
            step=0,
            round_index=0,
            metadata={"arm": arm, "kind": "base", "evidence_status": "local_exploratory"},
        )
    atomic_write_json(
        temporary / "DONE.json",
        {"schema_version": 1, "round": 0, "step": 0, "status": "base_checkpoint_complete"},
    )
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, final)


def _load_arm(
    checkpoint: Path,
    adapter: Any,
    optimizer: Any,
    scheduler: Any,
    config_digest: str,
) -> dict[str, int]:
    return load_checkpoint(
        checkpoint,
        model=adapter.model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_config_digest=config_digest,
    )


def _generate_and_select(
    *,
    arm: str,
    adapter: Any,
    frozen_observer: ObserverServiceClient,
    entries: list[PromptScheduleEntry],
    records: dict[str, Any],
    output_dir: Path,
    checkpoint_id: str,
) -> tuple[list[CandidateRecord], list[SelectionDecision]]:
    candidates_all: list[CandidateRecord] = []
    decisions: list[SelectionDecision] = []
    for entry in entries:
        scene = records[entry.prompt_id]["scene"]
        question = records[entry.prompt_id]["question"]
        generated = adapter.generate_images(
            [scene.prompt] * len(entry.candidate_seeds),
            entry.candidate_seeds,
            output_dir,
            checkpoint_id,
        )
        candidates = [
            replace(candidate, prompt_id=scene.scene_id, scene_id=scene.scene_id)
            for candidate in generated
        ]
        observations = {}
        for candidate in candidates:
            if arm == "naive":
                observation = adapter.observe_atoms(candidate.image_path, (question,))
            else:
                observation = frozen_observer.observe(
                    make_blind_request(candidate.image_path, (question,), candidate.candidate_id)
                )
            observations[candidate.candidate_id] = observation
        first_observation = next(iter(observations.values()))
        decisions.append(
            select_candidate(
                prompt_id=scene.scene_id,
                arm=arm,
                candidates=candidates,
                observations=observations,
                questions=(question,),
                selector_id=first_observation.observer_id,
                observer_revision=first_observation.observer_revision,
            )
        )
        candidates_all.extend(candidates)
    return candidates_all, decisions


def _paired_order(
    entries: list[PromptScheduleEntry],
    naive: list[SelectionDecision],
    rfo: list[SelectionDecision],
) -> dict[str, list[SelectionDecision]]:
    balanced_naive, balanced_rfo = balance_paired_decisions(naive, rfo)
    naive_map = {decision.prompt_id: decision for decision in balanced_naive}
    rfo_map = {decision.prompt_id: decision for decision in balanced_rfo}
    order = [entry.prompt_id for entry in entries if entry.prompt_id in naive_map]
    return {
        "naive": [naive_map[prompt_id] for prompt_id in order],
        "rfo_self": [rfo_map[prompt_id] for prompt_id in order],
    }


def _train_arm(
    *,
    arm: str,
    adapter: Any,
    optimizer: Any,
    scheduler: Any,
    decisions: list[SelectionDecision],
    candidates: list[CandidateRecord],
    records: dict[str, Any],
    replay_ids: list[str],
    config: Any,
    round_index: int,
) -> dict[str, Any]:
    import torch

    if not decisions:
        raise RuntimeError("Both training arms abstained on every prompt in the round")
    training = config.values["training"]
    candidate_map = {candidate.candidate_id: candidate for candidate in candidates}
    selected = [candidate_map[str(decision.selected_candidate_id)] for decision in decisions]
    micro_size = int(training["micro_batch_size"])
    accumulation = int(training["gradient_accumulation_steps"])
    optimizer_steps = int(training["optimizer_steps_per_round"])
    ratio = Fraction(str(float(training["understanding_replay_ratio"]))).limit_denominator(100)
    t2i_cursor = 0
    replay_cursor = 0
    t2i_losses: list[float] = []
    replay_losses: list[float] = []
    gradient_norms: list[float] = []
    for optimizer_step in range(optimizer_steps):
        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(accumulation):
            global_micro = optimizer_step * accumulation + micro_step
            use_replay = ratio.numerator > 0 and global_micro % ratio.denominator < ratio.numerator
            if use_replay:
                ids = [replay_ids[(replay_cursor + index) % len(replay_ids)] for index in range(micro_size)]
                replay_cursor += micro_size
                common = {
                    "images": tuple(
                        records[prompt_id]["reference_image"] for prompt_id in ids
                    ),
                    "questions": tuple(
                        records[prompt_id]["question"].text for prompt_id in ids
                    ),
                    "answers": tuple(
                        records[prompt_id]["question"].expected_answer for prompt_id in ids
                    ),
                    "sample_ids": tuple(ids),
                }
                if isinstance(adapter, Showo2Adapter):
                    batch = Showo2ReplayBatch(
                        **common,
                        latent_seed=_stable_seed(
                            config.values["seed"], round_index, optimizer_step, micro_step, "replay"
                        ),
                    )
                    loss = adapter.understanding_replay_loss(batch)
                else:
                    batch = ShowoReplayBatch(**common)
                    loss = adapter.mmu_replay_loss(batch)
                replay_losses.append(float(loss.detach().cpu()))
            else:
                examples = [selected[(t2i_cursor + index) % len(selected)] for index in range(micro_size)]
                t2i_cursor += micro_size
                common = {
                    "prompts": tuple(
                        records[item.prompt_id]["scene"].prompt for item in examples
                    ),
                    "images": tuple(item.image_path for item in examples),
                    "sample_ids": tuple(item.prompt_id for item in examples),
                }
                batch_seed = _stable_seed(
                    config.values["seed"], round_index, optimizer_step, micro_step
                )
                if isinstance(adapter, Showo2Adapter):
                    batch = Showo2GenerationBatch(**common, latent_seed=batch_seed)
                    loss = adapter.generation_loss(batch)
                else:
                    batch = ShowoSFTBatch(**common, mask_seed=batch_seed)
                    loss = adapter.sft_loss(batch)
                t2i_losses.append(float(loss.detach().cpu()))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite {arm} loss at round {round_index}")
            (loss / accumulation).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in adapter.model.parameters() if parameter.requires_grad],
            float(training["max_grad_norm"]),
        )
        gradient_norms.append(float(grad_norm.detach().cpu()))
        optimizer.step()
        scheduler.step()
    return {
        "arm": arm,
        "selected_samples": len(selected),
        "optimizer_steps": optimizer_steps,
        "t2i_microbatches": len(t2i_losses),
        "replay_microbatches": len(replay_losses),
        "mean_t2i_loss": sum(t2i_losses) / len(t2i_losses) if t2i_losses else None,
        "mean_replay_loss": sum(replay_losses) / len(replay_losses) if replay_losses else None,
        "mean_gradient_norm_before_clip": sum(gradient_norms) / len(gradient_norms),
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
    }


def run_real_paired_pilot(
    *,
    config_path: str | Path,
    train_manifest: str | Path,
    gate_report: str | Path,
    gradient_gate_report: str | Path,
    generated_domain_report: str | Path | None,
    frozen_observer_python: str | Path,
    output_dir: str | Path,
    resume: bool = False,
    joint_readiness_decision: str | Path | None = None,
    backbone_config: str | Path | None = None,
    lora_target_modules: Sequence[str] | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    eligible_families: tuple[str, ...] = ()
    if joint_readiness_decision is not None:
        if generated_domain_report is not None:
            raise ValueError("v2.2 E2 must not mix the legacy generated-domain Gate")
        gda_enabled, eligible_families = _assert_joint_prerequisites(
            joint_readiness_decision,
            gradient_gate_report,
        )
        if backbone_config is None or not lora_target_modules:
            raise ValueError("v2.2 E2 requires a backbone config and audited LoRA targets")
    else:
        if gate_report is None or generated_domain_report is None:
            raise ValueError("Legacy E2 requires Gate -1 and generated-domain reports")
        gda_enabled = _assert_prerequisites(
            gate_report,
            gradient_gate_report,
            generated_domain_report,
            generated_coverage_min=float(config.values["gates"]["verifier_coverage_min"]),
        )
    if joint_readiness_decision is not None:
        joint_path = Path(joint_readiness_decision).resolve()
        target_modules = tuple(str(item) for item in lora_target_modules or ())
        checkpoint_contract = {
            "schema_version": 2,
            "base_config_digest": config.digest,
            "joint_readiness_sha256": sha256_file(joint_path),
            "backbone_config_sha256": sha256_file(Path(str(backbone_config)).resolve()),
            "lora_target_modules": list(target_modules),
        }
        checkpoint_config_digest = sha256_json(checkpoint_contract)
        checkpoint_config_values = {
            "base_experiment_config": config.values,
            "joint_training_contract": checkpoint_contract,
        }
    else:
        checkpoint_contract = {
            "schema_version": 1,
            "base_config_digest": config.digest,
            "legacy_gate_minus_1_sha256": sha256_file(Path(str(gate_report)).resolve()),
        }
        checkpoint_config_digest = config.digest
        checkpoint_config_values = config.values
    run_root = Path(output_dir).resolve()
    if run_root.exists() and any(run_root.iterdir()) and not resume:
        raise FileExistsError(f"Refusing to overwrite non-empty run: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    existing_config = run_root / "resolved_config.json"
    if existing_config.exists():
        if _read_json(existing_config).get("digest") != config.digest:
            raise ValueError("Resume config digest does not match the existing run")
    else:
        write_config_snapshot(config, existing_config)
        write_host_manifest(run_root / "host_manifest.json")
        if joint_readiness_decision is not None:
            prerequisite_paths = {
                "gate_minus_2": Path(joint_readiness_decision).resolve(),
                "gate_minus_1b": Path(gradient_gate_report).resolve(),
                "backbone_config": Path(str(backbone_config)).resolve(),
            }
        else:
            prerequisite_paths = {
                "gate_minus_1": Path(gate_report).resolve(),
                "gate_minus_1b": Path(gradient_gate_report).resolve(),
                "generated_domain": Path(str(generated_domain_report)).resolve(),
            }
        atomic_write_json(
            run_root / "prerequisite_reports.json",
            {
                "schema_version": 1,
                "reports": {
                    key: {"path": str(path), "sha256": sha256_file(path)}
                    for key, path in prerequisite_paths.items()
                },
            },
        )
        is_formal_profile = str(config.values["profile"]).startswith("a800_80g")
        atomic_write_json(
            run_root / "EVIDENCE_STATUS.json",
            {
                "status": (
                    "formal_single_seed_pre_gate" if is_formal_profile else "local_single_seed_exploratory"
                ),
                "usable_for_formal_claims": False,
            },
        )
    contract_path = run_root / "training_contract.json"
    if contract_path.exists():
        if _read_json(contract_path).get("checkpoint_config_digest") != checkpoint_config_digest:
            raise ValueError("Resume training contract does not match the existing run")
    else:
        atomic_write_json(
            contract_path,
            {
                **checkpoint_contract,
                "checkpoint_config_digest": checkpoint_config_digest,
            },
        )
    records, record_order = _training_records(train_manifest)
    if eligible_families:
        eligible_set = set(eligible_families)
        record_order = [
            scene_id
            for scene_id in record_order
            if records[scene_id]["scene"].family.value in eligible_set
        ]
        records = {scene_id: records[scene_id] for scene_id in record_order}
    training = config.values["training"]
    train_count_key = (
        "full_train_prompts"
        if str(config.values["profile"]).startswith("a800_80g")
        else "local_train_prompts"
    )
    usable_records = stable_stratified_sample(
        [records[scene_id] for scene_id in record_order],
        int(config.values["data"][train_count_key]),
        stratum=lambda record: record["scene"].family.value,
        item_id=lambda record: record["scene"].scene_id,
        seed=int(config.values["seed"]),
    )
    usable_ids = [record["scene"].scene_id for record in usable_records]
    schedule = build_paired_schedule(
        usable_ids,
        rounds=int(training["rounds"]),
        prompts_per_round=int(training["prompts_per_round"]),
        candidate_k=int(training["candidate_k"]),
        seed=int(config.values["seed"]),
    )
    schedule_path = run_root / "schedule.jsonl"
    if not schedule_path.exists():
        atomic_write_jsonl(schedule_path, (as_serializable(entry) for entry in schedule))

    if joint_readiness_decision is not None:
        adapter = Showo2Adapter(
            backbone_config=Path(str(backbone_config)).resolve(),
            device=str(config.values["hardware"]["generator_device"]),
            lazy=False,
        )
    else:
        adapter = ShowoAdapter(
            device=str(config.values["hardware"]["generator_device"]),
            trainable=True,
            generation_timesteps=int(config.values["model"]["generation_timesteps"]),
            guidance_scale=float(config.values["model"]["guidance_scale"]),
            temperature=float(config.values["model"]["temperature"]),
        )
    lora = training["lora"]
    lora_summary = adapter.attach_lora(
        rank=int(lora["rank"]),
        alpha=int(lora["alpha"]),
        dropout=float(lora["dropout"]),
        target_modules=tuple(lora_target_modules or lora["target_modules"]),
        gradient_checkpointing=bool(training["gradient_checkpointing"]),
    )
    optimizer, scheduler = _optimizer_and_scheduler(adapter, config)
    _initialize_round_zero(
        run_root,
        adapter,
        optimizer,
        scheduler,
        config,
        checkpoint_config_digest,
        checkpoint_config_values,
    )
    completed = _completed_rounds(run_root)
    start_round = max(completed)
    if start_round >= int(training["rounds"]):
        return {
            "status": "already_complete",
            "run_root": str(run_root),
            "completed_round": start_round,
        }

    command = [
        str(Path(frozen_observer_python).resolve()),
        "-m",
        "selfsight.observers.service",
        "--backend",
        "showo2" if joint_readiness_decision is not None else "showo",
        "--model-id",
        adapter.model_id,
        "--revision",
        adapter.revision,
        "--device",
        str(config.values["hardware"]["observer_device"]),
        "--ready-report",
        str(run_root / "frozen_observer_ready.json"),
    ]
    if joint_readiness_decision is not None:
        command.extend(["--backbone-config", str(Path(str(backbone_config)).resolve())])
    with ObserverServiceClient(command, run_root / "frozen_observer_wire.jsonl") as frozen_observer:
        for round_index in range(start_round, int(training["rounds"])):
            next_index = round_index + 1
            temporary = run_root / f".round-{next_index:02d}.inprogress"
            final = run_root / "rounds" / f"round-{next_index:02d}"
            if (final / "DONE.json").exists():
                continue
            _abandon_incomplete(temporary)
            temporary.mkdir(parents=True)
            entries = [entry for entry in schedule if entry.round_index == round_index]
            arm_candidates: dict[str, list[CandidateRecord]] = {}
            arm_decisions: dict[str, list[SelectionDecision]] = {}
            previous = run_root / "rounds" / f"round-{round_index:02d}" / "arms"
            for arm in ARMS:
                _load_arm(
                    previous / arm,
                    adapter,
                    optimizer,
                    scheduler,
                    checkpoint_config_digest,
                )
                candidates, decisions = _generate_and_select(
                    arm=arm,
                    adapter=adapter,
                    frozen_observer=frozen_observer,
                    entries=entries,
                    records=records,
                    output_dir=temporary / "candidates" / arm,
                    checkpoint_id=f"{arm}-round-{round_index:02d}",
                )
                arm_candidates[arm] = candidates
                arm_decisions[arm] = decisions
            paired = _paired_order(entries, arm_decisions["naive"], arm_decisions["rfo_self"])
            training_reports = {}
            for arm in ARMS:
                _load_arm(
                    previous / arm,
                    adapter,
                    optimizer,
                    scheduler,
                    checkpoint_config_digest,
                )
                training_reports[arm] = _train_arm(
                    arm=arm,
                    adapter=adapter,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    decisions=paired[arm],
                    candidates=arm_candidates[arm],
                    records=records,
                    replay_ids=usable_ids,
                    config=config,
                    round_index=round_index,
                )
                save_checkpoint(
                    temporary / "arms" / arm,
                    model=adapter.model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config_digest=checkpoint_config_digest,
                    config_values=checkpoint_config_values,
                    step=next_index * int(training["optimizer_steps_per_round"]),
                    round_index=next_index,
                    metadata={
                        "arm": arm,
                        "selected_samples": len(paired[arm]),
                        "evidence_status": "local_exploratory",
                    },
                )
            for arm in ARMS:
                final_candidates = []
                for candidate in arm_candidates[arm]:
                    relative = Path(candidate.image_path).relative_to(temporary)
                    final_candidates.append(replace(candidate, image_path=str((final / relative).resolve())))
                CandidateManifest(temporary / f"candidate_manifest_{arm}.jsonl").write(
                    final_candidates, verify_rgb=False
                )
                atomic_write_jsonl(
                    temporary / f"selection_decisions_{arm}.jsonl",
                    (as_serializable(decision) for decision in arm_decisions[arm]),
                )
            round_report = {
                "schema_version": 1,
                "round": next_index,
                "step": next_index * int(training["optimizer_steps_per_round"]),
                "scheduled_prompts": len(entries),
                "paired_trainable_prompts": len(paired["naive"]),
                "abstain_naive": sum(decision.abstain for decision in arm_decisions["naive"]),
                "abstain_rfo": sum(decision.abstain for decision in arm_decisions["rfo_self"]),
                "training": training_reports,
            }
            atomic_write_json(temporary / "round_report.json", round_report)
            atomic_write_json(
                temporary / "DONE.json",
                {"schema_version": 1, "status": "complete", **round_report},
            )
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary, final)
            for arm in ARMS:
                for candidate in CandidateManifest(final / f"candidate_manifest_{arm}.jsonl").read():
                    if rgb_sha256(candidate.image_path) != candidate.rgb_sha256:
                        raise RuntimeError(f"Post-commit RGB hash mismatch: {candidate.candidate_id}")

    report = {
        "schema_version": 1,
        "status": "training_complete_pending_checkpoint_evaluation",
        "run_root": str(run_root),
        "rounds": int(training["rounds"]),
        "final_step": int(training["rounds"]) * int(training["optimizer_steps_per_round"]),
        "lora_trainable_parameters": int(
            lora_summary["trainable_parameters"]
            if isinstance(lora_summary, Mapping)
            else lora_summary.trainable_parameters
        ),
        "model_id": adapter.model_id,
        "revision": adapter.revision,
        "eligible_families": list(eligible_families),
        "formal_claims_allowed": False,
        "run_tier": (
            "formal_single_seed_pre_gate"
            if str(config.values["profile"]).startswith("a800_80g")
            else "local_single_seed_exploratory"
        ),
        "gda_enabled": gda_enabled,
        "gda_fallback_active": not gda_enabled,
    }
    atomic_write_json(run_root / "training_report.json", report)
    return report
