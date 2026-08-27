"""Resumable checkpoint evaluation for the local and A800 paired trajectories."""

from __future__ import annotations

import json
import math
import os
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import spearmanr

from selfsight.analysis.breakpoints import estimate_d_g, estimate_d_star, estimate_lead
from selfsight.analysis.figure1 import render_figure1_from_csv
from selfsight.analysis.gradient_gate import _batches, _gold_observation, _selected_map
from selfsight.backbones.showo2 import Showo2Adapter
from selfsight.config import load_config
from selfsight.data.candidates import CandidateManifest
from selfsight.data.questions import build_primary_atom, build_question
from selfsight.data.subsets import stable_stratified_sample
from selfsight.data.verifier import verify_image
from selfsight.observers.client import ObserverServiceClient
from selfsight.pilot.real_loop import _optimizer_and_scheduler, _stable_seed
from selfsight.rfo.isolation import make_blind_request
from selfsight.rfo.selection import select_candidate
from selfsight.schemas import CandidateRecord, SceneSpec, SelectionDecision, as_serializable
from selfsight.showo_adapter import ShowoAdapter
from selfsight.training.checkpoint import load_checkpoint
from selfsight.training.gradients import compare_gradients, noise_interval
from selfsight.utils.jsonl import atomic_write_json, atomic_write_jsonl, read_jsonl


def _records(
    path: str | Path,
    limit: int,
    seed: int,
    eligible_families: Sequence[str] = (),
) -> list[dict[str, Any]]:
    records = list(read_jsonl(path))
    eligible = {str(item) for item in eligible_families}
    if eligible:
        records = [
            record for record in records if str(record["scene"]["family"]) in eligible
        ]
    if len(records) < limit:
        raise RuntimeError(
            f"Evaluation manifest has only {len(records)} eligible records; {limit} required"
        )
    return stable_stratified_sample(
        records,
        limit,
        stratum=lambda record: str(record["atom"]["family"]),
        item_id=lambda record: str(record["scene"]["scene_id"]),
        seed=seed,
    )


def _entropy(values: list[str | None]) -> float:
    observed = [value for value in values if value is not None]
    if not observed:
        return float("nan")
    counts = Counter(observed)
    total = len(observed)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _evaluate_outcomes(
    *,
    adapter: Any,
    detector: ObserverServiceClient,
    records: list[dict[str, Any]],
    output: Path,
    checkpoint_id: str,
    images_per_prompt: int,
    seed: int,
    competent_families: set[str],
    final_output: Path,
) -> tuple[dict[str, Any], list[CandidateRecord]]:
    candidates_all: list[CandidateRecord] = []
    rows = []
    for record in records:
        scene = SceneSpec.from_dict(record["scene"])
        atom = build_primary_atom(scene)
        question = build_question(atom)
        seeds = tuple(_stable_seed(seed, "outcome", scene.scene_id, index) for index in range(images_per_prompt))
        generated = adapter.generate_images(
            [scene.prompt] * images_per_prompt,
            seeds,
            output / "images",
            checkpoint_id,
        )
        candidates = [
            replace(candidate, prompt_id=scene.scene_id, scene_id=scene.scene_id)
            for candidate in generated
        ]
        for candidate in candidates:
            internal = adapter.observe_atoms(candidate.image_path, (question,)).answers[0]
            verifier = verify_image(candidate.image_path, [atom])
            pixel_answer = verifier.answers[atom.atom_id]
            external = detector.observe(
                make_blind_request(candidate.image_path, (question,), f"eval-{candidate.candidate_id}")
            ).answers[0]
            detector_confirms_pixel = (
                pixel_answer is not None
                and external.normalized_answer is not None
                and external.normalized_answer == pixel_answer
                and question.family.value in competent_families
            )
            relative = Path(candidate.image_path).relative_to(output)
            final_path = str((final_output / relative).resolve())
            rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "prompt_id": scene.scene_id,
                    "family": question.family.value,
                    "image_path": final_path,
                    "sampling_seed": candidate.sampling_seed,
                    "intended_answer": question.expected_answer,
                    "internal_answer": internal.normalized_answer,
                    "internal_abstain": internal.abstain,
                    "pixel_answer": pixel_answer,
                    "verifier_coverage": verifier.coverage,
                    "detector_answer": external.normalized_answer,
                    "detector_abstain": external.abstain,
                    "competent_public_view": detector_confirms_pixel,
                    "internal_correct": internal.normalized_answer == question.expected_answer,
                    "external_correct": pixel_answer == question.expected_answer,
                    "false_confirmation": (
                        pixel_answer is not None
                        and pixel_answer != question.expected_answer
                        and internal.normalized_answer == question.expected_answer
                    ),
                }
            )
        candidates_all.extend(candidates)
    competent_errors = [
        row
        for row in rows
        if row["competent_public_view"] and not row["external_correct"]
    ]
    parsed = [row for row in rows if row["pixel_answer"] is not None]
    public_pairs = [
        row
        for row in rows
        if row["internal_answer"] is not None and row["detector_answer"] is not None
    ]
    metrics = {
        "samples": len(rows),
        "internal_score": sum(row["internal_correct"] for row in rows) / len(rows),
        "external_correctness": sum(row["external_correct"] for row in rows) / len(rows),
        "verifier_coverage": len(parsed) / len(rows),
        "scfr_competent": (
            sum(row["false_confirmation"] for row in competent_errors) / len(competent_errors)
            if competent_errors
            else float("nan")
        ),
        "scfr_denominator": len(competent_errors),
        "observer_answer_entropy": _entropy([row["internal_answer"] for row in rows]),
        "public_view_consistency": (
            sum(row["internal_answer"] == row["detector_answer"] for row in public_pairs)
            / len(public_pairs)
            if public_pairs
            else float("nan")
        ),
        "internal_abstain_rate": sum(row["internal_abstain"] for row in rows) / len(rows),
        "detector_abstain_rate": sum(row["detector_abstain"] for row in rows) / len(rows),
    }
    atomic_write_jsonl(output / "outcome_trials.jsonl", rows)
    return metrics, candidates_all


def _evaluate_gda(
    *,
    adapter: Any,
    detector: ObserverServiceClient,
    records: list[dict[str, Any]],
    output: Path,
    final_output: Path,
    checkpoint_id: str,
    config: Any,
) -> dict[str, Any]:
    scenes = {record["scene"]["scene_id"]: SceneSpec.from_dict(record["scene"]) for record in records}
    candidates_all: list[CandidateRecord] = []
    decisions: dict[str, list[SelectionDecision]] = {key: [] for key in ("naive", "rfo", "gold")}
    k = int(config.values["training"]["candidate_k"])
    for record in records:
        scene = scenes[record["scene"]["scene_id"]]
        atom = build_primary_atom(scene)
        question = build_question(atom)
        seeds = tuple(
            _stable_seed(config.values["seed"], "gradient", scene.scene_id, index) for index in range(k)
        )
        generated = adapter.generate_images(
            [scene.prompt] * k,
            seeds,
            output / "candidates",
            checkpoint_id,
        )
        candidates = [
            replace(candidate, prompt_id=scene.scene_id, scene_id=scene.scene_id)
            for candidate in generated
        ]
        observations = {key: {} for key in ("naive", "rfo", "gold")}
        for candidate in candidates:
            observations["naive"][candidate.candidate_id] = adapter.observe_atoms(
                candidate.image_path, (question,)
            )
            observations["rfo"][candidate.candidate_id] = detector.observe(
                make_blind_request(candidate.image_path, (question,), f"gda-{candidate.candidate_id}")
            )
            observations["gold"][candidate.candidate_id] = _gold_observation(candidate, question, atom)
        for criterion, criterion_decisions in decisions.items():
            first = next(iter(observations[criterion].values()))
            criterion_decisions.append(
                select_candidate(
                    prompt_id=scene.scene_id,
                    arm=criterion,
                    candidates=candidates,
                    observations=observations[criterion],
                    questions=(question,),
                    selector_id=first.observer_id,
                    observer_revision=first.observer_revision,
                )
            )
        candidates_all.extend(candidates)
    final_candidates = [
        replace(
            candidate,
            image_path=str((final_output / Path(candidate.image_path).relative_to(output)).resolve()),
        )
        for candidate in candidates_all
    ]
    CandidateManifest(output / "candidate_manifest.jsonl").write(
        final_candidates, verify_rgb=False
    )
    for criterion, values in decisions.items():
        atomic_write_jsonl(
            output / f"selection_{criterion}.jsonl",
            (as_serializable(value) for value in values),
        )
    selected = {criterion: _selected_map(values, candidates_all) for criterion, values in decisions.items()}
    common = [
        scene_id
        for scene_id in scenes
        if all(scene_id in selected[criterion] for criterion in ("naive", "rfo", "gold"))
    ]
    if len(common) < 8:
        return {
            "valid": False,
            "common_samples": len(common),
            "reason": "Fewer than 8 common non-abstaining gradient selections",
        }
    micro_size = int(config.values["training"]["micro_batch_size"])
    snapshots = {
        criterion: adapter.compute_lora_gradient_accumulated(
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
        for criterion in ("naive", "rfo", "gold")
    }
    free = compare_gradients(snapshots["naive"], snapshots["rfo"])
    gold = compare_gradients(snapshots["naive"], snapshots["gold"])
    noise_cosines = []
    rng = random.Random(_stable_seed(config.values["seed"], checkpoint_id, "noise"))
    for split_index in range(int(config.values["gradient_probe"]["noise_floor_splits"])):
        shuffled = list(common)
        rng.shuffle(shuffled)
        middle = len(shuffled) // 2
        halves = []
        for half_index, half in enumerate((shuffled[:middle], shuffled[middle:])):
            halves.append(
                adapter.compute_lora_gradient_accumulated(
                    _batches(
                        half,
                        selected["naive"],
                        scenes,
                        adapter=adapter,
                        micro_size=micro_size,
                        seed=_stable_seed(config.values["seed"], checkpoint_id, split_index, half_index),
                    ),
                    f"noise_{split_index}_{half_index}",
                )
            )
        noise_cosines.append(compare_gradients(*halves).cosine)
    return {
        "valid": True,
        "common_samples": len(common),
        "gda_free": as_serializable(free),
        "gda_gold": as_serializable(gold),
        "noise_floor": noise_interval(noise_cosines),
        "noise_cosines": noise_cosines,
    }


def _target_checkpoints(run_root: Path, rounds: int) -> list[tuple[str, int, Path]]:
    output = [("base", 0, run_root / "rounds" / "round-00" / "arms" / "naive")]
    for round_index in range(1, rounds + 1):
        for arm in ("naive", "rfo_self"):
            output.append(
                (
                    arm,
                    round_index,
                    run_root / "rounds" / f"round-{round_index:02d}" / "arms" / arm,
                )
            )
    return output


def evaluate_paired_run(
    *,
    config_path: str | Path,
    run_root: str | Path,
    outcome_manifest: str | Path,
    probe_manifest: str | Path,
    detector_audit_report: str | Path,
    detector_command: list[str],
    backbone_config: str | Path | None = None,
    lora_target_modules: Sequence[str] | None = None,
    eligible_families: Sequence[str] = (),
) -> dict[str, Any]:
    config = load_config(config_path)
    run_root = Path(run_root).resolve()
    training_report = json.loads((run_root / "training_report.json").read_text(encoding="utf-8"))
    if not str(training_report.get("status", "")).startswith("training_complete"):
        raise RuntimeError("Training is not complete")
    detector_audit = json.loads(Path(detector_audit_report).read_text(encoding="utf-8"))
    competent_families = {
        family
        for family, accuracy in detector_audit["family_open_accuracy"].items()
        if float(accuracy) >= 0.80
    }
    if eligible_families:
        competent_families.intersection_update(str(item) for item in eligible_families)
    outcome_key = (
        "tier_a_outcome"
        if str(config.values["profile"]).startswith("a800_80g")
        else "local_outcome"
    )
    subset_seed = int(config.values["seed"])
    outcomes = _records(
        outcome_manifest,
        int(config.values["data"][outcome_key]),
        subset_seed,
        eligible_families,
    )
    probes = _records(
        probe_manifest,
        int(config.values["gradient_probe"]["size"]),
        subset_seed,
        eligible_families,
    )
    if backbone_config is not None:
        if not lora_target_modules:
            raise ValueError("Show-o2 evaluation requires audited LoRA targets")
        adapter = Showo2Adapter(
            backbone_config=backbone_config,
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
    if (
        training_report.get("model_id") is not None
        and (
            training_report.get("model_id") != adapter.model_id
            or training_report.get("revision") != adapter.revision
        )
    ):
        raise RuntimeError("Evaluation backbone does not match the completed training run")
    lora = config.values["training"]["lora"]
    adapter.attach_lora(
        rank=int(lora["rank"]),
        alpha=int(lora["alpha"]),
        dropout=float(lora["dropout"]),
        target_modules=tuple(lora_target_modules or lora["target_modules"]),
        gradient_checkpointing=bool(config.values["training"]["gradient_checkpointing"]),
    )
    optimizer, scheduler = _optimizer_and_scheduler(adapter, config)
    evaluations = run_root / "evaluations"
    evaluations.mkdir(exist_ok=True)
    training_contract_path = run_root / "training_contract.json"
    checkpoint_config_digest = (
        str(
            json.loads(training_contract_path.read_text(encoding="utf-8"))[
                "checkpoint_config_digest"
            ]
        )
        if training_contract_path.is_file()
        else config.digest
    )
    command = [*detector_command, "--ready-report", str(evaluations / "detector_ready.json")]
    with ObserverServiceClient(command, evaluations / "detector_wire.jsonl") as detector:
        for arm, round_index, checkpoint in _target_checkpoints(
            run_root, int(config.values["training"]["rounds"])
        ):
            key = f"{arm}-round-{round_index:02d}"
            final = evaluations / key
            if (final / "DONE.json").is_file():
                continue
            temporary = evaluations / f".{key}.inprogress"
            if temporary.exists():
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                os.replace(temporary, temporary.with_name(f"{temporary.name}.abandoned-{stamp}"))
            temporary.mkdir(parents=True)
            load_checkpoint(
                checkpoint,
                model=adapter.model,
                optimizer=optimizer,
                scheduler=scheduler,
                expected_config_digest=checkpoint_config_digest,
            )
            outcome_metrics, candidates = _evaluate_outcomes(
                adapter=adapter,
                detector=detector,
                records=outcomes,
                output=temporary / "outcome",
                checkpoint_id=key,
                images_per_prompt=int(config.values["data"]["images_per_prompt"]),
                seed=int(config.values["seed"]),
                competent_families=competent_families,
                final_output=final / "outcome",
            )
            final_candidates = []
            for candidate in candidates:
                relative = Path(candidate.image_path).relative_to(temporary)
                final_candidates.append(replace(candidate, image_path=str((final / relative).resolve())))
            CandidateManifest(temporary / "outcome" / "candidate_manifest.jsonl").write(
                final_candidates, verify_rgb=False
            )
            gda = (
                _evaluate_gda(
                    adapter=adapter,
                    detector=detector,
                    records=probes,
                    output=temporary / "gradient",
                    final_output=final / "gradient",
                    checkpoint_id=key,
                    config=config,
                )
                if bool(training_report.get("gda_enabled"))
                else {"valid": False, "reason": "Gate -1b fallback active"}
            )
            report = {
                "schema_version": 1,
                "arm": arm,
                "round": round_index,
                "step": round_index * int(config.values["training"]["optimizer_steps_per_round"]),
                "outcomes": outcome_metrics,
                "gda": gda,
                "observer_capability": {
                    "observer_id": detector_audit["observer_id"],
                    "observer_revision": detector_audit["observer_revision"],
                    "family_open_accuracy": detector_audit["family_open_accuracy"],
                    "macro_open_accuracy": detector_audit["macro_open_accuracy"],
                    "predicted_yes_rate": detector_audit["predicted_yes_rate"],
                    "competent_families": sorted(competent_families),
                },
                "eligible_families": list(eligible_families),
            }
            atomic_write_json(temporary / "checkpoint_metrics.json", report)
            atomic_write_json(temporary / "DONE.json", {"status": "complete", **report})
            os.replace(temporary, final)

    rows = []
    for arm, round_index, _checkpoint in _target_checkpoints(
        run_root, int(config.values["training"]["rounds"])
    ):
        key = f"{arm}-round-{round_index:02d}"
        report = json.loads((evaluations / key / "checkpoint_metrics.json").read_text(encoding="utf-8"))
        target_arms = ("naive", "rfo_self") if arm == "base" else (arm,)
        for target_arm in target_arms:
            gda = report["gda"]
            rows.append(
                {
                    "seed": int(config.values["seed"]),
                    "arm": target_arm,
                    "round": round_index,
                    "step": report["step"],
                    "internal_score": report["outcomes"]["internal_score"],
                    "external_correctness": report["outcomes"]["external_correctness"],
                    "verifier_coverage": report["outcomes"]["verifier_coverage"],
                    "scfr_competent": report["outcomes"]["scfr_competent"],
                    "scfr_denominator": report["outcomes"]["scfr_denominator"],
                    "observer_answer_entropy": report["outcomes"]["observer_answer_entropy"],
                    "public_view_consistency": report["outcomes"]["public_view_consistency"],
                    "gda_free": gda.get("gda_free", {}).get("cosine") if gda.get("valid") else None,
                    "gda_gold": gda.get("gda_gold", {}).get("cosine") if gda.get("valid") else None,
                    "noise_low": gda.get("noise_floor", {}).get("low") if gda.get("valid") else None,
                    "noise_high": gda.get("noise_floor", {}).get("high") if gda.get("valid") else None,
                    "evidence_status": (
                        "formal_single_seed_pre_gate"
                        if str(config.values["profile"]).startswith("a800_80g")
                        else "local_single_seed_exploratory"
                    ),
                }
            )
    frame = pd.DataFrame(rows).sort_values(["arm", "step"])
    metrics_path = evaluations / "checkpoint_metrics.csv"
    frame.to_csv(metrics_path, index=False)
    naive = frame.loc[frame.arm == "naive"].sort_values("step")
    d_star = estimate_d_star(naive.step, naive.internal_score, naive.external_correctness)
    if naive.gda_free.notna().sum() >= 3:
        valid_gda = naive.dropna(subset=["gda_free", "noise_low", "noise_high"])
        d_g = estimate_d_g(
            valid_gda.step,
            valid_gda.gda_free,
            noise_low=valid_gda.noise_low,
            noise_high=valid_gda.noise_high,
            ema_alpha=float(config.values["gradient_probe"]["ema_alpha"]),
        )
        rho = float(spearmanr(valid_gda.gda_free, valid_gda.gda_gold).statistic)
    else:
        d_g = None
        rho = float("nan")
    breakpoints = {
        "evidence_status": (
            "formal_single_seed_pre_gate"
            if str(config.values["profile"]).startswith("a800_80g")
            else "local_single_seed_exploratory"
        ),
        "d_star": d_star.d_star,
        "d_star_reason": d_star.reason,
        "d_g": d_g.d_g if d_g else None,
        "d_g_reason": d_g.reason if d_g else "Gate -1b fallback active",
        "lead": estimate_lead(d_star.d_star, d_g.d_g if d_g else None),
        "gda_free_gold_spearman": rho,
    }
    atomic_write_json(evaluations / "breakpoints.json", breakpoints)
    figures = render_figure1_from_csv(
        metrics_path,
        evaluations / "figures" / "figure1_local",
        arm="naive",
        d_g=d_g.d_g if d_g else None,
        d_star=d_star.d_star,
        evidence_status=(
            "formal pre-gate"
            if str(config.values["profile"]).startswith("a800_80g")
            else "local exploratory"
        ),
    )
    report = {
        "schema_version": 1,
        "status": (
            "formal_single_seed_evaluation_complete"
            if str(config.values["profile"]).startswith("a800_80g")
            else "local_closed_loop_complete"
        ),
        "metrics": str(metrics_path),
        "breakpoints": breakpoints,
        "figures": figures,
        "formal_claims_allowed": False,
    }
    atomic_write_json(evaluations / "evaluation_report.json", report)
    return report
