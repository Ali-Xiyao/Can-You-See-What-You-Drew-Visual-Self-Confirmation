"""Non-scientific end-to-end smoke loop using deterministic synthetic backends."""

from __future__ import annotations

import random
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from selfsight.analysis.breakpoints import estimate_d_g, estimate_d_star, estimate_lead
from selfsight.analysis.figure1 import render_figure1_from_csv
from selfsight.config import load_config, write_config_snapshot
from selfsight.data.candidates import CandidateManifest
from selfsight.data.generator import build_splits
from selfsight.data.questions import build_primary_atom, build_question
from selfsight.data.renderer import render_scene
from selfsight.data.subsets import stable_stratified_sample
from selfsight.data.verifier import verify_image
from selfsight.observers.client import ObserverServiceClient
from selfsight.rfo.isolation import hard_render, make_blind_request
from selfsight.rfo.selection import select_candidate
from selfsight.schemas import (
    AtomicObservation,
    CandidateRecord,
    Color,
    ObservationResult,
    QuestionFamily,
    QuestionFormat,
    SceneSpec,
    Shape,
    Size,
    as_serializable,
)
from selfsight.training.checkpoint import load_checkpoint, save_checkpoint
from selfsight.training.paired import build_paired_schedule
from selfsight.utils.hashing import rgb_sha256
from selfsight.utils.jsonl import atomic_write_json, atomic_write_jsonl


def _corrupt_scene(scene: SceneSpec) -> SceneSpec:
    atom = build_primary_atom(scene)
    objects = list(scene.objects)
    metadata = dict(scene.metadata)
    if scene.family == QuestionFamily.EXISTENCE:
        target_shape = Shape(metadata["target_shape"])
        target_color = Color(metadata["target_color"])
        if atom.answer == "yes":
            target = next(item for item in objects if item.shape == target_shape and item.color == target_color)
            objects = [item for item in objects if item.object_id != target.object_id]
        else:
            objects[0] = replace(objects[0], shape=target_shape, color=target_color)
    elif scene.family == QuestionFamily.COUNT:
        target_shape = Shape(metadata["target_shape"])
        targets = [item for item in objects if item.shape == target_shape]
        if len(targets) > 1:
            objects = [item for item in objects if item.object_id != targets[0].object_id]
        else:
            other_index = next(index for index, item in enumerate(objects) if item.shape != target_shape)
            objects[other_index] = replace(objects[other_index], shape=target_shape)
    elif scene.family in {QuestionFamily.COLOR, QuestionFamily.BINDING}:
        target_id = str(metadata["target_object_id"])
        target = next(item for item in objects if item.object_id == target_id)
        new_color = next(color for color in Color if color != target.color)
        objects = [replace(item, color=new_color) if item.object_id == target_id else item for item in objects]
    elif scene.family == QuestionFamily.SIZE:
        target_id = str(metadata["target_object_id"])
        target = next(item for item in objects if item.object_id == target_id)
        new_size = Size.LARGE if target.size == Size.SMALL else Size.SMALL
        objects = [replace(item, size=new_size) if item.object_id == target_id else item for item in objects]
    else:
        subject_shape = Shape(metadata["subject_shape"])
        object_shape = Shape(metadata["object_shape"])
        subject = next(item for item in objects if item.shape == subject_shape)
        other = next(item for item in objects if item.shape == object_shape)
        should_be_true = atom.answer != "yes"
        relation = str(metadata["relation"])
        if relation == "left_of":
            subject_center = (112 if should_be_true else 400, subject.center[1])
            other_center = (400 if should_be_true else 112, other.center[1])
            objects = [
                replace(item, center=subject_center) if item.object_id == subject.object_id else
                replace(item, center=other_center) if item.object_id == other.object_id else item
                for item in objects
            ]
        elif relation == "above":
            subject_center = (subject.center[0], 112 if should_be_true else 400)
            other_center = (other.center[0], 400 if should_be_true else 112)
            objects = [
                replace(item, center=subject_center) if item.object_id == subject.object_id else
                replace(item, center=other_center) if item.object_id == other.object_id else item
                for item in objects
            ]
        else:
            subject_size = Size.LARGE if should_be_true else Size.SMALL
            other_size = Size.SMALL if should_be_true else Size.LARGE
            objects = [
                replace(item, size=subject_size) if item.object_id == subject.object_id else
                replace(item, size=other_size) if item.object_id == other.object_id else item
                for item in objects
            ]
    changed = replace(scene, objects=tuple(objects))
    changed_atom = build_primary_atom(changed)
    if scene.family in {QuestionFamily.EXISTENCE, QuestionFamily.COUNT, QuestionFamily.SPATIAL} and (
        changed_atom.answer != atom.answer
    ):
        raise AssertionError("Corruption must preserve intended metadata while changing visible evidence")
    if scene.family in {QuestionFamily.COLOR, QuestionFamily.SIZE, QuestionFamily.BINDING} and (
        changed_atom.answer == atom.answer
    ):
        raise AssertionError("Attribute corruption must change the target object's visible answer")
    verifier = verify_image(render_scene(changed), [atom])
    if verifier.answers[atom.atom_id] == atom.answer:
        raise AssertionError(f"Corruption did not change visible answer for {scene.scene_id}")
    return changed


def _observation(
    candidate_id: str,
    rgb_hash: str,
    question_id: str,
    answer: str | None,
    observer_id: str,
) -> ObservationResult:
    return ObservationResult(
        request_id=candidate_id,
        observer_id=observer_id,
        observer_revision="mock-v1-non-scientific",
        rgb_sha256=rgb_hash,
        answers=(
            AtomicObservation(
                question_id=question_id,
                raw_answer=answer or "unknown",
                normalized_answer=answer,
                abstain=answer is None,
            ),
        ),
    )


class _DummyLoraModel:
    @staticmethod
    def create():
        import torch

        class Tiny(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lora_A = torch.nn.Parameter(torch.zeros(8, 8))
                self.frozen = torch.nn.Parameter(torch.ones(2), requires_grad=False)

        return Tiny()


def _checkpoint_rounds(root: Path, config: Any, rounds: int) -> dict[str, Any]:
    import torch

    resume_reports = {}
    for arm_index, arm in enumerate(("naive", "rfo_self")):
        model = _DummyLoraModel.create()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        for round_index in range(rounds + 1):
            optimizer.zero_grad(set_to_none=True)
            loss = (model.lora_A - (round_index + arm_index) * 0.001).square().mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            save_checkpoint(
                root / "checkpoints" / arm / f"round-{round_index:02d}",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config_digest=config.digest,
                config_values=config.values,
                step=round_index * int(config.values["training"]["optimizer_steps_per_round"]),
                round_index=round_index,
                metadata={"evidence_status": "mock_non_scientific", "arm": arm},
            )
        restored = _DummyLoraModel.create()
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4)
        restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda _: 1.0)
        state = load_checkpoint(
            root / "checkpoints" / arm / f"round-{rounds:02d}",
            model=restored,
            optimizer=restored_optimizer,
            scheduler=restored_scheduler,
            expected_config_digest=config.digest,
        )
        resume_reports[arm] = state
    return resume_reports


def _metric_rows(rounds: int, steps_per_round: int, seed: int) -> list[dict[str, Any]]:
    rows = []
    for arm in ("naive", "rfo_self"):
        for round_index in range(rounds + 1):
            step = round_index * steps_per_round
            if arm == "naive":
                internal = min(0.95, 0.64 + 0.028 * round_index)
                external = 0.69 + 0.016 * min(round_index, 4) - 0.024 * max(0, round_index - 4)
                gda_free = 0.91 - 0.018 * min(round_index, 2) - 0.105 * max(0, round_index - 2)
                gda_gold = 0.93 - 0.015 * min(round_index, 2) - 0.112 * max(0, round_index - 2)
                scfr = 0.07 + 0.006 * min(round_index, 2) + 0.026 * max(0, round_index - 2)
            else:
                internal = min(0.94, 0.64 + 0.023 * round_index)
                external = min(0.90, 0.69 + 0.018 * round_index)
                gda_free = 0.91 - 0.012 * round_index
                gda_gold = 0.93 - 0.010 * round_index
                scfr = max(0.025, 0.07 - 0.004 * round_index)
            rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "round": round_index,
                    "step": step,
                    "internal_score": internal,
                    "external_correctness": external,
                    "gda_free": max(-0.25, gda_free),
                    "gda_gold": max(-0.25, gda_gold),
                    "noise_low": 0.82,
                    "noise_high": 0.96,
                    "scfr_competent": min(0.95, scfr),
                    "evidence_status": "mock_non_scientific",
                }
            )
    return rows


def run_mock_pilot(config_path: str | Path, output: str | Path | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path(output) if output else Path(config.section("paths")["run_root"]) / f"mock-pilot-{timestamp}"
    root = root.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty mock run: {root}")
    root.mkdir(parents=True, exist_ok=True)
    write_config_snapshot(config, root / "config.snapshot.json")
    atomic_write_json(root / "EVIDENCE_STATUS.json", {"status": "mock_non_scientific", "usable_for_claims": False})

    splits = build_splits(int(config.values["seed"]))
    training_scenes = stable_stratified_sample(
        splits["train"],
        int(config.values["data"]["local_train_prompts"]),
        stratum=lambda scene: scene.family.value,
        item_id=lambda scene: scene.scene_id,
        seed=int(config.values["seed"]),
    )
    by_id = {scene.scene_id: scene for scene in training_scenes}
    training = config.values["training"]
    schedule = build_paired_schedule(
        list(by_id),
        rounds=int(training["rounds"]),
        prompts_per_round=int(training["prompts_per_round"]),
        candidate_k=int(training["candidate_k"]),
        seed=int(config.values["seed"]),
    )
    atomic_write_jsonl(root / "schedule.jsonl", (as_serializable(item) for item in schedule))

    candidate_records = []
    decision_records = []
    for arm in ("naive", "rfo_self"):
        for entry in schedule:
            scene = by_id[entry.prompt_id]
            atom = build_primary_atom(scene)
            question = build_question(atom, QuestionFormat.OPEN)
            candidates = []
            internal_observations = {}
            pixel_observations = {}
            quality = (0.74 - 0.025 * max(0, entry.round_index - 3)) if arm == "naive" else min(0.92, 0.74 + 0.015 * entry.round_index)
            leakage = min(0.90, 0.08 + 0.075 * entry.round_index)
            for candidate_index, sampling_seed in enumerate(entry.candidate_seeds):
                local_rng = random.Random(sampling_seed + (0 if arm == "naive" else 91_337))
                visible_scene = scene if local_rng.random() < quality else _corrupt_scene(scene)
                candidate_id = f"{arm}-r{entry.round_index:02d}-{scene.scene_id}-k{candidate_index}"
                image_path = root / "candidates" / arm / f"round-{entry.round_index:02d}" / f"{candidate_id}.png"
                hard_render(render_scene(visible_scene), image_path)
                rgb_hash = rgb_sha256(image_path)
                pixel_answer = verify_image(image_path, [atom]).answers[atom.atom_id]
                internal_answer = atom.answer if pixel_answer != atom.answer and local_rng.random() < leakage else pixel_answer
                candidate = CandidateRecord(
                    candidate_id=candidate_id,
                    prompt_id=scene.scene_id,
                    scene_id=scene.scene_id,
                    sampling_seed=sampling_seed,
                    image_path=str(image_path.resolve()),
                    rgb_sha256=rgb_hash,
                    generator_id="mock/unified-generator",
                    generator_revision="deterministic-v1-non-scientific",
                    checkpoint_id=f"{arm}-round-{entry.round_index:02d}",
                    atom_answers={atom.atom_id: internal_answer or "abstain"},
                    verifier_answers={atom.atom_id: pixel_answer or "abstain"},
                    metadata={"evidence_status": "mock_non_scientific"},
                )
                candidates.append(candidate)
                candidate_records.append(candidate)
                internal_observations[candidate_id] = _observation(
                    candidate_id, rgb_hash, question.question_id, internal_answer, "mock/internal-leaky"
                )
                pixel_observations[candidate_id] = _observation(
                    candidate_id, rgb_hash, question.question_id, pixel_answer, "mock/rgb-blind"
                )
            for criterion, observations in (("naive", internal_observations), ("rfo", pixel_observations)):
                decision = select_candidate(
                    prompt_id=scene.scene_id,
                    arm=f"{arm}:{criterion}",
                    candidates=candidates,
                    observations=observations,
                    questions=(question,),
                    selector_id=f"mock/{criterion}",
                    observer_revision="mock-v1-non-scientific",
                )
                decision_records.append(as_serializable(decision))

    CandidateManifest(root / "candidate_manifest.jsonl").write(candidate_records, verify_rgb=False)
    atomic_write_jsonl(root / "selection_decisions.jsonl", decision_records)

    canary_scene = splits["tier_a_probe"][0]
    canary_atom = build_primary_atom(canary_scene)
    canary_question = build_question(canary_atom)
    canary_path = root / "canary" / "hard-render.png"
    hard_render(render_scene(canary_scene), canary_path)
    service_command = [sys.executable, "-m", "selfsight.observers.service", "--backend", "mock", "--device", "cpu"]
    with ObserverServiceClient(service_command, root / "canary" / "observer-wire.jsonl") as client:
        canary_result = client.observe(make_blind_request(canary_path, (canary_question,), "mock-canary"))
    atomic_write_json(root / "canary" / "observer-result.json", as_serializable(canary_result))

    resume_reports = _checkpoint_rounds(root, config, int(training["rounds"]))
    metrics = pd.DataFrame(
        _metric_rows(
            int(training["rounds"]),
            int(training["optimizer_steps_per_round"]),
            int(config.values["seed"]),
        )
    )
    metrics_path = root / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    naive = metrics.loc[metrics.arm == "naive"].sort_values("step")
    d_star = estimate_d_star(naive.step, naive.internal_score, naive.external_correctness)
    d_g = estimate_d_g(
        naive.step,
        naive.gda_free,
        noise_low=float(naive.noise_low.iloc[0]),
        noise_high=float(naive.noise_high.iloc[0]),
    )
    breakpoints = {
        "evidence_status": "mock_non_scientific",
        "d_star": d_star.d_star,
        "d_g": d_g.d_g,
        "lead": estimate_lead(d_star.d_star, d_g.d_g),
        "d_star_reason": d_star.reason,
        "d_g_reason": d_g.reason,
        "early_safe": d_g.early_safe,
    }
    atomic_write_json(root / "breakpoints.json", breakpoints)
    figure_outputs = render_figure1_from_csv(
        metrics_path,
        root / "figures" / "figure1_mock",
        arm="naive",
        d_g=d_g.d_g,
        d_star=d_star.d_star,
        evidence_status="mock",
    )
    report = {
        "schema_version": 1,
        "status": "complete_mock_non_scientific",
        "candidate_count": len(candidate_records),
        "decision_count": len(decision_records),
        "observer_canary_answer": canary_result.answers[0].normalized_answer,
        "observer_canary_expected": canary_question.expected_answer,
        "checkpoint_resume": resume_reports,
        "breakpoints": breakpoints,
        "figure_outputs": figure_outputs,
    }
    atomic_write_json(root / "run_report.json", report)
    return {"run_root": str(root), **report}
