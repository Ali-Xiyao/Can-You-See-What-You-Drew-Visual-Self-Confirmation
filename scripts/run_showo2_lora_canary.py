"""Select audited Show-o2 LoRA targets and run Gate -2A/A4 backward-resume canary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import yaml

from selfsight.backbones.lora_selection import (
    build_lora_target_selection,
    validate_lora_target_selection,
)
from selfsight.backbones.showo2 import (
    Showo2Adapter,
    Showo2GenerationBatch,
    Showo2ReplayBatch,
)
from selfsight.schemas import AtomicQuestion
from selfsight.training.checkpoint import lora_state_dict
from selfsight.utils.hashing import sha256_file, sha256_json
from selfsight.utils.jsonl import atomic_write_json, read_jsonl


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _lora_digest(model: object) -> str:
    digest = hashlib.sha256()
    state = lora_state_dict(model)
    if not state:
        raise RuntimeError("No LoRA tensors are present in the Show-o2 model")
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.float().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _assert_route(
    *,
    backbone: dict,
    canary: dict,
    reference: dict,
    generated: dict,
    human: dict,
) -> tuple[str, ...]:
    model_id = str(backbone["backbone_id"])
    revision = str(backbone["revision"])
    source_revision = str(backbone["source"]["revision"])
    dependencies = {str(key): str(value) for key, value in backbone["dependencies"].items()}
    for label, report in (
        ("A1", canary),
        ("A2", reference),
        ("A3-generated", generated),
        ("A3-human", human),
    ):
        if report.get("model_id") != model_id or report.get("revision") != revision:
            raise RuntimeError(f"{label} report identity mismatch")
        if report.get("source_revision") != source_revision:
            raise RuntimeError(f"{label} source revision mismatch")
        if report.get("dependency_revisions") != dependencies:
            raise RuntimeError(f"{label} dependency revision mismatch")
    if not bool(canary.get("passed")):
        raise RuntimeError("A1 engineering canary is red; A4 is forbidden")
    retained = tuple(str(item) for item in reference.get("passing_families", ()))
    if len(retained) < 4 or not bool(reference.get("passed")):
        raise RuntimeError("A2 has fewer than four retained families; A4 is forbidden")
    if not bool(generated.get("passed_without_human_precision")):
        raise RuntimeError("A3 generated measurability is red; A4 is forbidden")
    if not bool(human.get("passed")):
        raise RuntimeError("A3 blind precision audit is incomplete or red; A4 is forbidden")
    return retained


def _step(adapter, batch, replay, optimizer, scheduler, config: dict) -> dict:
    import torch

    trainable = [parameter for parameter in adapter.model.parameters() if parameter.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    generation = adapter.generation_loss(batch)
    replay_loss = adapter.understanding_replay_loss(replay)
    if not torch.isfinite(generation) or not torch.isfinite(replay_loss):
        raise RuntimeError("Show-o2 A4 produced a non-finite generation or replay loss")
    ratio = float(config["understanding_replay_ratio"])
    total = (1.0 - ratio) * generation + ratio * replay_loss
    total.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, float(config["max_grad_norm"]))
    if not torch.isfinite(grad_norm):
        raise RuntimeError("Show-o2 A4 produced a non-finite LoRA gradient norm")
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "generation_loss": float(generation.detach().cpu()),
        "understanding_replay_loss": float(replay_loss.detach().cpu()),
        "combined_loss": float(total.detach().cpu()),
        "gradient_norm_before_clip": float(grad_norm.detach().cpu()),
    }


def _run(args: argparse.Namespace) -> dict:
    import torch

    backbone = yaml.safe_load(args.backbone_config.read_text(encoding="utf-8"))
    readiness = yaml.safe_load(args.readiness_config.read_text(encoding="utf-8"))
    canary = _read_json(args.canary_report)
    reference = _read_json(args.reference_report)
    generated = _read_json(args.generated_report)
    human = _read_json(args.human_report)
    retained = _assert_route(
        backbone=backbone,
        canary=canary,
        reference=reference,
        generated=generated,
        human=human,
    )
    target_selection = validate_lora_target_selection(
        args.target_config, canary_report=args.canary_report
    )
    records = [
        row for row in read_jsonl(args.manifest) if str(row["scene"]["family"]) in retained
    ]
    if not records:
        raise RuntimeError("A4 manifest contains no A2-retained family")
    record = records[0]
    question = AtomicQuestion.from_dict(record["questions"][0])
    lora_config = readiness["lora_canary"]
    torch.cuda.reset_peak_memory_stats(torch.device(backbone["hardware"]["generator_device"]))
    adapter = Showo2Adapter(
        backbone_config=args.backbone_config,
        device=str(backbone["hardware"]["generator_device"]),
        lazy=False,
    )
    attached = adapter.attach_lora(
        target_modules=tuple(target_selection["target_modules"]),
        rank=int(lora_config["rank"]),
        alpha=int(lora_config["alpha"]),
        dropout=float(lora_config["dropout"]),
        gradient_checkpointing=True,
    )
    trainable = [parameter for parameter in adapter.model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(lora_config["learning_rate"]),
        weight_decay=float(lora_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    batch = Showo2GenerationBatch(
        prompts=(str(record["scene"]["prompt"]),),
        images=(str(record["reference_image"]),),
        sample_ids=(str(record["scene"]["scene_id"]),),
        latent_seed=int(readiness["audit_seed"]),
    )
    replay = Showo2ReplayBatch(
        images=(str(record["reference_image"]),),
        questions=(question.text,),
        answers=(question.expected_answer,),
        sample_ids=(str(record["scene"]["scene_id"]),),
        latent_seed=int(readiness["audit_seed"]),
    )
    started = perf_counter()
    digest_before = _lora_digest(adapter.model)
    first_step = _step(adapter, batch, replay, optimizer, scheduler, lora_config)
    digest_after_first = _lora_digest(adapter.model)
    config_values = {
        "backbone": backbone,
        "readiness_lora_canary": lora_config,
        "target_selection_digest": target_selection["selection_digest"],
    }
    config_digest = sha256_json(config_values)
    checkpoint_one = adapter.save_adapter(
        args.output.parent / "a4-checkpoint-step-1",
        optimizer=optimizer,
        scheduler=scheduler,
        config_digest=config_digest,
        config_values=config_values,
        step=1,
        round_index=0,
        metadata={"stage": "Gate -2A/A4", "sample_id": batch.sample_ids[0]},
    )
    with torch.no_grad():
        trainable[0].add_(0.125)
    digest_corrupted = _lora_digest(adapter.model)
    resumed = adapter.load_adapter(
        checkpoint_one,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_config_digest=config_digest,
    )
    digest_restored = _lora_digest(adapter.model)
    second_step = _step(adapter, batch, replay, optimizer, scheduler, lora_config)
    digest_after_resume_step = _lora_digest(adapter.model)
    checkpoint_two = adapter.save_adapter(
        args.output.parent / "a4-checkpoint-step-2",
        optimizer=optimizer,
        scheduler=scheduler,
        config_digest=config_digest,
        config_values=config_values,
        step=2,
        round_index=0,
        metadata={"stage": "Gate -2A/A4-resumed", "sample_id": batch.sample_ids[0]},
    )
    manifest_one = _read_json(checkpoint_one / "manifest.json")
    manifest_two = _read_json(checkpoint_two / "manifest.json")
    checks = {
        "only_lora_trainable": bool(trainable)
        and all("lora_" in name.lower() for name in attached["trainable_names"]),
        "first_optimizer_step_changed_adapter": digest_after_first != digest_before,
        "corruption_changed_adapter": digest_corrupted != digest_after_first,
        "adapter_exactly_restored": digest_restored == digest_after_first,
        "resume_step_changed_adapter": digest_after_resume_step != digest_restored,
        "optimizer_and_scheduler_resumed": resumed == {"step": 1, "round_index": 0}
        and scheduler.last_epoch == 2,
        "adapter_only_checkpoints": bool(manifest_one.get("adapter_only"))
        and bool(manifest_two.get("adapter_only")),
    }
    frozen_step0_supported = bool(canary["passed"]) and bool(reference["passed"]) and checks[
        "adapter_only_checkpoints"
    ]
    return {
        "schema_version": 2,
        "stage": "minus_2a_lora_backward_resume",
        "model_id": adapter.model_id,
        "revision": adapter.revision,
        "source_revision": adapter.identity.source_revision,
        "dependency_revisions": adapter.dependency_revisions(),
        "target_config": str(args.target_config),
        "target_config_sha256": sha256_file(args.target_config),
        "target_selection_digest": target_selection["selection_digest"],
        "lora": attached,
        "sample_id": batch.sample_ids[0],
        "first_step": first_step,
        "second_step_after_resume": second_step,
        "digests": {
            "before": digest_before,
            "after_first": digest_after_first,
            "corrupted": digest_corrupted,
            "restored": digest_restored,
            "after_resume_step": digest_after_resume_step,
        },
        "checkpoint_step_1": str(checkpoint_one),
        "checkpoint_step_1_manifest_sha256": sha256_file(checkpoint_one / "manifest.json"),
        "checkpoint_step_2": str(checkpoint_two),
        "checkpoint_step_2_manifest_sha256": sha256_file(checkpoint_two / "manifest.json"),
        "frozen_step0_supported": frozen_step0_supported,
        "elapsed_seconds": perf_counter() - started,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(adapter.device)),
        "checks": checks,
        "passed": all(checks.values()) and frozen_step0_supported,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select")
    select.add_argument("--canary-report", type=Path, required=True)
    select.add_argument("--suffix", action="append", required=True)
    select.add_argument("--output", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--backbone-config", type=Path, required=True)
    run.add_argument(
        "--readiness-config", type=Path, default=Path("configs/readiness_v2.2.yaml")
    )
    run.add_argument("--canary-report", type=Path, required=True)
    run.add_argument("--reference-report", type=Path, required=True)
    run.add_argument("--generated-report", type=Path, required=True)
    run.add_argument("--human-report", type=Path, required=True)
    run.add_argument("--target-config", type=Path, required=True)
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "select":
        args.canary_report = args.canary_report.resolve()
        args.output = args.output.resolve()
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite target selection: {args.output}")
        report = build_lora_target_selection(args.canary_report, suffixes=args.suffix)
        atomic_write_json(args.output, report)
    else:
        for field in (
            "backbone_config",
            "readiness_config",
            "canary_report",
            "reference_report",
            "generated_report",
            "human_report",
            "target_config",
            "manifest",
        ):
            setattr(args, field, getattr(args, field).resolve())
        args.output = args.output.resolve()
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite A4 report: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report = _run(args)
        atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2))
    if args.command == "run":
        raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
