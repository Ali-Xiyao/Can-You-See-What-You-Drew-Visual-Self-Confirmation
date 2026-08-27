"""One-image Show-o generation, RGB reload, MMU, LoRA, and resume canary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from selfsight.config import load_config, write_config_snapshot
from selfsight.data.generator import build_splits
from selfsight.data.questions import build_primary_atom, build_question
from selfsight.data.renderer import render_scene
from selfsight.showo_adapter import ShowoAdapter, ShowoReplayBatch, ShowoSFTBatch
from selfsight.training.checkpoint import load_checkpoint, lora_state_dict, save_checkpoint
from selfsight.utils.evidence import write_host_manifest
from selfsight.utils.hashing import sha256_file
from selfsight.utils.jsonl import atomic_write_json


def _lora_digest(model: object) -> str:
    """Stable value digest used to prove adapter restoration, independent of torch.save."""

    digest = hashlib.sha256()
    for name, tensor in sorted(lora_state_dict(model).items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.float().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/local_3090.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-lora", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_config_snapshot(config, output / "resolved_config.json")
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    adapter = ShowoAdapter(
        device="cuda:0",
        trainable=not args.skip_lora,
        generation_timesteps=int(config.values["model"]["generation_timesteps"]),
        guidance_scale=float(config.values["model"]["guidance_scale"]),
    )
    lora_summary = None
    if not args.skip_lora:
        lora = config.values["training"]["lora"]
        lora_summary = adapter.attach_lora(
            rank=int(lora["rank"]),
            alpha=int(lora["alpha"]),
            dropout=float(lora["dropout"]),
            target_modules=tuple(lora["target_modules"]),
            gradient_checkpointing=True,
        )
    scene = build_splits(int(config.values["seed"]))["tier_a_probe"][0]
    candidate = adapter.generate_images([scene.prompt], [int(config.values["seed"])], output / "images", "step-0")[0]
    atom = build_primary_atom(scene)
    question = build_question(atom)
    observation = adapter.observe_atoms(candidate.image_path, (question,))
    training_report = None
    if not args.skip_lora:
        trainable = [parameter for parameter in adapter.model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable,
            lr=float(config.values["training"]["learning_rate"]),
            weight_decay=float(config.values["training"]["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        before_digest = _lora_digest(adapter.model)
        batch = ShowoSFTBatch(
            prompts=(scene.prompt,),
            images=(candidate.image_path,),
            sample_ids=(scene.scene_id,),
            mask_seed=int(config.values["seed"]),
        )
        replay_image = render_scene(scene, output / "images" / "replay-reference.png")
        replay_batch = ShowoReplayBatch(
            images=(replay_image,),
            questions=(question.text,),
            answers=(question.expected_answer,),
            sample_ids=(scene.scene_id,),
        )
        optimizer.zero_grad(set_to_none=True)
        t2i_loss = adapter.sft_loss(batch)
        if not torch.isfinite(t2i_loss):
            raise RuntimeError(f"Non-finite Show-o T2I canary loss: {float(t2i_loss.detach().cpu())}")
        replay_ratio = float(config.values["training"]["understanding_replay_ratio"])
        ((1.0 - replay_ratio) * t2i_loss).backward()
        replay_loss = adapter.mmu_replay_loss(replay_batch)
        if not torch.isfinite(replay_loss):
            raise RuntimeError(
                f"Non-finite Show-o replay canary loss: {float(replay_loss.detach().cpu())}"
            )
        (replay_ratio * replay_loss).backward()
        loss_value = (
            (1.0 - replay_ratio) * float(t2i_loss.detach().cpu())
            + replay_ratio * float(replay_loss.detach().cpu())
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable, float(config.values["training"]["max_grad_norm"])
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        updated_digest = _lora_digest(adapter.model)
        if updated_digest == before_digest:
            raise RuntimeError("LoRA canary optimizer step did not change adapter parameters")
        checkpoint_dir = save_checkpoint(
            output / "checkpoint-step-1",
            model=adapter.model,
            optimizer=optimizer,
            scheduler=scheduler,
            config_digest=config.digest,
            config_values=config.values,
            step=1,
            round_index=0,
            metadata={"kind": "showo-windows-canary", "sample_id": scene.scene_id},
        )
        with torch.no_grad():
            first_trainable = trainable[0]
            first_trainable.add_(0.125)
        corrupted_digest = _lora_digest(adapter.model)
        resume_state = load_checkpoint(
            checkpoint_dir,
            model=adapter.model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_config_digest=config.digest,
        )
        restored_digest = _lora_digest(adapter.model)
        if restored_digest != updated_digest or restored_digest == corrupted_digest:
            raise RuntimeError("Adapter-only checkpoint did not exactly restore LoRA parameters")
        training_report = {
            "loss": loss_value,
            "t2i_loss": float(t2i_loss.detach().cpu()),
            "mmu_replay_loss": float(replay_loss.detach().cpu()),
            "understanding_replay_ratio": replay_ratio,
            "gradient_norm_before_clip": float(grad_norm.detach().cpu()),
            "lora_digest_before": before_digest,
            "lora_digest_after_step": updated_digest,
            "lora_digest_corrupted": corrupted_digest,
            "lora_digest_restored": restored_digest,
            "resume_state": resume_state,
            "checkpoint_path": str(checkpoint_dir),
            "checkpoint_manifest_sha256": sha256_file(checkpoint_dir / "manifest.json"),
        }
    report = {
        "schema_version": 1,
        "model_id": adapter.model_id,
        "revision": adapter.revision,
        "prompt": scene.prompt,
        "seed": int(config.values["seed"]),
        "image_path": candidate.image_path,
        "image_sha256": sha256_file(candidate.image_path),
        "rgb_sha256": candidate.rgb_sha256,
        "question": question.text,
        "expected_answer": question.expected_answer,
        "observation": observation.to_dict(),
        "lora_summary": lora_summary.__dict__ if lora_summary else None,
        "training_resume": training_report,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(0)),
        "gpu_name": torch.cuda.get_device_name(0),
    }
    atomic_write_json(output / "canary_report.json", report)
    write_host_manifest(output / "host_manifest.json")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
