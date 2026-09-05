"""Adapter-only checkpoints with optimizer, scheduler, RNG, and config state."""

from __future__ import annotations

import json
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from selfsight.utils.hashing import sha256_file
from selfsight.utils.jsonl import atomic_write_json


def _torch():
    import torch

    return torch


def _rng_state() -> dict[str, Any]:
    torch = _torch()
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    torch = _torch()
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def lora_state_dict(model: Any) -> dict[str, Any]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.state_dict().items()
        if "lora_" in name.lower()
    }


def capture_base_state(model: Any) -> dict[str, Any]:
    """What an arm needs to start a round exactly where the other arm started.

    Round 0 has no previous checkpoint to load, and the paired design runs both
    arms through one resident backbone. Without a base to come back to, the
    second arm of round 0 starts from the first arm's updated weights -- the
    two arms are then not two arms.

    The RNG goes with the adapter and not because it changes the weights. From
    round 1 on, `load_checkpoint` restores each arm's RNG along with its
    parameters; if round 0 skipped that, the arms would draw different LoRA
    dropout masks in the one round where they are supposed to be identical,
    and the pairing would be weaker exactly where it is strongest.
    """
    # `.clone()` and not just `lora_state_dict`: on a CPU tensor `.cpu()` is the
    # identity and hands back the live parameter, so the "base" would follow the
    # first arm's in-place optimizer step and restore to exactly the state it is
    # supposed to undo. `save_checkpoint` gets away with the same call because it
    # serialises immediately; a snapshot that outlives the step does not.
    return {
        "adapter": {name: tensor.clone() for name, tensor in lora_state_dict(model).items()},
        "rng": _rng_state(),
    }


def restore_base_state(model: Any, base: dict[str, Any]) -> None:
    """Counterpart to `capture_base_state`. Adapter only, like the checkpoints."""
    _missing, unexpected = model.load_state_dict(base["adapter"], strict=False)
    bad_unexpected = [name for name in unexpected if "lora_" in name.lower()]
    if bad_unexpected:
        raise ValueError(f"Unexpected LoRA keys restoring base: {bad_unexpected[:20]}")
    _restore_rng_state(base["rng"])


def save_checkpoint(
    directory: str | Path,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    config_digest: str,
    config_values: dict[str, Any],
    step: int,
    round_index: int,
    metadata: dict[str, Any] | None = None,
) -> Path:
    torch = _torch()
    destination = Path(directory).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=str(destination.parent)))
    try:
        adapter_path = temporary / "adapter.pt"
        state_path = temporary / "training_state.pt"
        torch.save(lora_state_dict(model), adapter_path)
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "rng": _rng_state(),
                "step": int(step),
                "round_index": int(round_index),
                "config_digest": config_digest,
            },
            state_path,
        )
        manifest = {
            "schema_version": 1,
            "adapter_only": True,
            "step": int(step),
            "round_index": int(round_index),
            "config_digest": config_digest,
            "config": config_values,
            "metadata": metadata or {},
            "files": {
                "adapter.pt": sha256_file(adapter_path),
                "training_state.pt": sha256_file(state_path),
            },
        }
        atomic_write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def load_checkpoint(
    directory: str | Path,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    expected_config_digest: str,
) -> dict[str, int]:
    torch = _torch()
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["config_digest"] != expected_config_digest:
        raise ValueError(
            f"Checkpoint config mismatch: {manifest['config_digest']} != {expected_config_digest}"
        )
    for file_name, expected in manifest["files"].items():
        actual = sha256_file(directory / file_name)
        if actual != expected:
            raise ValueError(f"Checkpoint file hash mismatch: {file_name}")
    adapter = torch.load(directory / "adapter.pt", map_location="cpu", weights_only=False)
    _missing, unexpected = model.load_state_dict(adapter, strict=False)
    bad_unexpected = [name for name in unexpected if "lora_" in name.lower()]
    if bad_unexpected:
        raise ValueError(f"Unexpected LoRA keys on resume: {bad_unexpected[:20]}")
    state = torch.load(directory / "training_state.pt", map_location="cpu", weights_only=False)
    # Both may be None: evaluation reloads a checkpoint to draw from it, and has
    # no optimiser to restore. The scheduler was already guarded; the optimiser
    # was not, and an evaluation pass would have died on the attribute lookup.
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state["scheduler"] is not None:
        scheduler.load_state_dict(state["scheduler"])
    _restore_rng_state(state["rng"])
    return {"step": int(state["step"]), "round_index": int(state["round_index"])}


def restore_arm_state(
    previous: str | Path,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    expected_config_digest: str,
    base: dict[str, Any],
) -> str:
    """Put `model` into `arm`'s state for the round about to run.

    The three places that swap arms mid-round must all go through here. Written
    out at each site it reads as `if previous.exists(): load_checkpoint(...)`,
    whose else branch is silent and wrong: doing nothing leaves whichever arm
    ran last in the weights. Returns which source it used, so callers can say so.
    """
    if Path(previous).exists():
        load_checkpoint(previous, model=model, optimizer=optimizer, scheduler=scheduler,
                        expected_config_digest=expected_config_digest)
        return "checkpoint"
    restore_base_state(model, base)
    return "base"
