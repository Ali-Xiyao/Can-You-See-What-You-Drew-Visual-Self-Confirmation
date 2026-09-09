"""Rebuild a trained checkpoint's backbone for an offline re-measurement pass.

Deviations 11.3 and 13.2 both ask for the same thing: go back to a checkpoint
the run already saved and ask it something the run did not record. Neither
retrains anything. What they need is that the model on the GPU is bit-for-bit
the model that drew the images, and the way to get that wrong is subtle enough
that it should exist once rather than once per script.

The subtle part is the seeding. `seed_training(config["seed"])` is called
twice: once before the adapter object is constructed, and again immediately
before `attach_lora`, which randomises LoRA A. Skip the second and the adapter
starts from different noise; the checkpoint then loads on top of a different
initialisation, and every number the pass produces is about a model the run
never had. `scripts/v4_train.py:503` and
`src/selfsight/v4/checkpoint_probe.py:373` both do it in that order, and this
is the third place that has to.

Import-time cost is why the heavy imports are inside the function: the
callers' `--dry-run` paths resolve their whole plan without loading torch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from selfsight.utils.hashing import sha256_json

ROOT = Path(__file__).resolve().parents[3]

# scripts/v4_train.py's LORA_TARGETS, as a path rather than a relative string
# so a caller's working directory cannot change which adapter shape is built.
LORA_TARGETS = ROOT / "runs/readiness/showo2-1p5b/a4-lora-targets-r1.json"


def load_trained_backbone(config: dict[str, Any], checkpoint: Path, *, device: str,
                          targets_path: Path | None = None) -> tuple[Any, str]:
    """(backbone, adapter digest) for one saved checkpoint.

    `checkpoint` is a directory `save_checkpoint` wrote. `load_checkpoint`
    checks it against `sha256_json(config)`, so passing a config that is not
    the run's own fails here rather than producing plausible numbers from the
    wrong training run.
    """

    from selfsight.backbones.showo2 import Showo2Adapter
    from selfsight.training.checkpoint import load_checkpoint
    from selfsight.v4.train import parameter_digest, seed_training, trainable_snapshot

    targets = json.loads((targets_path or LORA_TARGETS).read_text(encoding="utf-8"))
    if targets.get("forbidden_modules_selected"):
        raise ValueError("LoRA target audit includes forbidden modules")
    lora = config["training"]["lora"]

    seed_training(int(config["seed"]))
    backbone = Showo2Adapter(device=device, lazy=False)
    # Immediately before attach_lora, and not one statement earlier: this is
    # the draw that becomes LoRA A.
    seed_training(int(config["seed"]))
    backbone.attach_lora(target_modules=targets["target_modules"], rank=int(lora["rank"]),
                         alpha=int(lora["alpha"]), dropout=float(lora["dropout"]),
                         gradient_checkpointing=False)
    load_checkpoint(checkpoint, model=backbone.model, optimizer=None, scheduler=None,
                    expected_config_digest=sha256_json(config))
    return backbone, parameter_digest(trainable_snapshot(backbone.model))


def checkpoint_for(run: Path, arm: str, step: int, steps_per_round: int) -> Path:
    """Step 0 is the untrained base; step (i+1)*k is arm `arm`'s round i.

    Both arms share `checkpoints/base/round--01` at step 0, which is not a
    mistake to be tidied away: the two arms are the same model until the first
    optimizer step, and a pass still measures each of them there because the
    *images* differ from step 0 onwards.

    Off by one in either direction produces a complete, plausible curve: every
    checkpoint scored against the adapter from one round earlier, or step 0
    scored against a trained one.
    """

    if step == 0:
        return Path(run) / "checkpoints" / "base" / "round--01"
    if step % steps_per_round:
        raise ValueError(f"step {step} is not a multiple of {steps_per_round}")
    return Path(run) / "checkpoints" / arm / f"round-{step // steps_per_round - 1:03d}"
