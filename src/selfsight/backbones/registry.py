"""Which model answers, decided by a config file instead of by an import.

`stage_observe` used to name `Showo2Adapter` directly. E4 asks the same
questions of three model families that do not share an inference path, a
tokeniser, or even a python environment, so the choice has to become data.

The discriminator is a `family` key in the backbone config, defaulting to
`showo2`. The default is not laziness: `configs/backbones/showo2_1p5b.yaml` and
`showo2_7b.yaml` are referenced by rows already written to disk and by the
frozen 2026-09-08 packets, and adding a key to them would mean touching files
whose stems appear in answer filenames. New families declare themselves; the
old ones keep their shape.

Nothing here loads a model at import time. The three adapters live in three
environments (`envs/showo2`, `envs/core`, `envs/janus`), and only one of them
is importable in any given process -- which is why `environment` is recorded in
the config too, so a driver can spawn the right interpreter without a
family-to-path table hidden in a script.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

DEFAULT_FAMILY = "showo2"
FAMILIES = ("showo2", "showo_v1", "janus_pro")


def read_backbone_config(path: str | Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"Backbone config is not a mapping: {path}")
    family = raw.get("family", DEFAULT_FAMILY)
    if family not in FAMILIES:
        raise ValueError(f"Unknown backbone family {family!r} in {path}; expected one of {FAMILIES}")
    return raw


def backbone_family(config: Mapping[str, Any]) -> str:
    return str(config.get("family", DEFAULT_FAMILY))


def backbone_environment(config: Mapping[str, Any]) -> str | None:
    """The interpreter that can import this family's adapter, if declared."""

    value = config.get("environment")
    return None if value is None else str(value)


def build_observer(
    backbone_config: str | Path,
    *,
    device: str,
    lock_path: str | Path = "configs/models.lock.yaml",
) -> Any:
    """Return something with `observe_atoms(image_path, questions)`.

    Only the answering surface is promised. Show-o2 comes back as the full
    trainable adapter because that is the class that exists; the other two
    cannot train and are not asked to.
    """

    config = read_backbone_config(backbone_config)
    family = backbone_family(config)

    if family == "showo2":
        from selfsight.backbones.showo2 import Showo2Adapter

        observer: Any = Showo2Adapter(
            device=device, lazy=False, backbone_config=str(backbone_config)
        )
    elif family == "showo_v1":
        from selfsight.showo_adapter import ShowoAdapter

        # load_vision_tower is not a default: `_observe_one` raises without it,
        # and the failure would land after the weights are already resident.
        observer = ShowoAdapter(device=device, lock_path=lock_path, load_vision_tower=True)
    elif family == "janus_pro":
        from selfsight.backbones.janus_pro import JanusProAdapter

        observer = JanusProAdapter(device=device, lock_path=lock_path, lazy=False)
    else:  # pragma: no cover - read_backbone_config already rejected it
        raise ValueError(f"No observer for family {family!r}")

    check_identity(observer, config, source=backbone_config)
    return observer


def check_identity(observer: Any, config: Mapping[str, Any], *, source: str | Path) -> None:
    """Refuse a config that names one model and loads another.

    Two of the three families resolve their weights from the model lock rather
    than from this config, so the config's `backbone_id` is a claim, not a
    cause. A cross-model replication that quietly ran the same model twice would
    report a clean "no difference between architectures" and look like a
    finding, so the claim gets checked against the object that will answer.
    """

    for key, attribute in (("backbone_id", "model_id"), ("revision", "revision")):
        declared = config.get(key)
        actual = getattr(observer, attribute, None)
        if declared is None or actual is None:
            continue
        if str(declared) != str(actual):
            raise RuntimeError(
                f"{source} declares {key}={declared!r} but the loaded model "
                f"reports {actual!r}"
            )
