"""Configuration loading, validation, and immutable run snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in os.environ:
                raise KeyError(f"Required environment variable is not set: {name}")
            return os.environ[name]

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExperimentConfig:
    """A resolved config with a stable content digest."""

    source: Path
    values: dict[str, Any]
    digest: str

    def section(self, name: str) -> dict[str, Any]:
        value = self.values.get(name)
        if not isinstance(value, dict):
            raise KeyError(f"Missing mapping section: {name}")
        return value


def validate_config(values: Mapping[str, Any]) -> None:
    required = {"profile", "seed", "paths", "hardware", "model", "data", "training"}
    missing = sorted(required.difference(values))
    if missing:
        raise ValueError(f"Config is missing required keys: {missing}")
    paths = values["paths"]
    if not isinstance(paths, Mapping):
        raise TypeError("paths must be a mapping")
    for key, path_value in paths.items():
        path = Path(str(path_value))
        if os.name == "nt" and path.drive.upper() != "H:":
            raise ValueError(f"Large path {key} must be on H:, got {path}")
    training = values["training"]
    if int(training["gradient_accumulation_steps"]) * int(training["micro_batch_size"]) != 8:
        raise ValueError("Effective batch size must remain 8")
    if int(training["candidate_k"]) < 2:
        raise ValueError("candidate_k must be at least 2 for paired selection")


def load_config(path: str | Path) -> ExperimentConfig:
    source = Path(path).resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"Expected a YAML mapping in {source}")
    values = _expand_environment(raw)
    validate_config(values)
    return ExperimentConfig(source=source, values=values, digest=config_digest(values))


def write_config_snapshot(config: ExperimentConfig, destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": str(config.source),
        "digest": config.digest,
        "values": config.values,
    }
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    destination.write_text(text, encoding="utf-8")
    return destination
