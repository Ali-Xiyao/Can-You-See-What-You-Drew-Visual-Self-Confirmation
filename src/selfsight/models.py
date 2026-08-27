"""Resolve immutable local snapshots and repository checkouts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_model_lock(path: str | Path = "configs/models.lock.yaml") -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Invalid model lock")
    return value


def locked_model(model_id: str, lock_path: str | Path = "configs/models.lock.yaml") -> dict[str, Any]:
    lock = load_model_lock(lock_path)
    matches = [record for record in lock["models"] if record["id"] == model_id]
    if len(matches) != 1:
        raise KeyError(f"Expected exactly one lock for {model_id}, found {len(matches)}")
    return matches[0]


def snapshot_path(
    model_id: str,
    *,
    lock_path: str | Path = "configs/models.lock.yaml",
    model_root: str | Path | None = None,
    require_complete: bool = True,
) -> Path:
    record = locked_model(model_id, lock_path)
    root = Path(model_root or os.environ["SELFSIGHT_MODEL_ROOT"]).resolve()
    path = root / f"models--{model_id.replace('/', '--')}" / "snapshots" / record["revision"]
    if not path.is_dir():
        raise FileNotFoundError(f"Locked snapshot is not downloaded: {model_id}@{record['revision']} ({path})")
    if require_complete:
        required = ["config.json"]
        missing = [name for name in required if not (path / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Incomplete snapshot for {model_id}: missing {missing}")
    return path


def repository_path(
    repository_id: str,
    *,
    lock_path: str | Path = "configs/models.lock.yaml",
    model_root: str | Path | None = None,
) -> Path:
    import subprocess

    lock = load_model_lock(lock_path)
    record = next(item for item in lock["repositories"] if item["id"] == repository_id)
    root = Path(model_root or os.environ["SELFSIGHT_MODEL_ROOT"]).resolve()
    path = root / "repositories" / repository_id.split("/")[-1]
    if not (path / ".git").is_dir():
        raise FileNotFoundError(f"Locked repository is not synced: {path}")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=True
    ).stdout.strip()
    if head != record["revision"]:
        raise RuntimeError(f"Repository revision mismatch for {repository_id}: {head}")
    return path
