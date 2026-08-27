"""Evidence stamps and host manifests for auditable experiments."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from selfsight.utils.hashing import sha256_file
from selfsight.utils.jsonl import atomic_write_json


def _run(command: list[str]) -> dict[str, Any]:
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": type(exc).__name__, "detail": str(exc)}
    return {
        "ok": process.returncode == 0,
        "returncode": process.returncode,
        "stdout": process.stdout.strip(),
        "stderr": process.stderr.strip(),
    }


def capture_host_manifest() -> dict[str, Any]:
    roots = {}
    for name in (
        "SELFSIGHT_CACHE_ROOT",
        "SELFSIGHT_DATA_ROOT",
        "SELFSIGHT_RUN_ROOT",
        "SELFSIGHT_MODEL_ROOT",
        "SELFSIGHT_ENV_ROOT",
        "SELFSIGHT_TMP_ROOT",
    ):
        value = os.environ.get(name)
        if value:
            usage = shutil.disk_usage(Path(value).anchor or value)
            roots[name] = {
                "path": value,
                "drive_free_bytes": usage.free,
                "drive_total_bytes": usage.total,
            }
    return {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": {"version": sys.version, "executable": sys.executable},
        "paths": roots,
        "nvidia_smi": _run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,driver_version,pstate",
                "--format=csv,noheader,nounits",
            ]
        ),
        "git": _run(["git", "rev-parse", "HEAD"]),
        "pip_freeze": _run([sys.executable, "-m", "pip", "freeze"]),
    }


def write_host_manifest(destination: str | Path) -> Path:
    return atomic_write_json(destination, capture_host_manifest())


def create_evidence_stamp(
    *,
    command: list[str],
    config_path: str | Path,
    config_digest: str,
    artifacts: list[str | Path],
    gate: str | None = None,
    decision: str | None = None,
) -> dict[str, Any]:
    artifact_records = []
    for item in artifacts:
        path = Path(item).resolve()
        artifact_records.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        )
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "config_path": str(Path(config_path).resolve()),
        "config_digest": config_digest,
        "gate": gate,
        "decision": decision,
        "artifacts": artifact_records,
    }
