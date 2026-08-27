"""Clone or verify exact code revisions required by the locked experiment."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import yaml

URLS = {
    "showlab/Show-o": "https://github.com/showlab/Show-o.git",
    "deepseek-ai/Janus": "https://github.com/deepseek-ai/Janus.git",
}
SPARSE_PATHS = {
    "showlab/Show-o": ("configs", "models", "training", "llava", "parquet"),
    "deepseek-ai/Janus": ("janus",),
}


def _run(command: list[str], cwd: Path | None = None) -> str:
    process = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    if process.returncode:
        raise RuntimeError(f"Command failed ({process.returncode}): {command}\n{process.stderr}")
    return process.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=Path("configs/models.lock.yaml"))
    parser.add_argument("--repo", action="append", choices=tuple(URLS))
    args = parser.parse_args()
    root_value = os.environ.get("SELFSIGHT_MODEL_ROOT")
    if not root_value:
        raise SystemExit("Run scripts/set_h_env.ps1 first")
    root = Path(root_value).resolve() / "repositories"
    root.mkdir(parents=True, exist_ok=True)
    lock = yaml.safe_load(args.lock.read_text(encoding="utf-8"))
    selected = set(args.repo or URLS)
    for record in lock["repositories"]:
        if record["id"] not in selected:
            continue
        destination = root / record["id"].split("/")[-1]
        if not destination.exists():
            destination.mkdir(parents=True)
            print(f"INIT {record['id']} {destination}", flush=True)
            _run(["git", "init"], destination)
            _run(["git", "remote", "add", "origin", URLS[record["id"]]], destination)
            _run(["git", "sparse-checkout", "init", "--cone"], destination)
            _run(["git", "sparse-checkout", "set", *SPARSE_PATHS[record["id"]]], destination)
        origin = _run(["git", "remote", "get-url", "origin"], destination)
        if origin.rstrip("/").removesuffix(".git").lower() != URLS[record["id"]].rstrip("/").removesuffix(".git").lower():
            raise RuntimeError(f"Unexpected origin for {destination}: {origin}")
        print(f"FETCH {record['id']} {record['revision']}", flush=True)
        _run(
            ["git", "fetch", "--filter=blob:none", "--depth", "1", "origin", record["revision"]],
            destination,
        )
        _run(["git", "checkout", "--detach", "FETCH_HEAD"], destination)
        head = _run(["git", "rev-parse", "HEAD"], destination)
        if head != record["revision"]:
            raise RuntimeError(f"Repository revision mismatch: {record['id']} {head}")
        print(f"READY {record['id']} {head} {destination}", flush=True)


if __name__ == "__main__":
    main()
