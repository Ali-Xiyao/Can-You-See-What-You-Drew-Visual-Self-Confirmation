"""Download only immutable model revisions from configs/models.lock.yaml."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import quote

import yaml

from selfsight.utils.jsonl import atomic_write_json

DEFAULT_IGNORE_PATTERNS = (
    "onnx/*",
    "*.onnx",
    "*.onnx_data",
    "openvino/*",
    "coreml/*",
    "*.gguf",
    "*.ggml",
    "*.tflite",
    "tf_model.h5",
    "flax_model.msgpack",
    "rust_model.ot",
)


def _safe_name(repo_id: str) -> str:
    return repo_id.replace("/", "--")


def _load_lock(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("models"), list):
        raise TypeError("Invalid model lock file")
    for model in value["models"]:
        revision = str(model.get("revision", ""))
        if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
            raise ValueError(f"Model is not pinned to a full commit SHA: {model.get('id')}")
    return value


def _inventory(model_info: Any) -> list[dict[str, Any]]:
    output = []
    for sibling in model_info.siblings or ():
        lfs = getattr(sibling, "lfs", None)
        output.append(
            {
                "path": sibling.rfilename,
                "size": getattr(sibling, "size", None),
                "blob_id": getattr(sibling, "blob_id", None),
                "lfs_sha256": getattr(lfs, "sha256", None) if lfs else None,
            }
        )
    return output


def _filter_inventory(
    inventory: list[dict[str, Any]],
    *,
    allow_patterns: list[str] | None,
    ignore_patterns: list[str],
) -> list[dict[str, Any]]:
    from fnmatch import fnmatch

    output = inventory
    if allow_patterns:
        output = [
            item
            for item in output
            if any(fnmatch(str(item["path"]), pattern) for pattern in allow_patterns)
        ]
    return [
        item
        for item in output
        if not any(fnmatch(str(item["path"]), pattern) for pattern in ignore_patterns)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=Path("configs/models.lock.yaml"))
    parser.add_argument("--group", action="append", choices=("core", "observers", "audit", "late_eval"))
    parser.add_argument("--model-id", action="append")
    parser.add_argument("--plan", action="store_true", help="Resolve sizes and revisions without downloading")
    parser.add_argument("--force-low-space", action="store_true")
    parser.add_argument(
        "--large-file-transport",
        choices=("auto", "hub", "aria2"),
        default="auto",
        help="Use resumable segmented aria2 downloads for large LFS files when available.",
    )
    args = parser.parse_args()

    from huggingface_hub import HfApi, snapshot_download

    lock = _load_lock(args.lock.resolve())
    groups = set(args.group or ())
    ids = set(args.model_id or ())
    selected = [
        model
        for model in lock["models"]
        if (not groups and not ids) or model["group"] in groups or model["id"] in ids
    ]
    if not selected:
        raise SystemExit("No models matched the requested group/model ID")
    model_root_value = os.environ.get("SELFSIGHT_MODEL_ROOT")
    if not model_root_value:
        raise SystemExit("Run scripts/set_h_env.ps1 first; SELFSIGHT_MODEL_ROOT is not set")
    model_root = Path(model_root_value).resolve()
    model_root.mkdir(parents=True, exist_ok=True)
    if os.name == "nt" and model_root.drive.upper() != "H:":
        raise SystemExit(f"Refusing to download outside H:: {model_root}")

    api = HfApi()
    records = []
    total_bytes = 0
    for model in selected:
        info = api.model_info(model["id"], revision=model["revision"], files_metadata=True)
        if info.sha != model["revision"]:
            raise RuntimeError(f"Revision mismatch for {model['id']}: {info.sha}")
        allow_patterns = model.get("allow_patterns")
        ignore_patterns = [*DEFAULT_IGNORE_PATTERNS, *model.get("ignore_patterns", ())]
        inventory = _filter_inventory(
            _inventory(info),
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )
        size = sum(int(item["size"] or 0) for item in inventory)
        total_bytes += size
        records.append(
            {
                **model,
                "resolved_revision": info.sha,
                "expected_bytes": size,
                "effective_ignore_patterns": ignore_patterns,
                "inventory": inventory,
                "snapshot_path": None,
            }
        )
    free = shutil.disk_usage(model_root.anchor).free
    required = int(total_bytes * 1.15)
    print(json.dumps({"models": len(records), "expected_bytes": total_bytes, "free_bytes": free}, indent=2))
    if args.plan:
        return
    if free < required and not args.force_low_space:
        raise SystemExit(f"Need at least {required} free bytes for this batch; only {free} are free")

    aria2 = shutil.which("aria2c")
    use_aria2 = args.large_file_transport == "aria2" or (
        args.large_file_transport == "auto" and os.name == "nt" and aria2 is not None
    )
    if args.large_file_transport == "aria2" and aria2 is None:
        raise SystemExit("--large-file-transport aria2 was requested but aria2c is unavailable")

    def file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    for record in records:
        direct_snapshot = (
            model_root
            / f"models--{record['id'].replace('/', '--')}"
            / "snapshots"
            / record["revision"]
        )
        direct_snapshot.mkdir(parents=True, exist_ok=True)
        large_paths = []
        if use_aria2:
            for item in record["inventory"]:
                if int(item["size"] or 0) < 64 * 1024 * 1024 or not item.get("lfs_sha256"):
                    continue
                relative = str(item["path"])
                destination = direct_snapshot / relative
                expected_size = int(item["size"])
                expected_sha = str(item["lfs_sha256"])
                ready = (
                    destination.is_file()
                    and destination.stat().st_size == expected_size
                    and file_sha256(destination) == expected_sha
                )
                if not ready:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    url = (
                        f"https://huggingface.co/{record['id']}/resolve/{record['revision']}/"
                        f"{quote(relative)}?download=true"
                    )
                    subprocess.run(
                        [
                            str(aria2),
                            "--continue=true",
                            "--max-connection-per-server=16",
                            "--split=16",
                            "--min-split-size=1M",
                            "--file-allocation=none",
                            "--auto-file-renaming=false",
                            "--allow-overwrite=true",
                            "--console-log-level=warn",
                            "--summary-interval=30",
                            f"--dir={destination.parent}",
                            f"--out={destination.name}",
                            url,
                        ],
                        check=True,
                    )
                if destination.stat().st_size != expected_size:
                    raise RuntimeError(f"Size mismatch for {record['id']}/{relative}")
                if file_sha256(destination) != expected_sha:
                    raise RuntimeError(f"LFS SHA256 mismatch for {record['id']}/{relative}")
                large_paths.append(relative)
        small_patterns = [
            item["path"] for item in record["inventory"] if item["path"] not in large_paths
        ]
        snapshot_kwargs: dict[str, Any] = {
            "repo_id": record["id"],
            "revision": record["revision"],
            "token": os.environ.get("HF_TOKEN"),
            "allow_patterns": small_patterns if use_aria2 else record.get("allow_patterns"),
            "ignore_patterns": None if use_aria2 else record["effective_ignore_patterns"],
        }
        if os.name == "nt":
            # Hub cache snapshots are symlink farms. Standard Windows accounts often cannot
            # create symlinks, so materialize ordinary files directly in the locked snapshot.
            snapshot_kwargs.update(
                {
                    "local_dir": direct_snapshot,
                    "local_dir_use_symlinks": False,
                }
            )
        else:
            snapshot_kwargs["cache_dir"] = model_root
        snapshot_path = snapshot_download(
            **snapshot_kwargs,
        )
        if Path(snapshot_path).name != record["revision"]:
            raise RuntimeError(f"Downloaded snapshot path is not the locked revision: {snapshot_path}")
        record["snapshot_path"] = str(Path(snapshot_path).resolve())
        atomic_write_json(
            model_root / "registries" / f"{_safe_name(record['id'])}@{record['revision'][:12]}.json",
            {"schema_version": 1, "model": record},
        )
        print(f"READY {record['id']} {record['revision']} {snapshot_path}")


if __name__ == "__main__":
    main()
