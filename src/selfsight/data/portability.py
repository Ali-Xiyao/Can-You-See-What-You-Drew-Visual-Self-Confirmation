"""Audited manifest path rebasing for Windows-to-Linux dataset migration."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from selfsight.utils.hashing import rgb_sha256, sha256_file
from selfsight.utils.jsonl import atomic_write_json, atomic_write_jsonl, read_jsonl

SCENE_MANIFESTS = ("train", "tier_a_probe", "tier_a_outcome", "tier_d")


def _image_for_scene(data_root: Path, scene: dict[str, Any]) -> Path:
    split = str(scene["split"])
    return (data_root / "reference_images" / split / f"{scene['scene_id']}.png").resolve()


def _validate_scene_image(record: dict[str, Any], path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Rebased reference image is missing: {path}")
    if sha256_file(path) != str(record["reference_file_sha256"]):
        raise RuntimeError(f"Reference file SHA-256 changed during migration: {path}")
    if rgb_sha256(path) != str(record["reference_rgb_sha256"]):
        raise RuntimeError(f"Reference RGB SHA-256 changed during migration: {path}")


def rebase_scene_records(
    records: list[dict[str, Any]], data_root: str | Path
) -> list[dict[str, Any]]:
    """Rebase standard/Tier-D scene records and verify their immutable RGB evidence."""

    root = Path(data_root).resolve()
    output = []
    for source in records:
        record = deepcopy(source)
        path = _image_for_scene(root, record["scene"])
        _validate_scene_image(record, path)
        record["reference_image"] = str(path)
        output.append(record)
    return output


def rebase_tier_b_records(
    records: list[dict[str, Any]], data_root: str | Path
) -> list[dict[str, Any]]:
    """Rebase both members of every Tier-B pair and verify their RGB hashes."""

    root = Path(data_root).resolve()
    output = []
    for source in records:
        record = deepcopy(source)
        pair = record["pair"]
        source_path = _image_for_scene(root, pair["source"])
        counterfactual_path = _image_for_scene(root, pair["counterfactual"])
        if not source_path.is_file() or not counterfactual_path.is_file():
            raise FileNotFoundError(f"Rebased Tier-B pair is incomplete: {pair['pair_id']}")
        if rgb_sha256(source_path) != str(record["source_rgb_sha256"]):
            raise RuntimeError(f"Tier-B source RGB changed during migration: {pair['pair_id']}")
        if rgb_sha256(counterfactual_path) != str(record["counterfactual_rgb_sha256"]):
            raise RuntimeError(f"Tier-B counterfactual RGB changed during migration: {pair['pair_id']}")
        record["source_image"] = str(source_path)
        record["counterfactual_image"] = str(counterfactual_path)
        output.append(record)
    return output


def rebase_dataset_manifests(
    data_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write a new platform-local manifest view without modifying source manifests."""

    root = Path(data_root).resolve()
    source_dir = root / "manifests"
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty rebased manifest directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest_paths: dict[str, str] = {}
    manifest_hashes: dict[str, dict[str, str]] = {}
    row_counts: dict[str, int] = {}
    for name in SCENE_MANIFESTS:
        source_path = source_dir / f"{name}.jsonl"
        records = rebase_scene_records(list(read_jsonl(source_path)), root)
        destination = atomic_write_jsonl(output / source_path.name, records)
        manifest_paths[name] = str(destination.resolve())
        manifest_hashes[name] = {
            "source": sha256_file(source_path),
            "rebased": sha256_file(destination),
        }
        row_counts[name] = len(records)
    tier_b_source = source_dir / "tier_b.jsonl"
    tier_b_records = rebase_tier_b_records(list(read_jsonl(tier_b_source)), root)
    tier_b_destination = atomic_write_jsonl(output / tier_b_source.name, tier_b_records)
    manifest_paths["tier_b"] = str(tier_b_destination.resolve())
    manifest_hashes["tier_b"] = {
        "source": sha256_file(tier_b_source),
        "rebased": sha256_file(tier_b_destination),
    }
    row_counts["tier_b"] = len(tier_b_records)

    registry = json.loads((source_dir / "registry.json").read_text(encoding="utf-8"))
    registry["manifests"] = manifest_paths
    registry["path_rebase"] = {
        "source_manifest_dir": str(source_dir.resolve()),
        "data_root": str(root),
        "manifest_hashes": manifest_hashes,
    }
    atomic_write_json(output / "registry.json", registry)
    selection_path = source_dir / "tier_d_selection.json"
    if selection_path.is_file():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selection["manifest"] = manifest_paths["tier_d"]
        atomic_write_json(output / "tier_d_selection.json", selection)
    report = {
        "schema_version": 1,
        "status": "paths_rebased_rgb_verified",
        "source_manifest_dir": str(source_dir.resolve()),
        "output_manifest_dir": str(output),
        "data_root": str(root),
        "row_counts": row_counts,
        "manifest_hashes": manifest_hashes,
        "manifests": manifest_paths,
    }
    atomic_write_json(output / "rebase_report.json", report)
    return report
