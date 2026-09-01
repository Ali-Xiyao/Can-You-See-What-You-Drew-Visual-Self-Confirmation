"""Re-score an existing candidate bank under the current gold atoms. No GPU.

A candidate image depends only on the prompt. The v3 family redesign changed how
a candidate is *judged*, not what is generated, so the calibration sweep does not
need to be regenerated -- the banks on disk are still the right banks.

This is deliberately a separate script writing to a separate output root, because
`run_bank_probe` resumes from `packets/{index}-{scene_id}.json` and the redesign
does not change `scene_id`. Pointing the probe at the old directory would
silently reuse the old gold scores and report them as new evidence. Provenance
for every rescored setting is written to `rescore_provenance.json`.

    envs/core/python.exe scripts/rescore_v3_calibration.py \
        --source runs/v3/calib --records-root data/selfsight-v3-cal \
        --output runs/v3/calib-v2 --setting 3 --setting 4 --setting 5 --setting 6

It doubles as the shard merger. A Gate A run split across two cards leaves one
bank per card; pointing `--records` at the split's manifest folds every shard's
packets into a single report through the same scoring path as the calibration,
so the two are directly comparable:

    envs/core/python.exe scripts/rescore_v3_calibration.py \
        --source runs/v3/gate-a-v2/spatial \
        --records data/selfsight-v3/spatial/manifests/tier_a_probe.jsonl \
        --output runs/v3/gate-a-v2/spatial
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from selfsight.data.questions import build_gold_atoms
from selfsight.schemas import Atom, SceneSpec
from selfsight.utils.jsonl import atomic_write_json, atomic_write_jsonl, read_jsonl
from selfsight.v3.bank import BankCandidate, bank_supply_report, build_balanced_pool
from selfsight.v3.supply import _gold_score


def gold_atoms_for(record: dict, scene: SceneSpec) -> tuple[tuple[Atom, ...], str]:
    """Prefer the manifest's atoms; derive them when the manifest predates them."""

    serialized = record.get("gold_atoms")
    if serialized:
        return tuple(Atom.from_dict(dict(item)) for item in serialized), "manifest"
    return build_gold_atoms(scene), "derived_from_scene"


def rescore_setting(
    *,
    source: Path,
    records_path: Path,
    output: Path,
    bank_cfg: dict,
) -> dict:
    records = {}
    for record in read_jsonl(records_path):
        scene = SceneSpec.from_dict(dict(record["scene"]))
        records[scene.scene_id] = (record, scene)

    packet_dirs = sorted(source.glob("*/packets"))
    if not packet_dirs:
        raise SystemExit(f"no packets under {source}")

    candidate_k = int(bank_cfg["candidate_k"])
    min_per_side = int(bank_cfg["min_per_side"])
    pools, natural_rows = [], []
    natural_informative = 0
    atom_sources: set[str] = set()
    missing_images = 0
    seen: set[str] = set()

    for packet_dir in packet_dirs:
        for packet_path in sorted(packet_dir.glob("*.json")):
            packet = json.loads(packet_path.read_text(encoding="utf-8"))
            scene_id = str(packet["scene_id"])
            if scene_id in seen:
                continue
            entry = records.get(scene_id)
            if entry is None:
                continue
            seen.add(scene_id)
            record, scene = entry
            atoms, atom_source = gold_atoms_for(record, scene)
            atom_sources.add(atom_source)
            family = str(packet.get("family", scene.family.value))

            scored = []
            for item in packet["candidates"]:
                image_path = Path(item["image_path"])
                if not image_path.is_file():
                    missing_images += 1
                    continue
                score, abstained = _gold_score(str(image_path), atoms)
                scored.append(
                    BankCandidate(
                        candidate_id=str(item["candidate_id"]),
                        sampling_seed=int(item["sampling_seed"]),
                        gold_score=score,
                        abstained=abstained,
                    )
                )
            if not scored:
                continue

            natural = build_balanced_pool(
                scene_id, family, scored[:candidate_k], k=candidate_k, min_per_side=min_per_side
            )
            natural_informative += int(natural.informative)
            natural_rows.append(natural.to_dict())
            pools.append(
                build_balanced_pool(
                    scene_id, family, scored, k=candidate_k, min_per_side=min_per_side
                )
            )

    report = bank_supply_report(
        pools,
        families=sorted({pool.family for pool in pools}),
        min_balanced_rate=float(bank_cfg["min_balanced_rate"]),
        min_informative_pools=int(bank_cfg["min_informative_pools"]),
        natural_informative_rate=natural_informative / len(pools) if pools else None,
    )
    report.update(
        {
            "schema_version": 1,
            "benchmark_version": "3.0",
            "stage": "v3_gate_a_candidate_supply_rescored",
            "candidate_k": candidate_k,
            "min_per_side": min_per_side,
            "natural_informative_pools": natural_informative,
            "natural_pools": len(natural_rows),
        }
    )

    output.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(output / "pools.jsonl", (pool.to_dict() for pool in pools))
    atomic_write_jsonl(output / "natural_pools.jsonl", natural_rows)
    atomic_write_json(output / "gate_a.json", report)
    provenance = {
        "rescored_at": datetime.now(timezone.utc).isoformat(),
        "source_packets": [str(item.resolve()) for item in packet_dirs],
        "records": str(records_path.resolve()),
        "gold_atom_source": sorted(atom_sources),
        "prompts": len(pools),
        "missing_images": missing_images,
        "note": (
            "Images were generated by the earlier probe and are reused unchanged; "
            "only the gold atoms differ. Regenerating would produce the same "
            "images because the prompts are identical."
        ),
    }
    atomic_write_json(output / "rescore_provenance.json", provenance)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="runs/v3/calib")
    parser.add_argument("--records-root", help="data/selfsight-v3-cal (sweep mode)")
    parser.add_argument("--records", help="a single manifest (shard-merge mode)")
    parser.add_argument("--output", required=True, help="runs/v3/calib-v2")
    parser.add_argument("--setting", action="append", type=int)
    parser.add_argument("--config", default="configs/v3_main_seed.yaml")
    args = parser.parse_args()

    if bool(args.records_root) == bool(args.records):
        raise SystemExit("pass exactly one of --records-root (sweep) or --records (merge)")

    bank_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["candidate_bank"]
    source_root = Path(args.source)
    output_root = Path(args.output)

    if args.records:
        report = rescore_setting(
            source=source_root,
            records_path=Path(args.records),
            output=output_root,
            bank_cfg=bank_cfg,
        )
        print(
            f"{source_root.name}: prompts={report['prompts_searched']} "
            f"balanced_rate={report['balanced_rate']:.3f} "
            f"informative={report['informative_pools']} "
            f"natural_informative_rate={report['natural_informative_rate']:.3f}"
        )
        print(f"\n-> {output_root}")
        return 0

    if not args.setting:
        raise SystemExit("--records-root needs at least one --setting")
    records_root = Path(args.records_root)

    for setting in sorted(set(args.setting)):
        source = source_root / f"obj{setting}"
        records_path = records_root / f"obj{setting}" / "manifests" / "calibration.jsonl"
        if not source.is_dir() or not records_path.is_file():
            print(f"obj{setting}: no bank on disk, skipping")
            continue
        report = rescore_setting(
            source=source,
            records_path=records_path,
            output=output_root / f"obj{setting}" / "rescored",
            bank_cfg=bank_cfg,
        )
        print(
            f"obj{setting}: prompts={report['prompts_searched']} "
            f"balanced_rate={report['balanced_rate']:.3f} "
            f"informative={report['informative_pools']} "
            f"natural_informative_rate={report['natural_informative_rate']:.3f}"
        )
    print(f"\n-> {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
