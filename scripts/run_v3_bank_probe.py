"""Gate A: measure natural vs bounded-bank candidate supply on the local 3090s.

This is the first v3.0 experiment and the only one that must complete before any
other. Random K=4 sampling was informative for 9 of 42 prompts on this backbone
(21%); if a bounded bank cannot lift that rate, selection-based self-training has
no exploitable variance and the mainline stops.

Uses only already-materialized weights and manifests. Never downloads anything.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml

from selfsight.backbones.showo2 import Showo2Adapter
from selfsight.utils.evidence import write_host_manifest
from selfsight.utils.jsonl import read_jsonl
from selfsight.v3.supply import run_bank_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v3_main_seed.yaml")
    parser.add_argument("--backbone-config", default=None)
    parser.add_argument("--records", required=True, help="JSONL with scene+atom rows")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None, help="Override generator device, e.g. cuda:0")
    parser.add_argument("--limit", type=int, default=None, help="First N records only")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N records")
    parser.add_argument("--family", action="append", default=None, help="Restrict to family")
    parser.add_argument("--bank-size", type=int, default=None)
    parser.add_argument(
        "--min-informative-pools",
        type=int,
        default=None,
        help=(
            "Absolute supply floor. Defaults to the registered value; pass "
            "ceil(rate * n_prompts) when this run is only measuring the rate."
        ),
    )
    parser.add_argument(
        "--rate-only",
        action="store_true",
        help=(
            "Record the run as a rate measurement: the absolute supply floor is "
            "scaled to the prompt count and the deferral is written into the report."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    bank_cfg = config["candidate_bank"]

    records = list(read_jsonl(Path(args.records).resolve()))
    if args.family:
        wanted = {value.lower() for value in args.family}
        records = [row for row in records if str(row["scene"]["family"]).lower() in wanted]
    records = records[args.offset :]
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise SystemExit("No records selected")

    bank_size = int(args.bank_size or bank_cfg["bank_size"])
    min_rate = float(bank_cfg["min_balanced_rate"])
    if args.min_informative_pools is not None:
        min_pools = int(args.min_informative_pools)
    elif args.rate_only:
        min_pools = math.ceil(min_rate * len(records))
    else:
        min_pools = int(bank_cfg["min_informative_pools"])

    backbone_config = Path(
        args.backbone_config or config["model"]["backbone_config"]
    ).resolve()
    device = args.device or config["hardware"]["generator_device"]

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_host_manifest(output / "host_manifest.json")

    adapter = Showo2Adapter(backbone_config=backbone_config, device=device, lazy=False)
    if adapter.model_id != config["model"]["trainable_id"]:
        raise SystemExit(
            f"Backbone mismatch: adapter has {adapter.model_id}, "
            f"config expects {config['model']['trainable_id']}"
        )
    if adapter.revision != config["model"]["revision"]:
        raise SystemExit(
            f"Revision mismatch: adapter has {adapter.revision}, "
            f"config expects {config['model']['revision']}"
        )

    print(
        f"[gate-a] {len(records)} prompts x M={bank_size} = "
        f"{len(records) * bank_size} generations on {device}",
        flush=True,
    )

    report = run_bank_probe(
        records=records,
        adapter=adapter,
        output_dir=output,
        bank_size=bank_size,
        candidate_k=int(bank_cfg["candidate_k"]),
        min_per_side=int(bank_cfg["min_per_side"]),
        min_balanced_rate=min_rate,
        min_informative_pools=min_pools,
        reserved_seeds=(int(config["training"]["gradient_seed_base"]),),
    )
    report["model_id"] = adapter.model_id
    report["revision"] = adapter.revision
    report["device"] = device
    report["records_source"] = str(Path(args.records).resolve())
    report["model_downloads_used"] = False
    report["a800_used"] = False
    if args.rate_only:
        report["rate_measurement_only"] = True
        report["absolute_supply_requirement_deferred"] = int(
            bank_cfg["min_informative_pools"]
        )
        report["deferral_note"] = (
            "This run measures the balanced-pool rate. The registered absolute floor "
            f"of {bank_cfg['min_informative_pools']} informative pools requires a larger "
            "held-out prompt set and is evaluated separately."
        )
    (output / "gate_a.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(
        f"[gate-a] balanced_rate={report['balanced_rate']:.3f} "
        f"(threshold {min_rate}) | informative={report['informative_pools']} "
        f"| natural_rate={report.get('natural_informative_rate')} "
        f"| passed={report['passed']}",
        flush=True,
    )
    for family, stats in sorted(report["by_family"].items()):
        print(
            f"[gate-a]   {family:<10} balanced={stats['balanced']}/{stats['searched']} "
            f"rate={stats['rate']}",
            flush=True,
        )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
