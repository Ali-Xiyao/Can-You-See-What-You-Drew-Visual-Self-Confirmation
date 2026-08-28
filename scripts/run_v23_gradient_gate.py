"""Run/resume the local v2.3 RFO-Gold gradient survival gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from selfsight.backbones.lora_selection import validate_lora_target_selection
from selfsight.utils.hashing import sha256_file
from selfsight.v23.gradient import run_v23_gradient_gate
from selfsight.v23.protocol import validate_v23_authorization


def main() -> None:
    print("[v2.3 gradient] cli=ready", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--backbone-config", type=Path, required=True)
    parser.add_argument("--lora-target-config", type=Path, required=True)
    parser.add_argument("--frozen-observer-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    print("[v2.3 gradient] preflight=authorization", flush=True)
    authorization = validate_v23_authorization(
        args.authorization, stage="gradient_survival_gate", output_path=args.output
    )
    print("[v2.3 gradient] preflight=lora-targets", flush=True)
    target_path = args.lora_target_config.resolve()
    target_raw = json.loads(target_path.read_text(encoding="utf-8"))
    selection = validate_lora_target_selection(
        target_path, canary_report=Path(str(target_raw["canary_report"]))
    )
    a4_path = Path(str(authorization["evidence"]["a4"]["path"])).resolve()
    a4 = json.loads(a4_path.read_text(encoding="utf-8"))
    if (
        a4.get("passed") is not True
        or a4.get("target_config_sha256") != sha256_file(target_path)
        or a4.get("target_selection_digest") != selection.get("selection_digest")
    ):
        raise RuntimeError("v2.3 gradient LoRA target selection does not match the green A4")
    print("[v2.3 gradient] preflight=complete", flush=True)
    report = run_v23_gradient_gate(
        config_path=args.config,
        authorization_path=args.authorization,
        probe_manifest=args.probe_manifest,
        backbone_config=args.backbone_config,
        frozen_observer_python=args.frozen_observer_python,
        output_dir=args.output,
        lora_target_modules=tuple(str(item) for item in selection["target_modules"]),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
