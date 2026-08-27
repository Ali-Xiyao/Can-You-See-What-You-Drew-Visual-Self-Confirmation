"""Launch or resume the real paired dual-3090 training loop after both prerequisite Gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from selfsight.pilot.real_loop import run_real_paired_pilot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/local_3090.yaml"))
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument("--gradient-gate-report", type=Path, required=True)
    parser.add_argument("--generated-domain-report", type=Path, required=True)
    parser.add_argument(
        "--frozen-observer-python",
        type=Path,
        default=Path(r"H:\selfsight-envs\core\python.exe"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = run_real_paired_pilot(
        config_path=args.config,
        train_manifest=args.train_manifest,
        gate_report=args.gate_report,
        gradient_gate_report=args.gradient_gate_report,
        generated_domain_report=args.generated_domain_report,
        frozen_observer_python=args.frozen_observer_python,
        output_dir=args.output,
        resume=args.resume,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
