"""Freeze a held-out static probe bank and measure checkpoint gradient alignment.

Example:
  python scripts/v4_checkpoint_probe.py freeze --config CONFIG --split RUN/split.json --source runs/v4/gate-b-openct2 --outdir RUN/probe-bank
  python scripts/v4_checkpoint_probe.py run --config CONFIG --bank RUN/probe-bank --outdir RUN/probes/base --base --device cuda:0
  python scripts/v4_checkpoint_probe.py run --config CONFIG --bank RUN/probe-bank --outdir RUN/probes/naive/round-000 --checkpoint RUN/checkpoints/naive/round-000 --arm naive --reference RUN/probes/base --device cuda:0

CPU freeze copies existing RFO answers only after question/model/RGB validation.
GPU run uses <= 3*n*D*4 bytes of scratch space and keeps exact n-by-n Gram
matrices rather than multi-GB gradient vectors after successful completion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from selfsight.v4.checkpoint_probe import freeze_bank, run_probe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--config", required=True, type=Path)
    freeze.add_argument("--split", required=True, type=Path)
    freeze.add_argument("--source", required=True, type=Path)
    freeze.add_argument("--outdir", required=True, type=Path)
    freeze.add_argument("--max-prompts", type=int, default=16)
    freeze.add_argument("--min-prompts", type=int, default=4)
    run = sub.add_parser("run")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--bank", required=True, type=Path)
    run.add_argument("--outdir", required=True, type=Path)
    state = run.add_mutually_exclusive_group(required=True)
    state.add_argument("--checkpoint", type=Path)
    state.add_argument("--base", action="store_true")
    run.add_argument("--arm", default="base")
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--reference", type=Path)
    run.add_argument("--resamples", type=int, default=2000)
    run.add_argument("--scratch-limit-gib", type=float, default=8.0)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.stage == "freeze":
        result = freeze_bank(source=args.source, split_path=args.split, config=config,
                             destination=args.outdir, maximum=args.max_prompts, minimum=args.min_prompts)
        print(f"Frozen {result['actual_prompts']}/{args.max_prompts} held-out specs: {result['fingerprint']}")
    else:
        result = run_probe(config=config, bank_dir=args.bank, destination=args.outdir,
                           checkpoint=args.checkpoint, arm=args.arm, device=args.device,
                           resamples=args.resamples, reference=args.reference,
                           scratch_limit_gib=args.scratch_limit_gib)
        print(json.dumps({key: result[key] for key in ("checkpoint", "n_bank", "n_retained",
                                                      "gda_free", "gda_gold")}, indent=2))


if __name__ == "__main__":
    main()
