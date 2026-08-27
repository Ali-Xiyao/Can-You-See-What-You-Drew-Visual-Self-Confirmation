"""Export or score the preregistered blinded manual reference-verifier audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from selfsight.data.manual_audit import export_manual_audit, score_manual_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export")
    export.add_argument("--manifest", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--per-family", type=int, default=20)
    export.add_argument("--seed", type=int, default=20260827)
    score = subparsers.add_parser("score")
    score.add_argument("--review-csv", type=Path, required=True)
    score.add_argument("--answer-key", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        report = export_manual_audit(
            args.manifest,
            args.output,
            per_family=args.per_family,
            seed=args.seed,
        )
    else:
        report = score_manual_audit(args.review_csv, args.answer_key, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
