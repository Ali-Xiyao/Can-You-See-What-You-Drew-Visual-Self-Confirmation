"""Build v2.3 box calibration and local-only mechanism authorization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from selfsight.v23.protocol import build_v23_authorization, build_v23_calibration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--backbone-config", type=Path, required=True)
    parser.add_argument("--data-registry", type=Path, required=True)
    parser.add_argument("--human-report", type=Path, required=True)
    parser.add_argument("--answer-key", type=Path, required=True)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--a4-report", type=Path, required=True)
    parser.add_argument("--frozen-v22-decision", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    calibration_path = args.output_root.resolve() / "box-calibration.json"
    authorization_path = args.output_root.resolve() / "authorization.json"
    calibration = build_v23_calibration(
        human_report_path=args.human_report,
        answer_key_path=args.answer_key,
        review_csv_path=args.review_csv,
        output_path=calibration_path,
    )
    authorization = build_v23_authorization(
        config_path=args.config,
        backbone_config_path=args.backbone_config,
        data_registry_path=args.data_registry,
        calibration_path=calibration_path,
        a4_report_path=args.a4_report,
        frozen_v22_decision_path=args.frozen_v22_decision,
        output_root=args.output_root,
        output_path=authorization_path,
    )
    print(json.dumps({"calibration": calibration, "authorization": authorization}, indent=2))


if __name__ == "__main__":
    main()
