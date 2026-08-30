"""L0: emit the v3.0 Gate -2 decision from frozen HQ evidence (no GPU, no new measurement)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from selfsight.v3.readiness import recompute_v3_decision

FROZEN = Path("runs/readiness/showo2-1p5b-hq")
EXPLORATORY = Path("runs/exploratory-post-gate/showo2-1p5b-hq")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-decision", default=str(FROZEN / "decision-hq-red.json"))
    parser.add_argument("--reference-report", default=str(FROZEN / "a2-reference-r1.json"))
    parser.add_argument("--generated-report", default=str(FROZEN / "a3-generated-r1.json"))
    parser.add_argument("--human-report", default=str(FROZEN / "a3-human-r1.json"))
    parser.add_argument("--a4-report", default=str(EXPLORATORY / "a4-lora-r1.json"))
    parser.add_argument("--output", default="runs/v3/readiness/decision-v3.json")
    args = parser.parse_args()

    report = recompute_v3_decision(
        frozen_decision_path=args.frozen_decision,
        reference_report_path=args.reference_report,
        generated_report_path=args.generated_report,
        human_report_path=args.human_report,
        a4_report_path=args.a4_report,
        output_path=args.output,
    )
    print(json.dumps({k: report[k] for k in ("passed", "checks", "selected_eligible_families")}, indent=2))
    for family, row in report["family_eligibility"].items():
        print(f"  {family:<10} eligible={row['eligible']} {row.get('values', row)}")
    print(f"\nwrote {args.output}  digest={report['decision_digest'][:16]}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
