"""Demonstrate that estimate_d_star accepts a trajectory that never had coupling.

The user's added requirement is that "training caused decoupling" needs a
measurable coupled phase first. estimate_d_star computes slope_before for both
curves and then never consults it: the acceptance test at breakpoints.py:128-131
filters on slope_after only. So an external curve that was flat from step 0 --
one where the internal score never predicted anything -- is scored as a
decoupling event as soon as the internal curve bends upward.

This registers the defect as reproducible evidence. It does not modify the
function; PROTOCOL.md section 6 keeps that out of this round.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from selfsight.analysis.breakpoints import estimate_d_star  # noqa: E402

STEPS = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 64.0, 72.0, 80.0]


def flat(value, jitter=0.0):
    return [value + jitter * ((i % 3) - 1) for i in range(len(STEPS))]


def hinge(base, slope_before, slope_after, knot):
    return [
        base + slope_before * s + (slope_after - slope_before) * max(0.0, s - knot)
        for s in STEPS
    ]


CASES = {
    "A_never_coupled": {
        "why": "external flat from step 0 (no coupling ever); internal bends up at 32",
        "internal": hinge(0.50, 0.0, 0.004, 32.0),
        "external": flat(0.37, 0.002),
        "should_be_accepted": False,
    },
    "B_genuine_decoupling": {
        "why": "external rises then stalls at 32; internal keeps rising",
        "internal": hinge(0.50, 0.002, 0.004, 32.0),
        "external": hinge(0.30, 0.003, 0.0, 32.0),
        "should_be_accepted": True,
    },
    "C_external_falling_throughout": {
        "why": "external declines from step 0; nothing ever coupled, model just degrades",
        "internal": hinge(0.50, 0.001, 0.004, 32.0),
        "external": hinge(0.45, -0.002, -0.002, 32.0),
        "should_be_accepted": False,
    },
}


def main():
    report = {}
    for name, case in CASES.items():
        estimate = estimate_d_star(STEPS, case["internal"], case["external"])
        accepted = estimate.d_star is not None
        report[name] = {
            "why": case["why"],
            "d_star": estimate.d_star,
            "accepted": accepted,
            "should_be_accepted": case["should_be_accepted"],
            "external_slope_before": (
                estimate.fit_external.slope_before if estimate.fit_external else None
            ),
            "external_slope_after": estimate.external_post_slope,
            "internal_slope_after": estimate.internal_post_slope,
            "defect_exposed": accepted and not case["should_be_accepted"],
        }
    report["summary"] = {
        "cases_wrongly_accepted": sorted(
            k for k, v in report.items() if isinstance(v, dict) and v.get("defect_exposed")
        ),
        "proposed_additional_condition": (
            "require external_candidate.slope_before > delta, with delta built from "
            "the measured precision of the external curve, and require interval "
            "support for it; this is a strengthening and cannot create new D* claims"
        ),
        "not_applied_this_round": "PROTOCOL.md section 6",
    }
    path = Path(__file__).resolve().parent / "dstar-precondition-defect.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
