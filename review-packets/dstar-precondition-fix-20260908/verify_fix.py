"""Re-measure the D* precondition defect against the fixed estimator.

Procedure follows the standing rule for instrument defects: the old record is
kept as written, the defect is registered as evidence, and the re-measurement
is a new artifact that names its source. The source here is

    review-packets/bon-coupling-20260908/dstar-precondition-defect.json

which recorded, against the estimator as it stood on 2026-09-08 morning, that a
never-coupled flat external curve and a monotonically declining one both
returned a confident D* of 32. That file is not modified.

The same three synthetic cases are replayed here, unchanged, plus the two
questions the fix raises: whether a bare sign test is enough (it is not), and
whether the genuine case survives (it must, or the fix is not a fix).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from selfsight.analysis.breakpoints import estimate_d_star

PRIOR = ROOT / "review-packets/bon-coupling-20260908/dstar-precondition-defect.json"

STEPS = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 64.0, 72.0, 80.0]

# The synthetic curves carry no adjudication count, so the floor that
# v4.evaluate.external_noise_slope would derive is reconstructed here from the
# project's outcome-set size. Stated rather than tuned: n=120 prompts per
# checkpoint, binomial precision at the curve's own level, over the step span.
OUTCOME_N = 120


def flat(value, jitter=0.0):
    return [value + jitter * ((i % 3) - 1) for i in range(len(STEPS))]


def hinge(base, slope_before, slope_after, knot):
    return [
        base + slope_before * s + (slope_after - slope_before) * max(0.0, s - knot)
        for s in STEPS
    ]


def measured_floor(external):
    """What external_noise_slope would return for this curve at n=120."""
    span = STEPS[-1] - STEPS[0]
    sems = sorted(math.sqrt(max(0.0, p * (1.0 - p)) / OUTCOME_N) for p in external)
    return sems[len(sems) // 2] / span


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


def run(case, floor):
    estimate = estimate_d_star(
        STEPS, case["internal"], case["external"], min_coupled_external_slope=floor
    )
    return {
        "d_star": estimate.d_star,
        "accepted": estimate.d_star is not None,
        "external_pre_slope": estimate.external_pre_slope,
        "coupled_candidates": estimate.coupled_candidates,
        "admissible_candidates": estimate.admissible_candidates,
        "reason": estimate.reason,
    }


def main():
    prior = json.loads(PRIOR.read_text(encoding="utf-8"))
    report = {
        "source_record": {
            "path": "review-packets/bon-coupling-20260908/dstar-precondition-defect.json",
            "kept_unmodified": True,
            "recorded_wrongly_accepted": prior["summary"]["cases_wrongly_accepted"],
        },
        "outcome_n_assumed_for_floor": OUTCOME_N,
        "cases": {},
    }

    for name, case in CASES.items():
        floor = measured_floor(case["external"])
        sign_only = run(case, 0.0)
        measured = run(case, floor)
        report["cases"][name] = {
            "why": case["why"],
            "should_be_accepted": case["should_be_accepted"],
            "prior_d_star": prior[name]["d_star"],
            "measured_floor": floor,
            "sign_test_only": sign_only,
            "with_measured_floor": measured,
            "now_correct": measured["accepted"] == case["should_be_accepted"],
            "sign_test_alone_sufficient": sign_only["accepted"] == case["should_be_accepted"],
        }

    cases = report["cases"]
    report["summary"] = {
        "all_cases_now_correct": all(c["now_correct"] for c in cases.values()),
        "still_wrongly_accepted": sorted(k for k, c in cases.items() if not c["now_correct"]),
        "needed_a_measured_floor_not_just_a_sign_test": sorted(
            k for k, c in cases.items() if not c["sign_test_alone_sufficient"]
        ),
        "genuine_case_survives": cases["B_genuine_decoupling"]["with_measured_floor"]["accepted"],
        "still_open": (
            "fit_segmented has no no-break null, so it always returns some knot, and case B "
            "recovers 40.0 from noiseless data whose true knot is 32. Both are knot-location "
            "questions, separate from the precondition fixed here "
            "(.planning/2026-09-04-project-audit/findings.md line 131)."
        ),
    }

    (HERE / "verify-fix.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["summary"]["all_cases_now_correct"]:
        raise SystemExit("the fix does not settle every recorded case")


if __name__ == "__main__":
    main()
