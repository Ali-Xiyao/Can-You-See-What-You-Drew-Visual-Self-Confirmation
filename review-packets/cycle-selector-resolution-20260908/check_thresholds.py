"""Run the cycle scorer through the acceptance thresholds frozen on 2026-09-04.

The thresholds are taken unchanged from docs/prereg/2026-09-04-selector-resolution.md
section 2, and the handling of the one criterion that does not transfer to a
continuous score is fixed in docs/prereg/2026-09-08-cycle-selector.md section 4,
written before this script was run.

    primary   share of mixed pools where mean(correct) > mean(wrong)   >= 60%
    primary   share of mixed pools where mean(correct) = mean(wrong)   <= 25%
    secondary share of pools where every candidate ties                <= 25%
    secondary share of candidates at full marks                        N/A, unbounded score

No GPU: the cycle scores were produced in review-packets/bon-coupling-20260908.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "review-packets/bon-coupling-20260908"))

from bon_analysis import atomic_scores, cycle_scores, jsonl  # noqa: E402

CYCLE = ROOT / "review-packets/bon-coupling-20260908/cycle-scores.jsonl"

# gate-b is the original closed two-choice question set the 2026-09-04 thresholds
# were measured against. openct and openct2 are the section-1 repair of that
# prereg -- open counting, so the answer cannot be recovered from the prompt --
# already implemented. Running all three makes the comparison "does the repair
# clear its own bar" rather than "is cycle better than something unspecified".
BANKS = ("gate-b", "gate-b-openct", "gate-b-openct2")

# Reproducing the frozen numbers on the original bank is the admission check for
# this script: if separation() cannot recover them, it is measuring something else.
RECORDED_GATE_B = {
    "correct_beats_wrong": 0.349,
    "correct_equals_wrong": 0.612,
    "all_candidates_tied": 0.603,
}

# docs/prereg/2026-09-04-selector-resolution.md section 2, verbatim.
THRESHOLDS = {
    "primary_correct_beats_wrong": {"floor": 0.60, "atomic_recorded": 0.349, "reference": 0.746},
    "primary_correct_equals_wrong": {"ceiling": 0.25, "atomic_recorded": 0.612, "reference": 0.119},
    "secondary_all_tied": {"ceiling": 0.25, "atomic_recorded": 0.603, "reference": 0.112},
}


def separation(pools, by_pool):
    """Per mixed pool: does the correct group outscore the wrong group?"""
    beats = ties = loses = 0
    considered = 0
    all_tied = 0
    total = 0
    for pool in pools:
        scores = by_pool[pool["prompt_id"]]
        correct = [scores[c["candidate_id"]] for c in pool["candidates"] if c["correct"]]
        wrong = [scores[c["candidate_id"]] for c in pool["candidates"] if not c["correct"]]
        every = [scores[c["candidate_id"]] for c in pool["candidates"]]
        total += 1
        if len(set(every)) == 1:
            all_tied += 1
        if not correct or not wrong:
            continue
        considered += 1
        left = sum(correct) / len(correct)
        right = sum(wrong) / len(wrong)
        if left > right:
            beats += 1
        elif left == right:
            ties += 1
        else:
            loses += 1
    return {
        "mixed_pools": considered,
        "all_pools": total,
        "correct_beats_wrong": beats / considered,
        "correct_equals_wrong": ties / considered,
        "correct_loses_to_wrong": loses / considered,
        "all_candidates_tied": all_tied / total,
    }


def judge(measured):
    return {
        "primary_correct_beats_wrong": (
            measured["correct_beats_wrong"]
            >= THRESHOLDS["primary_correct_beats_wrong"]["floor"]
        ),
        "primary_correct_equals_wrong": (
            measured["correct_equals_wrong"]
            <= THRESHOLDS["primary_correct_equals_wrong"]["ceiling"]
        ),
        "secondary_all_tied": (
            measured["all_candidates_tied"] <= THRESHOLDS["secondary_all_tied"]["ceiling"]
        ),
    }


def main():
    scorers = {}
    for bank in BANKS:
        path = ROOT / "runs/v4" / bank
        pools = jsonl(path / "pools.jsonl")
        scorers[f"atomic::{bank}"] = (
            pools, atomic_scores(pools, jsonl(path / "observations.naive.jsonl"))
        )
    cycle_pools = jsonl(ROOT / "runs/v4/gate-b-openct2/pools.jsonl")
    scorers["cycle::gate-b-openct2"] = (cycle_pools, cycle_scores(cycle_pools, CYCLE))

    report = {
        "banks": list(BANKS),
        "thresholds_from": "docs/prereg/2026-09-04-selector-resolution.md section 2",
        "continuous_score_handling": "docs/prereg/2026-09-08-cycle-selector.md section 4",
        "scorers": {},
    }
    for name, (pools, by_pool) in scorers.items():
        measured = separation(pools, by_pool)
        verdicts = judge(measured)
        report["scorers"][name] = {
            "measured": measured,
            "verdicts": verdicts,
            "passes_all": all(verdicts.values()),
        }

    original = report["scorers"]["atomic::gate-b"]["measured"]
    report["reproduces_frozen_gate_b_numbers"] = {
        key: {"recorded": value, "recomputed": round(original[key], 3),
              "matches": abs(original[key] - value) < 0.0006}
        for key, value in RECORDED_GATE_B.items()
    }
    if not all(v["matches"] for v in report["reproduces_frozen_gate_b_numbers"].values()):
        raise SystemExit(f"admission check failed: {report['reproduces_frozen_gate_b_numbers']}")

    report["full_marks_criterion"] = (
        "N/A for cycle: log p(prompt|image) is unbounded, so there is no full-marks level "
        "to saturate. Declared not applicable in the 2026-09-08 prereg section 4 rather "
        "than replaced with a trivially-passed analogue."
    )
    report["already_known_before_this_run"] = (
        "The cycle scorer's unique-top rate on this bank (96.6%) was measured on the morning "
        "of 2026-09-08, before the 2026-09-08 prereg was written, so secondary_all_tied is "
        "recorded as already-known and not as a blind test. The primary criterion had never "
        "been computed for either scorer and is blind."
    )

    (HERE / "threshold-check.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for name, block in report["scorers"].items():
        m = block["measured"]
        print(f"\n=== {name} ({m['mixed_pools']} mixed of {m['all_pools']} pools) ===")
        print(f"  correct > wrong   {m['correct_beats_wrong']:6.1%}   need >= 60%   "
              f"{'PASS' if block['verdicts']['primary_correct_beats_wrong'] else 'FAIL'}")
        print(f"  correct = wrong   {m['correct_equals_wrong']:6.1%}   need <= 25%   "
              f"{'PASS' if block['verdicts']['primary_correct_equals_wrong'] else 'FAIL'}")
        print(f"  correct < wrong   {m['correct_loses_to_wrong']:6.1%}")
        print(f"  all tied          {m['all_candidates_tied']:6.1%}   need <= 25%   "
              f"{'PASS' if block['verdicts']['secondary_all_tied'] else 'FAIL'}")
        print(f"  -> {'PASSES' if block['passes_all'] else 'DOES NOT PASS'}")


if __name__ == "__main__":
    main()
