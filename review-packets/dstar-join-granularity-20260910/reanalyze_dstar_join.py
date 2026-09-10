"""Recompute the D* windows with the outcome join granularity repaired.

The frozen reporter keeps a spec only when *every* image in it carries both an
s_select score and a verdict. The writer scores exactly one image per spec (the
selected candidate) and adjudicates all four, so complete_specs is empty at
every checkpoint, external delta is null in every window, and the registered
rule cannot fire. That is a join defect, not a null result.

This changes one predicate and nothing else. Every estimator, threshold,
bootstrap count and seed is the frozen module's own: the module is imported and
driven, not copied. read_outcome is called for real first, so the frozen
parsing, the duplicate-image check and the spec-id mismatch check all still run;
only complete_specs is rebuilt afterwards.

Repaired predicate: a spec counts when at least one of its images is scored and
all of its images are adjudicated.
  s_select = mean over the images that are scored   (was: mean over all images)
  external = mean over all images                   (unchanged)

Nothing in the run directory is written.
"""

import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True

ROOT = Path("H:/Xiyao_Wang/062_Can You See What You Drew Visual Self-Confirmation")
OUTDIR = ROOT / "runs/v4/decoupling-main-20260908"
CONFIG = ROOT / "configs/v4_decoupling_main_20260908.yaml"
PROTOCOL = ROOT / "docs/prereg/2026-09-08-dstar-main-run.md"

spec = importlib.util.spec_from_file_location(
    "frozen_report", ROOT / "scripts/v4_decoupling_report.py")
frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)

original_read_outcome = frozen.read_outcome


def read_outcome_repaired(directory, expected_spec_ids=None):
    base = original_read_outcome(directory, expected_spec_ids=expected_spec_ids)
    manifest = frozen._jsonl(directory / "manifest.jsonl")
    verified = frozen._image_index(frozen._jsonl(directory / "verified.jsonl"),
                                   str(directory / "verified.jsonl"))
    selection = frozen._image_index(frozen._jsonl(directory / "s_select.jsonl"),
                                    str(directory / "s_select.jsonl"))
    grouped = defaultdict(list)
    for row in manifest:
        image_path = str(row["image_path"])
        score = frozen._finite(selection.get(image_path, {}).get("s_select"))
        verdict_row = verified.get(image_path, {})
        verdict = None
        if (verdict_row.get("resolution") != "pending_human"
                and isinstance(verdict_row.get("image_correct"), bool)):
            verdict = float(verdict_row["image_correct"])
        grouped[str(row["spec_id"])].append((score, verdict))

    complete = {}
    for spec_id, values in grouped.items():
        size = len(values)
        scores = [s for s, _ in values if s is not None]
        verdicts = [v for _, v in values if v is not None]
        if scores and len(verdicts) == size:
            complete[spec_id] = {"s_select": sum(scores) / len(scores),
                                 "external": sum(verdicts) / size,
                                 "n_images": size}

    expected = base["coverage"]["expected_specs"]
    base["complete_specs"] = complete
    base["coverage"] = {**base["coverage"], "complete_specs": len(complete),
                        "incomplete_specs": expected - len(complete),
                        "paired_measurement_coverage":
                            len(complete) / expected if expected else None}
    return base


def summarize(report, label):
    print(f"\n===================== {label} =====================")
    for arm, row in report["arms"].items():
        print(f"\n  [{arm}]  candidate={row['first_candidate_step']}  "
              f"bootstrap_rule={row['first_bootstrap_rule_supported_step']}  "
              f"robustness={row['first_robustness_supported_step']}")
        print(f"  {'end':>4}  {'pairs':>5}  {'int delta':>10}  {'ext delta':>10}  "
              f"{'ext 95% CI':>22}  {'cand':>5}  {'boot':>5}")
        for w in row["windows"]:
            ext = w["external"]
            ci = ("[{:+.4f}, {:+.4f}]".format(ext["ci_low"], ext["ci_high"])
                  if ext["ci_low"] is not None else "--")
            delta = "{:+.4f}".format(ext["delta"]) if ext["delta"] is not None else "--"
            internal = w.get("internal", {})
            idelta = ("{:+.4f}".format(internal["delta"])
                      if internal.get("delta") is not None else "--")
            print(f"  {w['available_at_step']:>4}  {w['n_paired_specs']:>5}  "
                  f"{idelta:>10}  {delta:>10}  {ci:>22}  "
                  f"{str(w['candidate']):>5}  {str(w['bootstrap_rule_supported']):>5}")


kwargs = dict(bootstrap_samples=2000, seed=20260906,
              config_path=CONFIG, protocol_path=PROTOCOL)

frozen_report = frozen.build_report(OUTDIR, **kwargs)
summarize(frozen_report, "FROZEN (as the run wrote it)")

frozen.read_outcome = read_outcome_repaired
repaired_report = frozen.build_report(OUTDIR, **kwargs)
summarize(repaired_report, "REPAIRED JOIN")

out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
if out:
    out.write_text(json.dumps({"frozen": frozen_report, "repaired": repaired_report},
                              ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    print(f"\nwrote {out}")
