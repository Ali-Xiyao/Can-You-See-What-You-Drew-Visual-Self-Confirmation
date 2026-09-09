"""Add step32 to the existing diagnostic, preserving the step24 report."""
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

SIDE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("side_selection_audit", SIDE / "analyze.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
audit.STEPS = (0, 8, 16, 24, 32)
result = audit.fixed_bank()
old = json.loads((SIDE / "report.json").read_text(encoding="utf-8"))["fixed_bank_trajectory"]
for cohort, data in result["cohorts"].items():
    for arm, points in data["arms"].items():
        assert points[:-1] == old["cohorts"][cohort]["arms"][arm], "Earlier diagnostics changed"
assert all(audit.sha(Path(path)) == value for path, value in audit.INPUTS.items())
report = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "CPU extension of existing retrospective fixed-bank diagnostic to step32",
    "source_sha256": {"analyze.py": audit.sha(SIDE / "analyze.py"), "extension": audit.sha(Path(__file__)), "prior_report": audit.sha(SIDE / "report.json")},
    "steps": audit.STEPS, "input_sha256": audit.INPUTS,
    "previous_0_8_16_24_points_identical": True, "fixed_bank_trajectory": result,
    "limitations": old["checkpoint_provenance_scope"] + " Balanced bank is not the natural candidate population; intervals are exploratory scene resamples, not a new event criterion or independent seed validation.",
}
with (SIDE / "fixed-bank-through32.json").open("x", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(json.dumps({cohort: {arm: [{"step": row["step"], "correct": row["selected_correct"], "gain": row["gain_over_random"], "gain_ci": row["gain_bootstrap"], "all_tied": row["n_all_tied_pools"]} for row in points] for arm, points in data["arms"].items()} for cohort, data in result["cohorts"].items()}, ensure_ascii=True))
