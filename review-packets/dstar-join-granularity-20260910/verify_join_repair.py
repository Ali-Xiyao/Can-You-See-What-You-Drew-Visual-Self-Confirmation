"""Prove the repair makes the xfail tests pass and breaks nothing else.

`scripts/v4_decoupling_report.py` is a frozen SOURCE and the main run checks
its digest at the top of every stage, so the repair is applied to the source
*text* in memory and written to a temp file. Nothing in the repo is touched.
"""
import importlib.util
import inspect
import sys
import tempfile
from pathlib import Path

# Derived rather than hardcoded so the packet runs from any checkout.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
sys.dont_write_bytecode = True
import test_read_outcome_join_granularity as suite

SOURCE = (ROOT / "scripts/v4_decoupling_report.py").read_text(encoding="utf-8")

EDITS = [
    ("""        if all(score is not None and verdict is not None for score, verdict in values):
            complete[spec_id] = {
                "s_select": sum(score for score, _ in values) / len(values),
                "external": sum(verdict for _, verdict in values) / len(values),
                "n_images": len(values),
            }""",
     """        if scores and len(verdicts) == size:
            complete[spec_id] = {
                "s_select": sum(scores) / len(scores),
                "external": sum(verdicts) / size,
                "n_images": size,
            }"""),
    ('            "s_select": sum(scores) / size if len(scores) == size else None,',
     '            "s_select": sum(scores) / len(scores) if scores else None,'),
]

repaired = SOURCE
for old, new in EDITS:
    if repaired.count(old) != 1:
        sys.exit(f"anchor appears {repaired.count(old)}x:\n{old[:80]}")
    repaired = repaired.replace(old, new)
print(f"both anchors matched exactly once; patch is {len(repaired)-len(SOURCE):+d} chars")

home = Path(tempfile.mkdtemp())
path = home / "v4_decoupling_report.py"
path.write_text(repaired, encoding="utf-8")
spec = importlib.util.spec_from_file_location("repaired_report", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

XFAIL = {name for name, fn in vars(suite).items()
         if name.startswith("test_") and getattr(fn, "pytestmark", None)}
tests = [(n, f) for n, f in vars(suite).items() if n.startswith("test_") and callable(f)]
print(f"{len(tests)} tests, {len(XFAIL)} of them currently xfail\n")

bad = []
for index, (name, fn) in enumerate(tests):
    kwargs = {}
    for parameter in inspect.signature(fn).parameters:
        if parameter == "report":
            kwargs[parameter] = module
        elif parameter in ("tmp_path", "r4"):
            scratch = home / f"t{index}"
            scratch.mkdir(exist_ok=True)
            kwargs[parameter] = (suite.block(scratch / "step-00040", suite.R4)
                                 if parameter == "r4" else scratch)
        else:
            sys.exit(f"{name} wants an unhandled fixture: {parameter}")
    try:
        fn(**kwargs)
        ok, detail = True, ""
    except Exception as exc:  # noqa: BLE001 - a harness must catch whatever a test raises
        ok, detail = False, f"   {type(exc).__name__}: {str(exc).splitlines()[0][:90]}"
    tag = "xfail->" if name in XFAIL else "       "
    print(("PASS  " if ok else "FAIL  ") + tag + name + ("\n" + detail if detail else ""))
    if not ok:
        bad.append(name)

print()
if bad:
    sys.exit(f"repair does not satisfy: {bad}")
print(f"repaired source passes all {len(tests)}; the {len(XFAIL)} strict xfails will fire")
