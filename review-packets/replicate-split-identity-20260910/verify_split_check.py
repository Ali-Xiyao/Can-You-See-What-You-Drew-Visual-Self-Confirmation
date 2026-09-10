"""Prove the patch applies, keeps the six old tests, and closes the two holes.

`scripts/v4_verify_replicates.py` does not exist on `paper/iclr-2028` -- it
arrives with the `staging/arm-b-merged` merge -- so nothing here touches the
working tree. The merged blob is read out of the object database, the patch is
applied to it with the real `git apply` in a temp repo, and the result is
compared byte for byte against the copy shipped in this packet.

Run from the repository root:
    envs/core/python.exe -B review-packets/replicate-split-identity-20260910/verify_split_check.py
"""
import importlib.util
import inspect
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TARGET = "scripts/v4_verify_replicates.py"
BLOB = "b5fd8b94bde40dcd4861aaf2fcb9b3a9ed63899f"
BRANCH = "staging/arm-b-merged"


def git(*args: str, binary: bool = False):
    done = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True)
    return done.stdout if binary else done.stdout.decode("utf-8")


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---- 1. the merged blob, and the patch applied to it with real git apply ----

actual = git("rev-parse", f"{BRANCH}:{TARGET}").strip()
if actual != BLOB:
    sys.exit(f"{BRANCH}:{TARGET} is {actual}, not the {BLOB} this patch was cut against")
print(f"{BRANCH}:{TARGET} is blob {BLOB[:12]} as recorded")

home = Path(tempfile.mkdtemp())
sandbox = home / "apply"
(sandbox / "scripts").mkdir(parents=True)
original = git("show", f"{BRANCH}:{TARGET}", binary=True)
(sandbox / TARGET).write_bytes(original)
# The sandbox has to carry the repository's eol rules or it is not a model of
# the landing. Measured 2026-09-10: without `.gitattributes`, `git apply` under
# core.autocrlf=true writes all 123 lines back as CRLF. `*.py text eol=lf` is
# what stops that, so this file is load-bearing for `git apply` as well as for
# the `*.patch` rule added the same day.
(sandbox / ".gitattributes").write_bytes((ROOT / ".gitattributes").read_bytes())
subprocess.run(["git", "init", "-q"], cwd=sandbox, check=True)
subprocess.run(["git", "apply", str(HERE / "split_check.patch")], cwd=sandbox, check=True)
applied = (sandbox / TARGET).read_bytes()
shipped = (HERE / "v4_verify_replicates.patched.py").read_bytes()
if applied != shipped:
    sys.exit("git apply produced something other than v4_verify_replicates.patched.py")
print(f"git apply reproduces the shipped copy exactly ({len(applied) - len(original):+d} bytes)")

(home / "before.py").write_bytes(original)
unpatched = load(home / "before.py", "unpatched")
patched = load(sandbox / TARGET, "patched")

# ---- 2. the six tests that came with the merged file still pass ----

(home / "suite").mkdir()
(home / "suite" / "old_suite.py").write_bytes(
    git("show", f"{BRANCH}:tests/test_v4_verify_replicates.py", binary=True))
sys.path.insert(0, str(home / "suite"))
sys.path.insert(0, str(ROOT / "tests"))
sys.dont_write_bytecode = True
import old_suite
import test_replicate_split_matches_main as new_suite


def run(suite, module, names, fixtures, label=""):
    failures = []
    for index, name in enumerate(names):
        fn = getattr(suite, name)
        scratch = home / f"{suite.__name__}-{label}-{index}"
        scratch.mkdir()
        kwargs = {key: (module if key == "verifier" else scratch)
                  for key in fixtures(fn)}
        try:
            fn(**kwargs)
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001 - a harness must catch whatever a test raises
            # A bare `assert x` raises an AssertionError whose str() is empty,
            # which is the shape half these reds take.
            detail = (str(exc).splitlines() or ["(no message)"])[0][:90]
            print(f"  FAIL  {name}\n          {type(exc).__name__}: {detail}")
            failures.append(name)
    return failures


def params(fn):
    return list(inspect.signature(fn).parameters)


old_names = [n for n in vars(old_suite) if n.startswith("test_")]
print(f"\nthe {len(old_names)} tests shipped with the merged file, against the patch:")
old_suite.SCRIPT = sandbox / TARGET
broke = run(old_suite, patched, old_names, params, "old")
if broke:
    sys.exit(f"the patch breaks tests that were already passing: {broke}")

# ---- 3. the new tests: red against the merged file, green against the patch ----

new_names = [n for n in vars(new_suite) if n.startswith("test_") and n != "test_the_recipe_digest_is_one_number_for_all_six_configs"]
print(f"\nthe {len(new_names)} new tests, against the merged file as it stands:")
# The command-line test reads VERIFIER rather than taking the fixture, so this
# has to move with the module or the "before" column silently tests the patch.
new_suite.VERIFIER = home / "before.py"
red = run(new_suite, unpatched, new_names, params, "before")
print(f"\nthe {len(new_names)} new tests, against the patch:")
new_suite.VERIFIER = sandbox / TARGET
green = run(new_suite, patched, new_names, params, "after")
if green:
    sys.exit(f"the patch does not satisfy: {green}")

# ---- 4. the substantive red, stated as a verdict rather than a TypeError ----

drift = home / "drift"
drift.mkdir()
main = new_suite.write_run(drift, "main", training_seed=20260906, outcome=new_suite.HELD_OUT)
runs = [new_suite.write_run(drift, f"e3-s{s}", training_seed=s,
                            outcome=["o1", "o2", "o9"] if s == 20260910 else new_suite.HELD_OUT)
        for s in new_suite.SEEDS]
old_report = unpatched.verify(runs)
new_report = patched.verify(runs, main=main)
print("\none replicate held out a different prompt set, same recipe:")
print(f"  merged file : verdict={old_report['verdict']!r}  split_shared={old_report['split_shared']}")
print(f"  patched     : verdict={new_report['verdict']!r}  "
      f"recipe_digests_shared={new_report['recipe_digests_shared']}")
if old_report["verdict"] != "five seeds" or new_report["verdict"] == "five seeds":
    sys.exit("the demonstration did not reproduce")

print(f"\n{len(old_names)} old + {len(new_names)} new pass on the patch; "
      f"{len(red)} of the new go red on the merged file as it stands")
print(json.dumps({"blob": BLOB, "old_pass": len(old_names), "new_pass": len(new_names),
                  "new_red_before": sorted(red)}, indent=2))
