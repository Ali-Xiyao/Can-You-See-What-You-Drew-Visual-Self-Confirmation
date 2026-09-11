"""Does `read_outcome_repair.patch` actually turn its three red tests green?

The patch cannot be tried in the working tree while the main run is alive:
`scripts/v4_decoupling_report.py` is in `SOURCES`, and `run_decoupling_pilot.py`
re-checks every `source_sha256` at the top of every stage, so editing it is
fatal to the run.  So this builds a sandbox under the system temp directory,
applies the patch there with real `git apply`, and runs the three
`NEEDS_REPAIR` tests against both copies with the markers stripped.

The judge has to be able to fail both ways: the three must be RED before and
GREEN after.  Three green before would mean both phases were testing the
patched file -- the mistake the split-check harness made on 2026-09-10 and had
to be fixed for.

Two other tests in that file fail inside the sandbox with FileNotFoundError,
because they read `scripts/v4_train.py` and the patch file itself and this
sandbox holds neither.  Both pass in the repository.  They are listed in
SANDBOX_BLIND so a future reader does not mistake them for patch damage.

Read-only with respect to the repository.  Writes only under a temp directory.

    python review-packets/dstar-join-granularity-20260910/verify_repair_patch.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT / "review-packets/dstar-join-granularity-20260910/read_outcome_repair.patch"
TARGET = "scripts/v4_decoupling_report.py"
TESTS = "tests/test_read_outcome_join_granularity.py"
WANTED = [
    "test_a_spec_scored_on_its_first_draw_and_fully_adjudicated_is_counted",
    "test_the_internal_mean_divides_by_the_scored_images_not_the_drawn_ones",
    "test_spec_measurements_carries_the_first_draw_score",
]
SANDBOX_BLIND = [
    "test_the_writer_still_stamps_first_draw_only",
    "test_the_patch_is_checked_out_with_lf_endings",
]


def run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)


def build(home: Path, patched: bool) -> Path:
    (home / "scripts").mkdir(parents=True)
    (home / "tests").mkdir(parents=True)
    # .gitattributes must come along, or git apply writes CRLF under
    # core.autocrlf=true and the applied file is not the shipped one.
    shutil.copy2(ROOT / ".gitattributes", home / ".gitattributes")
    shutil.copy2(ROOT / TARGET, home / TARGET)
    src = (ROOT / TESTS).read_text(encoding="utf-8")
    if src.count("@NEEDS_REPAIR") != 3:
        sys.exit(f"expected 3 NEEDS_REPAIR markers in {TESTS}, found {src.count('@NEEDS_REPAIR')}; "
                 "either the repair already landed or this harness is out of date")
    (home / TESTS).write_text(re.sub(r"^@NEEDS_REPAIR\n", "", src, flags=re.MULTILINE),
                              encoding="utf-8", newline="\n")

    run(["git", "init", "-q"], home)
    run(["git", "add", "-A"], home)
    run(["git", "-c", "user.email=x@y", "-c", "user.name=x", "commit", "-qm", "base"], home)
    if patched:
        out = run(["git", "apply", "-v", str(PATCH)], home)
        if out.returncode:
            sys.exit(f"git apply failed:\n{out.stderr}")
    return home


def phase(tmp: Path, name: str, patched: bool) -> dict[str, str]:
    home = build(tmp / name, patched)
    out = run([sys.executable, "-B", "-m", "pytest", TESTS, "-q", "--no-header",
               "-p", "no:cacheprovider", "-rA"], home)
    verdict = {}
    for line in out.stdout.splitlines():
        head = line.split(" ")[0]
        if head in ("PASSED", "FAILED", "ERROR"):
            for test in WANTED + SANDBOX_BLIND:
                if test in line:
                    verdict[test] = head
    tail = [ln for ln in out.stdout.splitlines() if " passed" in ln or " failed" in ln]
    print(f"--- {name} (patched={patched}) ---   {tail[-1] if tail else out.stdout[-300:]}")
    for test in WANTED:
        print(f"    {verdict.get(test, '(not reported)'):8s}  {test}")
    for test in SANDBOX_BLIND:
        print(f"    {verdict.get(test, '(not reported)'):8s}  {test}   (sandbox-blind, passes in the repo)")
    return verdict


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="repairbox-") as tmpdir:
        tmp = Path(tmpdir)
        before = phase(tmp, "before", patched=False)
        after = phase(tmp, "after", patched=True)

    print("\nverdict")
    ok = True
    for test in WANTED:
        was, now = before.get(test), after.get(test)
        good = was == "FAILED" and now == "PASSED"
        ok &= good
        print(f"  {'OK ' if good else 'BAD'}  {was or '?':6s} -> {now or '?':6s}  {test}")
    if not ok:
        sys.exit("\nthe patch does not turn all three red tests green -- do not land it as registered")
    print("\nall three go RED without the patch and GREEN with it, so the harness "
          "could have failed and did not")


if __name__ == "__main__":
    main()
