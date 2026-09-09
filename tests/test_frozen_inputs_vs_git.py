"""What the running experiment froze, against what the repository stores.

`run_manifest.json` freezes a sha256 for fourteen inputs: one config, one
protocol, and twelve sources. They are not enforced alike:

    config_sha256     re-checked before every stage (RuntimeError), and on
                      resume raises "use a new run version". No override.
    source_sha256     re-checked before every stage (RuntimeError); resume is
                      possible with --accept-code-update and a recorded repair.
    protocol_sha256   not checked per stage at all -- only on resume, in the
                      same no-override loop as the config.

Four of the fourteen have a CRLF working tree and an LF blob, because
`.gitattributes` says `eol=lf` and `git add` normalises the index without
touching the working tree. EXECUTION section 3.2 verified the second half of
that -- committing `v4_train.py` and `evaluate.py` did not change the bytes the
run reads -- and the corollary is what this file pins: **the committed blob is
therefore not the file the run froze**. A `git checkout` of one of those four
paths silently replaces it with the LF version. For a source, that stops the
run at the next stage boundary and can be repaired. For the config, it stops
the run and cannot. For the protocol it stops nothing now and makes the run
unresumable later, which is the quietest of the three and the easiest to do by
accident.

The difference is line endings and nothing else, which is checked here rather
than assumed. That is what makes this a handling rule instead of a defect:
there is no content divergence to reconcile, only files that must not be
re-materialised from git while the run is alive.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path, PureWindowsPath

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "runs/v4/decoupling-main-20260908/run_manifest.json"
SUPERVISOR = ROOT / "scripts/run_decoupling_pilot.py"

CRLF = bytes((13, 10))
LF = bytes((10,))

CONFIG = "configs/v4_decoupling_main_20260908.yaml"
PROTOCOL = "docs/prereg/2026-09-08-dstar-main-run.md"

# Measured 2026-09-09, at 33 h of 96. A fifth path appearing here means somebody
# committed another CRLF file the run depends on; a path leaving it means one
# was re-materialised from git, which is the accident this guards.
CRLF_WORKING_TREE = {
    CONFIG,
    PROTOCOL,
    "scripts/v4_train.py",
    "src/selfsight/v4/evaluate.py",
}

pytestmark = pytest.mark.skipif(not MANIFEST.exists(),
                                reason=f"{MANIFEST} is an untracked run artifact")


def frozen() -> dict[str, str]:
    """Every digest the manifest holds, keyed by a path git will accept.

    `protocol_path` is stored with Windows separators, so it is the one entry
    that has to be translated. Reading it from the manifest rather than
    hardcoding it is deliberate: if the run ever freezes a different protocol,
    this should follow it rather than keep checking the old one.
    """

    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    protocol = PureWindowsPath(payload["protocol_path"]).as_posix()
    assert protocol == PROTOCOL, f"the run froze a different protocol: {protocol}"
    return {CONFIG: payload["config_sha256"],
            protocol: payload["protocol_sha256"],
            **payload["source_sha256"]}


def blob(path: str) -> bytes | None:
    result = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=ROOT,
                            capture_output=True, check=False)
    return result.stdout if result.returncode == 0 else None


def supervisor() -> str:
    return SUPERVISOR.read_text(encoding="utf-8")


def test_every_frozen_file_still_matches_its_digest():
    """The run's own invariant, checked from outside the run.

    If this fails the supervisor is already dead or about to be: it makes the
    same comparison before every stage.
    """

    wrong = sorted(name for name, expected in frozen().items()
                   if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected)
    assert not wrong, f"working tree no longer matches the frozen digests: {wrong}"


def test_exactly_four_frozen_files_differ_from_their_committed_blob():
    differing = {name for name in frozen()
                 if blob(name) not in (None, (ROOT / name).read_bytes())}
    assert differing == CRLF_WORKING_TREE


def test_the_difference_is_line_endings_and_not_content():
    """No content divergence to reconcile -- only files not to check out."""

    for name in sorted(CRLF_WORKING_TREE):
        committed = blob(name)
        assert committed is not None, f"{name} is not committed"
        working = (ROOT / name).read_bytes()
        assert CRLF in working, f"{name} no longer has CRLF; the set above is stale"
        assert CRLF not in committed
        assert working.replace(CRLF, LF) == committed


def test_the_other_ten_are_byte_identical_to_the_repository():
    same = {name for name in frozen()
            if name not in CRLF_WORKING_TREE and blob(name) == (ROOT / name).read_bytes()}
    assert len(same) == len(frozen()) - len(CRLF_WORKING_TREE)


def test_the_resume_check_lets_sources_back_but_not_the_config_or_protocol():
    """Spelled out because the three digests read alike and are not alike.

    The `for key in (...)` loop raises with no way to say otherwise; the source
    comparison below it takes `--accept-code-update` and records a repair. If a
    future edit moves the override up into the first branch, this is the place
    to read again rather than the place to delete.
    """

    body = supervisor()
    head, rest = body.split('for key in ("config_sha256"', 1)
    no_override, override = rest.split('old["source_sha256"] != fingerprint', 1)

    assert '"protocol_sha256", "runs")' in no_override
    assert "use a new run version" in no_override
    assert "accept_code_update" not in no_override
    assert "accept_code_update" in override.split("manifest = old")[0]
    assert "accept_code_update" not in head


def test_the_protocol_is_not_re_checked_between_stages():
    """Not a complaint -- the reason its failure mode is the quiet one.

    A protocol edit mid-run costs nothing until something needs to resume, at
    which point the run is already unresumable and has been for a while.
    """

    body = supervisor()
    stage_check = body.split("def run(self, stage", 1)[1].split("complete =", 1)[0]
    assert "config_sha256" in stage_check
    assert "source_sha256" in stage_check
    assert "protocol_sha256" not in stage_check
