"""The five commands that start 400 GPU-hours, read out of the runbook.

EXECUTION section 1's original five omitted `--protocol`, so every replicate
would have taken the supervisor's default -- the *pilot's* protocol, which
registers a different scale (10 rounds, one image per prompt) from the one the
replicates run. Nothing would have failed. The wrong document freezes into
`run_manifest.json` at construction and into `decoupling_report.json` at the
first report, and neither has a way back: both raise "use a new run version".

Section 1.2 corrected them. This reads the runbook's commands from the
document itself, because a runbook that is only prose gets executed as prose.
Section 1's gates became `v4_e3_launch_preflight.py` for the same reason; the
preflight cannot help here, since it runs before a command line exists.

Section 1.3 corrects them again, and is what this now reads. Section 3.9 found
that `read_outcome` requires every image of a spec to carry both measurements
while the main run's protocol section 3 registers an internal curve scored on
the first draw only -- unsatisfiable together at R=4. The repair lands in
`scripts/` before the replicates start, so the reader they run is not the one
2026-09-08 describes, and the document they freeze has to be the one that does.
Section 1.2's five commands stay in the document; only the section this reads
moves.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / ".planning/2026-09-08-iclr-redesign/EXECUTION.md"
SECTION = "### 1.3 "

REGISTERED_SEEDS = (20260906, 20260907, 20260909, 20260910, 20260911)
REGISTERED_ARMS = ["naive", "blind_self"]
PROTOCOL = "docs/prereg/2026-09-10-join-granularity.md"
# The document the main run froze. The replicates no longer name it, so the
# chain from one to the other is a digest rather than a shared path.
MAIN_PROTOCOL = "docs/prereg/2026-09-08-dstar-main-run.md"
# What run_decoupling_pilot.py falls back to. Named here so that changing the
# default to the right document makes this file fail rather than let section
# 1.3 and the supervisor drift apart silently.
SUPERVISOR_DEFAULT = "docs/prereg/2026-09-06-decoupling-pilot.md"


def commands() -> list[list[str]]:
    body = DOC.read_text(encoding="utf-8")
    start = body.index(SECTION)
    fence = re.search(r"\n```\n(.*?)\n```\n", body[start:], re.DOTALL)
    assert fence, f"{SECTION.strip()} has no fenced command block"
    return [line.split() for line in fence.group(1).strip().splitlines()]


def flags(argv: list[str]) -> dict[str, list[str]]:
    """argparse's reading of the line, not a regex over it."""

    out: dict[str, list[str]] = {}
    key = None
    for token in argv:
        if token.startswith("--"):
            key = token
            out[key] = []
        elif key is not None:
            out[key].append(token)
    return out


def test_there_are_five_commands_one_per_registered_seed():
    assert len(commands()) == len(REGISTERED_SEEDS)


@pytest.mark.parametrize("index", range(len(REGISTERED_SEEDS)))
def test_each_command_is_internally_consistent(index):
    import yaml

    seed = REGISTERED_SEEDS[index]
    argv = commands()[index]
    assert argv[0].endswith("python.exe")
    assert "scripts/run_decoupling_pilot.py" in argv
    parsed = flags(argv)

    assert parsed["--outdir"] == [f"runs/v4/e3-s{seed}"]
    assert parsed["--config"] == [f"configs/v4_e3_replicate_s{seed}.yaml"]
    assert parsed["--arms"] == REGISTERED_ARMS

    config = ROOT / parsed["--config"][0]
    assert config.exists(), f"{config} is named by the runbook and does not exist"
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    # The directory name is what a human reads and the config is what trains.
    # These being the same number is the whole claim that five runs are five
    # seeds, and it is the one mismatch no artifact would record.
    assert payload["training"]["seed"] == seed


def test_every_command_names_the_protocol_rather_than_defaulting():
    for argv in commands():
        parsed = flags(argv)
        assert "--protocol" in parsed, (
            "a command without --protocol freezes the pilot's protocol into "
            "run_manifest.json and the first report, neither recoverable")
        assert parsed["--protocol"] == [PROTOCOL]


def test_the_protocol_the_runbook_names_exists():
    assert (ROOT / PROTOCOL).exists()


def test_the_named_protocol_is_not_the_supervisor_default():
    """If the default is ever fixed, change section 1.3 too, do not delete this.

    Two places that agree by coincidence are two places that stop agreeing
    without anyone editing either one.
    """

    assert PROTOCOL != SUPERVISOR_DEFAULT
    supervisor = (ROOT / "scripts/run_decoupling_pilot.py").read_text(encoding="utf-8")
    assert SUPERVISOR_DEFAULT in supervisor, (
        "the supervisor no longer defaults to the pilot protocol; re-read "
        "section 1.3 rather than deleting this test")


def test_the_replicate_protocol_pins_the_main_runs_protocol():
    """Endpoint 1 reads the main run and the five replicates together.

    They no longer name the same document, because they no longer run the same
    reader. What has to hold instead is that the chain is checkable: the
    replicate protocol names the main run's document and pins its digest, and
    that digest still matches the file. The last assertion is the useful one --
    it goes red the moment anyone edits the main run's protocol in place, which
    is what would break both its provenance and any resume.
    """

    manifest = ROOT / "runs/v4/decoupling-main-20260908/run_manifest.json"
    if not manifest.exists():
        pytest.skip(f"{manifest} is an untracked run artifact")
    frozen = json.loads(manifest.read_text(encoding="utf-8"))
    assert Path(frozen["protocol_path"]).as_posix() == MAIN_PROTOCOL
    text = (ROOT / PROTOCOL).read_text(encoding="utf-8")
    assert MAIN_PROTOCOL in text, "the replicate protocol does not name the main run's"
    assert frozen["protocol_sha256"] in text, (
        "the replicate protocol does not pin the digest the main run froze")
    assert hashlib.sha256((ROOT / MAIN_PROTOCOL).read_bytes()).hexdigest() ==         frozen["protocol_sha256"], "the main run's protocol has been edited in place"


def test_no_command_carries_through_round():
    """`--through-round` is what produces `canary_complete`.

    A replicate that stops short looks finished to anything reading state.json
    for a status rather than for a round count.
    """

    for argv in commands():
        assert "--through-round" not in argv


def test_no_command_carries_accept_code_update():
    """That flag rewrites the frozen source fingerprint in place.

    It exists for a reviewed repair on an existing run, and a fresh launch has
    nothing to repair -- passing it at launch would silence the one check that
    notices the tree moved underneath a replicate.
    """

    for argv in commands():
        assert "--accept-code-update" not in argv
