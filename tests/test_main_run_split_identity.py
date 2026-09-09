"""The three hashes of the main run's split, and which one replicates share.

EXECUTION section 1's first pre-launch check says the replicates' split must
match the main run's, quotes 83b7eaa6... as "the main run's value", and calls
it the sha256 of `runs/v4/decoupling-main-20260908/split.json`. That is a
correct description of that number and the wrong number to compare: the file
opens with a wall clock stamp, so a replicate that holds out exactly the same
prompts writes different bytes and a different sha256, and the registered
response to a mismatch is to stop. Followed literally, the check halts five
correct replicates.

There are three hashes and section 1 names two of them:

    file sha256   83b7eaa6...   bytes, includes `created`  -- differs always
    digest        9bab14d4...   the recipe: (runs, seed, outcome, probe)
    identity      41c5f1b2...   everything except `created`

`digest` is what `v4_train.split_digest` computes and what `read_split`
already enforces against each run's own config at every stage. `identity` is
what `selfsight.analysis.drift.split_digest` compares between two runs for
deviation 7.2, and it is the strict one: it covers the recipe and the prompt
lists the recipe produced, so a file whose lists were edited afterwards fails
it while keeping its `digest`.

The values are pinned here rather than left in prose because prose does not
run. The file lives under `runs/`, which is untracked, so this skips where
the artifact is absent -- a merge probe worktree, a fresh clone -- and checks
where the run is.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from selfsight.analysis.drift import split_digest

ROOT = Path(__file__).resolve().parents[1]
MAIN_RUN = ROOT / "runs/v4/decoupling-main-20260908"
SPLIT = MAIN_RUN / "split.json"

FILE_SHA256 = "83b7eaa696726a1f6261327a5ac70c2d494629e5e736bcae7799eed0482f7094"
RECIPE_DIGEST = "9bab14d427f2b61bea001b710126dcf18d610972985f723665534209231f9d58"
IDENTITY = "41c5f1b2a87470c1c946eb7ad64f6ef8cbfa1d101e326426fc7bbb9ccee028cf"

pytestmark = pytest.mark.skipif(not SPLIT.exists(),
                                reason=f"{SPLIT} is an untracked run artifact")


def test_the_file_sha256_is_the_number_section_1_quotes():
    assert hashlib.sha256(SPLIT.read_bytes()).hexdigest() == FILE_SHA256


def test_the_recipe_digest_is_the_other_number_section_1_quotes():
    assert json.loads(SPLIT.read_text(encoding="utf-8"))["digest"] == RECIPE_DIGEST


def test_the_identity_is_what_deviation_7_2_compares():
    assert split_digest(MAIN_RUN) == IDENTITY


def test_the_file_carries_a_wall_clock_stamp():
    """This is why the sha256 cannot be the thing compared.

    Not a hypothetical: the value below is the moment the main run's split
    stage ran, and every replicate's split stage runs at a different one.
    """

    payload = json.loads(SPLIT.read_text(encoding="utf-8"))
    assert payload["created"] == "2026-09-08T14:33:14Z"


def test_the_three_hashes_are_three_different_numbers():
    # Section 1 warns that two of them are "两个数,别混". There are three.
    assert len({FILE_SHA256, RECIPE_DIGEST, IDENTITY}) == 3
