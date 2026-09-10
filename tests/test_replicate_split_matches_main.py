"""Nothing compares the replicates' split to the main run's at launch time.

EXECUTION section 1 registers the hole and where it closes: deviation 7.2's
guard compares main against replicate in `load_columns`, which runs at
analysis time, after the 400 GPU-hours; `v4_verify_replicates.py` compares the
five replicates only with each other. That file lives on `staging/arm-b-merged`
and creating it on this branch would be an add/add conflict, so the tests that
need it skip until the merge lands, and go live the moment it does.

Measured on 2026-09-10 and pinned below: the quantity that file compares is
worse than "weaker than identity". `v4_train.split_digest` is a pure function
of `sorted(runs)`, `config["seed"]`, `data.local_outcome` and
`data.local_probe`; `run_decoupling_pilot.py` passes one module-level `RUNS`
constant to every run; and the five replicate configs share the partition seed
by design (that is what `v4_e3_launch_preflight.gate_configs` asserts). So the
recipe digest is one number, 9bab14d4..., for the main run and all five
replicates, computable from config text before a split stage has ever run.
Comparing it across the five is a condition that cannot fail.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from selfsight.analysis.drift import split_digest
from selfsight.utils.hashing import sha256_json

ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "scripts/v4_verify_replicates.py"
RUNS = ("runs/v4/main-2plus1", "runs/v4/main-1plus1plus1")  # run_decoupling_pilot.py:26
MAIN_CONFIG = ROOT / "configs/v4_decoupling_main_20260908.yaml"
RECIPE_DIGEST = "9bab14d427f2b61bea001b710126dcf18d610972985f723665534209231f9d58"

after_merge = pytest.mark.skipif(
    not VERIFIER.exists(),
    reason="scripts/v4_verify_replicates.py arrives with the staging/arm-b-merged merge")


def config_split_digest():
    """`v4_train.split_digest`, taken out of the source rather than retyped.

    Importing `scripts/v4_train.py` pulls the whole training stack in; lifting
    one function by AST keeps this a test of that function and not of a copy
    of it that can drift away from it.
    """

    tree = ast.parse((ROOT / "scripts/v4_train.py").read_text(encoding="utf-8"))
    fn = next(node for node in tree.body
              if isinstance(node, ast.FunctionDef) and node.name == "split_digest")
    namespace = {"sha256_json": sha256_json, "Any": object}
    code = compile(ast.Module(body=[fn], type_ignores=[]), "v4_train", "exec")
    exec(code, namespace)  # noqa: S102 - running the real function, not a copy of it
    return namespace["split_digest"]


def test_the_recipe_digest_is_one_number_for_all_six_configs():
    """The measurement behind the docstring, and it runs on this branch today.

    Goes red if a replicate is ever given its own partition seed or its own
    outcome/probe counts -- in which case the five no longer land on one split
    and the sign test in section 34 is comparing different populations.
    """

    digest = config_split_digest()
    configs = [MAIN_CONFIG] + sorted(ROOT.glob("configs/v4_e3_replicate_s*.yaml"))
    assert len(configs) == 6, [c.name for c in configs]
    computed = {c.name: digest(yaml.safe_load(c.read_text(encoding="utf-8")), RUNS)
                for c in configs}
    assert set(computed.values()) == {RECIPE_DIGEST}, computed


def write_run(root: Path, name: str, *, training_seed: int, outcome: list[str],
              digest: str = RECIPE_DIGEST, created: str = "2026-09-11T00:00:00Z") -> Path:
    """A run directory in the shape `stage_split` and the round writer leave behind."""

    run = root / name
    (run / "rounds" / "round-000").mkdir(parents=True)
    (run / "rounds" / "round-000" / "done.json").write_text(
        json.dumps({"round": 0, "initialization_seed": training_seed}), encoding="utf-8")
    (run / "split.json").write_text(json.dumps({
        "created": created, "digest": digest, "runs": list(RUNS), "seed": 20260906,
        "train": ["t1", "t2"], "outcome": outcome, "probe": ["p1"]}), encoding="utf-8")
    return run


HELD_OUT = ["o1", "o2", "o3"]
SEEDS = (20260906, 20260907, 20260909, 20260910, 20260911)


@pytest.fixture
def verifier():
    import importlib.util

    spec = importlib.util.spec_from_file_location("v4_verify_replicates", VERIFIER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@after_merge
def test_five_replicates_that_agree_only_with_each_other_are_refused(verifier, tmp_path):
    """The registered hole. Five perfect replicates of the wrong experiment."""

    main = write_run(tmp_path, "main", training_seed=20260906, outcome=HELD_OUT)
    runs = [write_run(tmp_path, f"e3-s{s}", training_seed=s, outcome=["o1", "o2", "o9"])
            for s in SEEDS]
    report = verifier.verify(runs, main=main)
    assert report["verdict"] == "NOT independent seeds"
    assert "not the main run's" in report["reason"]
    assert report["training_seeds_distinct"] and report["split_shared"]


@after_merge
def test_one_replicate_off_the_shared_split_is_caught_by_identity(verifier, tmp_path):
    """Same recipe, different held-out prompts: a corpus that grew mid-campaign.

    The five launches are spread over 304-352 h. A source run under `--runs`
    gaining a prompt in that window changes what `split_prompts` returns while
    leaving the recipe -- runs, seed, counts -- untouched.
    """

    main = write_run(tmp_path, "main", training_seed=20260906, outcome=HELD_OUT)
    runs = [write_run(tmp_path, f"e3-s{s}", training_seed=s,
                      outcome=["o1", "o2", "o9"] if s == 20260910 else HELD_OUT)
            for s in SEEDS]
    report = verifier.verify(runs, main=main)
    assert report["verdict"] == "NOT independent seeds"
    assert "do not share a split" in report["reason"]
    # And the field the old version compared saw nothing wrong.
    assert report["recipe_digests_shared"] is True


@after_merge
def test_the_wall_clock_stamp_is_not_part_of_the_comparison(verifier, tmp_path):
    """Every replicate's split stage runs at a different moment, and must pass.

    This is the failure `4d852dc` fixed inside drift.split_digest; the guard is
    repeated here because this is the caller that would have suffered it.
    """

    main = write_run(tmp_path, "main", training_seed=20260906, outcome=HELD_OUT,
                     created="2026-09-08T14:33:14Z")
    runs = [write_run(tmp_path, f"e3-s{s}", training_seed=s, outcome=HELD_OUT,
                      created=f"2026-09-{11 + index}T0{index}:00:00Z")
            for index, s in enumerate(SEEDS)]
    report = verifier.verify(runs, main=main)
    assert report["verdict"] == "five seeds", report["reason"]
    assert report["split_matches_main"] and report["checked_against_main"]


@after_merge
def test_the_command_line_refuses_to_run_without_the_main_run(tmp_path):
    """`verify()` keeps `main` optional so the six older tests still call it.

    That is only safe because the one path a human takes cannot skip it.
    """

    runs = [str(write_run(tmp_path, f"e3-s{s}", training_seed=s, outcome=HELD_OUT))
            for s in SEEDS]
    done = subprocess.run([sys.executable, str(VERIFIER), *runs],
                          capture_output=True, text=True, check=False)
    assert done.returncode != 0
    assert "--main" in done.stderr


@after_merge
def test_the_identity_pinned_for_the_main_run_is_what_this_compares(verifier):
    """Ties this file to `test_main_run_split_identity.py`'s third hash.

    Skips with the rest when the verifier is absent, and additionally when the
    run artefact is (a fresh clone, the merge probe worktree).
    """

    main_run = ROOT / "runs/v4/decoupling-main-20260908"
    if not (main_run / "split.json").exists():
        pytest.skip(f"{main_run}/split.json is an untracked run artefact")
    assert split_digest(main_run) == (
        "41c5f1b2a87470c1c946eb7ad64f6ef8cbfa1d101e326426fc7bbb9ccee028cf")
