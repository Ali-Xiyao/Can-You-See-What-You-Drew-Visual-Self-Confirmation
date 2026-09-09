"""A stage subprocess must import the checkout that launched it.

Every interpreter under `envs/` has selfsight installed editable against the
main tree. The E4 drivers put their own `src` on `sys.path` for themselves, but
the stages they shell out to are separate processes and inherit nothing, so
before this was fixed a driver running from a worktree read the branch while
`v4_run_pipeline.py observe` -- which is where every answer is graded -- read
main.

That failure is silent in the way that matters: the preflight passes, the
worktree's own tests pass, and the answers come back graded by whichever copy
of `grade` the main tree happens to hold.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DRIVERS = ("v4_cross_model.py", "v4_context_curve.py")


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module", params=DRIVERS)
def driver(request):
    return _load(f"_driver_{request.param.replace('.', '_')}", f"scripts/{request.param}")


def test_a_child_reads_the_checkout_that_launched_it(driver):
    """The point of the whole fix, measured rather than asserted.

    Spawns a real interpreter and asks it where selfsight came from. Passing
    `child_env()` has to override the editable install, which is a claim about
    sys.path precedence, so it is checked against the interpreter instead of
    being reasoned about.
    """

    probe = ("import selfsight, pathlib, json; "
             "print(json.dumps(str(pathlib.Path(selfsight.__file__).resolve().parent.parent)))")
    with_env = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                              text=True, check=True, env=driver.child_env())
    assert Path(json.loads(with_env.stdout.strip())) == driver.SRC


def test_the_editable_install_is_what_it_overrides(driver):
    """Without the fix the child lands somewhere else -- unless it cannot.

    Once the branch is merged the main tree and this checkout are the same
    directory and there is nothing to override, so the test says so rather than
    passing for a reason that has stopped being true.
    """

    probe = ("import selfsight, pathlib, json; "
             "print(json.dumps(str(pathlib.Path(selfsight.__file__).resolve().parent.parent)))")
    bare = os.environ.copy()
    bare.pop("PYTHONPATH", None)
    plain = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                           text=True, check=True, env=bare)
    landed = Path(json.loads(plain.stdout.strip()))
    if landed == driver.SRC:
        pytest.skip("this checkout is the one the interpreters are installed against")
    assert landed != driver.SRC, "the override is doing work; keep it"


def test_an_inherited_pythonpath_is_kept_behind_ours(driver):
    """A caller's PYTHONPATH survives, but does not get to win.

    Dropping it would break anyone running these from a shell that sets it;
    appending ours after it would put the main tree first again on the machines
    where PYTHONPATH already names it.
    """

    borrowed = str(Path(os.getcwd()) / "somewhere-else")
    env = dict(os.environ, PYTHONPATH=borrowed)
    saved = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = borrowed
    try:
        result = driver.child_env()["PYTHONPATH"]
    finally:
        if saved is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = saved
    assert result.split(os.pathsep)[0] == str(driver.SRC)
    assert borrowed in result.split(os.pathsep)
    assert env["PYTHONPATH"] == borrowed


def test_every_stage_subprocess_gets_the_environment(driver):
    """One call site without `env=` is the whole defect back.

    Reads the source because the observe stages cannot be run in a test. What
    it pins is that no `subprocess.run` in these drivers is left inheriting the
    ambient environment.
    """

    path = ROOT / "scripts" / Path(driver.__file__).name
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "run"
             and isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess"]
    assert calls, "the driver launches stages; if it stopped, this test is stale"
    for call in calls:
        passed = {keyword.arg: keyword.value for keyword in call.keywords}
        assert "env" in passed, f"{path.name}:{call.lineno} inherits the ambient environment"
        value = passed["env"]
        assert (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                and value.func.id == "child_env"), (
            f"{path.name}:{call.lineno} must use child_env(), not an ad-hoc dict")


def test_the_stage_script_does_not_depend_on_the_working_directory(driver):
    """PIPELINE was relative, so it named whichever checkout the shell sat in."""

    pipeline = Path(driver.PIPELINE)
    assert pipeline.is_absolute(), "a relative PIPELINE follows the cwd, not this file"
    assert pipeline == ROOT / "scripts" / "v4_run_pipeline.py"
    assert pipeline.exists()


def test_a_stage_cannot_reach_the_network_for_weights(driver):
    """The hard rule this project runs under, enforced where it can be broken.

    An adapter that cannot find a snapshot locally will fetch one, and E4 loads
    three models this driver has never loaded before. Offline turns a missing
    checkout into a loud failure instead of a quiet download.
    """

    assert driver.child_env()["HF_HUB_OFFLINE"] == "1"


def test_an_inherited_device_mask_does_not_renumber_the_cards(driver):
    """`--device cuda:1` means the second card, not the second visible one.

    A shell that exported CUDA_VISIBLE_DEVICES=1 would make `cuda:0` name
    physical card 1 and `cuda:1` fail outright, and the arms in this project are
    pinned per card. Dropping the mask keeps `--device` meaning what it says.
    """

    saved = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    try:
        env = driver.child_env()
    finally:
        if saved is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved
    assert "CUDA_VISIBLE_DEVICES" not in env


def test_the_stage_environment_matches_the_supervisor(driver):
    """The two drivers and run_decoupling_pilot.py launch the same stages.

    Read out of the supervisor rather than restated here, so that a setting
    added there for a reason does not stay missing from E4 for none. PYTHONPATH
    is excluded because the supervisor points it at the main tree by definition
    and these drivers point it at themselves -- that difference is the fix.
    """

    supervisor = ROOT / "scripts" / "run_decoupling_pilot.py"
    tree = ast.parse(supervisor.read_text(encoding="utf-8"))
    updates = [node for node in ast.walk(tree)
               if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and node.func.attr == "update"
               and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "env"]
    assert len(updates) == 1, "the supervisor builds its stage environment in one place"
    expected = {keyword.arg: keyword.value.value for keyword in updates[0].keywords
                if keyword.arg != "PYTHONPATH"}
    assert expected, "nothing to compare against; this test has gone stale"
    env = driver.child_env()
    for key, value in expected.items():
        assert env.get(key) == value, f"the supervisor sets {key}={value}; E4 does not"


def test_a_config_comes_from_the_checkout_and_data_from_the_shell():
    """Code and configs follow the file; runs/ and envs/ follow the cwd.

    E4 needs both halves at once: the branch's configs, which exist only in this
    checkout, and `envs/` and `runs/`, which exist only where the machine keeps
    them. Resolving configs against the cwd meant the driver could never see
    both from one directory.
    """

    driver = _load("_driver_models", "scripts/v4_cross_model.py")
    for model, config in driver.MODELS.items():
        if config is None:
            continue
        path = Path(config)
        assert path.is_absolute(), f"{model} resolves against the cwd"
        assert path == ROOT / "configs" / "backbones" / path.name
        assert path.exists(), f"{model} names a config this checkout does not have"
