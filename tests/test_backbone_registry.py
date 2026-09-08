"""E4 asks three model families the same questions; the config picks which one.

The failure this file is mostly about is not a crash. It is a cross-model
replication that quietly ran the same model three times, reported "the effect
does not depend on architecture", and looked like a finding. So the tests that
matter here are the ones that check the config's claim against the object that
actually answers, and the ones that check the declared revisions against the
model lock before a 17 GB load rather than after it.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

from selfsight.backbones import registry
from selfsight.models import load_model_lock

CONFIGS = Path(__file__).resolve().parents[1] / "configs/backbones"
E4 = {
    "showo2_7b.yaml": "showo2",
    "showo_v1.yaml": "showo_v1",
    "janus_pro_1b.yaml": "janus_pro",
}


def write(tmp_path: Path, **keys) -> Path:
    path = tmp_path / "backbone.yaml"
    path.write_text(yaml.safe_dump(keys), encoding="utf-8")
    return path


# ------------------------------------------------------------ reading configs


def test_a_config_without_a_family_is_the_backbone_it_always_was(tmp_path):
    """showo2_1p5b.yaml and showo2_7b.yaml have no `family` key and must not
    grow one: their stems appear in answer filenames already written to disk."""

    config = write(tmp_path, backbone_id="showlab/show-o2-1.5B")
    assert registry.backbone_family(registry.read_backbone_config(config)) == "showo2"


def test_the_two_frozen_showo2_configs_still_read_as_showo2():
    for name in ("showo2_1p5b.yaml", "showo2_7b.yaml"):
        config = registry.read_backbone_config(CONFIGS / name)
        assert registry.backbone_family(config) == "showo2", name


def test_an_unknown_family_is_refused_while_it_is_still_cheap(tmp_path):
    """Before the weights load, not three frames into someone else's package."""

    config = write(tmp_path, family="llava", backbone_id="x")
    with pytest.raises(ValueError, match="Unknown backbone family"):
        registry.read_backbone_config(config)


@pytest.mark.parametrize("name,family", sorted(E4.items()))
def test_each_e4_config_declares_its_family_and_interpreter(name, family):
    """The three adapters live in three environments and only one of them is
    importable in any given process, so the driver reads the interpreter off
    the config rather than keeping a family-to-path table in a script."""

    config = registry.read_backbone_config(CONFIGS / name)
    assert registry.backbone_family(config) == family
    if family == "showo2":
        return
    environment = registry.backbone_environment(config)
    assert environment is not None, f"{name} must name the interpreter that can load it"
    assert environment.startswith("envs/") and environment.endswith("python.exe"), environment
    # The interpreter is resolved from the working directory, like every other
    # path the pipeline is invoked with. A worktree checkout has no `envs/` at
    # all, so the existence half of the claim is only checkable where the jobs
    # actually run.
    if not Path("envs").is_dir():
        pytest.skip("no envs/ in this checkout")
    assert Path(environment).exists(), environment


@pytest.mark.parametrize("name", sorted(E4))
def test_every_declared_revision_matches_the_model_lock(name):
    """A typo'd revision would otherwise surface as a FileNotFoundError after
    the run has been queued behind two days of training."""

    config = registry.read_backbone_config(CONFIGS / name)
    lock = load_model_lock()
    revisions = {item["id"]: item["revision"] for item in lock["models"]}
    repositories = {item["id"]: item["revision"] for item in lock["repositories"]}

    assert revisions[config["backbone_id"]] == config["revision"]
    assert repositories[config["source"]["repository_id"]] == config["source"]["revision"]
    for model_id, revision in config["dependencies"].items():
        assert revisions[model_id] == revision, model_id


# --------------------------------------------------------------- dispatching


class Stub:
    """Stands in for an adapter: records how it was built, claims an identity."""

    model_id = "declared"
    revision = "declared-revision"

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def observe_atoms(self, image_path, questions):  # pragma: no cover - never called
        raise AssertionError("the registry must not answer anything itself")


def stub_family(monkeypatch, family: str) -> dict:
    """Replace one family's adapter class, leaving the others real."""

    built: dict = {}

    class Recorded(Stub):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            built.update(kwargs)

    module, attribute = {
        "showo2": ("selfsight.backbones.showo2", "Showo2Adapter"),
        "showo_v1": ("selfsight.showo_adapter", "ShowoAdapter"),
        "janus_pro": ("selfsight.backbones.janus_pro", "JanusProAdapter"),
    }[family]
    monkeypatch.setattr(f"{module}.{attribute}", Recorded)
    return built


@pytest.mark.parametrize("family", sorted(set(E4.values())))
def test_the_family_in_the_config_is_the_adapter_that_gets_built(monkeypatch, tmp_path, family):
    built = stub_family(monkeypatch, family)
    config = write(tmp_path, family=family, backbone_id="declared", revision="declared-revision")
    registry.build_observer(config, device="cpu")
    assert built, f"{family} config built some other family's adapter"
    assert built.get("device") == "cpu"


def test_show_o_v1_is_built_with_the_vision_tower_it_needs(monkeypatch, tmp_path):
    """`_observe_one` raises without it, and it raises after five gigabytes of
    weights are already resident on the card."""

    built = stub_family(monkeypatch, "showo_v1")
    config = write(tmp_path, family="showo_v1", backbone_id="declared",
                   revision="declared-revision")
    registry.build_observer(config, device="cpu")
    assert built["load_vision_tower"] is True


def test_showo2_still_receives_the_config_path_itself(monkeypatch, tmp_path):
    """Showo2Adapter reads its own YAML for geometry and dependencies; passing
    only the family would silently give every Show-o2 the 1.5B's geometry."""

    built = stub_family(monkeypatch, "showo2")
    config = write(tmp_path, family="showo2", backbone_id="declared",
                   revision="declared-revision")
    registry.build_observer(config, device="cpu")
    assert built["backbone_config"] == str(config)


# --------------------------------------------------------- the answer budget
#
# A truncated reply is not a wrong answer, it is a missing one, and a missing
# one is silent: the trial leaves the denominator and the accuracy is computed
# over whatever survived. Which trials survive depends on how much the model
# felt like saying, which is not independent of the condition E4 varies.


def test_a_config_that_says_nothing_keeps_the_budget_the_showo_models_use():
    assert registry.answer_length({}) == 16
    assert registry.answer_length({"official_profile": {}}) == 16


def test_the_budget_reaches_the_adapter_instead_of_its_default(monkeypatch, tmp_path):
    """The config key has to be read. A value sitting in YAML that nothing
    consults is the same as no value, and looks like one that was applied."""

    built = stub_family(monkeypatch, "janus_pro")
    config = write(tmp_path, family="janus_pro", backbone_id="declared",
                   revision="declared-revision",
                   official_profile={"mmu_max_new_tokens": 99})
    registry.build_observer(config, device="cpu")
    assert built["max_new_tokens"] == 99


def test_the_shipped_janus_config_leaves_room_for_a_sentence():
    """Janus narrates the picture and puts the letter at the end.

    At 16 tokens the reply stops before the letter and the trial is graded as an
    abstention -- measured at half of them on a two-image probe. This is the
    config value that stops that, and it is the same in both conditions, so it
    cannot move the within-model comparison E4 reads.
    """

    config = registry.read_backbone_config(CONFIGS / "janus_pro_1b.yaml")
    assert registry.answer_length(config) >= 64


def test_the_showo_configs_keep_the_budget_their_frozen_rows_were_answered_at():
    """showo_v1 is a frozen negative control. Its answers must be produced under
    the same decoding budget as the rest of the Show-o evidence."""

    config = registry.read_backbone_config(CONFIGS / "showo_v1.yaml")
    assert registry.answer_length(config) == 16


# ------------------------------------------------------------ identity guard


def test_a_config_that_names_one_model_and_loads_another_is_refused(monkeypatch, tmp_path):
    """Two of the three families resolve their weights from the model lock, not
    from this config, so `backbone_id` here is a claim rather than a cause. A
    replication that ran the same model twice would report a clean null."""

    stub_family(monkeypatch, "janus_pro")
    config = write(tmp_path, family="janus_pro", backbone_id="deepseek-ai/Janus-Pro-7B",
                   revision="declared-revision")
    with pytest.raises(RuntimeError, match="declares backbone_id="):
        registry.build_observer(config, device="cpu")


def test_a_config_that_names_the_wrong_revision_is_refused(monkeypatch, tmp_path):
    stub_family(monkeypatch, "showo_v1")
    config = write(tmp_path, family="showo_v1", backbone_id="declared", revision="some-other-sha")
    with pytest.raises(RuntimeError, match="declares revision="):
        registry.build_observer(config, device="cpu")


def test_a_config_that_claims_nothing_is_not_checked_against_nothing(monkeypatch, tmp_path):
    """The frozen showo2 configs predate the guard; absence must not raise."""

    stub_family(monkeypatch, "showo2")
    config = write(tmp_path, family="showo2")
    registry.build_observer(config, device="cpu")


# ----------------------------------------------------------- the contract


def test_every_adapter_answers_with_the_same_method():
    """`stage_observe` calls exactly one method on whatever it is given. An
    adapter that spells it differently fails at the first image, hours in."""

    import inspect

    from selfsight.backbones.janus_pro import JanusProAdapter
    from selfsight.backbones.showo2 import Showo2Adapter
    from selfsight.showo_adapter import ShowoAdapter

    for adapter in (Showo2Adapter, ShowoAdapter, JanusProAdapter):
        signature = inspect.signature(adapter.observe_atoms)
        assert list(signature.parameters) == ["self", "image_path", "questions"], adapter


# ------------------------------------------------------- interpreter resolution


def _fake_envs(root: Path, relative: str) -> Path:
    exe = root / relative
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_bytes(b"")
    return exe


def test_a_config_that_names_its_interpreter_gets_that_one(tmp_path):
    """A declared environment is never overridden by the family default."""

    config = tmp_path / "declared.yaml"
    config.write_text("family: janus_pro\nenvironment: envs/janus/Scripts/python.exe\n",
                      encoding="utf-8")
    exe = _fake_envs(tmp_path, "envs/janus/Scripts/python.exe")
    resolved = registry.read_backbone_config(config)
    assert registry.backbone_interpreter(resolved, root=tmp_path) == str(exe.resolve())


def test_a_showo2_config_that_names_none_gets_the_showo2_environment(tmp_path):
    """The case the older config test returns early on, which is where the gap was.

    `backbone_environment` answers None here and that is correct -- the config
    genuinely declares nothing. A caller that treats None as a path builds a
    command whose first element is None, which is what E4 did.
    """

    config = tmp_path / "bare.yaml"
    config.write_text("backbone_id: showlab/show-o2-7B\n", encoding="utf-8")
    exe = _fake_envs(tmp_path, "envs/showo2/python.exe")
    resolved = registry.read_backbone_config(config)
    assert registry.backbone_environment(resolved) is None
    assert registry.backbone_interpreter(resolved, root=tmp_path) == str(exe.resolve())


def test_the_path_is_absolute_because_createprocess_rejects_the_other_kind(tmp_path):
    """Windows will not launch `envs/showo2/python.exe` from the directory above it.

    Not a style preference. Measured in the project root:
    `Path("envs/showo2/python.exe").exists()` is True and
    `subprocess.run(["envs/showo2/python.exe", "-c", "print(1)"])` raises
    WinError 2 in the same process, because CreateProcess will not take a
    relative path spelled with forward slashes -- which is how every config in
    configs/backbones writes it. E4's preflight died on exactly this after
    resolving the interpreter correctly.
    """

    config = tmp_path / "bare.yaml"
    config.write_text("backbone_id: showlab/show-o2-7B\n", encoding="utf-8")
    _fake_envs(tmp_path, "envs/showo2/python.exe")
    resolved = registry.read_backbone_config(config)
    returned = registry.backbone_interpreter(resolved, root=tmp_path)
    assert Path(returned).is_absolute()
    assert "/" not in returned or os.sep == "/", returned


def test_a_family_with_neither_says_which_family(tmp_path):
    """Guessing an interpreter fails as a pile of import errors somewhere else."""

    config = tmp_path / "orphan.yaml"
    config.write_text("family: showo_v1\n", encoding="utf-8")
    resolved = registry.read_backbone_config(config)
    with pytest.raises(ValueError, match="showo_v1"):
        registry.backbone_interpreter(resolved, root=tmp_path)


def test_the_wrong_working_directory_says_so_rather_than_winerror_2(tmp_path):
    """WinError 2 renders as mojibake on this machine and names nothing.

    The interpreter is resolved from the cwd because `envs/` belongs to the
    machine, so getting the cwd wrong is the likely mistake and it should read
    as one.
    """

    config = tmp_path / "bare.yaml"
    config.write_text("backbone_id: showlab/show-o2-7B\n", encoding="utf-8")
    resolved = registry.read_backbone_config(config)
    with pytest.raises(FileNotFoundError, match="envs/ is resolved from"):
        registry.backbone_interpreter(resolved, root=tmp_path)


@pytest.mark.parametrize("name", sorted(E4))
def test_every_e4_config_resolves_to_an_interpreter_that_runs(name):
    """All three, including the two the older config test excuses.

    E4 loads these one after another over hours; an interpreter that cannot be
    launched should fail here, not once the second model is already resident.
    """

    if not Path("envs").is_dir():
        pytest.skip("no envs/ in this checkout")
    config = registry.read_backbone_config(CONFIGS / name)
    interpreter = registry.backbone_interpreter(config)
    finished = subprocess.run([interpreter, "-c", "print('ok')"], check=False,
                              capture_output=True, text=True, timeout=120)
    assert finished.returncode == 0, finished.stderr[-400:]


def test_the_driver_asks_for_an_interpreter_not_an_optional_one():
    """`backbone_environment` is the wrong function to build a command from.

    Reads the source because both call sites are inside stages that launch
    subprocesses. What it pins is that neither reaches for the Optional.
    """

    import ast

    source = Path(__file__).resolve().parents[1] / "scripts" / "v4_cross_model.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    used |= {alias.name for node in ast.walk(tree)
             if isinstance(node, ast.ImportFrom) for alias in node.names}
    assert "backbone_interpreter" in used, "the driver must resolve, not read the raw key"
    assert "backbone_environment" not in used, (
        "backbone_environment returns None for every Show-o2 config")


def test_a_relative_root_still_yields_something_launchable(tmp_path, monkeypatch):
    """`root="."` is the obvious way to say "here", and it must not stay relative.

    The absoluteness in the ordinary case comes from `Path.cwd()`, so this is
    the one path through the function where the normalisation is what supplies
    it -- and a relative result is the WinError 2 this all started with.
    """

    config = tmp_path / "bare.yaml"
    config.write_text("backbone_id: showlab/show-o2-7B\n", encoding="utf-8")
    _fake_envs(tmp_path, "envs/showo2/python.exe")
    monkeypatch.chdir(tmp_path)
    resolved = registry.read_backbone_config(config)
    returned = registry.backbone_interpreter(resolved, root=".")
    assert Path(returned).is_absolute(), returned
    assert Path(returned).exists()
