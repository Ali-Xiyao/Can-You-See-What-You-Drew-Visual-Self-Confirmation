"""E4 asks three model families the same questions; the config picks which one.

The failure this file is mostly about is not a crash. It is a cross-model
replication that quietly ran the same model three times, reported "the effect
does not depend on architecture", and looked like a finding. So the tests that
matter here are the ones that check the config's claim against the object that
actually answers, and the ones that check the declared revisions against the
model lock before a 17 GB load rather than after it.
"""
from __future__ import annotations

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
