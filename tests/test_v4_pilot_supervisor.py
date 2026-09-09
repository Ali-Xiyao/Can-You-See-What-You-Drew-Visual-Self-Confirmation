"""A completed file must not hide absent or invalid optimizer updates."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import pytest

from selfsight.v4.train import pending_rounds

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("pilot_supervisor", ROOT / "scripts/run_decoupling_pilot.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize("field,value", [
    ("parameter_delta_l2", 0),
    ("parameter_delta_l2", float("nan")),
    ("mean_gradient_norm_before_clip", float("inf")),
    ("mean_t2i_loss", float("nan")),
])
def test_done_with_invalid_updates_is_rejected(tmp_path, field, value):
    runner = object.__new__(MODULE.Pilot)
    runner.out = tmp_path
    runner.arms = list(MODULE.ARMS)
    runner.config = {"pilot": {"min_paired_prompts": 2}, "training": {"optimizer_steps_per_round": 8}}
    arm = {"mean_t2i_loss": 1, "mean_gradient_norm_before_clip": .2, "parameter_delta_l2": .1}
    arm[field] = value
    target = tmp_path / "rounds/round-000/DONE.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps({"paired": 3, "arms": [
        dict(arm, arm=name, round=0, optimizer_steps=8) for name in MODULE.ARMS]}))
    with pytest.raises(RuntimeError):
        runner.validate_round(0)


def test_done_with_too_few_paired_prompts_is_rejected(tmp_path):
    runner = object.__new__(MODULE.Pilot)
    runner.out = tmp_path
    runner.arms = list(MODULE.ARMS)
    runner.config = {"pilot": {"min_paired_prompts": 2}}
    target = tmp_path / "rounds/round-000/DONE.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps({"paired": 1, "arms": []}))
    with pytest.raises(RuntimeError, match="too few"):
        runner.validate_round(0)


@pytest.mark.parametrize("names", [[], ["naive"], ["naive", "naive"]])
def test_done_does_not_hide_missing_independent_training_arm(tmp_path, names):
    runner = object.__new__(MODULE.Pilot)
    runner.out = tmp_path
    runner.arms = list(MODULE.ARMS)
    runner.config = {"pilot": {"min_paired_prompts": 2}, "training": {"optimizer_steps_per_round": 8}}
    target = tmp_path / "rounds/round-000/DONE.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps({"paired": 3, "arms": [{"arm": name} for name in names]}))
    with pytest.raises(RuntimeError, match="missing or duplicate"):
        runner.validate_round(0)


def _resume_after_unrecorded_train(tmp_path, monkeypatch, through_round):
    """Run the real main/execute/run chain with CPU-only stage subprocesses.

    Training round 1 finished, but its supervisor sentinel was not persisted.
    The fake subprocess models both the old additional-round limit and the new
    exact-round target so the regression checks execution order, not just argv.
    """
    runner = object.__new__(MODULE.Pilot)
    runner.out = tmp_path
    runner.config_path = tmp_path / "config.yaml"
    runner.config_path.write_text("fixture: true\n", encoding="utf-8")
    # The real Pilot sets this in __init__ and `report` passes it to
    # v4_decoupling_report.py. This fixture builds the object with
    # object.__new__, so anything __init__ assigns has to be assigned here too,
    # or the stage that reads it fails only when a test happens to reach it.
    runner.protocol_path = tmp_path / "protocol.md"
    runner.protocol_path.write_text("# fixture protocol\n", encoding="utf-8")
    runner.audit_path = tmp_path / "retrospective-scene-audit" / "scene_overlap.json"
    runner.audit_path.parent.mkdir()
    runner.audit_path.write_text("{}", encoding="utf-8")
    runner.arms = list(MODULE.ARMS)
    runner.arm_device = dict(zip(runner.arms, MODULE.ARM_CARDS))
    runner.config = {
        "training": {"rounds": 4, "optimizer_steps_per_round": 8},
        "pilot": {"min_paired_prompts": 2, "min_free_gib": 0},
        "gradient_probe": {"source": "fixture", "size": 2, "resamples": 2},
    }
    runner.limit = through_round
    runner.started = time.time()
    runner.deadline = runner.started + 60
    runner.env = {}
    runner.common = ["--outdir", str(tmp_path)]
    runner.frozen = {
        "config_sha256": MODULE.digest(runner.config_path), "source_sha256": {},
    }
    runner.done_stages = tmp_path / "stage-completion"
    runner.done_stages.mkdir()
    (tmp_path / "split.json").write_text("{}", encoding="utf-8")

    def finish_round(index):
        MODULE.write_json(tmp_path / "rounds" / f"round-{index:03d}" / "DONE.json", {
            "paired": 3,
            "arms": [{"arm": arm, "round": index, "optimizer_steps": 8,
                      "mean_t2i_loss": 1, "mean_gradient_norm_before_clip": .2,
                      "parameter_delta_l2": .1} for arm in MODULE.ARMS],
        })

    for index in (0, 1):
        finish_round(index)
    completed = ["freeze-probe", "round-000.train", "base.step-00000.gradient"]
    for step in (0, 8):
        for arm in MODULE.ARMS:
            prefix = f"{arm}.step-{step:05d}"
            completed.extend(f"{prefix}.{suffix}" for suffix in (
                "generate", "detect.qwen3vl", "detect.internvl", "crop", "verify"))
            if step:
                completed.append(f"{prefix}.gradient")
        completed.extend(f"step-{step:05d}.{suffix}" for suffix in (
            "score", "report", "gradient-sensitivity", "plot"))
    for stage in completed:
        MODULE.write_json(runner.done_stages / f"{stage}.json", {"stage": stage})
    assert not (runner.done_stages / "round-001.train.json").exists()

    events = []
    commands = []

    class StageProcess:
        pid = 12345

        def __init__(self, command, **kwargs):
            commands.append(command)
            # Read from this stage's own environment, not from state.json. The
            # two arm chains run concurrently now and both publish that file, so
            # it reports whichever stage wrote last rather than this one. The
            # assertions below are unchanged: chains() joins before report() and
            # before the next round trains, so every measurement stage of a
            # round still lands before the next train, even though the order of
            # the two arms relative to each other is no longer fixed.
            events.append(("stage", kwargs["env"]["SELFSIGHT_STAGE"]))
            if Path(command[2]).name != "v4_train.py" or command[3] != "train":
                return
            done = {index for index in range(4)
                    if (tmp_path / "rounds" / f"round-{index:03d}" / "DONE.json").exists()}
            if "--round-index" in command:
                target = int(command[command.index("--round-index") + 1])
                pending = pending_rounds(4, sorted(done), None, round_index=target)
            else:
                count = int(command[command.index("--max-rounds") + 1])
                pending = pending_rounds(4, sorted(done), count)
            for index in pending:
                events.append(("train", index))
                finish_round(index)

        def wait(self, timeout):
            return 0

    def build_runner(args):
        runner.limit = args.through_round
        return runner

    monkeypatch.setattr(MODULE.subprocess, "Popen", StageProcess)
    monkeypatch.setattr(MODULE, "Pilot", build_runner)
    monkeypatch.setattr(MODULE.sys, "argv", ["run_decoupling_pilot.py", "--through-round", str(through_round)])
    MODULE.main()
    return runner, events, commands


def test_resume_measures_completed_round_before_training_next(tmp_path, monkeypatch):
    runner, events, commands = _resume_after_unrecorded_train(tmp_path, monkeypatch, 3)
    next_train = events.index(("train", 2))
    for arm in MODULE.ARMS:
        for suffix in ("generate", "detect.qwen3vl", "detect.internvl", "crop", "verify", "gradient"):
            assert events.index(("stage", f"{arm}.step-00016.{suffix}")) < next_train
    assert events.index(("stage", "step-00016.report")) < next_train
    assert [event for event in events if event[0] == "train"] == [("train", 2)]
    assert not (runner.out / "rounds/round-003/DONE.json").exists()
    trains = [command for command in commands if Path(command[2]).name == "v4_train.py"
              and command[3] == "train"]
    assert [command[command.index("--round-index") + 1] for command in trains] == ["1", "2"]


def test_repeated_completed_round_target_does_not_advance_pending_round(tmp_path, monkeypatch):
    runner, events, commands = _resume_after_unrecorded_train(tmp_path, monkeypatch, 2)
    assert [event for event in events if event[0] == "train"] == []
    assert not (runner.out / "rounds/round-002/DONE.json").exists()
    assert (runner.done_stages / "round-001.train.json").exists()
    assert ("stage", "naive.step-00016.gradient") in events
    assert ("stage", "rfo_gold.step-00016.gradient") in events
    trains = [command for command in commands if Path(command[2]).name == "v4_train.py"
              and command[3] == "train"]
    assert len(trains) == 1
    assert trains[0][trains[0].index("--round-index") + 1] == "1"


def test_the_report_is_stamped_with_the_protocol_the_run_registered():
    """A literal here disagrees with run_manifest.json and nothing notices.

    `__init__` records `protocol_sha256` from `--protocol` and refuses to start
    if it ever changes. `report` used to pass a hardcoded pilot document
    instead, so runs/v4/decoupling-main-20260908 registered the main-run
    protocol in its manifest and stamped the pilot's into
    decoupling_report.json. v4_decoupling_report.py freezes that provenance
    across steps, so the first report to land locks the wrong document in for
    every checkpoint after it.

    Read out of the source because `report` shells out three stages and the
    argv is built inline.
    """

    import ast

    tree = ast.parse((ROOT / "scripts/run_decoupling_pilot.py").read_text(encoding="utf-8"))
    method = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == "report")
    literals = [node.value for node in ast.walk(method)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
                and "prereg" in node.value]
    assert not literals, f"report names a protocol document directly: {literals}"
    attributes = {node.attr for node in ast.walk(method)
                  if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                  and node.value.id == "self"}
    assert "protocol_path" in attributes, "the stamp must come from what __init__ registered"


def test_gradient_sensitivity_is_told_which_scene_audit_to_read(tmp_path):
    """Letting the stage default lands it on the path the report also reads.

    `v4_gradient_sensitivity.py` defaults to RUN/audit-splits/scene_overlap.json,
    and `v4_decoupling_report.py` picks that same path up implicitly to choose an
    outcome subset -- which is only sound for an audit frozen before any outcome
    existed. A run whose audit was built mid-run has to name it, or the two
    consumers share one file across two different standards of evidence.
    """

    runner = object.__new__(MODULE.Pilot)
    runner.out = tmp_path
    runner.config_path = tmp_path / "config.yaml"
    runner.protocol_path = tmp_path / "protocol.md"
    runner.audit_path = tmp_path / "retrospective-scene-audit" / "scene_overlap.json"
    calls = []
    runner.run = lambda stage, environment, script, args: calls.append((script, args))
    MODULE.Pilot.report(runner, 0)

    sensitivity = [args for script, args in calls
                   if script.endswith("v4_gradient_sensitivity.py")]
    assert len(sensitivity) == 1, calls
    args = sensitivity[0]
    assert "--audit" in args, "without it the stage falls back to RUN/audit-splits/"
    assert args[args.index("--audit") + 1] == str(runner.audit_path)


def _write_audit(outdir, *, prospective: bool):
    """A scene audit where its own flag says it belongs."""

    folder = "audit-splits" if prospective else "retrospective-scene-audit"
    path = outdir / folder / "scene_overlap.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"created_before_any_outcome_evaluation_artifact": prospective}), encoding="utf-8")
    return path


def _constructible_run(tmp_path, monkeypatch):
    """The smallest tree `Pilot.__init__` accepts, with no scene audit in it yet."""

    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    monkeypatch.setattr(MODULE.subprocess, "check_output", lambda *args, **kwargs: "0" * 40)
    for name in MODULE.SOURCES:
        source = tmp_path / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("training:\n  rounds: 2\n  optimizer_steps_per_round: 8\n"
                      "pilot:\n  max_wall_hours: 1\n  min_paired_prompts: 2\n"
                      "gradient_probe:\n  source: fixture\n  size: 2\n", encoding="utf-8")
    protocol = tmp_path / "protocol.md"
    protocol.write_text("# protocol\n", encoding="utf-8")
    return argparse.Namespace(outdir=tmp_path / "runs" / "v4" / "audit-guard",
                              config=config, protocol=protocol, arms=list(MODULE.ARMS),
                              through_round=None, accept_code_update=False)


class _StopAtTrain(Exception):
    """The first stage that would touch a card, and the end of what we watch."""


def _opening_stages(runner, monkeypatch, *, the_stage_writes_the_audit=True):
    """Drive execute() up to the first training stage, recording what it ran."""

    calls = []

    def fake_run(stage, environment, script, stage_args):
        calls.append((stage, script, stage_args))
        if stage == "scene-audit" and the_stage_writes_the_audit:
            _write_audit(runner.out, prospective=True)
        if stage.endswith(".train"):
            raise _StopAtTrain(stage)

    monkeypatch.setattr(runner, "run", fake_run)
    with pytest.raises(_StopAtTrain):
        runner.execute()
    return calls


def test_a_run_without_a_scene_audit_builds_one_before_it_reserves_a_card(tmp_path,
                                                                         monkeypatch):
    """The stage that reads it runs after the first checkpoint, hours in.

    runs/v4/decoupling-main-20260908 died that way on 2026-09-08 with 5.83 h of
    compute already spent. Requiring the file at construction fixed the cost of
    the mistake but left it a mistake to make; the split and the probe bank the
    audit reads are this run's own first two stages, so it can be built here.
    """

    runner = MODULE.Pilot(_constructible_run(tmp_path, monkeypatch))
    assert runner.audit_path is None, "a run starting from nothing cannot have one yet"

    stages = [stage for stage, _script, _args in _opening_stages(runner, monkeypatch)]

    assert "scene-audit" in stages
    assert stages.index("freeze-probe") < stages.index("scene-audit"), "it reads the bank"
    first_train = next(i for i, stage in enumerate(stages) if stage.endswith(".train"))
    assert stages.index("scene-audit") < first_train, "before anything is an outcome"
    assert runner.audit_path == runner.out / "audit-splits" / "scene_overlap.json"


def test_the_audit_the_supervisor_builds_is_the_kind_the_report_can_read(tmp_path,
                                                                        monkeypatch):
    """Only the prospective kind is automatic. Settling for the retrospective
    one is a decision about what the run's evidence may support, and the audit
    script refuses to call a started run prospective anyway."""

    runner = MODULE.Pilot(_constructible_run(tmp_path, monkeypatch))
    calls = _opening_stages(runner, monkeypatch)

    stage_args = next(a for stage, _script, a in calls if stage == "scene-audit")
    assert "--prospective" in stage_args
    assert str(runner.out / "audit-splits" / "scene_overlap.json") in stage_args
    assert str(runner.config_path) in stage_args, "provenance is checked against it"


@pytest.mark.parametrize("prospective", [True, False])
def test_an_audit_this_run_already_has_is_not_rebuilt(tmp_path, monkeypatch, prospective):
    """Rebuilding a retrospective audit as a prospective one would upgrade the
    claim without anyone deciding to, and rebuilding the prospective one on a
    resume would date it after the outcomes it is supposed to predate."""

    args = _constructible_run(tmp_path, monkeypatch)
    audit = _write_audit(args.outdir, prospective=prospective)
    runner = MODULE.Pilot(args)

    stages = [stage for stage, _script, _args in _opening_stages(runner, monkeypatch)]

    assert "scene-audit" not in stages
    assert runner.audit_path == audit


def test_a_scene_audit_stage_that_wrote_nothing_is_not_taken_for_success(tmp_path,
                                                                        monkeypatch):
    """Exit code 0 and no file is the shape of a bad --output. The next reader
    is the gradient stage, which would be handed the string "None"."""

    runner = MODULE.Pilot(_constructible_run(tmp_path, monkeypatch))
    with pytest.raises(FileNotFoundError):
        _opening_stages(runner, monkeypatch, the_stage_writes_the_audit=False)


def test_every_script_the_opening_stages_shell_is_a_frozen_source(tmp_path, monkeypatch):
    """run() re-digests SOURCES before each stage, so a script that is not in it
    can be edited mid-run. The audit is written once and read by two consumers
    that check the artifact's provenance but not its producer's."""

    runner = MODULE.Pilot(_constructible_run(tmp_path, monkeypatch))
    scripts = {script for _stage, script, _args in _opening_stages(runner, monkeypatch)}

    assert scripts, "the opening stages shell something"
    assert scripts <= set(MODULE.SOURCES), f"unfrozen: {sorted(scripts - set(MODULE.SOURCES))}"


def test_the_registered_audit_is_not_the_one_the_report_reads(tmp_path, monkeypatch):
    """Same file for both consumers is the thing this whole arrangement avoids."""

    args = _constructible_run(tmp_path, monkeypatch)
    audit = _write_audit(args.outdir, prospective=False)

    runner = MODULE.Pilot(args)
    assert runner.audit_path == audit
    assert "audit-splits" not in runner.audit_path.parts


# ------------------------------------------------------- arm B's --arms


def _blind_self_run(tmp_path, monkeypatch, arms=("naive", "blind_self")):
    """A constructible run for the arm the prereg pairs against naive."""

    args = _constructible_run(tmp_path, monkeypatch)
    args.arms = list(arms)
    _write_audit(args.outdir, prospective=False)
    return args


def test_each_arm_gets_its_own_card_whatever_the_arms_are(tmp_path, monkeypatch):
    """ARM_DEVICE was a literal keyed on naive and rfo_gold, so arm B raised
    KeyError on blind_self at the first evaluate -- after the round had trained."""

    runner = MODULE.Pilot(_blind_self_run(tmp_path, monkeypatch))
    assert runner.arm_device == {"naive": "cuda:0", "blind_self": "cuda:1"}
    assert len(set(runner.arm_device.values())) == 2, "two chains, two cards"


def test_the_cards_follow_the_order_the_arms_were_named(tmp_path, monkeypatch):
    runner = MODULE.Pilot(_blind_self_run(tmp_path, monkeypatch, ("blind_self", "naive")))
    assert runner.arm_device == {"blind_self": "cuda:0", "naive": "cuda:1"}


@pytest.mark.parametrize("arms", [
    ["naive"],
    ["naive", "blind_self", "rfo_gold"],
])
def test_an_arm_set_that_does_not_fit_the_cards_is_refused_at_construction(
        tmp_path, monkeypatch, arms):
    """zip() truncates in silence: three arms would have started three chains
    and quietly given the third one no card of its own."""

    with pytest.raises(ValueError, match="one per card"):
        MODULE.Pilot(_blind_self_run(tmp_path, monkeypatch, arms))


def test_a_repeated_arm_is_refused_rather_than_collapsed(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="repeats"):
        MODULE.Pilot(_blind_self_run(tmp_path, monkeypatch, ("naive", "naive")))


def _with_config_arms(args, arms):

    """Add training.arms to the fixture config, the way every replicate has it."""

    text = args.config.read_text(encoding="utf-8")
    listed = ", ".join(arms)
    args.config.write_text(
        text.replace("training:\n",
                     f"training:\n  arms: [{listed}]\n", 1),
        encoding="utf-8")
    return args


def test_a_config_arm_set_that_contradicts_the_flag_is_refused(tmp_path, monkeypatch):
    """training.arms is read by nothing. The five replicate configs all carry
    it, so a launch that forgets --arms trains the registered pairing while
    the config, the outdir name and every log agree it is arm B. The frozen
    manifest catches it on resume, which is a whole checkpoint too late."""

    args = _with_config_arms(_blind_self_run(tmp_path, monkeypatch), ["naive", "blind_self"])
    args.arms = ["naive", "rfo_gold"]

    with pytest.raises(ValueError, match="training.arms"):
        MODULE.Pilot(args)


def test_a_config_arm_set_that_agrees_with_the_flag_is_fine(tmp_path, monkeypatch):
    args = _with_config_arms(_blind_self_run(tmp_path, monkeypatch), ["naive", "blind_self"])

    assert MODULE.Pilot(args).arms == ["naive", "blind_self"]


def test_the_order_of_the_config_arm_set_matters_because_it_is_the_card_order(
        tmp_path, monkeypatch):
    """ARM_CARDS is positional, so [naive, blind_self] and [blind_self, naive]
    put different arms on the gen3 x4 card. Treating them as the same set
    would let the flag silently swap the cards."""

    args = _with_config_arms(_blind_self_run(tmp_path, monkeypatch), ["blind_self", "naive"])
    args.arms = ["naive", "blind_self"]

    with pytest.raises(ValueError, match="training.arms"):
        MODULE.Pilot(args)


def test_a_config_without_training_arms_still_runs(tmp_path, monkeypatch):
    """decoupling-main-20260908 has no such key and has to stay resumable."""

    assert MODULE.Pilot(_blind_self_run(tmp_path, monkeypatch)).arms == ["naive", "blind_self"]


def test_round_validation_asks_for_the_arms_this_run_trains(tmp_path, monkeypatch):
    """Against the module constant, a blind-self round reported naive and
    blind_self and was rejected for not being naive and rfo_gold."""

    runner = MODULE.Pilot(_blind_self_run(tmp_path, monkeypatch))
    report = {"round": 0, "optimizer_steps": 8, "mean_t2i_loss": 1,
              "mean_gradient_norm_before_clip": .2, "parameter_delta_l2": .1}
    MODULE.write_json(runner.out / "rounds" / "round-000" / "DONE.json",
                      {"paired": 3, "arms": [dict(report, arm=arm) for arm in runner.arms]})
    runner.validate_round(0)

    MODULE.write_json(runner.out / "rounds" / "round-000" / "DONE.json",
                      {"paired": 3, "arms": [dict(report, arm=arm)
                                             for arm in MODULE.ARMS]})
    with pytest.raises(RuntimeError, match="missing or duplicate"):
        runner.validate_round(0)


def test_the_training_stage_is_told_which_arms_to_train(tmp_path, monkeypatch):
    """v4_train.py's --arms defaults to the registered pairing, so a supervisor
    that does not pass it trains rfo_gold no matter what it was launched with."""

    runner = MODULE.Pilot(_blind_self_run(tmp_path, monkeypatch))
    commands = []
    monkeypatch.setattr(runner, "run",
                        lambda stage, env, script, args: commands.append((script, args)))
    for name in ("validate_round", "chains", "probe", "report", "state"):
        monkeypatch.setattr(runner, name, lambda *a, **k: None)
    runner.limit = 1
    runner.execute()

    train = [args for script, args in commands
             if script.endswith("v4_train.py") and args[0] == "train"]
    assert len(train) == 1, commands
    assert "--arms" in train[0], "without it the round trains the registered pairing"
    assert train[0][train[0].index("--arms") + 1:] == ["naive", "blind_self"]


def test_relaunching_with_different_arms_does_not_resume_the_run(tmp_path, monkeypatch):
    """The launch command is long and --arms defaults to the pairing, so the
    likely mistake is relaunching arm B without it. That would train rfo_gold
    into a directory holding blind_self checkpoints and show up only in a round
    report, days later."""

    args = _blind_self_run(tmp_path, monkeypatch)
    MODULE.Pilot(args).lock.close()

    args.arms = list(MODULE.ARMS)
    with pytest.raises(ValueError, match="not"):
        MODULE.Pilot(args)


def test_a_run_frozen_before_arms_existed_still_resumes(tmp_path, monkeypatch):
    """decoupling-main-20260908's manifest has no `arms` key and was the
    registered pairing. Requiring the key would refuse to resume it."""

    args = _blind_self_run(tmp_path, monkeypatch, MODULE.ARMS)
    first = MODULE.Pilot(args)
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    del manifest["arms"]
    first.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    first.lock.close()

    assert MODULE.Pilot(args).arms == list(MODULE.ARMS)


def test_a_prospectively_frozen_audit_is_accepted_where_the_report_reads_it(tmp_path,
                                                                           monkeypatch):
    """arm B can have one: it has not run, so nothing in its directory could
    have been seen when the audit was fixed. The main run could not, and the
    supervisor was written when only that case existed."""

    args = _constructible_run(tmp_path, monkeypatch)
    audit = _write_audit(args.outdir, prospective=True)
    assert MODULE.Pilot(args).audit_path == audit


def test_neither_audit_says_how_to_make_either(tmp_path, monkeypatch):
    """Reachable from the stage that just tried and left nothing behind, which
    is where knowing the flag is worth most."""

    args = _constructible_run(tmp_path, monkeypatch)
    runner = MODULE.Pilot(args)
    with pytest.raises(FileNotFoundError) as failure:
        runner.resolve_audit()
    assert "--prospective" in str(failure.value)
    assert "v4_scene_audit.py" in str(failure.value), "say which script writes it"


def test_both_audits_present_is_refused_rather_than_resolved(tmp_path, monkeypatch):
    """v4_decoupling_report.py reads audit-splits/ implicitly and this stage
    would be told the other one. Picking a winner here would leave the two
    halves of one run measured against two different exclusion sets."""

    args = _constructible_run(tmp_path, monkeypatch)
    _write_audit(args.outdir, prospective=True)
    _write_audit(args.outdir, prospective=False)
    with pytest.raises(ValueError, match="Keep one"):
        MODULE.Pilot(args)


@pytest.mark.parametrize("prospective", [True, False])
def test_an_audit_whose_flag_contradicts_its_location_is_refused(tmp_path, monkeypatch,
                                                                 prospective):
    """The report checks half of this, hours into the run. Flipping the flag by
    hand and moving the file are the two ways to get a retrospective exclusion
    set into an outcome curve, and both are one edit."""

    args = _constructible_run(tmp_path, monkeypatch)
    path = _write_audit(args.outdir, prospective=prospective)
    path.write_text(json.dumps(
        {"created_before_any_outcome_evaluation_artifact": not prospective}),
        encoding="utf-8")
    with pytest.raises(ValueError, match="does not match where it is"):
        MODULE.Pilot(args)


def test_the_gradient_stage_is_told_the_audit_rather_than_left_to_default(tmp_path,
                                                                         monkeypatch):
    """The default is audit-splits/, so a run with a retrospective audit would
    get no exclusions at all and say nothing about it."""

    args = _constructible_run(tmp_path, monkeypatch)
    audit = _write_audit(args.outdir, prospective=False)
    runner = MODULE.Pilot(args)
    calls = []
    monkeypatch.setattr(runner, "run",
                        lambda stage, env, script, args_: calls.append((script, args_)))
    runner.report(0)

    sensitivity = [a for script, a in calls if script.endswith("v4_gradient_sensitivity.py")]
    assert len(sensitivity) == 1
    assert sensitivity[0][sensitivity[0].index("--audit") + 1] == str(audit)
