"""The launch gate that reads state.json, and the three ways a run can end.

This gate decides whether 400 GPU-hours of replicates start. It runs
unattended at whatever hour the main run finishes, and its output is read by
someone who was asleep, so a wrong *explanation* costs nearly as much as a
wrong verdict: an operator who is told the 96 h line fired when it did not
goes looking in the wrong place, and an operator told a dead run is "still
running" goes back to bed.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "v4_e3_launch_preflight", ROOT / "scripts" / "v4_e3_launch_preflight.py")
preflight = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(preflight)


def write_state(tmp_path: Path, **fields) -> Path:
    """A state.json with the keys run_decoupling_pilot.py actually writes."""

    path = tmp_path / "state.json"
    payload = {"supervisor_pid": 4242, "updated_unix": 1788977661.0,
               "started_unix": 1788877993.0, "elapsed_hours": 27.7, "through_round": 11}
    payload.update(fields)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_a_finished_run_passes(tmp_path):
    path = write_state(tmp_path, status="pilot_complete", completed_rounds=11,
                       stage="review_results")
    passed, message = preflight.gate_main_run_finished(path)
    assert passed is True
    assert "11 rounds" in message


def test_a_bounded_run_is_not_called_a_wall_clock_stop(tmp_path):
    """canary_complete means --through-round, not the 96 h line.

    run() raises RuntimeError when the deadline passes, and that exception
    leaves the round loop above the terminal state() call, so the 96 h case
    never writes canary_complete at all -- it writes nothing. The status is
    reached only by `self.limit != config["training"]["rounds"]`, which is
    the --through-round flag. STATUS 43.15 recorded this correctly for the
    2026-09-06 pilot resume; the first draft of this gate did not.
    """

    path = write_state(tmp_path, status="canary_complete", completed_rounds=3,
                       stage="review_results")
    passed, message = preflight.gate_main_run_finished(path)
    assert passed is False
    assert "through-round" in message
    lowered = message.lower()
    assert "96" not in lowered and "wall" not in lowered and "stop line" not in lowered


def test_a_run_whose_supervisor_is_gone_is_not_reported_as_still_running(tmp_path,
                                                                        monkeypatch):
    """This is the shape the 96 h line really leaves behind.

    Nothing rewrites state.json after run() raises, so the file says running
    for as long as the disk lasts. Reading that as "still going" is how an
    operator waits all night for a run that died before midnight.
    """

    monkeypatch.setattr(preflight, "_supervisor_alive", lambda pid: False)
    path = write_state(tmp_path, status="running", stage="naive.step-00024.detect.qwen3vl")
    passed, message = preflight.gate_main_run_finished(path)
    assert passed is False
    assert "4242" in message
    assert "without writing a terminal status" in message


def test_a_live_run_says_which_stage_it_is_on(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "_supervisor_alive", lambda pid: True)
    path = write_state(tmp_path, status="running", stage="naive.step-00024.detect.qwen3vl")
    passed, message = preflight.gate_main_run_finished(path)
    assert passed is False
    assert "naive.step-00024.detect.qwen3vl" in message
    assert "still running" in message


def test_an_unknown_liveness_answer_falls_back_to_the_ordinary_wording(tmp_path,
                                                                      monkeypatch):
    """None is "cannot tell", and it must not read as "dead"."""

    monkeypatch.setattr(preflight, "_supervisor_alive", lambda pid: None)
    path = write_state(tmp_path, status="running", stage="round-003.train")
    passed, message = preflight.gate_main_run_finished(path)
    assert passed is False
    assert "still running" in message


def test_liveness_never_uses_os_kill():
    """os.kill(pid, 0) is not a liveness probe on Windows.

    CPython implements os.kill on Windows by opening the process and calling
    TerminateProcess for any signal that is not a console-control event, so
    the portable idiom would kill the multi-day run this gate protects. The
    hazard is invisible at the call site, which is why it is pinned here.
    """

    tree = ast.parse((ROOT / "scripts" / "v4_e3_launch_preflight.py").read_text(
        encoding="utf-8"))
    # Parsed, not grepped: the docstring that explains the hazard says
    # os.kill, and a substring search cannot tell an explanation from a call.
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
             and node.func.attr == "kill"]
    assert calls == [], f"os.kill called at line(s) {[node.lineno for node in calls]}"
    imported = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                and node.module == "os" and any(a.name == "kill" for a in node.names)]
    assert imported == []


@pytest.mark.parametrize("fields", [
    {"status": "pilot_complete", "completed_rounds": 10},
    {"status": "pilot_complete"},
    {"status": "canary_complete", "completed_rounds": 11},
    {"status": "running", "completed_rounds": 11},
    {"status": "failed", "completed_rounds": 11},
    {"status": None, "completed_rounds": 11},
])
def test_only_eleven_completed_rounds_under_pilot_complete_passes(tmp_path, monkeypatch,
                                                                  fields):
    monkeypatch.setattr(preflight, "_supervisor_alive", lambda pid: True)
    passed, _ = preflight.gate_main_run_finished(write_state(tmp_path, **fields))
    assert passed is False


def test_an_absent_state_file_is_refused_by_path(tmp_path):
    passed, message = preflight.gate_main_run_finished(tmp_path / "state.json")
    assert passed is False
    assert "state.json" in message


# --- gate_card_schedule_decided -------------------------------------------
#
# This gate had no test, which is how it came to read a file and a key that
# nothing writes. `card_benchmark.py` lives in the arm B branch's review
# packet and this preflight lives here; until the merge rehearsal of
# 2026-09-09 the two had never been in one tree, so no run of anything could
# have noticed that the gate wanted `verdict.json["serialise_detect"]` and
# the benchmark wrote `card_benchmark.json["verdict"]["adopt_serial"]`.
#
# The fixture below is therefore built from the benchmark's own payload keys
# rather than from the gate's expectations, which is the only ordering that
# would have caught the original defect.

PACKET = "review-packets/card-scheduling-20260909"


def write_benchmark(root: Path, **fields) -> Path:
    """The shape `card_benchmark.py:main` writes, minus the timing detail."""

    payload = {"images": 64, "manifest": "runs/v4/.../manifest.jsonl",
               "no_load": True, "margin": 0.90,
               "results": [{"detector": "internvl"}, {"detector": "qwen3vl"}],
               "verdict": {"adopt_serial": True,
                           "reason": "serial clears the 10% margin on every detector"}}
    payload.update(fields)
    packet = root / PACKET
    packet.mkdir(parents=True, exist_ok=True)
    path = packet / "card_benchmark.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_the_gate_reads_the_file_the_benchmark_writes(tmp_path):
    write_benchmark(tmp_path)
    passed, message = preflight.gate_card_schedule_decided(tmp_path)
    assert passed is True
    assert "adopt_serial=True" in message
    # The reason travels with the decision: an operator reading this at four
    # in the morning should not have to open the JSON to learn which way the
    # criterion fell or why.
    assert "every detector" in message


def test_a_verdict_against_serialising_is_still_a_decision(tmp_path):
    """`adopt_serial: False` passes the gate. It is a decision, not a failure.

    The gate exists so the criterion cannot be skipped, not so it comes out
    a particular way -- the registered tie-break (keep the per-arm pinning)
    is one of its two legitimate answers.
    """

    write_benchmark(tmp_path, verdict={"adopt_serial": False,
                                       "reason": "serial clears the margin on "
                                                 "['internvl'], not on all"})
    passed, message = preflight.gate_card_schedule_decided(tmp_path)
    assert passed is True
    assert "adopt_serial=False" in message


def test_a_missing_benchmark_names_the_file_and_the_outdir(tmp_path):
    passed, message = preflight.gate_card_schedule_decided(tmp_path)
    assert passed is False
    # Naming the outdir is the point: the benchmark takes --outdir as a
    # required argument and will happily write a correct answer somewhere
    # this gate does not look.
    assert "card_benchmark.json" in message
    assert "--outdir" in message and PACKET.replace("/", "\\") in message.replace("/", "\\")


def test_the_old_verdict_json_no_longer_satisfies_the_gate(tmp_path):
    """A hand-written file with the old key must not pass.

    The old key was invented by this gate and appears in no pre-registration
    and in no other script. Leaving it as an accepted alternative would keep
    open exactly the path the defect created: a decision typed by hand at the
    moment the measurement is inconvenient.
    """

    packet = tmp_path / PACKET
    packet.mkdir(parents=True)
    (packet / "verdict.json").write_text(json.dumps({"serialise_detect": True}),
                                         encoding="utf-8")
    passed, message = preflight.gate_card_schedule_decided(tmp_path)
    assert passed is False
    assert "card_benchmark.json" in message


def test_a_benchmark_with_no_decision_is_refused(tmp_path):
    write_benchmark(tmp_path, verdict={})
    passed, message = preflight.gate_card_schedule_decided(tmp_path)
    assert passed is False
    assert "adopt_serial" in message


def test_a_busy_card_measurement_is_refused(tmp_path):
    """`--allow-busy-cards` records `no_load: false` and the gate must read it.

    Section 0.3 is a no-load benchmark; the numbers in it that are already
    known to be contaminated by another process are the reason it exists.
    """

    write_benchmark(tmp_path, no_load=False)
    passed, message = preflight.gate_card_schedule_decided(tmp_path)
    assert passed is False
    assert "no_load" in message


def test_a_margin_other_than_the_registered_one_is_refused(tmp_path):
    write_benchmark(tmp_path, margin=0.95)
    passed, message = preflight.gate_card_schedule_decided(tmp_path)
    assert passed is False
    assert "0.9" in message


def test_the_registered_margin_is_the_number_section_0_3_fixed():
    # 0.90, fixed before the measurement. The gate does not re-derive the
    # criterion -- card_benchmark.py owns that -- but it does refuse a report
    # produced under a different one.
    assert preflight.CARD_MARGIN == 0.90


def test_a_report_with_no_no_load_key_is_refused(tmp_path):
    """Absent is not the same as true, and the safe default is refuse.

    `card_benchmark.py` always writes the key, so a report without it did
    not come from the benchmark as it stands -- an older copy, a hand-edited
    file, or something else entirely. Every one of those is a reason to look
    rather than to assume the cards were empty.
    """

    payload = json.loads(write_benchmark(tmp_path).read_text(encoding="utf-8"))
    del payload["no_load"]
    (tmp_path / PACKET / "card_benchmark.json").write_text(json.dumps(payload),
                                                           encoding="utf-8")
    passed, message = preflight.gate_card_schedule_decided(tmp_path)
    assert passed is False
    assert "no_load" in message


# --- the three gates that had no test at all -------------------------------
#
# Six gates stand between the main run finishing and 400 GPU-hours starting.
# Two were tested. The card-schedule gate above shows what the other four were
# worth: it read a file and a key nothing writes, and no run of anything could
# have noticed. What follows exercises the *refusal* side of the remaining
# four, because a gate that cannot refuse is worse than no gate -- it is read,
# at whatever hour the run finishes, as permission.
#
# Fixtures are built from what the writer produces: the five real configs, and
# the arm B branch's real files. A gate checked against a fixture built from
# its own expectations agrees with itself.


def _load(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _dump(path: Path, payload: dict) -> None:
    import yaml

    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


REPLICATES = [ROOT / f"configs/v4_e3_replicate_s{seed}.yaml"
              for seed in preflight.REGISTERED_SEEDS]


def write_replicates(root: Path, mutate=None) -> Path:
    """The five real configs, re-emitted into `root` with one field changed.

    Returning None from `mutate` drops that config, which is how the missing
    file case is built without inventing what a config looks like.
    """

    (root / "configs").mkdir(parents=True, exist_ok=True)
    for seed, source in zip(preflight.REGISTERED_SEEDS, REPLICATES):
        payload = _load(source)
        if mutate is not None:
            payload = mutate(seed, payload)
        if payload is not None:
            _dump(root / "configs" / source.name, payload)
    return root


def test_the_five_real_configs_satisfy_the_gate():
    passed, message = preflight.gate_configs(ROOT)
    assert passed, message
    assert "one split" in message


def test_the_fixture_itself_passes_before_anything_is_mutated(tmp_path):
    """Otherwise every refusal below could be refusing the round trip."""

    passed, message = preflight.gate_configs(write_replicates(tmp_path))
    assert passed, message


def test_a_training_seed_that_is_not_its_filename_is_refused(tmp_path):
    """20260908 is the main run's seed, deliberately skipped by deviation 9.

    A replicate carrying it would look like a sixth measurement of the study
    and be a second measurement of the run it is supposed to replicate.
    """

    def collide(seed, payload):
        if seed == preflight.REGISTERED_SEEDS[2]:
            payload["training"]["seed"] = 20260908
        return payload

    passed, message = preflight.gate_configs(write_replicates(tmp_path, collide))
    assert not passed
    assert "training.seed is 20260908" in message


def test_a_missing_config_is_refused_by_name(tmp_path):
    def drop(seed, payload):
        return None if seed == preflight.REGISTERED_SEEDS[0] else payload

    passed, message = preflight.gate_configs(write_replicates(tmp_path, drop))
    assert not passed
    assert "v4_e3_replicate_s20260906.yaml is missing" in message


def test_the_plural_seeds_key_is_refused_as_dead_config(tmp_path):
    """`training.seeds` is read by the v2.x pipeline and never by v4.

    Setting it is a request that is granted silently and ignored completely,
    which is the one failure mode no downstream artifact records.
    """

    def revive(seed, payload):
        payload["training"]["seeds"] = [seed]
        return payload

    passed, message = preflight.gate_configs(write_replicates(tmp_path, revive))
    assert not passed
    assert "dead config in v4" in message


def test_an_arm_set_other_than_the_registered_pair_is_refused(tmp_path):
    """The exact confusion the supervisor's own guard was added for.

    naive+rfo_gold under a replicate's name trains the registered pairing and
    writes it into `e3-s20260906`; config, logs and directory name all agree,
    and only the frozen manifest disagrees, on resume.
    """

    def swap(seed, payload):
        payload["training"]["arms"] = ["naive", "rfo_gold"]
        return payload

    passed, message = preflight.gate_configs(write_replicates(tmp_path, swap))
    assert not passed
    assert "training.arms is ['naive', 'rfo_gold']" in message


def test_a_second_partition_seed_is_refused(tmp_path):
    """The failure that produces five runs nothing can pair.

    Top-level `seed` fixes the split, the split digest and the evaluation
    latents. One config carrying a different one is not a variant of the study
    but a different study, and endpoint 1 pairs over (prompt, draw) across
    replicates.
    """

    def drift(seed, payload):
        if seed == preflight.REGISTERED_SEEDS[4]:
            payload["seed"] = 20260907
        return payload

    passed, message = preflight.gate_configs(write_replicates(tmp_path, drift))
    assert not passed
    assert "top-level seed differs across configs" in message
    assert "nothing would pair" in message


def test_a_different_split_input_is_refused_even_at_one_seed(tmp_path):
    """Same seed, different pool sizes, so the same digest over a different
    corpus. The partition seed alone does not make two splits the same.
    """

    def shrink(seed, payload):
        if seed == preflight.REGISTERED_SEEDS[1]:
            payload["data"]["local_outcome"] = 63
        return payload

    passed, message = preflight.gate_configs(write_replicates(tmp_path, shrink))
    assert not passed
    assert "split inputs differ across configs" in message


# --- gate_cards_free -------------------------------------------------------


class _Smi:
    """Stands in for nvidia-smi. `rows` is its stdout, or an exception."""

    def __init__(self, rows):
        self.rows = rows

    def __call__(self, *args, **kwargs):
        if isinstance(self.rows, Exception):
            raise self.rows
        return type("R", (), {"stdout": self.rows})()


def _cards(monkeypatch, rows):
    monkeypatch.setattr(preflight.subprocess, "run", _Smi(rows))
    return preflight.gate_cards_free()


def test_two_idle_cards_pass(monkeypatch):
    passed, message = _cards(monkeypatch, "0, 4\n1, 11\n")
    assert passed
    assert message == "both cards idle"


def test_a_busy_card_is_refused_and_says_not_to_preempt(monkeypatch):
    """The machine carries jobs that are not this project's.

    The wording matters as much as the verdict: this runs unattended and the
    operator reading it has to be told not to clear the card.
    """

    passed, message = _cards(monkeypatch, "0, 16965\n1, 11\n")
    assert not passed
    assert "cuda:0 holds 16965 MiB" in message
    assert "do not preempt" in message


def test_one_visible_card_is_refused(monkeypatch):
    passed, message = _cards(monkeypatch, "0, 4\n")
    assert not passed
    assert "only 1 card visible" in message


def test_an_unreadable_nvidia_smi_is_not_treated_as_idle(monkeypatch):
    """Cannot tell must refuse. The alternative is a gate that opens widest
    exactly when the machine has stopped answering questions about itself.
    """

    passed, message = _cards(monkeypatch, OSError("nvidia-smi not found"))
    assert not passed
    assert "could not read nvidia-smi" in message


def test_no_card_at_all_is_refused_rather_than_counted_as_idle(monkeypatch):
    passed, message = _cards(monkeypatch, "\n")
    assert not passed
    assert "only 0 card visible" in message


def test_the_idle_threshold_is_the_registered_one():
    assert preflight.IDLE_MIB == 500


# --- gate_model_root -------------------------------------------------------


def test_an_unset_model_root_is_refused(monkeypatch):
    monkeypatch.delenv("SELFSIGHT_MODEL_ROOT", raising=False)
    passed, message = preflight.gate_model_root()
    assert not passed
    assert message == "SELFSIGHT_MODEL_ROOT is unset"


def test_a_model_root_that_is_not_a_directory_is_refused(monkeypatch, tmp_path):
    target = tmp_path / "weights"
    target.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv("SELFSIGHT_MODEL_ROOT", str(target))
    passed, message = preflight.gate_model_root()
    assert not passed
    assert "is not a directory" in message


def test_an_empty_model_root_is_refused_rather_than_accepted(monkeypatch):
    """An exported-but-empty variable is the shape a failed profile leaves."""

    monkeypatch.setenv("SELFSIGHT_MODEL_ROOT", "")
    passed, message = preflight.gate_model_root()
    assert not passed
    assert message == "SELFSIGHT_MODEL_ROOT is unset"


def test_a_real_directory_passes(monkeypatch, tmp_path):
    monkeypatch.setenv("SELFSIGHT_MODEL_ROOT", str(tmp_path))
    passed, message = preflight.gate_model_root()
    assert passed
    assert message == str(tmp_path)


# --- gate_merge_landed -----------------------------------------------------
#
# This gate names six things the merge has to provide. Nothing had checked
# that the merge provides them -- the gate lives here and the code it looks
# for lives on the other branch, which is the same separation that let the
# card-schedule gate read an invented filename.

ARM_B = "staging/arm-b-merged"
MERGE_FILES = ("scripts/v4_train.py", "scripts/run_decoupling_pilot.py",
               "scripts/v4_verify_replicates.py")
MARKERS = ("def training_seed(", "def partition_seed(", '"--arms"',
           "self.arm_device", "registered_arms")


def _branch_files():
    import subprocess

    out = {}
    for name in MERGE_FILES:
        result = subprocess.run(["git", "show", f"{ARM_B}:{name}"], cwd=ROOT,
                                capture_output=True, check=False)
        if result.returncode != 0:
            return None
        out[name] = result.stdout.decode("utf-8")
    return out


def _plant(root: Path, files: dict) -> Path:
    for name, body in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    return root


def test_the_arm_b_branch_really_provides_what_the_gate_asks_for(tmp_path):
    """The writer-first half. Skipped once the branch is gone."""

    files = _branch_files()
    if files is None:
        pytest.skip(f"{ARM_B} is not in this checkout")
    passed, message = preflight.gate_merge_landed(_plant(tmp_path, files))
    assert passed, message
    assert message == "merged"


@pytest.mark.parametrize("marker", MARKERS)
def test_removing_any_one_marker_is_caught(tmp_path, marker):
    files = _branch_files()
    if files is None:
        pytest.skip(f"{ARM_B} is not in this checkout")
    hits = sum(body.count(marker) for body in files.values())
    assert hits, f"{marker!r} is not in the merged files at all"
    files = {name: body.replace(marker, "REMOVED") for name, body in files.items()}
    passed, message = preflight.gate_merge_landed(_plant(tmp_path, files))
    assert not passed, f"removing {marker!r} left the gate green"
    assert message != "merged"


def test_an_absent_verify_script_is_caught(tmp_path):
    files = _branch_files()
    if files is None:
        pytest.skip(f"{ARM_B} is not in this checkout")
    del files["scripts/v4_verify_replicates.py"]
    passed, message = preflight.gate_merge_landed(_plant(tmp_path, files))
    assert not passed
    assert "v4_verify_replicates.py is absent" in message


def test_the_unmerged_side_would_be_refused_for_every_reason(tmp_path):
    """Five reasons, not one -- the sixth needs the file to be absent, not
    empty. An operator woken at 04:00 gets told what is missing rather
    than that something is.
    """

    _plant(tmp_path, {name: "" for name in MERGE_FILES})
    passed, message = preflight.gate_merge_landed(tmp_path)
    assert not passed
    assert message.count(";") == 4
    assert "training_seed" in message
    assert "the arm set is still hardcoded" in message


# --- gate_split_check_landed -----------------------------------------------
#
# The merge supplies the verifier at version 1; version 2 is a patch applied
# after it. Same separation as above, and the same danger: a gate that asks
# for something nobody provides, or names a file that does not exist.

SPLIT_PACKET = "review-packets/replicate-split-identity-20260910"
PATCHED = Path(ROOT) / SPLIT_PACKET / "v4_verify_replicates.patched.py"


def test_the_packet_really_provides_what_the_split_gate_asks_for(tmp_path):
    """The writer-first half, and it needs no branch: the provider is here."""

    assert PATCHED.exists(), f"{PATCHED} is what the gate is written against"
    _plant(tmp_path, {"scripts/v4_verify_replicates.py":
                      PATCHED.read_text(encoding="utf-8")})
    passed, message = preflight.gate_split_check_landed(tmp_path)
    assert passed, message
    assert message == "version 2, --main required"


def test_the_patch_the_gate_names_is_a_file_that_exists():
    """An operator reading the refusal has to be able to run what it says.

    The card-schedule gate once named a filename nobody had written. A gate
    whose remedy does not exist is worse than no gate: it is read as done.
    """

    assert (Path(ROOT) / preflight.SPLIT_PATCH).exists()
    assert preflight.SPLIT_CHECK == "scripts/v4_verify_replicates.py"


def test_the_merged_version_of_the_verifier_is_refused(tmp_path):
    """Version 1 is what the merge lands, and it is not enough on its own."""

    files = _branch_files()
    if files is None:
        pytest.skip(f"{ARM_B} is not in this checkout")
    _plant(tmp_path, {"scripts/v4_verify_replicates.py":
                      files["scripts/v4_verify_replicates.py"]})
    passed, message = preflight.gate_split_check_landed(tmp_path)
    assert not passed
    assert "is not version 2" in message
    assert preflight.SPLIT_PATCH in message


def test_an_absent_verifier_is_sent_to_the_merge_gate(tmp_path):
    """Two gates fail on one cause; only one of them should claim to explain it."""

    passed, message = preflight.gate_split_check_landed(tmp_path)
    assert not passed
    assert "see the arm B merge gate first" in message


def test_dropping_required_on_main_is_caught(tmp_path):
    """`verify()` keeps `main` optional so the older tests still call it.

    The command line is the only thing that makes that safe, so this gate has
    to notice if someone relaxes it.
    """

    source = PATCHED.read_text(encoding="utf-8").replace("required=True", "required=False")
    _plant(tmp_path, {"scripts/v4_verify_replicates.py": source})
    passed, message = preflight.gate_split_check_landed(tmp_path)
    assert not passed
    assert "--main is not required" in message


def test_dropping_the_main_comparison_is_caught(tmp_path):
    source = PATCHED.read_text(encoding="utf-8").replace("split_matches_main", "gone")
    _plant(tmp_path, {"scripts/v4_verify_replicates.py": source})
    passed, message = preflight.gate_split_check_landed(tmp_path)
    assert not passed
    assert "not the main run's" not in message  # that is the verifier's wording
    assert "compares the replicates' split to the main run's" in message


def test_the_gate_is_in_the_registry_under_its_own_name():
    """A gate that is never called is the failure family this repo keeps hitting."""

    assert preflight.GATES["replicate split check"] is preflight.gate_split_check_landed
