"""The scene audit must stay readable by the analysis it exists to feed."""
import importlib.util
import json
from pathlib import Path

import pytest

from selfsight.utils.hashing import sha256_json
from selfsight.v4.spec import SceneSpec, SpecObject
from selfsight.v4.train import ReplayExample, TrainingCorpus

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = _load("v4_scene_audit")
SENSITIVITY = _load("v4_gradient_sensitivity")


def spec(spec_id: str, prompt: str, objects: list[tuple[str, str | None, int]]) -> SceneSpec:
    return SceneSpec(spec_id=spec_id, prompt=prompt, objects=tuple(
        SpecObject.from_dict({"object": noun, "color": color, "count": count})
        for noun, color, count in objects))


# Six specs over four scenes. v4a-002 restates v4a-001 with a different word
# order and a synonym, and v4a-005 restates it again, so the probe bank prompt
# that shares that scene has to come out excluded.
SPECS = {
    "v4a-001": spec("v4a-001", "two red apples", [("apple", "red", 2)]),
    "v4a-002": spec("v4a-002", "a pair of red apples", [("apple", "red", 1), ("apples", "red", 1)]),
    "v4a-003": spec("v4a-003", "one blue mug", [("mug", "blue", 1)]),
    "v4a-004": spec("v4a-004", "one green leaf", [("leaf", "green", 1)]),
    "v4b-005": spec("v4b-005", "red apples, two of them", [("apples", "red", 2)]),
    "v4b-006": spec("v4b-006", "one yellow lemon", [("lemon", "yellow", 1)]),
    # Four more, used only by the prospective tests: two that restate a trained
    # scene, and two that restate each other without matching anything trained.
    "v4b-007": spec("v4b-007", "a lemon and a leaf", [("lemon", "yellow", 1),
                                                      ("leaf", "green", 1)]),
    "v4b-008": spec("v4b-008", "one green leaf beside one yellow lemon",
                    [("leaf", "green", 1), ("lemon", "yellow", 1)]),
    "v4b-009": spec("v4b-009", "a couple of red apples", [("apple", "red", 2)]),
    "v4b-010": spec("v4b-010", "a single blue mug", [("mug", "blue", 1)]),
}
# Deliberately not in sorted order: the audit sorts, and a reader of the file
# has to be able to tell that from the file rather than from the split.
PROSPECTIVE_OUTCOME = ["v4b-010", "v4b-006", "v4b-009", "v4b-008", "v4b-007"]
TRAIN = ["v4a-001", "v4a-002", "v4a-003", "v4a-004"]
CONFIG = "seed: 7\ntraining:\n  rounds: 2\n  prompts_per_round: 2\n  candidate_k: 2\n"


def build_run(tmp_path: Path, *, train=TRAIN, rounds: int = 2) -> tuple[Path, Path]:
    """A run directory holding only what the audit reads: a split and a bank."""
    run = tmp_path / "run"
    (run / "probe-bank").mkdir(parents=True)
    split = {"created": "2026-01-01T00:00:00Z", "digest": "d" * 64, "seed": 7,
             "runs": ["runs/v4/fake"], "train": list(train),
             "outcome": ["v4b-006"], "probe": ["v4b-005"]}
    (run / "split.json").write_text(json.dumps(split), encoding="utf-8")
    content = {"schema_version": "test", "split_sha256": "s" * 64,
               "pools": [{"spec_id": "v4b-005"}, {"spec_id": "v4b-006"}]}
    bank = dict(content, fingerprint=sha256_json(content))
    (run / "probe-bank" / "bank.json").write_text(json.dumps(bank), encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(CONFIG.replace("rounds: 2", f"rounds: {rounds}"), encoding="utf-8")
    return run, config


def build_prospective_run(tmp_path: Path) -> tuple[Path, Path]:
    """`build_run` with an outcome population worth auditing.

    Its one outcome prompt cannot show a duplicate scene group, an overlap
    ordering, or the difference between one exposed prompt and two.
    """
    run, config = build_run(tmp_path)
    split_path = run / "split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split["outcome"] = list(PROSPECTIVE_OUTCOME)
    split_path.write_text(json.dumps(split), encoding="utf-8")
    return run, config


@pytest.fixture
def corpus(monkeypatch):
    replay = (ReplayExample(image_path="a.png", question="q", answer="yes",
                            sample_id="s1", prompt_id="v4a-001"),)
    monkeypatch.setattr(AUDIT, "load_training_corpus",
                        lambda runs: TrainingCorpus(specs=dict(SPECS), replay=replay))


# Taken from runs/v4/decoupling-pilot-20260906's published audit. The pilot's
# file has no generator -- this script was written to replace it -- so these
# pin the key to what the frozen artifact already claims rather than to a
# reimplementation of it.
@pytest.mark.parametrize("scene,digest", [
    ([["leaf", "green", 1], ["lemon", "yellow", 1], ["tomato", "red", 1]],
     "04c6b990318f8c5bd1457ec6adb7d68994f79d8b7cf99b3d0e7d24e56078ae65"),
    ([["apple", "green", 1], ["mug", "blue", 1], ["mug", "red", 1]],
     "1ee5762b405610f4286f11550f030010ea7c43d35be5f72c045f2fc0a78a2d43"),
    ([["apple", "red", 1], ["bottle", "blue", 2]],
     "472a35116a2cd1b3e8d049463c0eed5a393c3672cb209734ebf30c46bbcb1651"),
])
def test_the_scene_key_still_hashes_the_way_the_frozen_pilot_audit_says(scene, digest):
    # Declared backwards on purpose. A spec lists its objects in whatever order
    # the prompt reads, so the key has to sort them or two spellings of one
    # scene hash differently and never cluster.
    built = AUDIT.canonical_scene(spec("x", "x", [(noun, color, count)
                                                  for noun, color, count in reversed(scene)]))
    assert built == scene
    assert sha256_json(built) == digest


def test_wording_that_the_correctness_instrument_cannot_see_is_one_scene():
    """Plural, synonym and order are all invisible to object/color/count."""
    keys = {sha256_json(AUDIT.canonical_scene(SPECS[spec_id]))
            for spec_id in ("v4a-001", "v4a-002", "v4b-005")}
    assert len(keys) == 1, "these three prompts describe the same scored scene"


def test_a_different_count_is_a_different_scene():
    one = AUDIT.canonical_scene(spec("a", "one red apple", [("apple", "red", 1)]))
    two = AUDIT.canonical_scene(spec("b", "two red apples", [("apple", "red", 2)]))
    assert sha256_json(one) != sha256_json(two)


def test_clusters_are_ordered_by_key_so_reruns_produce_the_same_bytes():
    # Two specs with different scenes, so the order they are first seen in
    # differs between the two calls and only sorting can reconcile them.
    forward = AUDIT.cluster(SPECS, ["v4a-001", "v4a-003"])
    backward = AUDIT.cluster(SPECS, ["v4a-003", "v4a-001"])
    assert forward == backward
    assert [group["scene_sha256"] for group in forward] == sorted(
        group["scene_sha256"] for group in forward)


def test_prompts_sharing_a_scene_land_in_one_cluster():
    # v4b-005 first, so the ids inside the group come out sorted rather than
    # in the order they were handed over.
    groups = AUDIT.cluster(SPECS, ["v4b-005", "v4a-003", "v4a-001"])
    assert [group["spec_ids"] for group in groups].count(["v4a-001", "v4b-005"]) == 1


def test_the_audit_it_writes_is_one_the_sensitivity_analysis_can_read(tmp_path, corpus):
    """The contract between the two files, checked by running it.

    read_partition re-derives every hash in the audit and refuses it on any
    disagreement, so this passing means the writer and the reader agree about
    the scene key, the cluster cover and the exclusion rows.
    """
    run, config = build_run(tmp_path)
    audit_path = tmp_path / "scene_overlap.json"
    audit_path.write_text(json.dumps(AUDIT.build_audit(run, config)), encoding="utf-8")

    audit, _bank, scenes, excluded = SENSITIVITY.read_partition(run, audit_path)
    assert set(scenes) == {"v4b-005", "v4b-006"}
    assert excluded == {"v4b-005"}, "v4b-005 restates a scheduled training scene"
    assert audit["canonical_scene_key_definition"]["algorithm"]


def test_it_never_claims_to_have_been_frozen_before_the_outcomes(tmp_path, corpus):
    """The flag is what stops v4_decoupling_report.py accepting this file."""
    run, config = build_run(tmp_path)
    audit = AUDIT.build_audit(run, config)
    assert audit["created_before_any_outcome_evaluation_artifact"] is False


def test_it_carries_nothing_that_could_choose_an_outcome_subset(tmp_path, corpus):
    """Defeating the flag must not be enough to get a retrospective subset.

    v4_decoupling_report.py reads within_outcome_clusters and
    scene_disjoint_outcome_sensitivity. Leaving them out means the worst case
    is a loud KeyError rather than a quietly retrospective outcome curve.
    """
    run, config = build_run(tmp_path)
    text = json.dumps(AUDIT.build_audit(run, config))
    assert "within_outcome_clusters" not in text
    assert "scene_disjoint" not in text


def test_a_schedule_that_leaves_training_prompts_unreached_is_refused(tmp_path, corpus):
    """Exposure is 'the whole train partition' only while the schedule covers it.

    At one round the audit would call two of the four training prompts unseen
    and stop excluding the probe prompts that match them, which understates the
    contamination instead of overstating it.
    """
    run, config = build_run(tmp_path, rounds=1)
    with pytest.raises(ValueError, match="whole train partition"):
        AUDIT.build_audit(run, config)


def test_a_bank_whose_contents_do_not_match_its_fingerprint_is_refused(tmp_path, corpus):
    run, config = build_run(tmp_path)
    bank_path = run / "probe-bank" / "bank.json"
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    bank["pools"].append({"spec_id": "v4a-004"})
    bank_path.write_text(json.dumps(bank), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        AUDIT.build_audit(run, config)


def test_a_reconstruction_that_contradicts_an_executed_round_is_refused(tmp_path, corpus):
    """The schedule is recomputed, so it has to be checked where it can be.

    A run that already trained a round holds the prompts it actually chose. If
    the recomputation disagrees, every round assignment in the audit is suspect
    and the exposure set may be wrong too.
    """
    run, config = build_run(tmp_path)
    scheduled = AUDIT.planned_exposure({"seed": 7, "training": {
        "rounds": 2, "prompts_per_round": 2, "candidate_k": 2}},
        {"train": TRAIN})
    # Round 0 as reconstructed, with one prompt swapped for one the schedule
    # puts in round 1. Derived rather than written out, so the test cannot
    # quietly become a no-op when the shuffle changes.
    first = sorted(prompt for prompt, index in scheduled.items() if index == 0)
    later = sorted(prompt for prompt, index in scheduled.items() if index == 1)
    executed = [first[0], later[0]]

    selection = run / "rounds" / "round-000" / "selection.json"
    selection.parent.mkdir(parents=True)
    selection.write_text(json.dumps({
        "round": 0, "decisions": {"naive": [{"prompt_id": p} for p in executed]},
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="not the round the run executed") as failure:
        AUDIT.build_audit(run, config)
    assert later[0] in str(failure.value), "name the prompt that disagrees"


def test_an_executed_round_that_matches_is_recorded_as_checked(tmp_path, corpus):
    run, config = build_run(tmp_path)
    scheduled = AUDIT.planned_exposure({"seed": 7, "training": {
        "rounds": 2, "prompts_per_round": 2, "candidate_k": 2}},
        {"train": TRAIN})
    first = sorted(prompt for prompt, index in scheduled.items() if index == 0)
    selection = run / "rounds" / "round-000" / "selection.json"
    selection.parent.mkdir(parents=True)
    selection.write_text(json.dumps({
        "round": 0, "decisions": {"naive": [{"prompt_id": prompt} for prompt in first]},
    }), encoding="utf-8")

    audit = AUDIT.build_audit(run, config)
    assert audit["provenance"]["schedule"]["rounds_checked_against_executed_selection"] == [0]


def test_the_cli_refuses_to_write_where_the_report_would_read_it(tmp_path, corpus, monkeypatch):
    run, config = build_run(tmp_path)
    monkeypatch.setattr("sys.argv", [
        "v4_scene_audit.py", "--outdir", str(run), "--config", str(config),
        "--output", str(run / "audit-splits" / "scene_overlap.json")])
    with pytest.raises(SystemExit) as failure:
        AUDIT.main()
    assert failure.value.code == 2
    assert not (run / "audit-splits").exists()


def test_a_bank_holding_two_pools_for_one_spec_is_refused(tmp_path, corpus):
    """Otherwise the spec appears twice in its cluster and the audit is malformed.

    read_partition rejects it downstream, but only after the file is on disk
    claiming to describe the bank.
    """
    run, config = build_run(tmp_path)
    bank_path = run / "probe-bank" / "bank.json"
    content = {"schema_version": "test", "split_sha256": "s" * 64,
               "pools": [{"spec_id": "v4b-005"}, {"spec_id": "v4b-005"}]}
    bank_path.write_text(json.dumps(dict(content, fingerprint=sha256_json(content))),
                         encoding="utf-8")
    with pytest.raises(ValueError, match="one pool per spec"):
        AUDIT.build_audit(run, config)


# ------------------------------------------ the prospective half, for arm B


def test_the_prospective_audit_is_the_one_the_report_will_accept(tmp_path, corpus):
    """The other contract, checked the same way as the gradient one.

    v4_decoupling_report.py validates the audit against the frozen split
    before it uses it: the flag, the provenance hashes, and that the audited
    outcome population is exactly the split's. Building the arguments it
    checks and running that validation is the only way to know the file is
    admissible without spending a run to find out.
    """
    run, config = build_run(tmp_path)
    audit = AUDIT.build_audit(run, config, prospective=True)

    assert audit["created_before_any_outcome_evaluation_artifact"] is True
    audited = [sid for group in audit["within_outcome_clusters"] for sid in group["spec_ids"]]
    split = json.loads((run / "split.json").read_text(encoding="utf-8"))
    assert sorted(audited) == sorted(split["outcome"]), "the report compares these as sets"
    assert len(audited) == len(set(audited)), "the report refuses a repeated outcome spec"
    sensitivity = set(audit["scene_disjoint_outcome_sensitivity"]["spec_ids"])
    assert sensitivity <= set(audited), "the report refuses ids outside the population"
    assert audit["provenance"]["config_sha256"] == AUDIT.sha256_file(config)
    assert audit["provenance"]["split_sha256"] == AUDIT.sha256_file(run / "split.json")


def test_the_prospective_audit_still_reads_as_a_gradient_audit(tmp_path, corpus):
    """One file, two consumers. Adding the outcome half must not move the
    probe-bank half out from under read_partition."""
    run, config = build_run(tmp_path)
    audit_path = tmp_path / "scene_overlap.json"
    audit_path.write_text(json.dumps(AUDIT.build_audit(run, config, prospective=True)),
                          encoding="utf-8")

    _audit, _bank, scenes, excluded = SENSITIVITY.read_partition(run, audit_path)
    assert set(scenes) == {"v4b-005", "v4b-006"}
    assert excluded == {"v4b-005"}


def test_an_outcome_prompt_restating_a_trained_scene_is_not_in_the_disjoint_set(
        tmp_path, corpus, monkeypatch):
    """The whole point of the subset. v4b-006 is a lemon nothing trains on, so
    it is disjoint; make it restate the apples and it must drop out."""
    run, config = build_run(tmp_path)
    assert AUDIT.build_audit(run, config, prospective=True)[
        "scene_disjoint_outcome_sensitivity"]["spec_ids"] == ["v4b-006"]

    restated = dict(SPECS)
    restated["v4b-006"] = spec("v4b-006", "two red apples again", [("apple", "red", 2)])
    monkeypatch.setattr(AUDIT, "load_training_corpus", lambda runs: TrainingCorpus(
        specs=restated,
        replay=(ReplayExample(image_path="a.png", question="q", answer="yes",
                              sample_id="s1", prompt_id="v4a-001"),)))
    audit = AUDIT.build_audit(run, config, prospective=True)
    assert audit["scene_disjoint_outcome_sensitivity"]["spec_ids"] == []
    assert audit["summary"]["outcome_overlap_n"] == 1


@pytest.mark.parametrize("artifact", [
    "evaluations", "checkpoints", "gradient-probes", "rounds",
    "checkpoint_metrics.csv", "decoupling_report.json",
])
def test_a_run_that_has_already_produced_something_cannot_be_frozen_prospectively(
        tmp_path, corpus, artifact):
    """The flag is a claim about the world, so the world is what gets checked.

    An allowlist, so an artifact nobody thought of stops the audit instead of
    passing through it.
    """
    run, config = build_run(tmp_path)
    target = run / artifact
    if target.suffix:
        target.write_text("{}", encoding="utf-8")
    else:
        target.mkdir()
    with pytest.raises(ValueError, match="prospective audit claims") as failure:
        AUDIT.build_audit(run, config, prospective=True)
    assert artifact in str(failure.value), "say which artifact stopped it"


def test_the_two_stages_that_run_before_any_arm_draws_anything_are_allowed(tmp_path, corpus):
    """split and freeze-probe produce the audit's own inputs. Refusing them
    would leave no moment at which a prospective audit could be built."""
    run, config = build_run(tmp_path)
    (run / "stage-completion").mkdir()
    for stage in ("split", "freeze-probe"):
        (run / "stage-completion" / f"{stage}.json").write_text("{}", encoding="utf-8")
    (run / "run_manifest.json").write_text("{}", encoding="utf-8")
    (run / "logs").mkdir()

    assert AUDIT.outcome_evaluation_artifacts(run) == []
    assert AUDIT.build_audit(run, config, prospective=True)[
        "created_before_any_outcome_evaluation_artifact"] is True


def test_a_completed_training_round_is_not_one_of_those_two_stages(tmp_path, corpus):
    run, _config = build_run(tmp_path)
    (run / "stage-completion").mkdir()
    (run / "stage-completion" / "round-000.train.json").write_text("{}", encoding="utf-8")
    assert AUDIT.outcome_evaluation_artifacts(run) == ["stage-completion/round-000.train.json"]


def test_the_cli_will_not_write_a_prospective_audit_anywhere_the_report_cannot_see_it(
        tmp_path, corpus, monkeypatch):
    """A freeze nobody reads is not a freeze. The report only ever looks at
    RUN/audit-splits/scene_overlap.json."""
    run, config = build_run(tmp_path)
    monkeypatch.setattr("sys.argv", [
        "v4_scene_audit.py", "--outdir", str(run), "--config", str(config),
        "--output", str(tmp_path / "somewhere-else.json"), "--prospective"])
    with pytest.raises(SystemExit) as failure:
        AUDIT.main()
    assert failure.value.code == 2
    assert not (tmp_path / "somewhere-else.json").exists()


def test_the_cli_writes_the_prospective_audit_where_the_report_reads_it(tmp_path, corpus,
                                                                       monkeypatch, capsys):
    run, config = build_run(tmp_path)
    output = run / "audit-splits" / "scene_overlap.json"
    monkeypatch.setattr("sys.argv", [
        "v4_scene_audit.py", "--outdir", str(run), "--config", str(config),
        "--output", str(output), "--prospective"])
    AUDIT.main()

    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["kind"] == AUDIT.PROSPECTIVE_KIND
    assert written["created_before_any_outcome_evaluation_artifact"] is True
    assert "retrospective_notice" not in written, "it is not one"
    assert "scene-disjoint" in capsys.readouterr().out


def test_outcome_prompts_that_restate_each_other_are_one_cluster_and_are_named(tmp_path, corpus):
    """`within_outcome_duplicate_scene_groups` is what tells the reader the
    outcome population is not 5 independent prompts but 4 scenes."""
    run, config = build_prospective_run(tmp_path)
    audit = AUDIT.build_audit(run, config, prospective=True)

    assert [group["spec_ids"] for group in audit["within_outcome_clusters"]
            if len(group["spec_ids"]) > 1] == [["v4b-007", "v4b-008"]]
    assert audit["within_outcome_duplicate_scene_groups"] == [
        group for group in audit["within_outcome_clusters"] if len(group["spec_ids"]) > 1]
    assert audit["summary"]["outcome_unique_canonical_scenes"] == 4
    assert audit["summary"]["outcome_repeated_scene_groups"] == 1
    assert audit["summary"]["outcome_prompts_in_repeated_scene_groups"] == 2
    assert audit["summary"]["scene_disjoint_outcome_n"] == 3


def test_the_overlap_rows_come_out_sorted_whatever_order_the_split_lists(tmp_path, corpus):
    """Two runs of this over the same frozen split have to produce the same
    bytes, and the split's own ordering is not something the audit controls."""
    run, config = build_prospective_run(tmp_path)
    rows = AUDIT.build_audit(run, config, prospective=True)["overlap"]["outcome"]

    assert [row["spec_id"] for row in rows] == ["v4b-009", "v4b-010"]
    assert PROSPECTIVE_OUTCOME.index("v4b-010") < PROSPECTIVE_OUTCOME.index("v4b-009"), (
        "the split has to disagree with the sort or this proves nothing")


def test_an_overlap_row_says_which_round_trains_the_scene_it_matches(tmp_path, corpus):
    """Without the round the row says a scene is contaminated but not when,
    which is the part that decides whether a checkpoint predates it."""
    run, config = build_prospective_run(tmp_path)
    split = json.loads((run / "split.json").read_text(encoding="utf-8"))
    scheduled = AUDIT.planned_exposure(
        AUDIT.yaml.safe_load(config.read_text(encoding="utf-8")), split)
    rows = AUDIT.build_audit(run, config, prospective=True)["overlap"]["outcome"]

    for row in rows:
        for match in row["training_matches"]:
            assert match["scheduled_rounds"] == [scheduled[match["spec_id"]]]
    assert {match["scheduled_rounds"][0]
            for row in rows for match in row["training_matches"]}, "no rounds recorded at all"
