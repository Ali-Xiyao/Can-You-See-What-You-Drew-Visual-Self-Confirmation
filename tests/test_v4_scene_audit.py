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
}
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
