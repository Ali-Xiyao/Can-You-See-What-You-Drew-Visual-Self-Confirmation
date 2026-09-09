"""Tests for the endpoint 3 driver's not-done logic and its two verdict wordings.

The arithmetic is tested in `tests/test_endpoint3.py`. What is tested here is
what the driver decides before any arithmetic -- whether the selection pass
finished -- and what it says afterwards. Deviation 13.2 point 7 makes "not
done" a registered outcome with a required consequence (no corpus replay
standing in), and point 6 makes the downgrade as binding as the prediction, so
both wordings are pinned rather than left to whichever branch runs first.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from selfsight.analysis.endpoint3 import DOWNGRADE, CheckpointPoint, Endpoint3Verdict

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "v4_e3_endpoint3.py"

SEED = 20260906


def _module():
    spec = importlib.util.spec_from_file_location("v4_e3_endpoint3", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


driver = _module()


def _adjudicated(run: Path, arm: str, steps: list[int]) -> None:
    for step in steps:
        directory = run / "evaluations" / arm / f"step-{step:05d}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "verified.jsonl").write_text(
            json.dumps({"spec_id": "p1", "candidate_index": 0,
                        "image_correct": True, "resolution": "agreed"}) + "\n",
            encoding="utf-8")


def _selection(run: Path, arm: str, steps: list[int], *, scorable: bool = True,
               blind_hits: dict[int, int] | None = None) -> None:
    """Two prompts, two candidates, both conditions, one conflict trial each.

    `blind_hits[step]` is how many of the four conflict trials the blind
    condition gets right at that step, which is the dose: prompted always gets
    zero, so x = blind_hits / 4. The steps need different doses or the fit has
    no spread to work with, which the analysis module correctly refuses.

    `scorable=False` makes every trial an abstention, which is the shape of a
    checkpoint whose pass ran and produced nothing usable -- a real outcome,
    and not one deviation 13.2 gives a rule for dropping.
    """

    directory = run / "analysis" / "selection" / arm
    directory.mkdir(parents=True, exist_ok=True)
    for step in steps:
        hits = (blind_hits or {}).get(step, 4)
        rows = []
        conflicts_seen = 0
        for spec_id in ("p1", "p2"):
            for index in (0, 1):
                conflicts_seen += 1
                for condition in ("blind", "prompted"):
                    blind_right = condition == "blind" and conflicts_seen <= hits
                    for source, correct in (("image_differs_from_spec", blind_right),
                                            ("spec_matches_image", index == 1)):
                        rows.append({"condition": condition, "spec_id": spec_id,
                                     "candidate_index": index, "gold_source": source,
                                     "correct": None if not scorable else correct,
                                     "abstain": not scorable,
                                     "image_correct": index == 1, "arm": arm,
                                     "step": step})
        (directory / f"step-{step:05d}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _replicate(tmp_path: Path, seed: int, *, steps: list[int], measured: list[int] | None,
               arms: tuple[str, ...] = ("naive", "blind_self")) -> Path:
    run = tmp_path / f"e3-s{seed}"
    for arm in arms:
        _adjudicated(run, arm, steps)
        if measured:
            _selection(run, arm, measured, blind_hits={0: 4, 8: 2, 16: 1})
    return run


# --- has the pass finished --------------------------------------------------


def test_no_selection_pass_at_all_says_which_script_to_run(tmp_path: Path):
    """Nothing measured is a different sentence from some steps missing.

    Both are not done, but only one of them is answered by launching the pass,
    and the reason string is what a reader of the packet gets. Matching on the
    script name rather than on "no selection pass" is deliberate: the
    partial-pass branch says that phrase too.
    """

    run = tmp_path / f"e3-s{SEED}"
    for arm in ("naive", "blind_self"):
        _adjudicated(run, arm, [0, 8, 16])
    with pytest.raises(driver.NotDone, match="run scripts/v4_e3_selection_observe"):
        driver.points_for(run, SEED, ["naive", "blind_self"])


def test_a_missing_arm_checkpoint_stops_the_endpoint(tmp_path: Path):
    """The one place endpoint 3 is stricter than endpoint 2, on purpose.

    Deviation 13.1 point 6 lets endpoint 2 fit a shorter series as long as the
    gaps are named. Deviation 13.2 has no such clause -- its point 7 is a plain
    "did not finish -> not done" -- so fitting the checkpoints that happen to
    exist would be inventing the rule that 13.1 spells out and 13.2 does not.
    """

    run = tmp_path / f"e3-s{SEED}"
    for arm in ("naive", "blind_self"):
        _adjudicated(run, arm, [0, 8, 16, 24])
        _selection(run, arm, [0, 8, 16])
    with pytest.raises(driver.NotDone, match=r"step\(s\) \[24\]"):
        driver.points_for(run, SEED, ["naive", "blind_self"])


def test_one_arm_finishing_is_not_the_pass_finishing(tmp_path: Path):
    run = tmp_path / f"e3-s{SEED}"
    for arm in ("naive", "blind_self"):
        _adjudicated(run, arm, [0, 8])
    _selection(run, "naive", [0, 8])
    _selection(run, "blind_self", [0])
    with pytest.raises(driver.NotDone, match="blind_self"):
        driver.points_for(run, SEED, ["naive", "blind_self"])


def test_a_selection_pass_with_no_verdicts_to_score_against_is_not_done(tmp_path: Path):
    # The shape of a stale output directory: the pass ran, the adjudication for
    # that checkpoint was moved or never finished, and the scores sit there
    # looking complete.
    run = tmp_path / f"e3-s{SEED}"
    for arm in ("naive", "blind_self"):
        _adjudicated(run, arm, [0, 8])
        _selection(run, arm, [0, 8, 16])
    with pytest.raises(driver.NotDone, match=r"step\(s\) \[16\]"):
        driver.points_for(run, SEED, ["naive", "blind_self"])


def test_a_checkpoint_that_kept_no_scorable_pool_is_not_done(tmp_path: Path):
    run = tmp_path / f"e3-s{SEED}"
    for arm in ("naive", "blind_self"):
        _adjudicated(run, arm, [0, 8])
        _selection(run, arm, [0, 8], scorable=False)
    with pytest.raises(driver.NotDone, match="no prompt kept"):
        driver.points_for(run, SEED, ["naive", "blind_self"])


def test_both_arms_land_in_the_same_cluster(tmp_path: Path):
    """Deviation 13.2 point 5: the cluster is (seed, checkpoint), arms inside.

    If the arms clustered separately the bootstrap would treat the two arms of
    one checkpoint of one run as independent draws of the dose, which is the
    dependence the deviation exists to respect.
    """

    run = tmp_path / f"e3-s{SEED}"
    for arm in ("naive", "blind_self"):
        _adjudicated(run, arm, [0, 8])
        _selection(run, arm, [0, 8])
    points = driver.points_for(run, SEED, ["naive", "blind_self"])
    assert len(points) == 4
    assert {point.cluster for point in points} == {(SEED, 0), (SEED, 8)}
    assert {point.arm for point in points} == {"naive", "blind_self"}


# --- what it says -----------------------------------------------------------


def test_not_done_writes_the_registered_sentence_and_no_slope(tmp_path: Path):
    driver.write_not_done(tmp_path, [f"e3-s{SEED} naive: no selection pass"],
                          arms=["naive", "blind_self"])
    payload = json.loads((tmp_path / "endpoint3.json").read_text(encoding="utf-8"))
    assert payload["status"] == "not_done"
    assert payload["substituted_corpus_replay"] is False
    assert "slope" not in json.dumps(payload)
    text = (tmp_path / "endpoint3.txt").read_text(encoding="utf-8")
    assert "NOT DONE" in text
    assert "13.2" in text


def test_not_done_has_its_own_exit_code():
    # Not 0: a wrapper must not carry on as though endpoint 3 had an answer.
    # Not 1: it is a registered outcome, not a crash.
    assert driver.NOT_DONE == 2


def _point(step: int, dose: float, gain: float) -> CheckpointPoint:
    return CheckpointPoint(seed=SEED, arm="naive", step=step, context_effect=dose,
                           selection_gain=gain, selection_gain_first_index=gain,
                           pools=2, candidates=4,
                           conflict_trials={"blind_trials": 2, "prompted_trials": 2})


def _verdict(low: float, high: float) -> Endpoint3Verdict:
    positive = low > 0.0
    return Endpoint3Verdict(
        slope=0.42, interval=(low, high), dose_response=positive,
        wording=("dose-response: selection gain scales with the context effect"
                 if positive else DOWNGRADE),
        points=2, seeds=1, clusters=2, discarded_resamples=0)


def _report(low: float, high: float) -> str:
    return driver.report([_point(0, 0.1, 0.2), _point(8, 0.3, 0.4)], _verdict(low, high),
                         arms=["naive", "blind_self"], resamples=20000,
                         bootstrap_seed=20260908, confirmatory=True)


def test_a_positive_interval_is_reported_as_the_dose_response():
    text = _report(0.05, 0.80)
    assert "SUPPORTED" in text
    assert DOWNGRADE not in text


def test_an_interval_that_spans_zero_takes_the_registered_downgrade():
    """A large positive slope with a wide interval is still the downgrade.

    This is the branch a reader will want softened, so it is the branch worth
    pinning: same +0.42 point estimate as the supported case above.
    """

    text = _report(-0.05, 0.90)
    assert "DOWNGRADED" in text
    assert DOWNGRADE in text
    assert "SUPPORTED" not in text


def test_the_downgrade_wording_is_the_analysis_module_s():
    # Restating it here would let the paper and the code drift apart in the one
    # sentence the pre-registration wrote out in full.
    assert driver.DOWNGRADE is DOWNGRADE


def test_a_seed_set_that_is_not_the_registered_five_says_so():
    text = driver.report([_point(0, 0.1, 0.2), _point(8, 0.3, 0.4)], _verdict(0.05, 0.8),
                         arms=["naive", "blind_self"], resamples=20000,
                         bootstrap_seed=20260908, confirmatory=False)
    assert "NOT THE REGISTERED ANALYSIS" in text


def _main(argv: list[str], monkeypatch) -> int:
    monkeypatch.setattr("sys.argv", ["v4_e3_endpoint3.py", *argv])
    return driver.main()


def test_one_seed_short_is_the_whole_endpoint_short(tmp_path: Path, monkeypatch):
    """The seed that finished does not get reported on its own.

    Deviation 13.2's analysis is over the five replicates. Printing a slope
    from the ones that happened to finish would be answering a question nobody
    registered, with a number that looks exactly like the registered one.
    """

    done = _replicate(tmp_path, 20260906, steps=[0, 8, 16], measured=[0, 8, 16])
    short = _replicate(tmp_path, 20260907, steps=[0, 8, 16], measured=None)
    out = tmp_path / "packet"
    code = _main(["--runs", str(done), str(short), "--outdir", str(out),
                  "--allow-partial", "--resamples", "200"], monkeypatch)
    assert code == driver.NOT_DONE
    payload = json.loads((out / "endpoint3.json").read_text(encoding="utf-8"))
    assert payload["status"] == "not_done"
    assert len(payload["reasons"]) == 1
    assert "e3-s20260907" in payload["reasons"][0]
    assert "slope" not in json.dumps(payload)


def test_a_finished_pass_produces_a_slope_and_the_points_behind_it(tmp_path: Path,
                                                                   monkeypatch):
    run = _replicate(tmp_path, 20260906, steps=[0, 8, 16], measured=[0, 8, 16])
    out = tmp_path / "packet"
    code = _main(["--runs", str(run), "--outdir", str(out), "--allow-partial",
                  "--resamples", "200"], monkeypatch)
    assert code == 0
    payload = json.loads((out / "endpoint3.json").read_text(encoding="utf-8"))
    assert payload["status"] == "done"
    assert payload["confirmatory"] is False
    # Three checkpoints, two arms, one seed.
    assert len(payload["points"]) == 6
    assert {tuple(point["cluster"]) for point in payload["points"]} == {
        (20260906, 0), (20260906, 8), (20260906, 16)}
    assert payload["downgrade_wording"] == DOWNGRADE
    assert payload["bootstrap"]["unit"] == "(seed, checkpoint)"
    assert "NOT THE REGISTERED ANALYSIS" in (out / "endpoint3.txt").read_text(
        encoding="utf-8")


def test_a_seed_set_that_is_not_the_five_is_refused_without_allow_partial(
        tmp_path: Path, monkeypatch):
    run = _replicate(tmp_path, 20260906, steps=[0, 8], measured=[0, 8])
    with pytest.raises(SystemExit, match="registered over seeds"):
        _main(["--runs", str(run), "--outdir", str(tmp_path / "packet")], monkeypatch)


def test_the_registered_seeds_are_the_ones_every_other_script_uses():
    """Four literal copies of one registered fact, pinned to each other.

    Deviation 9 fixed these five in advance and they are what "confirmatory"
    means in three drivers and what the launch gate checks the configs against.
    A copy drifting would not fail anything: it would relabel a confirmatory
    result as exploratory, or the reverse, and print the sentence for it.
    """

    names = ("v4_e3_endpoint1", "v4_e3_endpoint2", "v4_e3_launch_preflight")
    for name in names:
        spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.REGISTERED_SEEDS == driver.REGISTERED_SEEDS, name
    # 20260908 is the bootstrap seed section 3 spent; deviation 9 skips it here.
    assert 20260908 not in driver.REGISTERED_SEEDS
    assert len(set(driver.REGISTERED_SEEDS)) == 5
