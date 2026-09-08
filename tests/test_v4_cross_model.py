"""E4's driver: the file `observe` writes must be the file `report` reads.

The two stages compute the answer filename independently -- one from the
backbone config path, one from a model key -- and they run hours apart in
different interpreters. If they disagree, the cheap outcome is `report` saying
"not measured" forever. The expensive one is `report` reading the 1.5B's rows
under the 7B's name and publishing a perfect replication of the model against
itself.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def driver():
    return _load("v4_cross_model", "scripts/v4_cross_model.py")


@pytest.fixture(scope="module")
def pipeline():
    return _load("v4_run_pipeline_cross", "scripts/v4_run_pipeline.py")


# --------------------------------------------------------------- the coupling


def test_the_two_stages_agree_on_every_filename(driver, pipeline, tmp_path):
    for model, config in driver.MODELS.items():
        for condition in driver.CONDITIONS:
            written = pipeline.answer_file(
                tmp_path,
                prompted=condition == "prompted",
                backbone=config or pipeline.DEFAULT_BACKBONE,
            )
            read = tmp_path / driver.answer_name(model, condition)
            assert written == read, f"{model}/{condition}"


def test_the_baseline_reads_the_files_the_frozen_result_lives_in(driver):
    """Section 1 of the paper is answers.jsonl and answers.prompted.jsonl. The
    baseline column must be those exact files, not a re-measurement."""

    assert driver.answer_name(driver.BASELINE, "image_only") == "answers.jsonl"
    assert driver.answer_name(driver.BASELINE, "prompted") == "answers.prompted.jsonl"


def test_no_two_models_share_a_filename(driver):
    names = [driver.answer_name(model, condition)
             for model in driver.MODELS for condition in driver.CONDITIONS]
    assert len(names) == len(set(names))


def test_the_baseline_is_not_re_run(driver):
    """Its rows are frozen evidence; re-answering them would overwrite the
    measurement every number in section 1 comes from."""

    assert driver.MODELS[driver.BASELINE] is None


# ------------------------------------------------------------- the verdict


def measured(delta: float, p_value: float, *, skew: float = 1.0) -> dict:
    """A `measure` result with only the fields the verdict reads."""

    return {
        "image_only": {"point": 0.6, "low": 0.5, "high": 0.7, "n": 100},
        "prompted": {"point": 0.6 + delta, "low": 0.5, "high": 0.7, "n": 100},
        "delta": delta,
        "pairs": 100, "blind_only": 30, "told_only": 5,
        "p_one_sided": p_value,
        "decisive_trials": 120, "blind_only_abstained": 5, "told_only_abstained": 5,
        "p_abstention_imbalance": skew,
        "replicates": delta < 0 and p_value < 0.05,
    }


@pytest.fixture
def verdict(driver, monkeypatch, tmp_path, capsys):
    def run(results: dict, models: list[str] | None = None) -> str:
        monkeypatch.setattr(driver, "measure", lambda runs, model: results.get(model))
        args = SimpleNamespace(runs=["runs/v4/main-2plus1"],
                               models=models or list(driver.MODELS),
                               out=str(tmp_path / "out.json"))
        driver.stage_report(args)
        return capsys.readouterr().out
    return run


def test_a_result_in_the_wrong_direction_does_not_count_as_replication(driver):
    """One-sided means one side. A model where the description *helps* is
    evidence against the claim, however small its p-value."""

    assert measured(-0.3, 1e-9)["replicates"] is True
    assert measured(+0.3, 1e-9)["replicates"] is False


def test_the_verdict_waits_until_the_registered_three_are_all_measured(verdict):
    text = verdict({"showo2_1p5b": measured(-0.3, 1e-9), "showo2_7b": measured(-0.2, 1e-4)})
    assert "VERDICT DEFERRED" in text
    assert "showo_v1" in text


def test_two_of_the_registered_three_carries_the_claim(verdict):
    text = verdict({"showo2_1p5b": measured(-0.3, 1e-9),
                    "showo2_7b": measured(-0.2, 1e-4),
                    "showo_v1": measured(-0.01, 0.4)})
    assert "PREREG E4 met" in text
    assert "negative controls" in text, "the labelling constraint must travel with the verdict"


def test_one_of_three_shrinks_the_claim_instead_of_adding_models(verdict):
    text = verdict({"showo2_1p5b": measured(-0.3, 1e-9),
                    "showo2_7b": measured(-0.01, 0.4),
                    "showo_v1": measured(+0.02, 0.9)})
    assert "PREREG E4 not met" in text
    assert "do not add models" in text


def test_janus_cannot_supply_the_majority_it_was_not_registered_for(verdict):
    """Deviation 1 added a fourth model after E4 was written. Counting it into
    a 2-of-3 rule registered over three would be choosing the denominator after
    seeing the numbers."""

    text = verdict({"showo2_1p5b": measured(-0.3, 1e-9),
                    "showo2_7b": measured(-0.01, 0.4),
                    "showo_v1": measured(+0.02, 0.9),
                    "janus_pro_1b": measured(-0.25, 1e-6)})
    assert "PREREG E4 not met" in text
    assert "beyond the registered set: janus_pro_1b replicates" in text
    assert "not counted toward the 2-of-3" in text


def test_janus_is_reported_even_when_it_disagrees(verdict):
    text = verdict({"showo2_1p5b": measured(-0.3, 1e-9),
                    "showo2_7b": measured(-0.2, 1e-4),
                    "showo_v1": measured(-0.2, 1e-4),
                    "janus_pro_1b": measured(+0.05, 0.8)})
    assert "PREREG E4 met" in text
    assert "beyond the registered set: janus_pro_1b does not replicate" in text


# --------------------------------------------------------------- measuring


def test_a_model_with_only_one_condition_answered_is_not_half_reported(driver, tmp_path):
    """A crashed prompted pass would otherwise leave a blind-only column that
    reads as a huge effect."""

    run = tmp_path / "run"
    run.mkdir()
    (run / "answers.showo2_7b.jsonl").write_text("", encoding="utf-8")
    assert driver.measure([str(run)], "showo2_7b") is None


def test_a_run_missing_from_the_pool_is_not_silently_dropped(driver, tmp_path):
    """Pooling over two corpora and finding one is the difference between
    `n=667` and `n=323`, which changes every interval in the table."""

    first, second = tmp_path / "a", tmp_path / "b"
    for run in (first, second):
        run.mkdir()
    for condition in ("answers.showo2_7b.jsonl", "answers.prompted.showo2_7b.jsonl"):
        (first / condition).write_text("", encoding="utf-8")
    assert driver.measure([str(first), str(second)], "showo2_7b") is None


# ------------------------------------------------- which trials were answerable
#
# `conflict_pairs` drops a trial either condition declined, which is right. What
# it cannot do is notice the surviving set was chosen by the model: Janus answers
# an absence question with "The image does not contain any notebooks", correct
# and matching neither option, so it abstains -- and how often it phrases things
# that way is not independent of the condition.


def row(question, abstain, image="i.png", source="image_differs_from_spec"):
    return {"image_path": image, "question": question, "abstain": abstain,
            "correct": None if abstain else True, "gold_source": source}


def test_abstention_is_counted_on_the_same_pairing_as_the_result(driver):
    from selfsight.analysis.context import abstention_pairs

    blind = [row("q1", False), row("q2", True), row("q3", False)]
    prompted = [row("q1", True), row("q2", True), row("q3", False)]
    assert abstention_pairs(blind, prompted) == [(False, True), (True, True), (False, False)]


def test_the_same_question_about_two_pictures_is_two_trials(driver):
    """The question text is templated, so it repeats across images.

    Keyed on the words alone, one picture's trial silently overwrites the
    other's and half the corpus vanishes from the diagnostic.
    """

    from selfsight.analysis.context import abstention_pairs

    question = "Which of these is in this picture? Answer A or B only.\nA. plate\nB. banana"
    blind = [row(question, False, image="a.png"), row(question, False, image="b.png")]
    prompted = [row(question, True, image="a.png"), row(question, False, image="b.png")]
    assert abstention_pairs(blind, prompted) == [(False, True), (False, False)]


def test_a_trial_that_is_not_decisive_is_not_counted_either_way(driver):
    """The result is measured on the conflict trials, so the abstention
    diagnostic has to describe that same set and not a larger one."""

    from selfsight.analysis.context import abstention_pairs

    rows = [row("q1", True), row("q2", True, source="spec_matches_image")]
    assert abstention_pairs(rows, rows) == [(True, True)]


def test_a_model_that_declines_differently_in_the_two_conditions_is_named(verdict):
    text = verdict({"showo2_1p5b": measured(-0.3, 1e-9),
                    "showo2_7b": measured(-0.2, 1e-4, skew=1e-6),
                    "showo_v1": measured(-0.2, 1e-4)})
    assert "showo2_7b: declines to answer at different rates" in text
    assert "subsets the model chose" in text
    assert "do not drop the model" in text, (
        "dropping it would be choosing the sample after seeing the numbers")


def test_the_quiet_case_says_so_rather_than_saying_nothing(verdict):
    """A diagnostic that only prints on failure reads as absent when it passes,
    and the reader cannot tell it from one that was never run."""

    text = verdict({"showo2_1p5b": measured(-0.3, 1e-9),
                    "showo2_7b": measured(-0.2, 1e-4),
                    "showo_v1": measured(-0.2, 1e-4)})
    assert "no model declines to answer at a different rate" in text
