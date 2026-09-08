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


def measured(delta: float, p_value: float) -> dict:
    """A `measure` result with only the fields the verdict reads."""

    return {
        "image_only": {"point": 0.6, "low": 0.5, "high": 0.7, "n": 100},
        "prompted": {"point": 0.6 + delta, "low": 0.5, "high": 0.7, "n": 100},
        "delta": delta,
        "pairs": 100, "blind_only": 30, "told_only": 5,
        "p_one_sided": p_value,
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
