"""The frozen instrument must preserve prompts, labels and the gradient estimand."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from selfsight.schemas import AtomicObservation, ObservationResult, as_serializable
from selfsight.utils.hashing import rgb_sha256, sha256_file, sha256_json
from selfsight.v3.paired import cosine_from_gram
from selfsight.v4.checkpoint_probe import (
    bind_run, choose_pools, freeze_bank, gram_from_arrays, load_bank, questions_for,
    reference_differences, run_probe, select_pool, summarize_gram, validate_observation,
)
from selfsight.v4.probe import spec_questions
from selfsight.v4.spec import SceneSpec, SpecObject


def _observation(pool, candidate, *, identity=("rfo", "revision"), correct=True):
    answers = []
    for question in questions_for(pool):
        value = question.expected_answer if correct else ("no" if question.expected_answer == "yes" else "0")
        raw = chr(65 + question.choices.index(value)) if question.choices else value
        answers.append(AtomicObservation(question.question_id, raw, value, False))
    return ObservationResult("request", *identity, candidate["rgb_sha256"], tuple(answers))


def _fixture(tmp_path):
    source, corpus = tmp_path / "source", tmp_path / "corpus"
    source.mkdir()
    corpus.mkdir()
    pools, manifests, verdicts, observations = [], [], [], []
    for index in range(4):
        spec = SceneSpec(f"s{index}", "one red mug", (SpecObject("mug", "red", 1),))
        pool = {"prompt_id": f"corpus:s{index}", "run": "corpus", "spec_id": spec.spec_id,
                "prompt": spec.prompt, "questions": [as_serializable(q) for q in spec_questions(spec)],
                "candidates": []}
        for ci in range(2):
            image = corpus / f"s{index}-{ci}.png"
            Image.new("RGB", (2, 2), (index * 30, ci * 100, 0)).save(image)
            candidate = {"candidate_id": f"s{index}:{ci}", "image_path": str(image),
                         "sampling_seed": ci, "correct": ci == 0}
            pool["candidates"].append(candidate)
            manifests.append({"spec": spec.to_dict(), "candidate_index": ci,
                              "image_path": str(image), "seed": ci})
            verdicts.append({"image_path": str(image), "resolution": "agreed", "image_correct": ci == 0})
            observed = candidate | {"rgb_sha256": rgb_sha256(image)}
            observations.append({"prompt_id": pool["prompt_id"], "candidate_id": candidate["candidate_id"],
                "observation": as_serializable(_observation(pool, observed, correct=ci == 0))})
        pools.append(pool)
    for path, rows in ((source / "pools.jsonl", pools), (source / "observations.rfo.jsonl", observations),
                       (corpus / "manifest.jsonl", manifests), (corpus / "verified.jsonl", verdicts)):
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (source / "runs.json").write_text(json.dumps({"runs": [str(corpus)]}), encoding="utf-8")
    split = {"train": ["training"], "outcome": ["outcome"], "probe": [f"s{i}" for i in range(4)]}
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    rfo = tmp_path / "rfo.yaml"
    rfo.write_text("observer_id: rfo\nrevision: revision\ntrainable: false\n", encoding="utf-8")
    config = {"seed": 123, "model": {"backbone_config": "configs/backbones/showo2_1p5b.yaml",
                                    "trainable_id": "test-model"},
              "hardware": {"precision": "bf16"},
              "training": {"lora": {"rank": 1, "alpha": 2, "dropout": 0}, "gradient_checkpointing": True}}
    bankdir = tmp_path / "bank"
    bank = freeze_bank(source=source, split_path=split_path, config=config,
                       destination=bankdir, maximum=4, rfo_config=rfo)
    return bank, bankdir, config, source, split_path, rfo


def test_frozen_bank_rejects_question_drift_and_modified_image(tmp_path):
    bank, bankdir, config, source, split, rfo = _fixture(tmp_path)
    assert bank["actual_prompts"] == 4
    assert load_bank(bankdir, config) == bank
    rows = [json.loads(x) for x in (source / "pools.jsonl").read_text().splitlines()]
    rows[0]["questions"][1]["text"] = "How many unqualified mugs?"
    (source / "pools.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    with pytest.raises(ValueError, match="different question"):
        freeze_bank(source=source, split_path=split, config=config, destination=tmp_path / "changed",
                    maximum=4, rfo_config=rfo)
    Image.new("RGB", (2, 2), "white").save(bank["pools"][0]["candidates"][0]["image_path"])
    with pytest.raises(ValueError, match="image changed"):
        load_bank(bankdir, config)


def test_pool_holdout_and_unique_spec_are_enforced(tmp_path):
    bank, *_ = _fixture(tmp_path)
    split = bank["split"]
    pool = bank["pools"][0]
    duplicate = pool | {"prompt_id": "zz:" + pool["spec_id"]}
    chosen = choose_pools(bank["pools"] + [duplicate], split, maximum=4, seed=3)
    assert len({p["spec_id"] for p in chosen}) == 4
    with pytest.raises(ValueError, match="overlap"):
        choose_pools(bank["pools"], split | {"train": [pool["spec_id"]]}, maximum=4, seed=3)


def test_one_missing_answer_drops_whole_paired_pool(tmp_path):
    bank, *_ = _fixture(tmp_path)
    pool = bank["pools"][0]
    observations = {c["candidate_id"]: _observation(pool, c, correct=c["correct"]) for c in pool["candidates"]}
    assert not select_pool(pool, observations)["dropped"]
    cid = pool["candidates"][1]["candidate_id"]
    previous = observations[cid]
    observations[cid] = replace(previous, answers=(replace(previous.answers[0], abstain=True,
        normalized_answer=None), *previous.answers[1:]))
    assert select_pool(pool, observations)["dropped"]
    with pytest.raises(ValueError, match="normalization"):
        validate_observation(observations[cid], pool, pool["candidates"][1], identity=("rfo", "revision"))


def test_gram_preserves_cosine_of_mean_not_mean_cosine():
    left = np.array([[100, 0], [0, 1]], dtype=np.float32)
    right = np.array([[1, 0], [0, 100]], dtype=np.float32)
    gram = gram_from_arrays(left, right, prompt_ids=("a", "b"), left_name="naive", right_name="rfo", chunk_elements=2)
    report = summarize_gram(gram, resamples=200, seed=3)
    expected = np.dot(left.mean(0), right.mean(0)) / (np.linalg.norm(left.mean(0)) * np.linalg.norm(right.mean(0)))
    assert report["cosine"] == pytest.approx(expected)
    assert report["mean_per_prompt_cosine"] == pytest.approx(1)
    assert report["cosine"] < .03


def test_paired_gram_bootstrap_matches_direct_same_prompt_resamples():
    rng = np.random.default_rng(8)
    left, right = rng.normal(size=(6, 19)), rng.normal(size=(6, 19))
    gram = gram_from_arrays(left, right, prompt_ids=tuple(map(str, range(6))), left_name="naive", right_name="gold", chunk_elements=12)
    report = summarize_gram(gram, resamples=200, seed=8)
    counts = np.random.default_rng(8).multinomial(6, np.full(6, 1/6), size=200)
    a, b = counts @ left, counts @ right
    direct = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    assert [report["ci_low"], report["ci_high"]] == pytest.approx(np.quantile(direct, [.025, .975]))


def test_resume_rejects_checkpoint_or_estimator_change(tmp_path):
    bind_run(tmp_path / "probe", {"checkpoint": "a", "bootstrap": 100})
    bind_run(tmp_path / "probe", {"checkpoint": "a", "bootstrap": 100})
    with pytest.raises(ValueError, match="fingerprint"):
        bind_run(tmp_path / "probe", {"checkpoint": "b", "bootstrap": 100})


def test_end_to_end_small_backbone_keeps_grams_and_removes_vectors(tmp_path, monkeypatch):
    import torch
    import selfsight.backbones.showo2 as showo
    bank, bankdir, config, *_ = _fixture(tmp_path)
    by_image = {c["image_path"]: (p, c) for p in bank["pools"] for c in p["candidates"]}

    class TinyBackbone:
        model_id, revision = "test-model", "revision"
        def __init__(self, **kwargs):
            self.model = torch.nn.Module()
            self.model.register_parameter("lora_vector", torch.nn.Parameter(torch.zeros(3)))
        def attach_lora(self, **kwargs):
            return {"trainable_parameters": 3}
        def observe_atoms(self, image, questions):
            pool, candidate = by_image[image]
            assert 'You were asked to draw' in questions[0].text
            return _observation(pool, candidate, identity=(self.model_id, self.revision), correct=candidate["correct"])
        def compute_lora_gradient(self, batch, criterion):
            index = int(batch.sample_ids[0].split(":s")[1])
            return SimpleNamespace(vector=torch.tensor([1., float(index), 2.]), loss=1.)

    monkeypatch.setattr(showo, "Showo2Adapter", TinyBackbone)
    destination = tmp_path / "probe"
    report = run_probe(config=config, bank_dir=bankdir, destination=destination, checkpoint=None,
                       device="cpu", resamples=100)
    assert report["n_retained"] == 4
    assert report["gradient_evaluations"] == 4  # Three identical choices share one computation.
    assert report["gda_free"]["cosine"] == pytest.approx(1)
    assert not list((destination / "gradient-tmp").glob("*.f32"))
    assert (destination / "grams.npz").stat().st_size < 10_000
    assert run_probe(config=config, bank_dir=bankdir, destination=destination, checkpoint=None,
                     device="cpu", resamples=100) == report
    with np.load(destination / "grams.npz") as saved:
        ids = tuple(str(p) for p in saved["prompt_ids"])
        from selfsight.v3.paired import GramMatrices
        grams = {name: GramMatrices("naive", other, ids, *[saved[f"{name}_{part}"] for part in ("ll", "rr", "lr")])
                 for name, other in (("gda_free", "rfo"), ("gda_gold", "gold"))}
    delta = reference_differences(grams, destination, bank_fingerprint=bank["fingerprint"], resamples=100, seed=123)
    assert delta["gda_free"]["point_difference"] == pytest.approx(0)
    assert delta["gda_free"]["ci_low"] == pytest.approx(0)
    assert delta["gda_free"]["n_common"] == 4
