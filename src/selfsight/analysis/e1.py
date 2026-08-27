"""E1 context-matrix and Tier-B pixel-override experiment."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from selfsight.data.questions import build_primary_atom, build_question
from selfsight.data.subsets import stable_stratified_sample
from selfsight.observers.client import ObserverServiceClient
from selfsight.rfo.isolation import hard_render, make_blind_request
from selfsight.rfo.metrics import ContextTrial, compute_e1_metrics
from selfsight.schemas import SceneSpec
from selfsight.utils.jsonl import atomic_write_json, read_jsonl


def _model_answer(adapter: Any, image: Image.Image, text: str, question: Any) -> str | None:
    from selfsight.data.questions import normalize_answer

    return normalize_answer(adapter._observe_one(image, text), question)


def run_e1_tier_b(
    *,
    adapter: Any,
    detector: ObserverServiceClient,
    manifest_path: str | Path,
    output_dir: str | Path,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run the fixed five-context matrix without exposing intent to the detector process."""

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    blank_path = hard_render(
        Image.new("RGB", (512, 512), (245, 245, 245)), output / "blank.png"
    )["path"]
    with Image.open(blank_path) as opened:
        blank = opened.convert("RGB")
    trials: list[ContextTrial] = []
    rows: list[dict[str, Any]] = []
    by_category: dict[str, list[ContextTrial]] = defaultdict(list)
    records = list(read_jsonl(manifest_path))
    if limit is not None:
        records = stable_stratified_sample(
            records,
            limit,
            stratum=lambda record: str(record["pair"]["category"]),
            item_id=lambda record: str(record["pair"]["pair_id"]),
            seed=20260827,
        )
    for index, record in enumerate(records):
        pair = record["pair"]
        source = SceneSpec.from_dict(pair["source"])
        atom = build_primary_atom(source)
        question = build_question(atom)
        intended = str(pair["source_answer"])
        pixel = str(pair["counterfactual_answer"])
        if intended == pixel:
            raise AssertionError(f"Tier-B pair does not conflict: {pair['pair_id']}")
        counterfactual_path = Path(record["counterfactual_image"]).resolve()
        hard_path = output / "hard_rgb" / f"{pair['pair_id']}.png"
        with Image.open(counterfactual_path) as opened:
            counterfactual = opened.convert("RGB")
            hard_render(counterfactual, hard_path)
        with Image.open(hard_path) as opened:
            hard_rgb = opened.convert("RGB")
        prompt_context = f"Original instruction: {pair['intent_prompt']}\n{question.text}"
        rgb_only = _model_answer(adapter, counterfactual, question.text, question)
        prompt_only = _model_answer(adapter, blank, prompt_context, question)
        rgb_prompt = _model_answer(adapter, counterfactual, prompt_context, question)
        hard_answer = _model_answer(adapter, hard_rgb, question.text, question)
        detector_result = detector.observe(
            make_blind_request(hard_path, (question,), f"e1-{index:04d}")
        )
        detector_answer = detector_result.answers[0].normalized_answer
        trial = ContextTrial(
            trial_id=str(pair["pair_id"]),
            intended_answer=intended,
            pixel_answer=pixel,
            rgb_only_answer=rgb_only,
            prompt_only_answer=prompt_only,
            rgb_prompt_answer=rgb_prompt,
            hard_render_answer=hard_answer,
            counterfactual_answer=detector_answer,
        )
        trials.append(trial)
        by_category[str(pair["category"])].append(trial)
        rows.append(
            {
                "pair_id": pair["pair_id"],
                "category": pair["category"],
                "family": question.family.value,
                "question_id": question.question_id,
                "intended_answer": intended,
                "pixel_answer": pixel,
                "rgb_only_answer": rgb_only,
                "prompt_only_answer": prompt_only,
                "rgb_prompt_answer": rgb_prompt,
                "hard_render_answer": hard_answer,
                "matched_detector_answer": detector_answer,
                "detector_id": detector_result.observer_id,
                "detector_revision": detector_result.observer_revision,
            }
        )
    if not trials:
        raise ValueError("E1 received no Tier-B records")
    report = {
        "schema_version": 1,
        "samples": len(trials),
        "metrics": compute_e1_metrics(trials),
        "by_category": {
            category: compute_e1_metrics(category_trials)
            for category, category_trials in sorted(by_category.items())
        },
        "rows": rows,
        "isolation_statement": (
            "The matched detector received only absolute hard-rendered RGB paths, hashes, and atomic "
            "questions. Intent prompts were used only inside the deliberately prompted Show-o conditions."
        ),
    }
    atomic_write_json(output / "e1_report.json", report)
    return report
