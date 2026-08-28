"""E1 context-matrix and Tier-B pixel-override experiment."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

from PIL import Image

from selfsight.data.questions import build_primary_atom, build_question
from selfsight.data.subsets import stable_stratified_sample
from selfsight.observers.client import ObserverServiceClient
from selfsight.rfo.isolation import hard_render, make_blind_request
from selfsight.rfo.metrics import ContextTrial, compute_e1_metrics
from selfsight.schemas import SceneSpec
from selfsight.utils.hashing import sha256_json
from selfsight.utils.jsonl import atomic_write_json, read_jsonl


def _model_answer(adapter: Any, image_path: str | Path, text: str, question: Any) -> str | None:
    contextualized = replace(
        question,
        question_id=f"{question.question_id}-{sha256_json(text)[:12]}",
        text=text,
    )
    result = adapter.observe_atoms(image_path, (contextualized,))
    if len(result.answers) != 1:
        raise RuntimeError("E1 adapter must return exactly one answer per context")
    return result.answers[0].normalized_answer


def _filter_eligible_records(
    records: list[dict[str, Any]],
    eligible_families: tuple[str, ...] | None,
    *,
    require_all_eligible_present: bool = True,
) -> tuple[list[dict[str, Any]], list[str]]:
    if eligible_families is None:
        return records, []
    eligible = set(eligible_families)
    if not eligible:
        raise ValueError("E1 requires at least one eligible family")
    observed = {str(record["pair"]["source"]["family"]) for record in records}
    unknown = sorted(eligible.difference(observed))
    if unknown and require_all_eligible_present:
        raise ValueError(f"E1 eligible families are absent from the manifest: {unknown}")
    retained = [
        record
        for record in records
        if str(record["pair"]["source"]["family"]) in eligible
    ]
    return retained, sorted(observed.difference(eligible))


def run_e1_tier_b(
    *,
    adapter: Any,
    detector: ObserverServiceClient,
    manifest_path: str | Path,
    output_dir: str | Path,
    limit: int | None = None,
    eligible_families: tuple[str, ...] | None = None,
    evidence_bindings: dict[str, Any] | None = None,
    non_formal: bool = False,
    exploratory_authorization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the fixed five-context matrix without exposing intent to the detector process."""

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    resolution = int(getattr(adapter, "native_resolution", 512))
    blank_path = hard_render(
        Image.new("RGB", (resolution, resolution), (245, 245, 245)), output / "blank.png"
    )["path"]
    trials: list[ContextTrial] = []
    rows: list[dict[str, Any]] = []
    by_category: dict[str, list[ContextTrial]] = defaultdict(list)
    records, excluded_families = _filter_eligible_records(
        list(read_jsonl(manifest_path)),
        eligible_families,
        require_all_eligible_present=not non_formal,
    )
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
        prompt_context = f"Original instruction: {pair['intent_prompt']}\n{question.text}"
        rgb_only = _model_answer(adapter, counterfactual_path, question.text, question)
        prompt_only = _model_answer(adapter, blank_path, prompt_context, question)
        rgb_prompt = _model_answer(adapter, counterfactual_path, prompt_context, question)
        hard_answer = _model_answer(adapter, hard_path, question.text, question)
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
        "non_formal": bool(non_formal),
        "exploratory_authorization": exploratory_authorization,
        "samples": len(trials),
        "eligible_families": sorted(eligible_families) if eligible_families else None,
        "manifest_present_eligible_families": sorted(
            {
                str(record["pair"]["source"]["family"])
                for record in records
            }
        ),
        "excluded_manifest_families": excluded_families,
        "context_resolution": resolution,
        "evidence_bindings": evidence_bindings or {},
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
