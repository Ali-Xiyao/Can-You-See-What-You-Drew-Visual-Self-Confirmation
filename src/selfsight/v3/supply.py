"""Gate A runner: measure natural vs bounded-bank candidate supply.

This is the first v3.0 experiment and the only one that must complete before any
other. It answers a single question: does this backbone produce enough pools that
contain both a verifier-correct and a verifier-incorrect candidate for a
selection-based experiment to be possible at all?

It reports the natural K=4 rate and the bank rate on the *same prompts*, so the
cost of the bank is explicit rather than hidden. Generation is resumable per
prompt; an interrupted run re-reads its packets instead of regenerating.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

from selfsight.data.generated_verifier import verify_generated_image
from selfsight.schemas import Atom, CandidateRecord, SceneSpec
from selfsight.utils.jsonl import atomic_write_json, atomic_write_jsonl
from selfsight.v3.bank import (
    BankCandidate,
    assert_disjoint_seed_domains,
    bank_supply_report,
    build_balanced_pool,
    search_seeds,
)


class GenerationAdapter(Protocol):
    """The subset of the backbone contract this probe needs."""

    def generate_images(
        self,
        prompts: Sequence[str],
        seeds: Sequence[int],
        output_dir: Path,
        tag: str,
    ) -> Sequence[CandidateRecord]: ...


def _gold_score(image_path: str, atoms: Sequence[Atom]) -> tuple[float, bool]:
    """Verifier score over the gold atoms. Binary; abstention is explicit.

    A conjunction: the candidate is correct only if every gold atom holds. Any
    single abstention makes the whole candidate unscoreable rather than wrong,
    so an atom set that abstains stays visible in the pool statistics instead of
    being silently counted as a failure.
    """

    result = verify_generated_image(image_path, atoms)
    answers = [result.answers[atom.atom_id] for atom in atoms]
    if any(answer is None for answer in answers):
        return (float("nan"), True)
    correct = all(answer == atom.answer for answer, atom in zip(answers, atoms, strict=True))
    return (1.0 if correct else 0.0, False)


def _score_bank(
    candidates: Sequence[CandidateRecord],
    atoms: Sequence[Atom],
) -> list[BankCandidate]:
    scored = []
    for candidate in candidates:
        score, abstained = _gold_score(candidate.image_path, atoms)
        scored.append(
            BankCandidate(
                candidate_id=candidate.candidate_id,
                sampling_seed=candidate.sampling_seed,
                gold_score=score,
                abstained=abstained,
            )
        )
    return scored


def run_bank_probe(
    *,
    records: Sequence[Mapping[str, Any]],
    adapter: GenerationAdapter,
    output_dir: str | Path,
    bank_size: int,
    candidate_k: int,
    min_per_side: int,
    min_balanced_rate: float,
    min_informative_pools: int,
    reserved_seeds: Sequence[int] = (),
) -> dict[str, Any]:
    """Search a bounded bank per prompt and report Gate A.

    The first `candidate_k` bank entries double as the natural-sampling control:
    they are an unfiltered draw, so comparing them with the balanced subset
    isolates the effect of the bank rather than confounding it with a different
    prompt set.
    """

    if bank_size < candidate_k:
        raise ValueError("bank_size must be at least candidate_k")
    root = Path(output_dir)
    packets = root / "packets"
    packets.mkdir(parents=True, exist_ok=True)

    pools = []
    natural_informative = 0
    natural_rows = []
    manifest: list[CandidateRecord] = []

    for index, record in enumerate(records):
        scene = SceneSpec.from_dict(dict(record["scene"]))
        atom = Atom.from_dict(dict(record["atom"]))
        # Manifests built before gold atoms existed score on the question atom.
        gold_atoms = tuple(
            Atom.from_dict(dict(item)) for item in record.get("gold_atoms", ())
        ) or (atom,)
        family = str(record.get("family", scene.family.value))
        seeds = search_seeds(scene.scene_id, bank_size)
        assert_disjoint_seed_domains(seeds, reserved_seeds)

        packet_path = packets / f"{index:04d}-{scene.scene_id}.json"
        if packet_path.is_file():
            payload = json.loads(packet_path.read_text(encoding="utf-8"))
            candidates = [CandidateRecord.from_dict(item) for item in payload["candidates"]]
            scored = [
                BankCandidate(
                    candidate_id=str(item["candidate_id"]),
                    sampling_seed=int(item["sampling_seed"]),
                    gold_score=float(item["gold_score"]),
                    abstained=bool(item["abstained"]),
                )
                for item in payload["scored"]
            ]
        else:
            generated = adapter.generate_images(
                (scene.prompt,) * bank_size,
                seeds,
                root / "candidates",
                f"v3-bank-{index:04d}",
            )
            candidates = [
                replace(item, prompt_id=scene.scene_id, scene_id=scene.scene_id)
                for item in generated
            ]
            scored = _score_bank(candidates, gold_atoms)
            atomic_write_json(
                packet_path,
                {
                    "schema_version": 1,
                    "scene_id": scene.scene_id,
                    "family": family,
                    "search_seeds": list(seeds),
                    "candidates": [item.to_dict() for item in candidates],
                    "scored": [item.to_dict() for item in scored],
                },
            )
        manifest.extend(candidates)

        natural = build_balanced_pool(
            scene.scene_id, family, scored[:candidate_k], k=candidate_k, min_per_side=min_per_side
        )
        natural_informative += int(natural.informative)
        natural_rows.append(natural.to_dict())

        pools.append(
            build_balanced_pool(
                scene.scene_id, family, scored, k=candidate_k, min_per_side=min_per_side
            )
        )

    families = sorted({pool.family for pool in pools})
    report = bank_supply_report(
        pools,
        families=families,
        min_balanced_rate=min_balanced_rate,
        min_informative_pools=min_informative_pools,
        natural_informative_rate=natural_informative / len(pools) if pools else None,
    )
    report.update(
        {
            "schema_version": 1,
            "benchmark_version": "3.0",
            "stage": "v3_gate_a_candidate_supply",
            "bank_size": bank_size,
            "candidate_k": candidate_k,
            "min_per_side": min_per_side,
            "natural_informative_pools": natural_informative,
            "natural_pools": len(natural_rows),
        }
    )
    atomic_write_jsonl(root / "pools.jsonl", (pool.to_dict() for pool in pools))
    atomic_write_jsonl(root / "natural_pools.jsonl", natural_rows)
    atomic_write_jsonl(root / "candidate_manifest.jsonl", (item.to_dict() for item in manifest))
    atomic_write_json(root / "gate_a.json", report)
    return report
