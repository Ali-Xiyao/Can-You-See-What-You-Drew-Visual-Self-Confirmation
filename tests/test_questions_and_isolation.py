from __future__ import annotations

import json
import sys

from selfsight.data.questions import build_primary_atom, build_question, normalize_answer
from selfsight.data.renderer import render_scene
from selfsight.observers.client import ObserverServiceClient
from selfsight.observers.protocol import assert_blind_wire_payload, encode_request
from selfsight.rfo.isolation import hard_render, make_blind_request
from selfsight.schemas import QuestionFamily, QuestionFormat
from selfsight.utils.hashing import rgb_sha256


def test_answer_normalization_and_option_order(registered_splits):
    scene = registered_splits["tier_a_outcome"][0]
    atom = build_primary_atom(scene)
    open_question = build_question(atom)
    assert normalize_answer(f"The answer is {atom.answer}.", open_question) == atom.answer
    assert normalize_answer("I cannot tell from the image", open_question) is None
    forced = build_question(atom, QuestionFormat.FORCED_CHOICE, choice_order_seed=1)
    index = forced.choices.index(atom.answer)
    assert normalize_answer(chr(65 + index), forced) == atom.answer


def test_generated_count_normalization_accepts_nonnegative_counts_outside_ontology(
    registered_splits,
):
    scene = next(
        item for item in registered_splits["tier_a_outcome"] if item.family == QuestionFamily.COUNT
    )
    question = build_question(build_primary_atom(scene))
    assert normalize_answer("There are 6 visible squares.", question) == "6"
    assert normalize_answer("06", question) == "6"
    assert normalize_answer("-6", question) is None
    assert normalize_answer("6 or 7", question) is None


def test_hard_render_and_wire_context_are_blind(tmp_path, registered_splits):
    scene = registered_splits["tier_a_probe"][0]
    atom = build_primary_atom(scene)
    question = build_question(atom)
    destination = tmp_path / "hard-render.png"
    evidence = hard_render(render_scene(scene), destination)
    assert evidence["rgb_sha256"] == rgb_sha256(destination)
    request = make_blind_request(destination, (question,), "isolation-test")
    wire = encode_request(request)
    payload = json.loads(wire)
    assert_blind_wire_payload(payload)
    assert "expected_answer" not in wire
    assert "original_prompt" not in wire
    assert "scene_graph" not in wire
    assert scene.prompt not in wire


def test_mock_observer_subprocess_only_reads_rgb(tmp_path, registered_splits):
    scene = registered_splits["tier_a_probe"][1]
    atom = build_primary_atom(scene)
    question = build_question(atom)
    image_path = tmp_path / "observer.png"
    hard_render(render_scene(scene), image_path)
    command = [
        sys.executable,
        "-m",
        "selfsight.observers.service",
        "--backend",
        "mock",
        "--device",
        "cpu",
    ]
    log_path = tmp_path / "wire.jsonl"
    with ObserverServiceClient(command, log_path) as client:
        result = client.observe(make_blind_request(image_path, (question,), "subprocess-test"))
    assert result.answers[0].normalized_answer == atom.answer
    wire = log_path.read_text(encoding="utf-8")
    assert "expected_answer" not in wire
    assert scene.prompt not in wire
