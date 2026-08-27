from __future__ import annotations

from collections import Counter

from selfsight.data.counterfactuals import build_tier_b
from selfsight.data.generated_verifier import verify_generated_image
from selfsight.data.questions import build_primary_atom
from selfsight.data.renderer import render_scene
from selfsight.data.verifier import verify_image
from selfsight.schemas import QuestionFamily


def test_reference_renderer_and_verifier_by_family(registered_splits):
    scenes = registered_splits["tier_a_outcome"]
    sampled = []
    for family in QuestionFamily:
        sampled.extend([scene for scene in scenes if scene.family == family][:12])
    for scene in sampled:
        atom = build_primary_atom(scene)
        result = verify_image(render_scene(scene), [atom])
        assert result.coverage == 1.0
        assert result.answers[atom.atom_id] == atom.answer


def test_generated_contour_verifier_preserves_reference_semantics(registered_splits):
    scenes = registered_splits["tier_a_outcome"]
    sampled = []
    for family in QuestionFamily:
        sampled.extend([scene for scene in scenes if scene.family == family][:12])
    for scene in sampled:
        atom = build_primary_atom(scene)
        result = verify_generated_image(render_scene(scene), [atom])
        assert result.coverage == 1.0
        assert result.answers[atom.atom_id] == atom.answer


def test_tier_b_registered_composition_and_pixel_flip(registered_splits):
    pairs = build_tier_b(registered_splits["tier_a_outcome"])
    assert Counter(pair.category for pair in pairs) == {
        "count_delete": 100,
        "color_change": 100,
        "relation_left_right": 50,
        "relation_size": 50,
        "binding_swap": 100,
    }
    categories = sorted({pair.category for pair in pairs})
    sampled = [next(pair for pair in pairs if pair.category == category) for category in categories]
    for pair in sampled:
        source_atom = build_primary_atom(pair.source)
        changed_atom = build_primary_atom(pair.counterfactual)
        source_answer = verify_image(render_scene(pair.source), [source_atom]).answers[source_atom.atom_id]
        changed_answer = verify_image(render_scene(pair.counterfactual), [changed_atom]).answers[changed_atom.atom_id]
        assert source_answer == source_atom.answer
        assert changed_answer == changed_atom.answer
        assert source_answer != changed_answer
