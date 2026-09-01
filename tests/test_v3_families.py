"""The three v3 primary families: existence, spatial, binding.

`binding` used to be a relabelled `color`: both branches of `build_primary_atom`
emitted `subject="shape=X"; predicate="color"`, so the question was answerable
from shape alone and the only difference was a hard-coded object count. These
tests pin the property that makes it a distinct family -- two objects share the
bound shape, so the subject must carry size -- and the reference-uniqueness rule
that lets object count act as a difficulty knob at all.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from PIL import Image

from selfsight.data.generator import MAX_OBJECTS_PER_SCENE, generate_split
from selfsight.data.questions import (
    build_gold_atoms,
    build_primary_atom,
    build_question,
    parse_subject,
)
from selfsight.data.renderer import render_scene
from selfsight.data.verifier import (
    _matches,
    detect_objects,
    evaluate_atom,
    verify_image,
)
from selfsight.schemas import QuestionFamily, QuestionFormat

PRIMARY = (QuestionFamily.EXISTENCE, QuestionFamily.SPATIAL, QuestionFamily.BINDING)
DIFFICULTIES = (3, 4, 5, 6)


def scenes_for(family, objects_per_scene, total=24, seed=20260901):
    return generate_split(
        split="tier_a_probe",
        total=total,
        seed=seed,
        families=[family],
        objects_per_scene=objects_per_scene,
    )


@pytest.mark.parametrize("objects_per_scene", DIFFICULTIES)
def test_binding_scenes_have_exactly_two_same_shape_competitors(objects_per_scene):
    for scene in scenes_for(QuestionFamily.BINDING, objects_per_scene):
        bound = scene.objects[0].shape
        competitors = [item for item in scene.objects if item.shape == bound]

        assert len(competitors) == 2, "binding needs a genuine shape collision"
        assert competitors[0].size != competitors[1].size, "size must separate them"
        assert competitors[0].color != competitors[1].color, "colour must be the answer"


@pytest.mark.parametrize("objects_per_scene", DIFFICULTIES)
def test_binding_subject_carries_size_and_shape_alone_is_ambiguous(objects_per_scene):
    """The defining property: `shape=X` is not enough to answer."""

    for scene in scenes_for(QuestionFamily.BINDING, objects_per_scene):
        atom = build_primary_atom(scene)
        fields = parse_subject(atom.subject)[0]

        assert fields["size"], "binding subject must include size"

        detections = detect_objects(render_scene(scene))
        assert len(_matches(detections, {"shape": fields["shape"]})) == 2
        assert len(_matches(detections, fields)) == 1


@pytest.mark.parametrize("objects_per_scene", DIFFICULTIES)
def test_binding_question_names_the_size(objects_per_scene):
    for scene in scenes_for(QuestionFamily.BINDING, objects_per_scene):
        atom = build_primary_atom(scene)
        text = build_question(atom, question_format=QuestionFormat.OPEN).text

        assert parse_subject(atom.subject)[0]["size"] in text


@pytest.mark.parametrize("objects_per_scene", DIFFICULTIES)
def test_spatial_referenced_shapes_each_occur_once(objects_per_scene):
    """Distractors may repeat a shape; the two referenced shapes may not."""

    for scene in scenes_for(QuestionFamily.SPATIAL, objects_per_scene):
        atom = build_primary_atom(scene)
        shapes = [item.shape.value for item in scene.objects]
        for group in parse_subject(atom.subject):
            assert shapes.count(group["shape"]) == 1


@pytest.mark.parametrize("family", PRIMARY)
@pytest.mark.parametrize("objects_per_scene", DIFFICULTIES)
def test_reference_renders_verify_at_every_difficulty(family, objects_per_scene):
    """A family whose own reference render fails is broken before any model runs."""

    for scene in scenes_for(family, objects_per_scene, total=12):
        atom = build_primary_atom(scene)
        result = verify_image(render_scene(scene), [atom])

        assert result.coverage == 1.0
        assert result.answers[atom.atom_id] == atom.answer


def test_object_count_is_a_real_knob_not_capped_at_the_shape_count():
    """The old cap was len(SHAPES)=3, which left only two usable settings."""

    assert MAX_OBJECTS_PER_SCENE > 3

    for objects_per_scene in range(2, MAX_OBJECTS_PER_SCENE + 1):
        scenes = scenes_for(QuestionFamily.EXISTENCE, objects_per_scene, total=6)
        assert all(len(scene.objects) == objects_per_scene for scene in scenes)

    with pytest.raises(ValueError, match="between 2 and"):
        scenes_for(QuestionFamily.EXISTENCE, MAX_OBJECTS_PER_SCENE + 1, total=6)


def test_binding_rejects_a_population_too_small_to_collide():
    with pytest.raises(ValueError, match="binding needs"):
        scenes_for(QuestionFamily.BINDING, 2, total=6)


def test_every_object_in_a_scene_is_visually_distinct():
    """(shape, colour) uniqueness keeps distractors from duplicating a referent."""

    for family in PRIMARY:
        for objects_per_scene in DIFFICULTIES:
            for scene in scenes_for(family, objects_per_scene, total=12):
                keys = [(item.shape, item.color) for item in scene.objects]
                assert len(set(keys)) == len(keys)


# --------------------------------------------------------------------- gold atoms
#
# The v3 calibration sweep found the difficulty knob was not measuring
# difficulty. `color` and the strict spatial relations abstain unless the
# subject resolves to exactly one detection; that is guaranteed on a reference
# render and false on 60-85% of Show-o2 generations, so most candidates came
# back unscoreable and `p` was estimated on the surviving minority. These tests
# pin the two properties that fixed it.


def blank_image():
    return Image.new("RGB", (512, 512), (255, 255, 255))


@pytest.mark.parametrize("family", PRIMARY)
@pytest.mark.parametrize("objects_per_scene", DIFFICULTIES)
def test_gold_atoms_hold_on_the_reference_render(family, objects_per_scene):
    """A gold set its own reference render fails is broken before any model runs."""

    for scene in scenes_for(family, objects_per_scene, total=12):
        atoms = build_gold_atoms(scene)
        result = verify_image(render_scene(scene), atoms)

        assert result.coverage == 1.0
        for atom in atoms:
            assert result.answers[atom.atom_id] == atom.answer


@pytest.mark.parametrize("family", PRIMARY)
@pytest.mark.parametrize("objects_per_scene", DIFFICULTIES)
def test_gold_atoms_are_total_on_degenerate_images(family, objects_per_scene):
    """Never abstain: a bad render must score wrong, not unscoreable.

    Blank, and the reference painted twice at an offset so every shape occurs
    more than once -- the duplicate case that made `color` and `left_of` return
    None on generated pixels.
    """

    for scene in scenes_for(family, objects_per_scene, total=8):
        atoms = build_gold_atoms(scene)
        reference = render_scene(scene)
        doubled = reference.copy()
        doubled.paste(reference.resize((256, 256)), (0, 0))

        for image in (blank_image(), doubled):
            detections = detect_objects(image)
            for atom in atoms:
                assert evaluate_atom(detections, atom) is not None


@pytest.mark.parametrize("family", PRIMARY)
@pytest.mark.parametrize("objects_per_scene", DIFFICULTIES)
def test_a_blank_image_never_scores_correct(family, objects_per_scene):
    """The omission loophole.

    A negative claim is nearly free in a generative setting: a model that draws
    nothing satisfies it. Existence negatives scored p=0.82-0.93 against
    0.25-0.42 for positives. Every gold set therefore asserts that something is
    present, so drawing nothing can never be selected as correct.
    """

    detections = detect_objects(blank_image())
    for scene in scenes_for(family, objects_per_scene, total=12):
        atoms = build_gold_atoms(scene)
        answers = [evaluate_atom(detections, atom) for atom in atoms]

        assert any(
            atom.predicate == "exists" and atom.answer == "yes" for atom in atoms
        ), "a gold set with no positive claim can be satisfied by an empty canvas"
        assert not all(
            answer == atom.answer for answer, atom in zip(answers, atoms, strict=True)
        )


def test_binding_gold_rejects_the_miscombination():
    """Right shape and size, the competitor's colour: the binding failure mode."""

    for scene in scenes_for(QuestionFamily.BINDING, 3, total=8):
        target = next(
            item
            for item in scene.objects
            if item.object_id == scene.metadata["target_object_id"]
        )
        competitor = next(
            item
            for item in scene.objects
            if item.shape == target.shape and item.object_id is not target.object_id
        )
        swapped = replace(
            scene,
            objects=tuple(
                replace(item, color=competitor.color) if item.object_id == target.object_id
                else replace(item, color=target.color) if item.object_id == competitor.object_id
                else item
                for item in scene.objects
            ),
        )
        detections = detect_objects(render_scene(swapped))
        atoms = build_gold_atoms(scene)

        assert not all(
            evaluate_atom(detections, atom) == atom.answer for atom in atoms
        ), "swapping the two same-shape colours must not still score correct"


def test_spatial_gold_survives_a_duplicated_referent():
    """Strict `left_of` abstains here; the existential form must still answer."""

    for scene in scenes_for(QuestionFamily.SPATIAL, 3, total=8):
        relation_atom, grounding_atom = build_gold_atoms(scene)

        assert relation_atom.predicate.startswith("exists_")
        assert grounding_atom.predicate == "exists"

        reference = render_scene(scene)
        doubled = reference.copy()
        doubled.paste(reference.resize((256, 256)), (0, 0))
        detections = detect_objects(doubled)

        assert evaluate_atom(detections, relation_atom) in {"yes", "no"}


def test_manifest_records_carry_gold_atoms():
    from selfsight.data.questions import build_gold_atoms as build

    scene = scenes_for(QuestionFamily.BINDING, 3, total=1)[0]
    atoms = build(scene)

    assert len(atoms) == 2
    assert [atom.atom_id for atom in atoms] == [
        f"{scene.scene_id}:gold0",
        f"{scene.scene_id}:gold1",
    ]
