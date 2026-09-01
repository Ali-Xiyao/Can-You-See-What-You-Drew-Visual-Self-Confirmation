"""Tests for the Tier B edits.

The point of Tier B is that the truth comes from the edit rather than from a
detector, so an edit that quietly does the wrong thing produces confidently
mislabelled trials. These pin the properties the labels depend on.
"""

from __future__ import annotations

import numpy as np
import pytest

from selfsight.v4.tierb import (
    COLOUR_HUES,
    RecolourPlan,
    edit_accepted,
    flip,
    flip_question,
    hue_distance,
    recolour,
    recolour_question,
    target_colours,
)


def _solid(colour: tuple[int, int, int], size: int = 40) -> np.ndarray:
    return np.tile(np.array(colour, dtype=np.uint8), (size, size, 1))


def _scene() -> np.ndarray:
    """A grey field with a blue square on the left half."""
    image = np.full((40, 80, 3), 128, dtype=np.uint8)
    image[10:30, 5:25] = (30, 60, 200)
    return image


def test_recolour_changes_the_named_colour_of_the_box():
    image = _scene()
    edited, share, _ = recolour(image, (5, 10, 25, 30), "blue", "red")
    assert share > 0.5
    patch = edited[15:25, 10:20].reshape(-1, 3).mean(axis=0)
    assert patch[0] > patch[2], "should now read as red, not blue"


def test_recolour_leaves_everything_outside_the_box_alone():
    image = _scene()
    edited, _, _ = recolour(image, (5, 10, 25, 30), "blue", "red")
    outside = np.ones(image.shape[:2], dtype=bool)
    outside[10:30, 5:25] = False
    assert np.array_equal(edited[outside], image[outside])


def test_recolour_preserves_shading_within_the_object():
    """A flat fill would make the object a single colour and stop looking like
    an object. Rotating the hue must keep the variation in value."""
    image = _scene()
    image[10:20, 5:25] = (15, 30, 100)  # a darker half, as shading
    edited, _, _ = recolour(image, (5, 10, 25, 30), "blue", "green")
    top = edited[12:18, 8:22].astype(float).max(axis=-1).mean()
    bottom = edited[22:28, 8:22].astype(float).max(axis=-1).mean()
    assert bottom - top > 30, "the light and dark halves must stay different"


def test_recolour_skips_the_grey_background_inside_the_box():
    """The saturation floor is what keeps the surface out of the mask."""
    image = _scene()
    edited, share, _ = recolour(image, (0, 0, 80, 40), "blue", "red")
    assert share < 0.5, "only the square is blue; the grey field must be left"
    assert np.array_equal(edited[35:40, 60:80], image[35:40, 60:80])


def test_recolour_of_an_absent_colour_changes_nothing():
    image = _scene()
    edited, share, _ = recolour(image, (5, 10, 25, 30), "yellow", "red")
    assert share == 0.0
    assert np.array_equal(edited, image)


def test_a_background_the_same_colour_as_the_object_is_not_edited():
    """The failure the first sample sheets showed twice: a green wall behind a
    green pear is inside the hue window, is fully saturated, and forms a bigger
    blob than the pear. Taking the largest blob rotated the wall and left the
    pear alone, giving a picture with a coloured rectangle in it.

    The wall reaches the border of the padded patch and the object cannot, so
    the wall is dropped and the edit reports nothing rather than something
    wrong."""
    image = np.full((80, 80, 3), (40, 180, 60), dtype=np.uint8)  # green wall
    image[30:50, 30:50] = (200, 190, 40)  # a yellow object, not green
    edited, share, _ = recolour(image, (28, 28, 52, 52), "green", "yellow")
    assert share == 0.0
    assert np.array_equal(edited, image)


def test_an_object_inside_a_padded_box_is_still_edited():
    """The guard must not throw away the ordinary case along with the wall."""
    image = np.full((80, 80, 3), 128, dtype=np.uint8)
    image[30:50, 30:50] = (30, 60, 200)  # a blue object on grey
    edited, share, centre = recolour(image, (28, 28, 52, 52), "blue", "red")
    assert share > 0.5 and centre > 0.9
    patch = edited[35:45, 35:45].reshape(-1, 3).mean(axis=0)
    assert patch[0] > patch[2]


def test_an_object_running_off_the_image_edge_is_refused():
    """Its own pixels reach the patch border, so object and background cannot be
    told apart. Refusing is right; guessing would mislabel the pair."""
    image = np.full((40, 40, 3), 128, dtype=np.uint8)
    image[:, :20] = (30, 60, 200)
    _, share, _ = recolour(image, (0, 0, 20, 40), "blue", "red")
    assert share == 0.0


def test_a_mask_that_found_the_object_scores_high_in_the_centre():
    image = _scene()
    _, _, centre = recolour(image, (5, 10, 25, 30), "blue", "red")
    assert centre > 0.9


def test_flip_is_its_own_inverse():
    image = _scene()
    assert np.array_equal(flip(flip(image)), image)


def test_flip_moves_the_object_to_the_other_side():
    image = _scene()
    left_before = image[:, :40].astype(int).sum()
    left_after = flip(image)[:, :40].astype(int).sum()
    assert left_before != left_after


def test_a_target_colour_worn_by_another_object_is_refused():
    """Two red things in one picture make "what colour is the mug" ambiguous."""
    assert "red" not in target_colours("mug", "blue", ["red"], None)


def test_a_target_outside_the_plausible_set_is_refused():
    assert target_colours("lemon", "yellow", [], {"yellow", "green"}) == ["green"]


def test_brown_is_never_a_target():
    assert "brown" not in target_colours("mug", "blue", [], None)


def test_a_target_too_close_in_hue_is_refused():
    """Orange onto red is not a visible edit, and a trial whose two options look
    the same is scored against the model for our failure."""
    assert "orange" not in target_colours("mug", "red", [], None)


def test_hue_distance_wraps_around_the_wheel():
    assert hue_distance(0.98, 0.02) == pytest.approx(0.04)


@pytest.mark.parametrize("gold_first", [True, False])
def test_the_pair_carries_opposite_gold(gold_first):
    question = recolour_question("mug", "blue", "red", 1, gold_first)
    assert question["gold_original"] != question["gold_edited"]
    spatial = flip_question("mug", "plate", True, gold_first)
    assert spatial["gold_original"] != spatial["gold_edited"]


def test_both_members_are_asked_the_identical_question():
    """The pair differs in pixels only. If the wording differed too, a change in
    the answer could not be attributed to the pixels."""
    question = recolour_question("mug", "blue", "red", 1, True)
    assert "A. blue" in question["question"] and "B. red" in question["question"]


def _plan() -> RecolourPlan:
    return RecolourPlan("i.png", "mug", (0, 0, 10, 10), "blue", "red", False)


def test_an_edit_that_moved_only_the_target_is_accepted():
    before = [{"object": "mug", "color": "blue"}, {"object": "plate", "color": "white"}]
    after = [{"object": "mug", "color": "red"}, {"object": "plate", "color": "white"}]
    assert edit_accepted(_plan(), before, after) == (True, "ok")


def test_an_edit_the_detectors_did_not_notice_is_rejected():
    before = [{"object": "mug", "color": "blue"}]
    assert edit_accepted(_plan(), before, list(before))[0] is False


def test_an_edit_that_also_changed_a_second_object_is_rejected():
    before = [{"object": "mug", "color": "blue"}, {"object": "plate", "color": "blue"}]
    after = [{"object": "mug", "color": "red"}, {"object": "plate", "color": "red"}]
    accepted, reason = edit_accepted(_plan(), before, after)
    assert accepted is False and reason == "collateral_change"


def test_an_edit_that_made_the_object_a_third_colour_is_rejected():
    before = [{"object": "mug", "color": "blue"}]
    after = [{"object": "mug", "color": "purple"}]
    accepted, reason = edit_accepted(_plan(), before, after)
    assert accepted is False and reason == "target_not_recoloured"


def test_the_table_is_not_counted_as_collateral():
    """Surfaces are excluded from the object list everywhere else; if they were
    compared here, every edit near a cloth would be thrown away."""
    before = [{"object": "mug", "color": "blue"}, {"object": "table", "color": "brown"}]
    after = [{"object": "mug", "color": "red"}]
    assert edit_accepted(_plan(), before, after) == (True, "ok")
