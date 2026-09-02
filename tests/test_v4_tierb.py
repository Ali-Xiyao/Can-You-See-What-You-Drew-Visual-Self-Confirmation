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
    DeletePlan,
    RecolourPlan,
    delete,
    delete_accepted,
    delete_confirmed,
    delete_question,
    deletion_mask,
    edit_accepted,
    object_mask,
    sham_box,
    flip,
    flip_question,
    hue_distance,
    recolour,
    recolour_question,
    target_colours,
)

import random


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


# ------------------------------------------------------------------ deletion


def _still_life() -> np.ndarray:
    """A grey field with a blue square touching a white square."""
    image = np.full((60, 100, 3), 120, dtype=np.uint8)
    image[15:45, 10:40] = (30, 60, 200)
    image[15:45, 40:70] = (250, 250, 250)
    return image


def _fill_black(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """A stand-in inpainter, so the tests do not need the real one."""
    out = image.copy()
    out[mask] = 0
    return out


def test_a_white_object_is_segmented_by_value_not_hue():
    """The bug that left half a bottle standing: white has no hue, so the old
    code refused it and fell back to protecting its whole box."""
    mask, share, _ = object_mask(_still_life(), (40, 15, 70, 45), "white")
    assert share > 0.8
    assert mask[30, 55] and not mask[30, 20]


def test_the_deletion_covers_the_whole_target_box():
    image = _still_life()
    mask, centre = deletion_mask(image, (10, 15, 40, 45), "blue",
                                 [((40, 15, 70, 45), "white")])
    assert centre > 0.95
    assert mask[20:40, 15:35].all()


def test_the_deletion_spares_the_neighbour_it_touches():
    image = _still_life()
    mask, _ = deletion_mask(image, (10, 15, 40, 45), "blue",
                            [((40, 15, 70, 45), "white")])
    assert not mask[30, 55]


def test_only_masked_pixels_move():
    """`nothing else changed` has to be true by construction, not by hope: the
    detectors are asked whether one object is gone, and a filler that quietly
    resampled the rest of the frame would make that question unanswerable."""
    image = _still_life()
    out, mask, _ = delete(image, (10, 15, 40, 45), "blue", _fill_black,
                          [((40, 15, 70, 45), "white")])
    assert (out[~mask] == image[~mask]).all()
    assert not (out[mask] == image[mask]).all()


def test_a_neighbour_sitting_in_the_target_scores_low_centre_coverage():
    """A spoon lying on the plate being deleted. Rejecting is right: the
    alternative is an image with half an object still in it."""
    image = np.full((60, 100, 3), 120, dtype=np.uint8)
    image[10:50, 10:60] = (250, 250, 250)      # a white plate
    image[20:40, 20:50] = (30, 60, 200)        # a blue spoon lying on it
    _, centre = deletion_mask(image, (10, 10, 60, 50), "white",
                              [((20, 20, 50, 40), "blue")])
    assert centre < 0.8


def _delete_plan(sham: bool = False) -> DeletePlan:
    return DeletePlan("a.png", "apple", "red", (0.0, 0.0, 1.0, 1.0), sham)


def test_a_deletion_the_detectors_confirm_is_accepted():
    before = [{"object": "apple", "color": "red"},
              {"object": "mug", "color": "blue"}]
    after = [{"object": "mug", "color": "blue"}]
    assert delete_accepted(_delete_plan(), before, after) == (True, "ok")


def test_a_deletion_that_invented_a_replacement_is_rejected():
    """The pilot's red mug came out as a beige egg. The mask cannot leave the
    object behind -- it is the whole box -- so this is the failure that matters,
    and catching it here is what licenses a crude editor."""
    before = [{"object": "apple", "color": "red"},
              {"object": "mug", "color": "blue"}]
    after = [{"object": "mug", "color": "blue"},
             {"object": "egg", "color": "white"}]
    assert delete_accepted(_delete_plan(), before, after)[1] == "collateral_change"


def test_a_deletion_that_took_the_neighbour_with_it_is_rejected():
    before = [{"object": "apple", "color": "red"},
              {"object": "mug", "color": "blue"}]
    assert delete_accepted(_delete_plan(), before, [])[1] == "collateral_change"


def test_an_object_the_filler_left_standing_is_rejected():
    before = [{"object": "apple", "color": "red"},
              {"object": "mug", "color": "blue"}]
    assert delete_accepted(_delete_plan(), before, before)[1] == "unchanged"


def test_a_sham_is_accepted_only_when_the_list_did_not_move():
    before = [{"object": "apple", "color": "red"}]
    assert delete_accepted(_delete_plan(sham=True), before, before) == (True, "ok")
    assert not delete_accepted(_delete_plan(sham=True), before, [])[0]


def test_confirmation_asks_the_opposite_question_of_a_sham():
    gone = [{"object": "mug", "color": "blue"}]
    assert delete_confirmed(_delete_plan(), gone)
    assert not delete_confirmed(_delete_plan(sham=True), gone)


@pytest.mark.parametrize("gold_first", [True, False])
def test_the_deletion_pair_carries_opposite_gold(gold_first):
    built = delete_question("apple", "red", gold_first)
    assert built["gold_original"] != built["gold_edited"]


def test_both_members_of_a_deletion_pair_are_asked_the_same_question():
    """A gap between the members has to be about the picture, so the wording
    cannot differ between them."""
    first = delete_question("apple", "red", True)
    assert first["question"] == delete_question("apple", "red", True)["question"]
    assert "red apple" in first["question"]


def test_the_article_agrees_with_the_colour_not_the_noun():
    assert "an orange mug" in delete_question("mug", "orange", True)["question"]
    assert "a red apple" in delete_question("apple", "red", True)["question"]


def test_the_sham_window_avoids_every_detection():
    box = sham_box((200, 200), [(0, 0, 100, 100)], 30 * 30, random.Random(0))
    assert box is not None
    x0, y0, x1, y1 = box
    assert x0 >= 100 or y0 >= 100


def test_a_sham_window_that_cannot_fit_is_refused():
    assert sham_box((40, 40), [(0, 0, 40, 40)], 30 * 30, random.Random(0)) is None
