"""Tests for the v4 detector adapters and the LLM-authored corpus checks.

These cover the two places where something outside the pipeline hands us data we
did not produce: a vision model's free-text reply, and an LLM's attempt at a
scene spec. Both are parsed defensively, and the cases here are the ones where a
lenient parse would silently corrupt a measurement rather than fail loudly.
"""

from __future__ import annotations

import pytest

from selfsight.v4.detectors import cached_detections, parse_reply
from selfsight.v4.questions import build_questions, to_atomic
from selfsight.v4.spec import SceneSpec
from selfsight.v4.tasks import TOTAL_OBJECTS, parse_scenes

SPEC = SceneSpec.from_dict(
    {
        "spec_id": "s1",
        "prompt": "two green pears and one white mug on a wooden table",
        "objects": [
            {"object": "pear", "color": "green", "count": 2},
            {"object": "mug", "color": "white", "count": 1},
        ],
    }
)
EXACT = [
    {"object": "pear", "color": "green", "center": [30, 30]},
    {"object": "pear", "color": "green", "center": [80, 30]},
    {"object": "mug", "color": "white", "center": [230, 45]},
]


# ------------------------------------------------------------ reply parsing


def test_parse_reply_normalises_case_and_derives_centre():
    got, error = parse_reply(
        '[{"object":"Apple","color":"Green","box":[100,200,300,400]}]', 512, 512
    )
    assert error is None
    assert got == [
        {
            "object": "apple",
            "color": "green",
            "bbox": [100.0, 200.0, 300.0, 400.0],
            "center": [200.0, 300.0],
        }
    ]


@pytest.mark.parametrize(
    "box,expected",
    [
        ([0, 0, 1000, 1000], [0.0, 0.0, 512.0, 512.0]),  # Qwen's 0-1000 frame
        ([0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 512.0, 512.0]),  # normalised 0-1
        ([10, 10, 300, 300], [10.0, 10.0, 300.0, 300.0]),  # already pixels
    ],
)
def test_box_frame_is_detected_not_assumed(box, expected):
    """A model that honours the pixel request must not be rescaled into nonsense."""
    got, _ = parse_reply(
        f'[{{"object":"mug","color":"white","box":{box}}}]', 512, 512
    )
    assert got[0]["bbox"] == expected


def test_unparseable_reply_is_an_error_not_an_empty_scene():
    """Returning [] here would be scored as "the model drew nothing".

    That is a wrong answer invented by the harness rather than read off the
    pixels, and it would land on exactly the images hardest to read.
    """
    got, error = parse_reply("I'm not able to see the image.", 512, 512)
    assert got == []
    assert error == "no_json_array"


def test_object_without_a_name_is_dropped_but_the_rest_survive():
    got, error = parse_reply(
        '[{"object":"","color":"red"},{"object":"book","color":"red"}]', 512, 512
    )
    assert error is None
    assert [item["object"] for item in got] == ["book"]


def test_missing_colour_stays_none_rather_than_becoming_a_guess():
    got, _ = parse_reply('[{"object":"book"}]', 512, 512)
    assert got[0]["color"] is None


def test_cached_detections_skips_rows_that_are_a_todo_list(tmp_path):
    """A to-do file and a results file have the same shape.

    Merging the former over the latter once blanked real detections; a row
    without a `detections` key is now skipped rather than stored empty.
    """
    path = tmp_path / "d.jsonl"
    path.write_text(
        '{"image_path": "a.png", "detections": [{"object": "mug"}]}\n'
        '{"image_path": "b.png"}\n'
        '{"image_path": "c.png", "error": "no_json_array"}\n',
        encoding="utf-8",
    )
    table = cached_detections(path)
    assert set(table) == {"a.png"}


# -------------------------------------------------------- corpus authoring


def _scene(objects, prompt):
    return {"prompt": prompt, "objects": objects, "relations": [], "surface": "table"}


def test_scene_wider_than_the_measured_band_is_rejected():
    """Five objects measured p = 0.000, so the selection experiment has no headroom."""
    scenes = [
        _scene(
            [{"object": "apple", "color": "red", "count": 2},
             {"object": "mug", "color": "white", "count": 2}],
            "two red apples and two white mugs on a table",
        )
    ]
    accepted, rejected = parse_scenes(str(scenes).replace("'", '"'), prefix="t")
    assert not accepted
    assert f"want {TOTAL_OBJECTS}" in rejected[0]["reason"]


def test_spec_the_prompt_does_not_state_is_rejected():
    """The defining property of a verifiable scene.

    A spec entry the prompt never mentions cannot be ticked off against the
    image, because the generator was never asked for it.
    """
    scenes = [
        _scene(
            [{"object": "apple", "color": "red", "count": 2},
             {"object": "candle", "color": "white", "count": 1}],
            "two red apples on a table",
        )
    ]
    accepted, rejected = parse_scenes(str(scenes).replace("'", '"'), prefix="t")
    assert not accepted
    assert "does not mention candle" in rejected[0]["reason"]


def test_a_well_formed_three_object_scene_is_accepted():
    scenes = [
        _scene(
            [{"object": "apple", "color": "red", "count": 2},
             {"object": "mug", "color": "white", "count": 1}],
            "two red apples and one white mug on a table",
        )
    ]
    accepted, rejected = parse_scenes(str(scenes).replace("'", '"'), prefix="t")
    assert not rejected
    assert accepted[0].n_objects == TOTAL_OBJECTS


# ------------------------------------------------------- observer protocol


def test_to_atomic_preserves_the_gold_through_the_frozen_protocol():
    """The v4 trial must survive conversion for `observe_atoms` without drift."""
    from selfsight.data.questions import normalize_answer

    for question in build_questions(SPEC, EXACT, seed=7):
        atomic = to_atomic(question)
        assert atomic.choices == (question.option_a, question.option_b)
        gold_option = question.option_a if question.gold == "A" else question.option_b
        assert atomic.expected_answer == gold_option
        assert normalize_answer(question.gold, atomic) == gold_option
        wrong = "B" if question.gold == "A" else "A"
        assert normalize_answer(wrong, atomic) != gold_option


def test_to_atomic_leaves_an_unreadable_reply_as_an_abstention():
    from selfsight.data.questions import normalize_answer

    atomic = to_atomic(build_questions(SPEC, EXACT, seed=7)[0])
    assert normalize_answer("I am not sure", atomic) is None


def test_implausible_colour_on_a_natural_kind_is_rejected():
    """Asking was not enough; the check has to enforce it.

    The instruction has said "never write a blue banana" since the first draft.
    The first authored corpus still came back with blue apples x10, black x4,
    brown x3 and white x2 out of 55 apple entries. A generator's object prior
    fights a prompt like that, so the image fails for a reason unrelated to the
    capability being measured -- and by a different amount per object, which
    makes object identity a confound of the difficulty tiers.
    """
    scenes = [
        _scene(
            [{"object": "apple", "color": "blue", "count": 2},
             {"object": "mug", "color": "red", "count": 1}],
            "two blue apples and one red mug on a table",
        )
    ]
    accepted, rejected = parse_scenes(str(scenes).replace("'", '"'), prefix="t")
    assert not accepted
    assert rejected[0]["reason"] == "implausible colour: blue apple"


def test_a_manufactured_object_may_be_any_colour():
    """Constraining these would remove variation the corpus needs.

    A mug really does come in blue; an apple does not. Only natural kinds are
    listed, and a category absent from the table is unconstrained.
    """
    scenes = [
        _scene(
            [{"object": "mug", "color": "blue", "count": 2},
             {"object": "book", "color": "purple", "count": 1}],
            "two blue mugs and one purple book on a table",
        )
    ]
    accepted, rejected = parse_scenes(str(scenes).replace("'", '"'), prefix="t")
    assert not rejected
    assert len(accepted) == 1


def test_transparency_is_not_accepted_as_a_colour():
    """"clear" is the absence of a colour, not one of them.

    Three entries of the first clean corpus asked for clear jars. A detector
    asked for an object's dominant colour will never answer "clear", so the entry
    is a guaranteed mismatch that says nothing about the generator.
    """
    scenes = [
        _scene(
            [{"object": "jar", "color": "clear", "count": 2},
             {"object": "bottle", "color": "red", "count": 1}],
            "two clear jars and one red bottle on a table",
        )
    ]
    accepted, rejected = parse_scenes(str(scenes).replace("'", '"'), prefix="t")
    assert not accepted
    assert rejected[0]["reason"] == "not a colour word: clear"

# ------------------------------------------------------------- crop scaling


def test_a_sliver_of_a_box_is_not_scaled_into_an_out_of_memory_error(tmp_path):
    """Short-side-only upsampling ran away on the shapes the ladder gets most.

    A padded box around a spoon or a book spine is a strip. Scaling 500x40 to a
    448 short side gives 4200x450, and a model that tokenises by area then OOMs
    -- which killed a 412-query crop pass at query 50. Both the long side and
    the factor are capped now, and the crop still gets more pixels than it had.
    """
    from PIL import Image

    from selfsight.v4.detectors import VlmDetector

    path = tmp_path / "wide.png"
    Image.new("RGB", (512, 512), "white").save(path)

    seen: list[tuple[int, int]] = []

    def run(image, instruction):
        seen.append((image.width, image.height))
        return '[{"object":"spoon","color":"silver"}]'

    detector = VlmDetector("probe", run)
    detector.detect_crop(str(path), (0, 200, 500, 240))
    width, height = seen[0]
    assert max(width, height) <= 1344
    assert (width, height) != (500, 40)  # it was enlarged, just not without limit


def test_crop_boxes_are_stripped_because_the_caller_has_no_crop_frame():
    """A box in crop coordinates would be read as a box in image coordinates."""
    from PIL import Image

    from selfsight.v4.detectors import VlmDetector

    import tempfile, os

    handle, name = tempfile.mkstemp(suffix=".png")
    os.close(handle)
    Image.new("RGB", (512, 512), "white").save(name)
    try:
        detector = VlmDetector(
            "probe",
            lambda image, instruction:
                '[{"object":"mug","color":"blue","box":[0,0,100,100]}]',
        )
        found = detector.detect_crop(name, (100, 100, 300, 300))
        assert found and "bbox" not in found[0] and "center" not in found[0]
    finally:
        os.unlink(name)
