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
