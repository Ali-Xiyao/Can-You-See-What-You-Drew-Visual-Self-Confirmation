"""Tests for the escalation ladder's second level.

Two detectors returned the same object list on 49.6% of a 468-image run and the
same verdict on 86.9%. Sending every list disagreement to a person spent 121 of
157 adjudications on images whose answer was never in doubt, and blew a budget
that is 10%. These pin down what the verdict level may and may not conclude.
"""

from __future__ import annotations

from typing import Any

from selfsight.v4.spec import SceneSpec
from selfsight.v4.verifier import Resolution, verify


class _Fixed:
    """A detector that returns a fixed list, and nothing from a crop."""

    def __init__(self, detector_id: str, detections: list[dict[str, Any]],
                 crop: list[dict[str, Any]] | None = None):
        self.detector_id = detector_id
        self._detections = detections
        self._crop = crop or []

    def detect(self, image_path: str) -> list[dict[str, Any]]:
        return [dict(d) for d in self._detections]

    def detect_crop(self, image_path: str, bbox: Any) -> list[dict[str, Any]]:
        return [dict(d) for d in self._crop]


def _spec(objects, spec_id="s1"):
    return SceneSpec.from_dict({
        "spec_id": spec_id,
        "prompt": "a scene",
        "objects": [{"object": o, "color": c, "count": n} for o, c, n in objects],
    })


def _obj(noun, colour, x=100.0):
    return {"object": noun, "color": colour,
            "bbox": [x, 100.0, x + 50, 150.0], "center": [x + 25, 125.0]}


def test_a_dispute_that_cannot_change_the_verdict_is_settled_without_a_human():
    """Asked for two blue mugs; one model sees three and a candle, one sees
    three. The image is wrong on either reading and no person is needed."""
    spec = _spec([("mug", "blue", 2)])
    primary = _Fixed("a", [_obj("mug", "blue", 10), _obj("mug", "blue", 80),
                           _obj("mug", "blue", 150), _obj("candle", "white", 220)])
    secondary = _Fixed("b", [_obj("mug", "blue", 10), _obj("mug", "blue", 80),
                             _obj("mug", "blue", 150)])
    result = verify("img.png", spec, primary, secondary)
    assert result.resolution is Resolution.AGREED_VERDICT
    assert result.image_correct is False
    assert result.verifier_agreement is False


def test_the_settled_list_drops_the_object_only_one_model_saw():
    """Nothing downstream may ask a question about a disputed object.

    Its gold would be a coin flip between two strong detectors, and the model
    would be scored against that.
    """
    spec = _spec([("mug", "blue", 2)])
    primary = _Fixed("a", [_obj("mug", "blue", 10), _obj("mug", "blue", 80),
                           _obj("mug", "blue", 150), _obj("candle", "white", 220)])
    secondary = _Fixed("b", [_obj("mug", "blue", 10), _obj("mug", "blue", 80),
                             _obj("mug", "blue", 150)])
    result = verify("img.png", spec, primary, secondary)
    assert [d["object"] for d in result.detections] == ["mug", "mug", "mug"]
    assert [d["object"] for d in result.disputed] == ["candle"]


def test_the_verdict_is_not_recomputed_from_the_agreed_core():
    """The trap this level would fall into if written the obvious way.

    Both models call the image wrong, for different reasons: one saw a spare
    spoon, the other a spare fork. Their agreed core is exactly the spec, so
    recomputing image_correct from it would return True -- an image neither
    detector thought was right.
    """
    spec = _spec([("mug", "blue", 2)])
    primary = _Fixed("a", [_obj("mug", "blue", 10), _obj("mug", "blue", 80),
                           _obj("spoon", "silver", 150)])
    secondary = _Fixed("b", [_obj("mug", "blue", 10), _obj("mug", "blue", 80),
                             _obj("fork", "silver", 150)])
    result = verify("img.png", spec, primary, secondary)
    assert result.resolution is Resolution.AGREED_VERDICT
    assert result.image_correct is False  # not True, which the core alone gives


def test_a_dispute_that_flips_the_verdict_still_goes_down_the_ladder():
    """One model says the spec was met, the other says an extra object is there.

    This is the case the crop step exists for, and it must not be swallowed by
    the verdict level.
    """
    spec = _spec([("mug", "blue", 2)])
    primary = _Fixed("a", [_obj("mug", "blue", 10), _obj("mug", "blue", 80),
                           _obj("spoon", "silver", 150)],
                     crop=[{"object": "spoon", "color": "silver"}])
    secondary = _Fixed("b", [_obj("mug", "blue", 10), _obj("mug", "blue", 80)])
    result = verify("img.png", spec, primary, secondary)
    assert result.resolution is Resolution.RESOLVED_BY_CROP
    assert result.image_correct is False


def test_a_verdict_conflict_the_crop_cannot_settle_reaches_a_human():
    spec = _spec([("mug", "blue", 2)])
    primary = _Fixed("a", [_obj("mug", "blue", 10), _obj("mug", "blue", 80),
                           _obj("spoon", "silver", 150)], crop=[])
    secondary = _Fixed("b", [_obj("mug", "blue", 10), _obj("mug", "blue", 80)])
    result = verify("img.png", spec, primary, secondary)
    assert result.resolution is Resolution.PENDING_HUMAN
    assert result.disputed[0]["reason"] == "not_confirmed_in_crop"


def test_identical_lists_still_resolve_at_level_one():
    spec = _spec([("mug", "blue", 2)])
    items = [_obj("mug", "blue", 10), _obj("mug", "blue", 80)]
    result = verify("img.png", spec, _Fixed("a", items), _Fixed("b", items))
    assert result.resolution is Resolution.AGREED
    assert result.image_correct is True
    assert result.verifier_agreement is True
