"""Display vocabulary: the claim and the visible text must not diverge.

The v3 probe manifest shipped `{"square_display_name": "box"}` on all 256 rows
while 183 still said "square" in the prompt or the question. Downstream code
reads the claim, so nothing failed. These tests pin the audit that makes that
combination loud, and the rewrite that prevents it.
"""

from __future__ import annotations

import pytest

from selfsight.v3.vocabulary import (
    ASPECT_RATIO_RANGE,
    DISPLAY_NAME,
    INTERNAL_NAME,
    assert_display_vocabulary,
    audit_rows,
    claims_display_vocabulary,
    display_text,
    has_internal_wording,
    scene_vocabulary_metadata,
    visible_texts,
)


def _row(prompt, question, *, claim=True, scene_id="s0", choices=()):
    row = {
        "scene": {
            "scene_id": scene_id,
            "prompt": prompt,
            "metadata": {"subject_shape": "square", "object_shape": "circle"},
        },
        "atom": {"subject": "shape=square;color=red", "answer": "yes"},
        "questions": [{"text": question, "choices": list(choices)}],
    }
    if claim:
        row["vocabulary"] = {f"{INTERNAL_NAME}_display_name": DISPLAY_NAME}
    return row


def test_rewrite_covers_singular_and_plural():
    assert display_text("a red square and two blue squares") == "a red box and two blue boxes"


def test_rewrite_preserves_capitalisation():
    assert display_text("Square first") == "Box first"
    assert display_text("SQUARE") == "BOX"
    assert display_text("Squares here") == "Boxes here"


def test_rewrite_respects_word_boundaries():
    assert display_text("squarely a squarish square") == "squarely a squarish box"


def test_rewrite_is_idempotent():
    once = display_text("a large red square")

    assert display_text(once) == once
    assert has_internal_wording(once) is False


def test_visible_texts_cover_prompt_question_and_choices():
    row = _row("a red square", "Is there a red square?", choices=("square", "circle"))

    assert visible_texts(row) == (
        "a red square",
        "Is there a red square?",
        "square",
        "circle",
    )


def test_visible_texts_exclude_internal_geometry_codes():
    """`atom.subject` and `scene.metadata.*_shape` drive the verifier lookup.

    Rewriting them would break shape resolution, and neither is ever shown to a
    model or a reviewer, so they must not be audited as visible text.
    """

    row = _row("a red box", "Is there a red box?")

    assert visible_texts(row) == ("a red box", "Is there a red box?")
    assert audit_rows([row])["clean"] is True


def test_audit_flags_a_row_that_claims_the_display_name_but_does_not_use_it():
    report = audit_rows([_row("a red square", "Is there a red square?", claim=True)])

    assert report["clean"] is False
    assert report["rows_with_internal_wording"] == 1
    assert report["rows_claiming_but_violating"] == 1
    assert report["examples"][0]["scene_id"] == "s0"


def test_audit_separates_claiming_from_violating():
    rows = [
        _row("a red box", "Is there a red box?", claim=True, scene_id="ok"),
        _row("a red square", "Is there a red box?", claim=False, scene_id="no_claim"),
        _row("a red square", "Is there a red box?", claim=True, scene_id="claims"),
    ]

    report = audit_rows(rows)

    assert report["rows"] == 3
    assert report["rows_claiming_display_vocabulary"] == 2
    assert report["rows_with_internal_wording"] == 2
    assert report["rows_claiming_but_violating"] == 1


def test_audit_catches_the_question_even_when_the_prompt_is_clean():
    report = audit_rows([_row("a red box", "Is there a red square?")])

    assert report["clean"] is False
    assert report["examples"][0]["texts"] == ["Is there a red square?"]


def test_claim_detection_requires_the_exact_display_name():
    assert claims_display_vocabulary(_row("a", "b", claim=True)) is True
    assert claims_display_vocabulary(_row("a", "b", claim=False)) is False
    assert claims_display_vocabulary({"vocabulary": {"square_display_name": "square"}}) is False
    assert claims_display_vocabulary({"vocabulary": "box"}) is False


def test_assert_fails_closed_with_a_locatable_example():
    with pytest.raises(ValueError, match="1/1 rows still use 'square'"):
        assert_display_vocabulary([_row("a red square", "Is there a red box?")])


def test_assert_passes_on_rewritten_rows():
    rows = [_row(display_text("a red square"), display_text("Is there a red square?"))]

    assert_display_vocabulary(rows)


def test_scene_metadata_records_the_alias_and_the_tolerance():
    metadata = scene_vocabulary_metadata()

    assert metadata["shape_display_alias"] == {INTERNAL_NAME: DISPLAY_NAME}
    assert metadata["quadrilateral_aspect_ratio_range"] == [0.5, 2.0]


def test_tolerance_is_the_verifier_constant_not_a_copy():
    """The display name only means anything if it names what the verifier accepts.

    Re-exported rather than restated: a second literal is how the claim and the
    rule drift apart (EVIDENCE_LOG section 11).
    """

    from selfsight.data.generated_verifier import QUADRILATERAL_ASPECT_RANGE

    assert ASPECT_RATIO_RANGE is QUADRILATERAL_ASPECT_RANGE


def _rectangle(tmp_path, width, height, name):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(image)
    left = (256 - width) // 2
    top = (256 - height) // 2
    draw.rectangle([left, top, left + width, top + height], fill=(220, 20, 20))
    path = tmp_path / name
    image.save(path)
    return path


@pytest.mark.parametrize(
    ("width", "height", "expected_square"),
    [
        (120, 120, True),   # aspect 1.00 - square under any reading
        (190, 100, True),   # aspect 1.90 - inside tolerance, NOT a square in plain English
        (100, 190, True),   # aspect 0.53 - the low side of the same tolerance
        (230, 60, False),   # aspect 3.83 - outside tolerance
    ],
)
def test_verifier_accepts_the_range_the_display_name_claims(
    tmp_path, width, height, expected_square
):
    """Behavioural check: what `box` actually denotes, measured on pixels.

    The 1.90 case is the whole reason the display vocabulary exists - the verifier
    calls it SQUARE while a reader asked about a "square" would reject it.
    """

    from selfsight.data.generated_verifier import detect_generated_objects
    from selfsight.schemas import Shape

    path = _rectangle(tmp_path, width, height, f"{width}x{height}.png")
    detections = detect_generated_objects(path)

    assert len(detections) == 1, f"expected one contour, got {len(detections)}"
    assert (detections[0].shape is Shape.SQUARE) is expected_square
