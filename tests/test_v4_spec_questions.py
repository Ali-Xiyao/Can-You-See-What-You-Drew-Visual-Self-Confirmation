"""Tests for the v4 scene spec, 2AFC questions, and the escalation ladder.

The cases here are the ones that went wrong in v3 and must not come back:
omission must fail rather than pass, the gold must follow the image rather than
the prompt, and a corpus-level filter must never silently drop hard images.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from selfsight.v4.questions import (
    LABELLED_CHOICE,
    Family,
    ForcedChoice,
    build_questions,
    grade,
)
from selfsight.v4.spec import SceneSpec, image_correct, match_report
from selfsight.v4.verifier import Resolution, ladder_summary, verify

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
    {"object": "pear", "color": "green", "bbox": [10, 10, 50, 50], "center": [30, 30]},
    {"object": "pear", "color": "green", "bbox": [60, 10, 100, 50], "center": [80, 30]},
    {"object": "mug", "color": "white", "bbox": [200, 10, 260, 80], "center": [230, 45]},
]


class FakeDetector:
    def __init__(self, detector_id, detections, crop=None):
        self.detector_id = detector_id
        self._detections = detections
        self._crop = detections if crop is None else crop

    def detect(self, image_path):
        return list(self._detections)

    def detect_crop(self, image_path, bbox):
        return list(self._crop)


def test_exact_match_is_correct():
    assert image_correct(SPEC, EXACT)


def test_omission_fails():
    """v3's central defect: an image missing objects scored correct.

    Under the per-family gold, dropping the mug satisfied every negative claim
    for free. Scene match has no negative claim to satisfy.
    """
    assert not image_correct(SPEC, EXACT[:2])


def test_extra_object_fails():
    extra = EXACT + [{"object": "pear", "color": "green", "center": [140, 30]}]
    assert not image_correct(SPEC, extra)


def test_wrong_colour_fails():
    recoloured = [dict(EXACT[0]), dict(EXACT[1]), dict(EXACT[2], color="black")]
    assert not image_correct(SPEC, recoloured)


def test_unspecified_colour_does_not_match_a_coloured_request():
    """An unasserted colour must not be treated as whatever the spec wanted."""
    uncoloured = [dict(item, color=None) for item in EXACT]
    assert not image_correct(SPEC, uncoloured)


def test_match_report_isolates_the_failing_level():
    report = match_report(SPEC, EXACT[:2])
    assert report["n_requested"] == 3
    assert report["n_detected"] == 2
    assert not report["count_ok"]
    assert report["missing"] == [{"object": "mug", "color": "white", "n": 1}]


def test_gold_follows_the_image_not_the_spec():
    """The experiment's core trial.

    The generator drew three pears when two were asked for. "Three" is correct,
    because the question is what the model drew, not what it was told to draw.
    A model reciting its prompt answers "two" and is scored wrong.
    """
    drew_three = EXACT + [
        {"object": "pear", "color": "green", "bbox": [110, 10, 150, 50],
         "center": [130, 30]}
    ]
    counting = [
        q for q in build_questions(SPEC, drew_three, seed=3)
        if q.family is Family.COUNTING
    ]
    assert counting, "counting question should be constructible"
    question = counting[0]
    assert question.gold_source == "image_differs_from_spec"
    correct_option = question.option_a if question.gold == "A" else question.option_b
    assert correct_option == "three"


def test_forced_choice_has_no_constant_safe_answer():
    """Gold must not sit on one letter, or position bias beats chance."""
    golds = []
    for seed in range(60):
        for question in build_questions(SPEC, EXACT, seed=seed):
            golds.append(question.gold)
    assert set(golds) == {"A", "B"}
    share = golds.count("A") / len(golds)
    assert 0.35 < share < 0.65, f"gold letter is skewed: A={share:.2f}"


def test_absence_question_names_a_category_that_was_not_drawn():
    absence = [
        q for q in build_questions(SPEC, EXACT, seed=11)
        if q.family is Family.ABSENCE
    ]
    assert absence
    assert absence[0].metadata["absent"] not in {"pear", "mug"}


def test_spatial_needs_two_singleton_objects_and_skips_ties():
    """A near tie is not a fact about the image, so no question is emitted."""
    tied = [
        {"object": "mug", "color": "white", "center": [100, 30]},
        {"object": "book", "color": "red", "center": [110, 30]},
    ]
    spec = SceneSpec.from_dict(
        {"spec_id": "s2", "prompt": "x", "objects": [
            {"object": "mug", "color": "white", "count": 1},
            {"object": "book", "color": "red", "count": 1}]}
    )
    assert not [
        q for q in build_questions(spec, tied, seed=1) if q.family is Family.SPATIAL
    ]

    apart = [dict(tied[0]), dict(tied[1], center=[400, 30])]
    spatial = [
        q for q in build_questions(spec, apart, seed=1) if q.family is Family.SPATIAL
    ]
    assert spatial
    correct = spatial[0].option_a if spatial[0].gold == "A" else spatial[0].option_b
    subject = spatial[0].metadata["subject"]
    assert correct == ("left" if subject == "mug" else "right")


@pytest.mark.parametrize(
    "reply,expected", [("A", True), ("B", False), ("A.", True), ("", None),
                       ("I am not sure", None)]
)
def test_grade_marks_unparseable_replies_as_abstention(reply, expected):
    question = next(
        q for q in build_questions(SPEC, EXACT, seed=5) if q.gold == "A"
    )
    assert grade(reply, question) is expected


# ------------------------------------------------------- grading a narrator
#
# Show-o2-1.5B answers with a bare letter, so the grader that produced every
# frozen number was never asked to read a sentence. E4 puts the same questions
# to models that narrate first, and a grader that abstains on those does not
# report "no answer" -- it reports a lower accuracy, which is the shape of the
# effect the paper claims.

NARRATOR = ForcedChoice(
    question_id="n1", spec_id="s1", family=Family.EXISTENCE,
    prompt_text="Which of these is in this picture? Answer A or B only.\nA. plate\nB. banana",
    option_a="plate", option_b="banana", gold="A",
    gold_source="image_differs_from_spec", metadata={},
)


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("Answer: A", True),
        ("Answer: B", False),
        ("The answer is A because the fruit sits on it", True),
        ("The image shows a table.\n\nAnswer: B", False),
        ("Option B", False),
        ("Choice: (A)", True),
        ("**Answer:** B", False),
        ("Answer:A", True),
    ],
)
def test_a_letter_behind_a_label_is_read_as_the_answer(reply, expected):
    assert grade(reply, NARRATOR) is expected


def test_the_word_after_the_label_is_not_read_as_a_letter():
    """"Answer: Apples" is not a choice of A.

    The label rule captures the first A or B after the marker, and without a
    word boundary it would take the first letter of whatever word follows. That
    is the same collision as "Based" being read as B, one word to the right, and
    it is the reason the pattern ends where it does.
    """

    assert grade("Answer: Apples are not shown here", NARRATOR) is None


def test_a_trailing_article_is_not_an_answer():
    """The false positive that a bare-trailing-letter rule produces.

    Measured, not imagined: an earlier version of the pattern accepted a letter
    at the end of a reply, and turned a Janus reply truncated mid-phrase into a
    confident -- and by luck correct -- choice of option A. Requiring the marker
    word is what stops it, so this must stay an abstention.
    """

    assert grade("It looks like a still life, possibly a lemon or a", NARRATOR) is None


def test_a_letter_loose_in_prose_is_still_an_abstention():
    """The label rule must not widen into "find an A or a B somewhere".

    A reply that names both letters and commits to neither is a refusal, and
    recording it as a choice would put a coin flip in the accuracy column.
    """

    assert grade("Neither A nor B is clearly visible", NARRATOR) is None


@pytest.mark.parametrize(
    "reply,picked",
    [("A", True), ("A.", True), ("A. plate", True), ("B", False), ("B)", False)],
)
def test_a_reply_that_is_only_the_letter_still_grades(reply, picked):
    """The rule the frozen numbers were produced by. It has to keep working."""

    assert grade(reply, NARRATOR) is picked


def test_an_opening_word_is_not_the_letter_it_starts_with():
    """"Based" is not B.

    The bare-letter rule accepted any reply starting with the character, and a
    narrating model opens with an A- or B-word constantly. Behind that rule this
    reply graded as B: the wrong option, from a reply that names the right one,
    attributed to the model with no sign that anything went wrong. It is worse
    than an abstention, and it is invisible in the output.
    """

    assert grade("Based on the image, it is the plate", NARRATOR) is True


@pytest.mark.parametrize(
    "reply", ["A plate and a banana are both possible",
              "Both options describe something plausible"],
)
def test_an_indefinite_article_is_not_a_choice(reply):
    """The same collision one character later.

    "A plate" opens with the letter A followed by a space, which is exactly what
    a chosen letter looks like to a rule that only checks the first character.
    A reply that will not commit has to stay uncommitted in the output.
    """

    assert grade(reply, NARRATOR) is None


def _frozen_answers() -> list[dict]:
    """Every answer row behind the numbers in the report.

    runs/ is gitignored, so a worktree checkout does not have it and this skips
    there; the working directory is tried too, which is how the same test can be
    pointed at the tree the rows actually live in.
    """

    import json

    roots = (Path(__file__).resolve().parents[1], Path.cwd())
    rows = []
    for run in ("runs/v4/main-2plus1", "runs/v4/main-1plus1plus1"):
        for name in ("answers.jsonl", "answers.prompted.jsonl"):
            candidates = [root / run / name for root in roots]
            path = next((item for item in candidates if item.exists()), None)
            if path is None:
                pytest.skip(f"{run}/{name} is not in this checkout")
            rows.extend(json.loads(line)
                        for line in path.read_text(encoding="utf-8").splitlines() if line)
    return rows


def test_the_grader_still_reproduces_every_grade_already_published():
    """A change to `grade` is a change to results that are already written down.

    Re-grading the frozen replies with the current code has to return the
    `correct` field those files were written with, for all of them. Anything
    else means a number in the report no longer follows from the code that
    produced it, and the honest response is a new measurement under a new name
    rather than a quiet edit of the old one.
    """

    rows = _frozen_answers()
    assert len(rows) > 7000, f"only {len(rows)} rows found -- wrong tree?"

    changed = []
    for row in rows:
        question = ForcedChoice(
            question_id="q", spec_id=row["spec_id"], family=Family(row["family"]),
            prompt_text=row["question"], option_a=row["option_a"],
            option_b=row["option_b"], gold=row["gold"],
            gold_source=row["gold_source"], metadata={},
        )
        if grade(row["raw_answer"], question) is not row["correct"]:
            changed.append((row["raw_answer"], row["correct"]))
    assert not changed, f"{len(changed)} of {len(rows)} frozen grades move: {changed[:5]}"


def test_the_frozen_rows_actually_exercise_the_new_rule():
    """The test above passes trivially if no frozen reply ever reaches the
    label rule. Some do, and that is what makes 'nothing moves' informative."""

    rows = _frozen_answers()
    labelled = [row for row in rows
                if LABELLED_CHOICE.search(row["raw_answer"].strip().upper())]
    assert len(labelled) >= 20, f"only {len(labelled)} frozen replies carry a label"


def test_ladder_agreement():
    detector = FakeDetector("qwen", EXACT)
    result = verify("i.png", SPEC, detector, FakeDetector("internvl", EXACT))
    assert result.resolution is Resolution.AGREED
    assert result.verifier_agreement
    assert result.image_correct


def test_ladder_resolves_a_single_object_dispute_by_cropping():
    result = verify(
        "i.png", SPEC,
        FakeDetector("qwen", EXACT),
        FakeDetector("internvl", EXACT[:2]),
    )
    assert result.resolution is Resolution.RESOLVED_BY_CROP
    assert result.image_correct
    assert [d["object"] for d in result.disputed] == ["mug"]


def test_ladder_escalates_when_the_crop_does_not_settle_it():
    result = verify(
        "i.png", SPEC,
        FakeDetector("qwen", EXACT, crop=[]),
        FakeDetector("internvl", EXACT[:2]),
    )
    assert result.resolution is Resolution.PENDING_HUMAN
    assert not result.verifier_agreement


def test_human_label_overrides_both_detectors():
    result = verify(
        "i.png", SPEC,
        FakeDetector("qwen", EXACT[:1]),
        FakeDetector("internvl", EXACT[:2]),
        human_labels={"i.png": EXACT},
    )
    assert result.resolution is Resolution.HUMAN
    assert result.image_correct


def test_ladder_summary_keeps_every_image():
    """No filtering: hard images must stay in, flagged rather than dropped.

    Filtering to the agreeing subset resamples the corpus toward easy scenes and
    biases the family comparison, because clutter is not evenly distributed
    across families.
    """
    results = [
        verify("a.png", SPEC, FakeDetector("q", EXACT), FakeDetector("i", EXACT)),
        verify("b.png", SPEC, FakeDetector("q", EXACT),
               FakeDetector("i", EXACT[:2])),
        verify("c.png", SPEC, FakeDetector("q", EXACT, crop=[]),
               FakeDetector("i", EXACT[:2])),
    ]
    summary = ladder_summary(results)
    assert summary["n"] == 3
    assert sum(summary["by_resolution"].values()) == 3
    assert summary["pending_human"] == 1


def test_single_detector_does_not_claim_agreement_it_did_not_check():
    result = verify("i.png", SPEC, FakeDetector("qwen", EXACT))
    assert result.resolution is Resolution.AGREED
    assert result.verifier_agreement is True
    assert result.disputed == ()
