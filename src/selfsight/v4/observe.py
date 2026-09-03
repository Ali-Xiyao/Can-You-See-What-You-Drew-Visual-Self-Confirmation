"""The two arms' ways of looking at a generated image.

The whole experiment turns on one difference, and it is entirely contained in
this file: the naive arm asks the backbone about a picture while telling it what
the picture was supposed to be, and the RFO arm asks a frozen external observer
that has never seen the description. Everything downstream -- selection,
gradients, the training curves, D* -- inherits that difference and nothing else.

It lives in the library rather than in whichever script needs it because it has
now been needed by three: the Gate B probe, the training loop, and per-checkpoint
evaluation. Two copies of `PROMPTED_PREAMBLE` that drift apart would silently
stop those three from measuring the same thing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from selfsight.observers.protocol import decode_request, encode_request, execute_request
from selfsight.schemas import AtomicQuestion, BlindObservationRequest, ObservationResult
from selfsight.utils.hashing import rgb_sha256, sha256_json

PROMPTED_PREAMBLE = """You were asked to draw a picture from this description:
"{prompt}"

Here is the picture you drew. Answer about what is actually in the picture.

{question}"""
"""The leak, verbatim.

`scripts/v4_run_pipeline.py` and the Tier B runner carry this same string. It is
duplicated there rather than imported for a reason that has expired -- those
predate this module -- and `tests/test_v4_probe.py` asserts the copies are
byte-identical. If that test ever fails, the fix is to make the other copies
import this one, never to edit one side until the test passes.
"""


def prompted(prompt: str, questions: Sequence[AtomicQuestion]) -> tuple[AtomicQuestion, ...]:
    """The same questions, each wrapped in the description the image came from."""

    return tuple(
        replace(question, text=PROMPTED_PREAMBLE.format(prompt=prompt, question=question.text))
        for question in questions
    )


def observe_naive(
    backbone: Any,
    *,
    prompt: str,
    questions: Sequence[AtomicQuestion],
    image_path: str | Path,
) -> ObservationResult:
    """The leaky arm: the backbone judging its own drawing, told what it drew."""

    return backbone.observe_atoms(str(image_path), prompted(prompt, questions))


def observe_rfo(
    observer: Any,
    *,
    namespace: str,
    prompt_id: str,
    candidate_id: str,
    questions: Sequence[AtomicQuestion],
    image_path: str | Path,
) -> ObservationResult:
    """The blind arm, over the allow-listed wire.

    Going through encode/decode rather than calling the observer directly is not
    ceremony: `assert_blind_wire_payload` is what makes "this observer never saw
    the prompt" a checked property rather than a claim about this file. Note that
    `prompt` is not a parameter here, and could not be smuggled through if it
    were -- the wire rejects the key.
    """

    resolved = Path(image_path).resolve()
    request = BlindObservationRequest(
        request_id=sha256_json([namespace, prompt_id, candidate_id]),
        image_path=str(resolved),
        rgb_sha256=rgb_sha256(resolved),
        questions=tuple(questions),
    )
    return execute_request(observer, decode_request(encode_request(request)))
