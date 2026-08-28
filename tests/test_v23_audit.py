from __future__ import annotations

import pytest

from selfsight.schemas import BlindObservationRequest


def test_blind_wire_rejects_context_leak():
    payload = {
        "schema_version": 1,
        "request_id": "r1",
        "image_path": "C:/tmp/image.png",
        "rgb_sha256": "0" * 64,
        "questions": [],
        "prompt": "forbidden source prompt",
    }
    with pytest.raises(ValueError, match="forbidden fields"):
        BlindObservationRequest.from_wire(payload)
