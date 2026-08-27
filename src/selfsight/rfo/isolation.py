"""Hard RGB materialization and context-redacted observer requests."""

from __future__ import annotations

import os
import tempfile
import uuid
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from selfsight.schemas import AtomicQuestion, BlindObservationRequest
from selfsight.utils.hashing import rgb_sha256, sha256_file


def hard_render(source: Image.Image | str | Path, destination: str | Path) -> dict[str, str | int]:
    """Write pixels to PNG, close the writer, reopen from disk, and hash decoded RGB."""

    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".png", dir=str(destination.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        if isinstance(source, Image.Image):
            rgb = source.convert("RGB")
        else:
            with Image.open(source) as opened:
                rgb = opened.convert("RGB")
        rgb.save(temporary, format="PNG", optimize=False)
        del rgb
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        with Image.open(destination) as reopened:
            reopened_rgb = reopened.convert("RGB")
            reopened_rgb.load()
            width, height = reopened_rgb.size
            decoded_hash = rgb_sha256(reopened_rgb)
        return {
            "path": str(destination),
            "file_sha256": sha256_file(destination),
            "rgb_sha256": decoded_hash,
            "width": width,
            "height": height,
        }
    finally:
        temporary.unlink(missing_ok=True)


def make_blind_request(
    image_path: str | Path,
    questions: Sequence[AtomicQuestion],
    request_id: str | None = None,
) -> BlindObservationRequest:
    path = Path(image_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return BlindObservationRequest(
        request_id=request_id or str(uuid.uuid4()),
        image_path=str(path),
        rgb_sha256=rgb_sha256(path),
        questions=tuple(questions),
    )
