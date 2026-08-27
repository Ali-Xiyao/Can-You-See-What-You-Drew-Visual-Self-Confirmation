"""Stable hashes for files, RGB pixels, and structured evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_bytes(payload.encode("utf-8"))


def rgb_sha256(image_or_path: Image.Image | str | Path) -> str:
    if isinstance(image_or_path, Image.Image):
        image = image_or_path
    else:
        with Image.open(image_or_path) as opened:
            image = opened.convert("RGB")
    array = np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))
    header = f"{array.shape[1]}x{array.shape[0]}:RGB:".encode("ascii")
    return sha256_bytes(header + array.tobytes())
