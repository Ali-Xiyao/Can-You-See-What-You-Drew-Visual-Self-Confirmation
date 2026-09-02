"""The filler used to close the hole a deleted object leaves behind.

Not a research component -- nothing is concluded from what it paints. Its
output goes through the same two-detector gate the recoloured images go
through, and an image where the filler invented a replacement object or ate the
neighbour is rejected there. What this module has to guarantee is narrower: that
only masked pixels move, which `tierb.delete` enforces, and that the same filler
runs over the deletion arm and its sham control, which is what makes the edit
artifact a shared property of both arms rather than a confound of one.

OpenCV's Telea was measured first because it needs no download. It smeared, and
on the still lifes in this corpus it dragged the neighbouring object's colour
across the hole, which is a collateral change the gate would reject on almost
every image. LaMa is the smallest model that does not do that.

Model: Carve/LaMa-ONNX, `lama_fp32.onnx`, revision
c3c0c9e468934d62e79c329e35d82dd09ff8c444, 208MB, at $SELFSIGHT_MODEL_ROOT/
lama-onnx. Downloaded with the user's explicit approval. Substituted for
smartywu/big-lama, which ships a zip that needs the LaMa repo's hydra config to
load at all. CPU only, about 3s per edit, which is why the area cap in
`tierb.MAX_DELETE_SHARE` is applied before this runs rather than after.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

SIZE = 512
"""The resolution the export was traced at. Not a choice."""


def _model_path() -> Path:
    root = os.environ.get("SELFSIGHT_MODEL_ROOT")
    if not root:
        raise RuntimeError("SELFSIGHT_MODEL_ROOT is not set; "
                           "source scripts/set_h_env.sh")
    path = Path(root) / "lama-onnx" / "lama_fp32.onnx"
    if not path.exists():
        raise FileNotFoundError(f"no inpainting weights at {path}")
    return path


class LamaInpainter:
    """Callable of (image, mask) -> filled image, for `tierb.delete`.

    The whole frame is resized to the traced 512 rather than a crop around the
    hole. A crop was tried: at the hole sizes this corpus produces, a window
    with enough surrounding context to be worth cropping is the whole image
    anyway, and the output was bit-identical.
    """

    def __init__(self, path: Path | None = None) -> None:
        import onnxruntime as ort

        self.path = Path(path) if path else _model_path()
        self.session = ort.InferenceSession(
            str(self.path), providers=["CPUExecutionProvider"])

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        import cv2

        height, width = image.shape[:2]
        small = cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
        # Nearest, not area: a resampled mask with intermediate values would
        # feed the network a half-masked rim and get half the object back.
        small_mask = cv2.resize(mask.astype(np.uint8) * 255, (SIZE, SIZE),
                                interpolation=cv2.INTER_NEAREST)
        out = self.session.run(None, {
            "image": (small.astype(np.float32) / 255.0).transpose(2, 0, 1)[None],
            "mask": (small_mask > 127).astype(np.float32)[None, None],
        })[0][0]
        out = np.clip(out.transpose(1, 2, 0), 0, 255).astype(np.uint8)
        return cv2.resize(out, (width, height), interpolation=cv2.INTER_CUBIC)
