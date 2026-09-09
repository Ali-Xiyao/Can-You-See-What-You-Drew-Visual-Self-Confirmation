"""Janus-Pro-1B, so the context ablation can be asked of a second model family.

Show-o v1 and Show-o2 share a lineage: the same lab, the same discrete-token
framing, an overlapping training recipe. A reviewer who sees the ablation
replicate on both is entitled to say it replicated on one idea twice. Janus-Pro
splits the visual pathway in two -- SigLIP for understanding, a separate VQ
tokeniser for generation -- which is the architectural choice most often cited
as the fix for exactly the interference this project measures. If the
description still overrides the pixels there, the finding is about unified
models, not about Show-o.

E4 is inference only. This adapter exposes `observe_atoms` and nothing else: no
LoRA, no training loss, no generation. Janus never draws anything here -- it
answers questions about images Show-o2 drew, which is what makes the comparison
across models a comparison of readers rather than of painters.

The vision encoder runs once per question rather than once per image. Four
questions on a 1B model costs about a second of duplicated SigLIP; hoisting it
would mean reaching inside `prepare_inputs_embeds` and reproducing the
placeholder-splicing by hand, which is the kind of divergence from the official
path that E4 exists to avoid.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image

from selfsight.data.questions import normalize_answer
from selfsight.models import locked_model, repository_path, snapshot_path
from selfsight.schemas import AtomicObservation, AtomicQuestion, ObservationResult
from selfsight.utils.hashing import rgb_sha256, sha256_json

MODEL_ID = "deepseek-ai/Janus-Pro-1B"
SOURCE_REPOSITORY = "deepseek-ai/Janus"

# The tag the processor splices the 576 image embeddings into. It is read from
# processor_config.json at load time; this is the value that file is expected to
# hold, checked rather than assumed so a different snapshot cannot silently
# produce a text-only answer.
IMAGE_TAG = "<image_placeholder>"


class JanusProAdapter:
    """Answer atomic questions about an image with Janus-Pro's official path."""

    def __init__(
        self,
        *,
        device: str = "cuda:0",
        dtype: str = "bf16",
        lock_path: str | Path = "configs/models.lock.yaml",
        model_root: str | Path | None = None,
        max_new_tokens: int = 16,
        lazy: bool = True,
    ) -> None:
        import torch

        self.model_id = MODEL_ID
        self.revision = locked_model(self.model_id, lock_path)["revision"]
        self.device = torch.device(device)
        self.dtype = torch.bfloat16 if dtype == "bf16" else torch.float32
        self.lock_path = lock_path
        self.model_root = model_root
        self.max_new_tokens = int(max_new_tokens)
        self.model: Any = None
        self.processor: Any = None
        self.tokenizer: Any = None
        if not lazy:
            self._load()

    def _load(self) -> None:
        if self.model is not None:
            return
        from transformers import AutoModelForCausalLM

        # `janus` is an editable install pointing at the locked checkout, but an
        # environment that only has the weights should fail on the checkout
        # rather than on an import three frames deeper.
        repo = repository_path(
            SOURCE_REPOSITORY, lock_path=self.lock_path, model_root=self.model_root
        )
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from janus.models import VLChatProcessor

        snapshot = snapshot_path(
            self.model_id, lock_path=self.lock_path, model_root=self.model_root
        )
        self.processor = VLChatProcessor.from_pretrained(str(snapshot))
        if self.processor.image_tag != IMAGE_TAG:
            raise RuntimeError(
                f"Janus image tag is {self.processor.image_tag!r}, not {IMAGE_TAG!r}: "
                "the question would be answered without the picture"
            )
        self.tokenizer = self.processor.tokenizer
        model = AutoModelForCausalLM.from_pretrained(str(snapshot), local_files_only=True)
        self.model = model.to(self.dtype).to(self.device).eval()

    @property
    def identity(self) -> dict[str, Any]:
        """What answered, for the row the pipeline writes."""

        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "source_repository": SOURCE_REPOSITORY,
            "implementation": "janus_pro_understanding_only",
            "native_resolution": 384,
        }

    def _observe_one(self, image: Image.Image, question: str) -> str:
        import torch

        # No `images` key: that one is read by `janus.utils.io.load_pil_images`,
        # which opens paths. The picture is already decoded, so it goes to the
        # processor directly and the conversation carries only the placeholder.
        conversation = [
            {"role": "<|User|>", "content": f"{IMAGE_TAG}\n{question}"},
            {"role": "<|Assistant|>", "content": ""},
        ]
        prepared = self.processor(
            conversations=conversation, images=[image], force_batchify=True
        ).to(self.device, dtype=self.dtype)
        with torch.inference_mode():
            embeddings = self.model.prepare_inputs_embeds(**prepared)
            tokens = self.model.language_model.generate(
                inputs_embeds=embeddings,
                attention_mask=prepared.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.max_new_tokens,
                # Greedy, like the Show-o adapters. A sampled answer would make
                # the two conditions differ by noise as well as by context.
                do_sample=False,
                use_cache=True,
            )
        return self.tokenizer.decode(tokens[0].cpu().tolist(), skip_special_tokens=True).strip()

    def observe_atoms(
        self,
        image_path: str | Path,
        questions: Sequence[AtomicQuestion],
    ) -> ObservationResult:
        self._load()
        started = datetime.now(timezone.utc).isoformat()
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        image_hash = rgb_sha256(image)
        answers = []
        for question in questions:
            question_started = perf_counter()
            raw = self._observe_one(image, question.text)
            normalized = normalize_answer(raw, question)
            answers.append(
                AtomicObservation(
                    question_id=question.question_id,
                    raw_answer=raw,
                    normalized_answer=normalized,
                    abstain=normalized is None,
                    latency_ms=(perf_counter() - question_started) * 1000.0,
                )
            )
        return ObservationResult(
            request_id=sha256_json(
                {"rgb_sha256": image_hash, "questions": [item.question_id for item in questions]}
            ),
            observer_id=self.model_id,
            observer_revision=self.revision,
            rgb_sha256=image_hash,
            answers=tuple(answers),
            started_at=started,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
