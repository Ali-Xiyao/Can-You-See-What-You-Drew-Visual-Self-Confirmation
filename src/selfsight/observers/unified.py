"""Inference-only observers for the unified Show-o and Janus backbones."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from selfsight.models import locked_model, repository_path, snapshot_path
from selfsight.observers.base import BaseObserver
from selfsight.schemas import AtomicQuestion


class ShowoBlindObserver(BaseObserver):
    """Step-0 Show-o copy used only behind the blind JSONL boundary."""

    def __init__(self, model_id: str, revision: str, device: str) -> None:
        from selfsight.showo_adapter import ShowoAdapter

        expected = str(locked_model(model_id)["revision"])
        if model_id != "showlab/show-o-w-clip-vit-512x512" or revision != expected:
            raise ValueError(f"Unsupported or unpinned Show-o observer: {model_id}@{revision}")
        self.observer_id = model_id
        self.revision = revision
        self.adapter = ShowoAdapter(device=device, trainable=False, load_vision_tower=True)

    def answer(self, image_path: str | Path, questions: Sequence[AtomicQuestion]) -> list[str]:
        from PIL import Image

        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        return [self.adapter._observe_one(image, question.text) for question in questions]


class DiscreteShowoBlindObserver(BaseObserver):
    """Pinned pure-discrete Show-o MMU path; inference audit only."""

    def __init__(self, model_id: str, revision: str, device: str) -> None:
        import torch
        from transformers import AutoTokenizer

        expected = str(locked_model(model_id)["revision"])
        if model_id != "showlab/show-o-512x512" or revision != expected:
            raise ValueError(f"Unsupported or unpinned discrete Show-o observer: {model_id}@{revision}")
        repo = repository_path("showlab/Show-o")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from models import MAGVITv2, Showo
        from training.prompting_utils import UniversalPrompting, create_attention_mask_for_mmu
        from training.utils import image_transform

        self.observer_id = model_id
        self.revision = revision
        self.device = torch.device(device)
        # The official discrete mmu_generate path grows its attention bias in FP32.
        # Keeping this audit-only model in FP32 avoids a query/bias dtype mismatch.
        self.dtype = torch.float32
        self.torch = torch
        self.create_attention_mask_for_mmu = create_attention_mask_for_mmu
        self.image_transform = image_transform
        phi_path = snapshot_path("microsoft/phi-1_5")
        self.tokenizer = AutoTokenizer.from_pretrained(
            phi_path, padding_side="left", local_files_only=True
        )
        self.prompting = UniversalPrompting(
            self.tokenizer,
            max_text_len=128,
            special_tokens=(
                "<|soi|>",
                "<|eoi|>",
                "<|sov|>",
                "<|eov|>",
                "<|t2i|>",
                "<|mmu|>",
                "<|t2v|>",
                "<|v2v|>",
                "<|lvg|>",
            ),
            ignore_id=-100,
            cond_dropout_prob=0.1,
        )
        self.vq_model = (
            MAGVITv2.from_pretrained(snapshot_path("showlab/magvitv2"))
            .to(self.device, dtype=self.dtype)
            .eval()
        )
        self.vq_model.requires_grad_(False)
        self.model = (
            Showo.from_pretrained(
                snapshot_path(model_id),
                llm_model_path=str(phi_path),
                local_files_only=True,
                torch_dtype=self.dtype,
            )
            .to(self.device, dtype=self.dtype)
            .eval()
        )
        self.model.requires_grad_(False)

    def _token(self, name: str, batch_size: int = 1):
        value = int(self.prompting.sptids_dict[name])
        return self.torch.full((batch_size, 1), value, device=self.device, dtype=self.torch.long)

    def answer(self, image_path: str | Path, questions: Sequence[AtomicQuestion]) -> list[str]:
        from PIL import Image

        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        pixels = self.image_transform(image, resolution=512).unsqueeze(0).to(
            self.device, dtype=self.dtype
        )
        with self.torch.inference_mode():
            image_tokens = self.vq_model.get_code(pixels) + len(self.tokenizer)
        answers = []
        for question in questions:
            question_ids = self.tokenizer(
                [f"USER: \n{question.text} ASSISTANT:"], return_tensors="pt"
            ).input_ids.to(self.device)
            input_ids = self.torch.cat(
                [
                    self._token("<|mmu|>"),
                    self._token("<|soi|>"),
                    image_tokens,
                    self._token("<|eoi|>"),
                    self._token("<|sot|>"),
                    question_ids,
                ],
                dim=1,
            )
            attention = self.create_attention_mask_for_mmu(
                input_ids, eoi_id=int(self.prompting.sptids_dict["<|eoi|>"])
            )
            with self.torch.inference_mode():
                generated = self.model.mmu_generate(
                    input_ids,
                    attention_mask=attention,
                    max_new_tokens=16,
                    top_k=1,
                    eot_token=int(self.prompting.sptids_dict["<|eot|>"]),
                )
            token_ids = self.torch.stack(generated).unsqueeze(0)
            answers.append(self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)[0].strip())
        return answers


class JanusProObserver(BaseObserver):
    """Pinned Janus-Pro understanding path; generation remains out of training scope."""

    def __init__(self, model_id: str, revision: str, device: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM

        expected = str(locked_model(model_id)["revision"])
        if model_id != "deepseek-ai/Janus-Pro-1B" or revision != expected:
            raise ValueError(f"Unsupported or unpinned Janus observer: {model_id}@{revision}")
        repo = repository_path("deepseek-ai/Janus")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from janus.models import VLChatProcessor

        model_path = snapshot_path(model_id)
        self.observer_id = model_id
        self.revision = revision
        self.device = torch.device(device)
        self.processor = VLChatProcessor.from_pretrained(str(model_path))
        self.tokenizer = self.processor.tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(self.device).eval()

    def answer(self, image_path: str | Path, questions: Sequence[AtomicQuestion]) -> list[str]:
        import torch
        from PIL import Image

        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        answers: list[str] = []
        for question in questions:
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\n{question.text}",
                    "images": [image],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
            prepared = self.processor(
                conversations=conversation,
                images=[image],
                force_batchify=True,
            ).to(self.device)
            with torch.inference_mode():
                embeddings = self.model.prepare_inputs_embeds(**prepared)
                tokens = self.model.language_model.generate(
                    inputs_embeds=embeddings,
                    attention_mask=prepared.attention_mask,
                    pad_token_id=self.tokenizer.eos_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=16,
                    do_sample=False,
                    use_cache=True,
                )
            answers.append(self.tokenizer.decode(tokens[0].cpu().tolist(), skip_special_tokens=True))
        return answers
