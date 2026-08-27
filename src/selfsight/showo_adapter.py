"""Native-Windows Show-o adapter using the locked official repository and SDPA."""

from __future__ import annotations

import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image

from selfsight.adapters import GradientResult, ModelAdapter
from selfsight.data.questions import normalize_answer
from selfsight.models import locked_model, repository_path, snapshot_path
from selfsight.rfo.isolation import hard_render
from selfsight.schemas import (
    AtomicObservation,
    AtomicQuestion,
    CandidateRecord,
    ObservationResult,
)
from selfsight.training.gradients import collect_lora_gradient, collect_lora_gradient_accumulated
from selfsight.training.lora import attach_showo_lora
from selfsight.utils.hashing import rgb_sha256, sha256_json

SPECIAL_TOKENS = (
    "<|soi|>",
    "<|eoi|>",
    "<|sov|>",
    "<|eov|>",
    "<|t2i|>",
    "<|mmu|>",
    "<|t2v|>",
    "<|v2v|>",
    "<|lvg|>",
)
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)
SYSTEM_PROMPT_LEN = 28


@dataclass(frozen=True)
class ShowoSFTBatch:
    prompts: tuple[str, ...]
    images: tuple[str | Path | Image.Image, ...]
    sample_ids: tuple[str, ...]
    mask_seed: int


@dataclass(frozen=True)
class ShowoReplayBatch:
    images: tuple[str | Path | Image.Image, ...]
    questions: tuple[str, ...]
    answers: tuple[str, ...]
    sample_ids: tuple[str, ...]


class ShowoAdapter(ModelAdapter):
    def __init__(
        self,
        *,
        device: str = "cuda:0",
        dtype: str = "bf16",
        lock_path: str | Path = "configs/models.lock.yaml",
        trainable: bool = False,
        load_vision_tower: bool = True,
        generation_timesteps: int = 25,
        guidance_scale: float = 5.0,
        temperature: float = 1.0,
    ) -> None:
        import torch
        from omegaconf import OmegaConf
        from transformers import AutoTokenizer, CLIPImageProcessor

        self.model_id = "showlab/show-o-w-clip-vit-512x512"
        self.revision = locked_model(self.model_id, lock_path)["revision"]
        self.device = torch.device(device)
        self.dtype = torch.bfloat16 if dtype == "bf16" else torch.float32
        self.generation_timesteps = generation_timesteps
        self.guidance_scale = guidance_scale
        self.temperature = temperature
        self.repo_root = repository_path("showlab/Show-o", lock_path=lock_path)
        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))
        from llava.llava import conversation as conversation_lib
        from models import CLIPVisionTower, MAGVITv2, Showo, get_mask_chedule
        from training.prompting_utils import UniversalPrompting

        conversation_lib.default_conversation = conversation_lib.conv_templates["phi1.5"]
        self.conversation_lib = conversation_lib
        self.create_attention_mask_predict_next = __import__(
            "training.prompting_utils", fromlist=["create_attention_mask_predict_next"]
        ).create_attention_mask_predict_next
        self.create_attention_mask_for_mmu_vit = __import__(
            "training.prompting_utils", fromlist=["create_attention_mask_for_mmu_vit"]
        ).create_attention_mask_for_mmu_vit
        self.image_transform = __import__("training.utils", fromlist=["image_transform"]).image_transform
        self.mask_or_random_replace_tokens = __import__(
            "training.utils", fromlist=["mask_or_random_replace_tokens"]
        ).mask_or_random_replace_tokens
        self.mask_schedule = get_mask_chedule("cosine")

        phi_path = snapshot_path("microsoft/phi-1_5", lock_path=lock_path)
        showo_path = snapshot_path(self.model_id, lock_path=lock_path)
        vq_path = snapshot_path("showlab/magvitv2", lock_path=lock_path)
        vision_path = snapshot_path("openai/clip-vit-large-patch14-336", lock_path=lock_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            phi_path, padding_side="left", local_files_only=True
        )
        self.uni_prompting = UniversalPrompting(
            self.tokenizer,
            max_text_len=128,
            special_tokens=SPECIAL_TOKENS,
            ignore_id=-100,
            cond_dropout_prob=0.1,
        )
        self.vq_model = MAGVITv2.from_pretrained(vq_path).to(self.device, dtype=self.dtype)
        self.vq_model.requires_grad_(False).eval()
        self.model = Showo.from_pretrained(
            showo_path,
            llm_model_path=str(phi_path),
            local_files_only=True,
            torch_dtype=self.dtype,
        ).to(self.device, dtype=self.dtype)
        self.model.train(trainable)
        if not trainable:
            self.model.requires_grad_(False)
        self.vision_tower = None
        self.clip_image_processor = None
        if load_vision_tower:
            self.vision_tower = CLIPVisionTower(str(vision_path)).to(self.device, dtype=self.dtype)
            self.vision_tower.requires_grad_(False).eval()
            self.clip_image_processor = CLIPImageProcessor.from_pretrained(
                vision_path, local_files_only=True
            )
        self.config = OmegaConf.create(
            {
                "model": {
                    "showo": {
                        "w_clip_vit": True,
                        "vocab_size": 58498,
                        "llm_vocab_size": 50295,
                        "codebook_size": 8192,
                        "num_vq_tokens": 1024,
                        "num_new_special_tokens": 10,
                    }
                },
                "dataset": {"preprocessing": {"max_seq_length": 128, "resolution": 512}},
                "training": {
                    "min_masking_rate": 0.0,
                    "noise_type": "mask",
                    "predict_all_tokens": False,
                },
            }
        )

    def attach_lora(self, **kwargs: Any):
        self.model, summary = attach_showo_lora(self.model, **kwargs)
        self.model.train()
        return summary

    def _attention_mask(self, input_ids: Any) -> Any:
        mask = self.create_attention_mask_predict_next(
            input_ids,
            pad_id=int(self.uni_prompting.sptids_dict["<|pad|>"]),
            soi_id=int(self.uni_prompting.sptids_dict["<|soi|>"]),
            eoi_id=int(self.uni_prompting.sptids_dict["<|eoi|>"]),
            rm_pad_in_image=True,
        )
        return mask.to(device=self.device, dtype=self.dtype)

    def _generate_one(self, prompt: str, seed: int) -> Image.Image:
        import torch

        mask_token_id = self.model.config.mask_token_id
        image_tokens = torch.full(
            (1, self.config.model.showo.num_vq_tokens),
            fill_value=mask_token_id,
            dtype=torch.long,
            device=self.device,
        )
        input_ids, _ = self.uni_prompting(([prompt], image_tokens), "t2i_gen")
        input_ids = input_ids.to(self.device)
        uncond_ids, _ = self.uni_prompting(([""], image_tokens), "t2i_gen")
        uncond_ids = uncond_ids.to(self.device)
        attention_mask = self._attention_mask(torch.cat([input_ids, uncond_ids], dim=0))
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        with torch.inference_mode():
            token_ids = self.model.t2i_generate(
                input_ids=input_ids,
                uncond_input_ids=uncond_ids,
                attention_mask=attention_mask,
                guidance_scale=self.guidance_scale,
                temperature=self.temperature,
                timesteps=self.generation_timesteps,
                noise_schedule=self.mask_schedule,
                noise_type="mask",
                seq_len=self.config.model.showo.num_vq_tokens,
                uni_prompting=self.uni_prompting,
                config=self.config,
                generator=generator,
            )
            token_ids = torch.clamp(token_ids, 0, self.config.model.showo.codebook_size - 1)
            images = self.vq_model.decode_code(token_ids)
            images = torch.clamp((images + 1.0) / 2.0, 0.0, 1.0)
        array = (images[0].permute(1, 2, 0).float().cpu().numpy() * 255.0).round().astype(np.uint8)
        return Image.fromarray(array, mode="RGB")

    def generate_images(
        self,
        prompts: Sequence[str],
        seeds: Sequence[int],
        output_dir: str | Path,
        checkpoint_id: str,
    ) -> list[CandidateRecord]:
        if len(prompts) != len(seeds):
            raise ValueError("prompts and seeds must have equal length")
        output_dir = Path(output_dir).resolve()
        records = []
        was_training = self.model.training
        self.model.eval()
        try:
            for index, (prompt, seed) in enumerate(zip(prompts, seeds)):
                prompt_id = sha256_json({"prompt": prompt})[:20]
                candidate_id = f"{checkpoint_id}-{prompt_id}-{seed}-{index}"
                image_path = output_dir / f"{candidate_id}.png"
                evidence = hard_render(self._generate_one(prompt, int(seed)), image_path)
                records.append(
                    CandidateRecord(
                        candidate_id=candidate_id,
                        prompt_id=prompt_id,
                        scene_id=prompt_id,
                        sampling_seed=int(seed),
                        image_path=str(image_path),
                        rgb_sha256=str(evidence["rgb_sha256"]),
                        generator_id=self.model_id,
                        generator_revision=self.revision,
                        checkpoint_id=checkpoint_id,
                        metadata={
                            "prompt": prompt,
                            "generation_timesteps": self.generation_timesteps,
                            "guidance_scale": self.guidance_scale,
                        },
                    )
                )
        finally:
            self.model.train(was_training)
        return records

    def _embed_tokens(self, input_ids: Any) -> Any:
        return self.model.showo.get_input_embeddings()(input_ids)

    def _mmu_generate_embeddings(self, embeddings: Any, attention_mask: Any, max_new_tokens: int) -> Any:
        import torch

        results = []
        for _ in range(max_new_tokens):
            logits = self.model(input_ids=None, input_embeddings=embeddings, attention_mask=attention_mask)
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            results.append(next_id[0, 0])
            if int(next_id[0, 0]) == self.tokenizer.eos_token_id:
                break
            embeddings = torch.cat([embeddings, self._embed_tokens(next_id)], dim=1)
            length = attention_mask.shape[-1]
            squeezed = attention_mask.squeeze(0).squeeze(0)
            expanded = torch.full(
                (length + 1, length + 1),
                fill_value=torch.finfo(self.dtype).min,
                dtype=self.dtype,
                device=self.device,
            )
            expanded[:length, :length] = squeezed
            expanded[length, :length] = squeezed[-1, :]
            expanded[length, length] = 0
            attention_mask = expanded.unsqueeze(0).unsqueeze(0)
        return torch.stack(results).unsqueeze(0) if results else torch.empty((1, 0), dtype=torch.long)

    def _observe_one(self, image: Image.Image, question: str, max_new_tokens: int = 16) -> str:
        import torch

        if self.vision_tower is None or self.clip_image_processor is None:
            raise RuntimeError("Show-o observer requires load_vision_tower=True")
        conv = self.conversation_lib.default_conversation.copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        question_prompt = conv.get_prompt().strip()
        system_ids = self.tokenizer(
            SYSTEM_PROMPT, return_tensors="pt", padding="longest"
        ).input_ids.to(self.device)
        if system_ids.shape[-1] != SYSTEM_PROMPT_LEN:
            raise RuntimeError(f"Unexpected Show-o system prompt length: {system_ids.shape[-1]}")
        question_ids = self.tokenizer(
            question_prompt, return_tensors="pt", padding="longest"
        ).input_ids.to(self.device)
        mmu = torch.full((1, 1), int(self.uni_prompting.sptids_dict["<|mmu|>"]), device=self.device)
        soi = torch.full((1, 1), int(self.uni_prompting.sptids_dict["<|soi|>"]), device=self.device)
        eoi = torch.full((1, 1), int(self.uni_prompting.sptids_dict["<|eoi|>"]), device=self.device)
        input_ids = torch.cat([mmu, system_ids, soi, eoi, question_ids], dim=1).long()
        pixels = self.clip_image_processor.preprocess(image, return_tensors="pt")["pixel_values"].to(
            self.device, dtype=self.dtype
        )
        with torch.inference_mode():
            image_embeddings = self.model.mm_projector(self.vision_tower(pixels))
            text_embeddings = self._embed_tokens(input_ids)
            split = 2 + SYSTEM_PROMPT_LEN
            embeddings = torch.cat(
                [text_embeddings[:, :split, :], image_embeddings, text_embeddings[:, split:, :]], dim=1
            )
            attention = self.create_attention_mask_for_mmu_vit(
                embeddings, system_prompt_len=SYSTEM_PROMPT_LEN
            ).to(self.device, dtype=self.dtype)
            token_ids = self._mmu_generate_embeddings(embeddings, attention, max_new_tokens)
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)[0].strip()

    def observe_atoms(
        self,
        image_path: str | Path,
        questions: Sequence[AtomicQuestion],
    ) -> ObservationResult:
        from datetime import datetime, timezone

        started = datetime.now(timezone.utc).isoformat()
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        answers = []
        was_training = self.model.training
        self.model.eval()
        try:
            for question in questions:
                start = perf_counter()
                raw = self._observe_one(image, question.text)
                normalized = normalize_answer(raw, question)
                answers.append(
                    AtomicObservation(
                        question_id=question.question_id,
                        raw_answer=raw,
                        normalized_answer=normalized,
                        abstain=normalized is None,
                        latency_ms=(perf_counter() - start) * 1000.0,
                    )
                )
        finally:
            self.model.train(was_training)
        return ObservationResult(
            request_id=sha256_json({"image": str(Path(image_path).resolve()), "questions": [q.question_id for q in questions]}),
            observer_id=self.model_id,
            observer_revision=self.revision,
            rgb_sha256=rgb_sha256(image_path),
            answers=tuple(answers),
            started_at=started,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )

    def _image_tensor(self, image_or_path: Image.Image | str | Path):
        if isinstance(image_or_path, Image.Image):
            image = image_or_path.convert("RGB")
        else:
            with Image.open(image_or_path) as opened:
                image = opened.convert("RGB")
        return self.image_transform(image, resolution=512)

    def encode_image_targets(self, images: Sequence[Image.Image | str | Path]) -> Any:
        import torch

        pixels = torch.stack([self._image_tensor(image) for image in images]).to(
            self.device, dtype=self.dtype
        )
        with torch.no_grad():
            return self.vq_model.get_code(pixels) + len(self.uni_prompting.text_tokenizer)

    def sft_loss(self, batch: ShowoSFTBatch):
        import torch
        import torch.nn.functional as F

        if len(batch.prompts) != len(batch.images):
            raise ValueError("SFT prompts/images length mismatch")
        python_state = random.getstate()
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        try:
            random.seed(batch.mask_seed)
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(batch.mask_seed)
                if self.device.type == "cuda":
                    torch.cuda.manual_seed(batch.mask_seed)
                image_tokens = self.encode_image_targets(batch.images)
                input_image_ids, labels, _, _ = self.mask_or_random_replace_tokens(
                    image_tokens,
                    self.model.config.mask_token_id,
                    self.config,
                    self.mask_schedule,
                    is_train=True,
                )
                input_ids, _, labels = self.uni_prompting(
                    (list(batch.prompts), input_image_ids, labels), "t2i"
                )
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(input_ids=input_ids, attention_mask=self._attention_mask(input_ids))
                start = self.config.dataset.preprocessing.max_seq_length + 1
                return F.cross_entropy(
                    logits[:, start:].contiguous().view(-1, self.model.output_size),
                    labels[:, start:].contiguous().view(-1),
                    ignore_index=-100,
                )
        finally:
            random.setstate(python_state)

    def mmu_replay_loss(self, batch: ShowoReplayBatch):
        """Causal answer loss on program-rendered train-split QA using the frozen CLIP tower."""

        import torch
        import torch.nn.functional as F

        if self.vision_tower is None or self.clip_image_processor is None:
            raise RuntimeError("Understanding replay requires the frozen CLIP vision tower")
        size = len(batch.images)
        if not (size == len(batch.questions) == len(batch.answers) == len(batch.sample_ids)):
            raise ValueError("MMU replay fields must have equal lengths")
        if size == 0:
            raise ValueError("MMU replay batch cannot be empty")

        system_ids = self.tokenizer(
            SYSTEM_PROMPT,
            add_special_tokens=True,
            return_tensors="pt",
        ).input_ids[0].tolist()
        if len(system_ids) != SYSTEM_PROMPT_LEN:
            raise RuntimeError(f"Unexpected Show-o system prompt length: {len(system_ids)}")
        mmu_id = int(self.uni_prompting.sptids_dict["<|mmu|>"])
        soi_id = int(self.uni_prompting.sptids_dict["<|soi|>"])
        eoi_id = int(self.uni_prompting.sptids_dict["<|eoi|>"])
        pad_id = int(self.uni_prompting.sptids_dict["<|pad|>"])
        text_rows: list[list[int]] = []
        answer_starts: list[int] = []
        for question, answer in zip(batch.questions, batch.answers):
            conv = self.conversation_lib.default_conversation.copy()
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prefix = conv.get_prompt().strip()
            prefix_ids = self.tokenizer(prefix, add_special_tokens=False).input_ids
            answer_ids = self.tokenizer(f" {answer}", add_special_tokens=False).input_ids
            answer_ids = list(answer_ids) + [int(self.tokenizer.eos_token_id)]
            text_rows.append([mmu_id, *system_ids, soi_id, eoi_id, *prefix_ids, *answer_ids])
            answer_starts.append(1 + len(system_ids) + 1 + 576 + 1 + len(prefix_ids))

        max_tokens = max(len(row) for row in text_rows)
        input_ids = torch.full(
            (size, max_tokens),
            fill_value=pad_id,
            dtype=torch.long,
            device=self.device,
        )
        for index, row in enumerate(text_rows):
            input_ids[index, : len(row)] = torch.tensor(row, dtype=torch.long, device=self.device)
        if isinstance(batch.images[0], Image.Image):
            pil_images = [image.convert("RGB") for image in batch.images]  # type: ignore[union-attr]
        else:
            pil_images = []
            for item in batch.images:
                with Image.open(item) as opened:  # type: ignore[arg-type]
                    pil_images.append(opened.convert("RGB"))
        pixels = self.clip_image_processor.preprocess(pil_images, return_tensors="pt")[
            "pixel_values"
        ].to(self.device, dtype=self.dtype)
        with torch.no_grad():
            vision_features = self.vision_tower(pixels)
            image_embeddings = self.model.mm_projector(vision_features)
        text_embeddings = self._embed_tokens(input_ids)
        split = 2 + len(system_ids)
        embeddings = torch.cat(
            [text_embeddings[:, :split, :], image_embeddings, text_embeddings[:, split:, :]], dim=1
        )
        labels = torch.full(
            embeddings.shape[:2],
            fill_value=-100,
            dtype=torch.long,
            device=self.device,
        )
        for index, (row, answer_start) in enumerate(zip(text_rows, answer_starts)):
            answer_ids = row[(answer_start - 576) :]
            labels[index, answer_start : answer_start + len(answer_ids)] = torch.tensor(
                answer_ids, dtype=torch.long, device=self.device
            )
        attention = self.create_attention_mask_for_mmu_vit(
            embeddings, system_prompt_len=len(system_ids)
        ).to(self.device, dtype=self.dtype)
        logits = self.model(input_ids=None, input_embeddings=embeddings, attention_mask=attention)
        return F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, self.model.output_size),
            labels[:, 1:].contiguous().view(-1),
            ignore_index=-100,
        )

    def compute_lora_gradient(self, batch: ShowoSFTBatch, criterion: str) -> GradientResult:
        snapshot = collect_lora_gradient(
            self.model,
            lambda: self.sft_loss(batch),
            criterion=criterion,
            sample_ids=batch.sample_ids,
        )
        return GradientResult(
            criterion=criterion,
            vector=snapshot.vector,
            per_block=snapshot.per_block,
            loss=snapshot.loss,
            sample_ids=snapshot.sample_ids,
        )

    def compute_lora_gradient_accumulated(
        self,
        batches: Sequence[ShowoSFTBatch],
        criterion: str,
    ) -> GradientResult:
        snapshot = collect_lora_gradient_accumulated(
            self.model,
            [lambda batch=batch: self.sft_loss(batch) for batch in batches],
            criterion=criterion,
            sample_ids=(sample_id for batch in batches for sample_id in batch.sample_ids),
        )
        return GradientResult(
            criterion=criterion,
            vector=snapshot.vector,
            per_block=snapshot.per_block,
            loss=snapshot.loss,
            sample_ids=snapshot.sample_ids,
        )
