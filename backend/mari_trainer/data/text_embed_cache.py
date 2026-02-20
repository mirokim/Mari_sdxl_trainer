import hashlib
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TextEmbedCache:
    """텍스트 인코더 출력 사전 인코딩 및 디스크 캐싱.

    SDXL의 두 텍스트 인코더(CLIP ViT-L + OpenCLIP ViT-G)를
    학습 시 VRAM에서 해제할 수 있어 ~2-3GB 메모리 절약.
    """

    def __init__(
        self,
        text_encoder_one,
        text_encoder_two,
        tokenizer_one,
        tokenizer_two,
        cache_dir: str,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        max_token_length: int = 77,
    ):
        self.text_encoder_one = text_encoder_one
        self.text_encoder_two = text_encoder_two
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dtype = dtype
        self.device = device
        self.max_token_length = max_token_length

    @torch.no_grad()
    def encode_and_cache(
        self,
        items: list,
        progress_callback=None,
    ) -> int:
        """캡션들을 텍스트 인코더로 인코딩하여 캐시.

        Args:
            items: [{"path": str, "caption": str}, ...]

        Returns:
            캐시된 항목 수
        """
        self.text_encoder_one.to(self.device)
        self.text_encoder_two.to(self.device)
        self.text_encoder_one.eval()
        self.text_encoder_two.eval()

        cached_count = 0
        unique_items = {item["path"]: item for item in items}

        for i, (path, item) in enumerate(tqdm(unique_items.items(), desc="텍스트 임베딩 캐싱")):
            cache_name = Path(path).stem
            cache_path = self.cache_dir / f"{cache_name}.pt"

            if cache_path.exists():
                cached_count += 1
                if progress_callback:
                    progress_callback(i + 1, len(unique_items))
                continue

            caption = item.get("caption", "")
            prompt_embeds, pooled_prompt_embeds = self._encode_prompt(caption)

            torch.save({
                "prompt_embeds": prompt_embeds.cpu(),
                "pooled_prompt_embeds": pooled_prompt_embeds.cpu(),
            }, cache_path)

            cached_count += 1
            if progress_callback:
                progress_callback(i + 1, len(unique_items))

        logger.info(f"텍스트 임베딩 캐싱 완료: {cached_count}개")
        return cached_count

    def _encode_prompt(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """캡션을 양 텍스트 인코더로 인코딩.

        Returns:
            (prompt_embeds, pooled_prompt_embeds)
        """
        # 토크나이저 1 (CLIP ViT-L)
        tokens_one = self.tokenizer_one(
            prompt,
            max_length=self.max_token_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        # 토크나이저 2 (OpenCLIP ViT-G)
        tokens_two = self.tokenizer_two(
            prompt,
            max_length=self.max_token_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        # 인코더 1
        output_one = self.text_encoder_one(
            tokens_one.input_ids, output_hidden_states=True
        )
        prompt_embeds_one = output_one.hidden_states[-2]  # penultimate layer

        # 인코더 2
        output_two = self.text_encoder_two(
            tokens_two.input_ids, output_hidden_states=True
        )
        prompt_embeds_two = output_two.hidden_states[-2]
        pooled_prompt_embeds = output_two[0]  # pooled output

        # 결합
        prompt_embeds = torch.cat([prompt_embeds_one, prompt_embeds_two], dim=-1)

        return prompt_embeds.squeeze(0), pooled_prompt_embeds.squeeze(0)

    def is_cached(self, image_path: str) -> bool:
        cache_name = Path(image_path).stem
        return (self.cache_dir / f"{cache_name}.pt").exists()

    def load_cached(self, image_path: str) -> Optional[dict]:
        cache_name = Path(image_path).stem
        cache_path = self.cache_dir / f"{cache_name}.pt"
        if cache_path.exists():
            return torch.load(cache_path, weights_only=True)
        return None

    def clear_cache(self):
        for f in self.cache_dir.glob("*.pt"):
            f.unlink()
        logger.info("텍스트 임베딩 캐시 삭제 완료")
