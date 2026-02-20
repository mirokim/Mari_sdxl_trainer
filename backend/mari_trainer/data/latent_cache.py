import logging
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from tqdm import tqdm

from .preprocessing import resize_and_crop, image_to_tensor

logger = logging.getLogger(__name__)


class LatentCache:
    """VAE를 사용한 잠재벡터 사전 인코딩 및 디스크 캐싱.

    학습 시 VAE를 VRAM에서 해제할 수 있어 메모리 절약 효과.
    """

    def __init__(
        self,
        vae,
        cache_dir: str,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.vae = vae
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dtype = dtype
        self.device = device

    @torch.no_grad()
    def encode_and_cache(
        self,
        items: list,
        batch_size: int = 1,
        progress_callback=None,
    ) -> int:
        """이미지를 VAE로 인코딩하여 캐시 저장.

        Args:
            items: [{"path": str, "bucket": (w, h)}, ...]
            progress_callback: (current, total) 콜백

        Returns:
            캐시된 항목 수
        """
        self.vae.to(self.device)
        self.vae.eval()

        cached_count = 0
        unique_items = {item["path"]: item for item in items}

        for i, (path, item) in enumerate(tqdm(unique_items.items(), desc="잠재벡터 캐싱")):
            cache_name = Path(path).stem
            cache_path = self.cache_dir / f"{cache_name}.pt"

            if cache_path.exists():
                cached_count += 1
                if progress_callback:
                    progress_callback(i + 1, len(unique_items))
                continue

            bucket_w, bucket_h = item["bucket"]

            image = Image.open(path).convert("RGB")
            image, crop_coords = resize_and_crop(image, bucket_w, bucket_h, random_crop=False)
            pixel_values = image_to_tensor(image).unsqueeze(0).to(self.device, dtype=self.dtype)

            latents = self.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            latents = latents.squeeze(0).cpu()

            torch.save(latents, cache_path)
            cached_count += 1

            if progress_callback:
                progress_callback(i + 1, len(unique_items))

        logger.info(f"잠재벡터 캐싱 완료: {cached_count}개")
        return cached_count

    def is_cached(self, image_path: str) -> bool:
        """해당 이미지의 캐시 존재 여부."""
        cache_name = Path(image_path).stem
        return (self.cache_dir / f"{cache_name}.pt").exists()

    def load_cached(self, image_path: str) -> Optional[torch.Tensor]:
        """캐시된 잠재벡터 로드."""
        cache_name = Path(image_path).stem
        cache_path = self.cache_dir / f"{cache_name}.pt"
        if cache_path.exists():
            return torch.load(cache_path, weights_only=True)
        return None

    def clear_cache(self):
        """캐시 디렉토리 정리."""
        for f in self.cache_dir.glob("*.pt"):
            f.unlink()
        logger.info("잠재벡터 캐시 삭제 완료")
