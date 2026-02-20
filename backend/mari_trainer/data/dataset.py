import logging
import random
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from .bucketing import AspectRatioBucketer
from .caption_utils import read_caption, apply_caption_dropout
from .preprocessing import (
    resize_and_crop,
    image_to_tensor,
    random_flip_image,
    get_image_dimensions,
    compute_sdxl_time_ids,
)
from ..utils.file_utils import get_image_files, parse_kohya_folder

logger = logging.getLogger(__name__)


class SDXLDataset(Dataset):
    """SDXL 학습용 데이터셋.

    Kohya 스타일 폴더 구조 지원: {repeats}_{concept}/
    종횡비 버켓팅, 캡션 드롭아웃, 랜덤 플립 지원.
    """

    def __init__(
        self,
        dataset_path: str,
        resolution: int = 1024,
        enable_bucketing: bool = True,
        random_flip: bool = True,
        caption_dropout_rate: float = 0.05,
        cached_latents_dir: Optional[str] = None,
        cached_text_embeds_dir: Optional[str] = None,
    ):
        self.dataset_path = dataset_path
        self.resolution = resolution
        self.enable_bucketing = enable_bucketing
        self.random_flip = random_flip
        self.caption_dropout_rate = caption_dropout_rate
        self.cached_latents_dir = cached_latents_dir
        self.cached_text_embeds_dir = cached_text_embeds_dir

        # 데이터셋 로딩
        self.items = self._load_items()
        logger.info(f"데이터셋 로드: {len(self.items)}개 항목 ({dataset_path})")

        # 버켓팅
        if enable_bucketing:
            self.bucketer = AspectRatioBucketer(base_resolution=resolution)
            self._assign_buckets()
        else:
            for item in self.items:
                item["bucket"] = (resolution, resolution)

    def _load_items(self) -> List[dict]:
        """Kohya 폴더 구조에서 학습 항목 로딩."""
        items = []
        folders = parse_kohya_folder(self.dataset_path)

        if not folders:
            # Kohya 구조가 아닌 경우 플랫 폴더로 처리
            images = get_image_files(self.dataset_path)
            for img_path in images:
                w, h = get_image_dimensions(str(img_path))
                caption = read_caption(str(img_path))
                items.append({
                    "path": str(img_path),
                    "caption": caption,
                    "width": w,
                    "height": h,
                    "repeats": 1,
                })
        else:
            for folder in folders:
                for img_path in folder["images"]:
                    w, h = get_image_dimensions(img_path)
                    caption = read_caption(img_path)
                    for _ in range(folder["repeats"]):
                        items.append({
                            "path": img_path,
                            "caption": caption,
                            "width": w,
                            "height": h,
                            "repeats": folder["repeats"],
                        })

        return items

    def _assign_buckets(self):
        """각 아이템에 버켓 할당."""
        for item in self.items:
            item["bucket"] = self.bucketer.assign_bucket(item["width"], item["height"])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        bucket_w, bucket_h = item["bucket"]

        # 캐시된 잠재벡터 사용
        if self.cached_latents_dir:
            return self._get_cached_item(item, idx)

        # 이미지 로딩 & 전처리
        image = Image.open(item["path"]).convert("RGB")
        original_size = (image.width, image.height)

        if self.random_flip:
            image = random_flip_image(image)

        image, crop_coords = resize_and_crop(image, bucket_w, bucket_h)
        pixel_values = image_to_tensor(image)

        # 캡션
        caption = item["caption"]
        caption = apply_caption_dropout(caption, self.caption_dropout_rate)

        # SDXL 시간 조건
        time_ids = compute_sdxl_time_ids(
            original_size=original_size,
            crop_coords_top_left=crop_coords,
            target_size=(bucket_w, bucket_h),
        )

        return {
            "pixel_values": pixel_values,
            "caption": caption,
            "original_size": original_size,
            "crop_coords": crop_coords,
            "target_size": (bucket_w, bucket_h),
            "time_ids": time_ids.squeeze(0),
        }

    def _get_cached_item(self, item: dict, idx: int) -> dict:
        """캐시된 잠재벡터/텍스트 임베딩 로드."""
        cache_name = Path(item["path"]).stem
        bucket_w, bucket_h = item["bucket"]

        result = {
            "caption": apply_caption_dropout(item["caption"], self.caption_dropout_rate),
            "original_size": (item["width"], item["height"]),
            "target_size": (bucket_w, bucket_h),
        }

        # 잠재벡터 캐시
        latent_path = Path(self.cached_latents_dir) / f"{cache_name}.pt"
        if latent_path.exists():
            result["cached_latents"] = torch.load(latent_path, weights_only=True)

        # 텍스트 임베딩 캐시
        if self.cached_text_embeds_dir:
            embed_path = Path(self.cached_text_embeds_dir) / f"{cache_name}.pt"
            if embed_path.exists():
                embeds = torch.load(embed_path, weights_only=True)
                result["cached_prompt_embeds"] = embeds["prompt_embeds"]
                result["cached_pooled_prompt_embeds"] = embeds["pooled_prompt_embeds"]

        return result

    def get_stats(self) -> dict:
        """데이터셋 통계."""
        unique_images = len(set(item["path"] for item in self.items))
        return {
            "total_items": len(self.items),
            "unique_images": unique_images,
            "total_with_repeats": len(self.items),
            "captioned": sum(1 for item in self.items if item["caption"]),
        }
