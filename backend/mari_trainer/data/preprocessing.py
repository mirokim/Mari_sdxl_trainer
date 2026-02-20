import random
from typing import Tuple
from PIL import Image
import numpy as np
import torch
from torchvision import transforms


def resize_and_crop(
    image: Image.Image,
    target_width: int,
    target_height: int,
    random_crop: bool = True,
) -> Tuple[Image.Image, Tuple[int, int]]:
    """이미지를 목표 크기에 맞게 리사이즈 + 크롭.

    Returns:
        (처리된 이미지, crop_coords_top_left)
    """
    w, h = image.size
    target_aspect = target_width / target_height
    image_aspect = w / h

    # 종횡비에 맞게 리사이즈 (빈 공간 없이)
    if image_aspect > target_aspect:
        # 이미지가 더 넓음 → 높이 기준으로 리사이즈
        new_h = target_height
        new_w = int(h * target_aspect * (w / h) / target_aspect)
        new_w = max(int(w * target_height / h), target_width)
    else:
        # 이미지가 더 높음 → 너비 기준으로 리사이즈
        new_w = target_width
        new_h = max(int(h * target_width / w), target_height)

    image = image.resize((new_w, new_h), Image.LANCZOS)

    # 크롭
    if random_crop:
        left = random.randint(0, max(0, new_w - target_width))
        top = random.randint(0, max(0, new_h - target_height))
    else:
        left = (new_w - target_width) // 2
        top = (new_h - target_height) // 2

    image = image.crop((left, top, left + target_width, top + target_height))
    return image, (top, left)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """PIL Image → [-1, 1] 범위의 텐서."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    arr = np.array(image).astype(np.float32) / 255.0
    arr = (arr * 2.0) - 1.0  # [0,1] → [-1,1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # HWC → CHW
    return tensor


def random_flip_image(image: Image.Image, probability: float = 0.5) -> Image.Image:
    """랜덤 수평 뒤집기."""
    if random.random() < probability:
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    return image


def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """이미지 크기 (width, height) 반환. 메모리 효율적."""
    with Image.open(image_path) as img:
        return img.size


def compute_sdxl_time_ids(
    original_size: Tuple[int, int],
    crop_coords_top_left: Tuple[int, int],
    target_size: Tuple[int, int],
) -> torch.Tensor:
    """SDXL add_time_ids 계산.

    SDXL은 original_size, crop_coords, target_size를 조건으로 사용.
    """
    add_time_ids = list(original_size) + list(crop_coords_top_left) + list(target_size)
    return torch.tensor([add_time_ids], dtype=torch.float32)
