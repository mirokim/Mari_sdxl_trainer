import math
from typing import List, Tuple, Dict
from collections import defaultdict


# SDXL 표준 버켓 해상도 (총 픽셀 수 ≈ 1024*1024 = 1,048,576 기준)
SDXL_BUCKET_RESOLUTIONS = [
    (512, 2048), (512, 1984), (512, 1920), (512, 1856),
    (576, 1792), (576, 1728), (576, 1664),
    (640, 1600), (640, 1536),
    (704, 1472), (704, 1408),
    (768, 1344), (768, 1280),
    (832, 1216), (832, 1152),
    (896, 1088), (896, 1024),
    (960, 1024), (960, 960),
    (1024, 960), (1024, 896), (1024, 1024),
    (1088, 896),
    (1152, 832),
    (1216, 832),
    (1280, 768),
    (1344, 768),
    (1408, 704),
    (1472, 704),
    (1536, 640),
    (1600, 640),
    (1664, 576),
    (1728, 576),
    (1792, 576),
    (1856, 512),
    (1920, 512),
    (1984, 512),
    (2048, 512),
]


class AspectRatioBucketer:
    """SDXL 종횡비 버켓팅."""

    def __init__(
        self,
        min_resolution: int = 512,
        max_resolution: int = 2048,
        step: int = 64,
        base_resolution: int = 1024,
    ):
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.step = step
        self.base_resolution = base_resolution
        self.buckets = self._generate_buckets()

    def _generate_buckets(self) -> List[Tuple[int, int]]:
        """버켓 해상도 목록 생성."""
        target_pixels = self.base_resolution * self.base_resolution
        buckets = set()

        for w in range(self.min_resolution, self.max_resolution + 1, self.step):
            h = round(target_pixels / w / self.step) * self.step
            if self.min_resolution <= h <= self.max_resolution:
                buckets.add((w, h))

        return sorted(buckets)

    def assign_bucket(self, width: int, height: int) -> Tuple[int, int]:
        """이미지의 종횡비에 가장 가까운 버켓 할당."""
        aspect = width / height
        best_bucket = self.buckets[0]
        best_diff = float("inf")

        for bw, bh in self.buckets:
            bucket_aspect = bw / bh
            diff = abs(aspect - bucket_aspect)
            if diff < best_diff:
                best_diff = diff
                best_bucket = (bw, bh)

        return best_bucket

    def create_bucket_groups(
        self, items: List[dict]
    ) -> Dict[Tuple[int, int], List[dict]]:
        """이미지 목록을 버켓별로 그룹화.

        각 item은 {"path": str, "width": int, "height": int, ...} 형태.
        """
        groups = defaultdict(list)
        for item in items:
            bucket = self.assign_bucket(item["width"], item["height"])
            item["bucket"] = bucket
            groups[bucket].append(item)
        return dict(groups)

    def create_batches(
        self, items: List[dict], batch_size: int
    ) -> List[List[dict]]:
        """버켓별 배치 생성. 같은 버켓 내 이미지끼리 배치."""
        groups = self.create_bucket_groups(items)
        batches = []

        for bucket, group_items in groups.items():
            for i in range(0, len(group_items), batch_size):
                batch = group_items[i:i + batch_size]
                batches.append(batch)

        return batches
