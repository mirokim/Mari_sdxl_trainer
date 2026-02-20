from abc import ABC, abstractmethod
from typing import List, Optional
from PIL import Image


class BaseCaptioner(ABC):
    """자동 캡셔닝 기본 인터페이스."""

    @abstractmethod
    def load_model(self):
        """모델 로드."""
        pass

    @abstractmethod
    def caption_image(self, image: Image.Image, prompt: Optional[str] = None) -> str:
        """단일 이미지 캡셔닝."""
        pass

    @abstractmethod
    def unload_model(self):
        """모델 언로드 (VRAM 해제)."""
        pass

    def caption_batch(
        self,
        images: List[Image.Image],
        prompt: Optional[str] = None,
    ) -> List[str]:
        """배치 캡셔닝 (기본: 순차 처리)."""
        return [self.caption_image(img, prompt) for img in images]
