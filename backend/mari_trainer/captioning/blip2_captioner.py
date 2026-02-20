import logging
from typing import Optional

import torch
from PIL import Image

from .base_captioner import BaseCaptioner

logger = logging.getLogger(__name__)


class BLIP2Captioner(BaseCaptioner):
    """Salesforce BLIP-2 기반 자동 캡셔닝.

    모델: Salesforce/blip2-opt-2.7b (~5GB)
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip2-opt-2.7b",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.model = None
        self.processor = None

    def load_model(self):
        """BLIP-2 모델 로드."""
        from transformers import AutoProcessor, Blip2ForConditionalGeneration

        logger.info(f"BLIP-2 로딩: {self.model_name}")
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype=self.dtype
        ).to(self.device)
        self.model.eval()
        logger.info("BLIP-2 로딩 완료")

    def caption_image(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
    ) -> str:
        """이미지 캡셔닝.

        Args:
            image: PIL Image
            prompt: 조건부 프롬프트 (예: "a photo of")
        """
        if self.model is None:
            self.load_model()

        if image.mode != "RGB":
            image = image.convert("RGB")

        if prompt:
            inputs = self.processor(
                image, text=prompt, return_tensors="pt"
            ).to(self.device, self.dtype)
        else:
            inputs = self.processor(
                image, return_tensors="pt"
            ).to(self.device, self.dtype)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=100
            )

        return self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

    def unload_model(self):
        """모델 언로드."""
        if self.model is not None:
            self.model.cpu()
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("BLIP-2 언로드 완료")
