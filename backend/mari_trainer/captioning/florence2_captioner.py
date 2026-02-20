import logging
from typing import Optional

import torch
from PIL import Image

from .base_captioner import BaseCaptioner

logger = logging.getLogger(__name__)


class Florence2Captioner(BaseCaptioner):
    """Microsoft Florence-2 기반 자동 캡셔닝.

    모델 옵션:
    - microsoft/Florence-2-base (~500MB)
    - microsoft/Florence-2-large (~1.5GB)

    캡션 모드:
    - <CAPTION>: 짧은 캡션
    - <DETAILED_CAPTION>: 상세 캡션
    - <MORE_DETAILED_CAPTION>: 매우 상세한 캡션 (권장)
    """

    MODELS = {
        "florence-2-base": "microsoft/Florence-2-base",
        "florence-2-large": "microsoft/Florence-2-large",
    }

    CAPTION_MODES = [
        "<CAPTION>",
        "<DETAILED_CAPTION>",
        "<MORE_DETAILED_CAPTION>",
    ]

    def __init__(
        self,
        model_name: str = "microsoft/Florence-2-large",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.model = None
        self.processor = None

    def load_model(self):
        """Florence-2 모델 로드."""
        from transformers import AutoProcessor, AutoModelForCausalLM

        logger.info(f"Florence-2 로딩: {self.model_name}")
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        logger.info("Florence-2 로딩 완료")

    def caption_image(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
    ) -> str:
        """이미지 캡셔닝.

        Args:
            image: PIL Image
            prompt: 캡션 모드 ("<MORE_DETAILED_CAPTION>" 등)
        """
        if self.model is None:
            self.load_model()

        if prompt is None:
            prompt = "<MORE_DETAILED_CAPTION>"

        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(self.device, self.dtype)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False,
            )

        text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            text, task=prompt, image_size=(image.width, image.height)
        )

        return parsed.get(prompt, text).strip()

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

        logger.info("Florence-2 언로드 완료")
