import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SDXLModelLoader:
    """SDXL 모델 컴포넌트를 개별적으로 로드."""

    @staticmethod
    def load_components(
        model_path: str,
        dtype: torch.dtype = torch.float16,
        vae_path: Optional[str] = None,
        use_fp16_vae_fix: bool = True,
    ) -> dict:
        """SDXL 파이프라인 컴포넌트를 학습용으로 로드.

        Returns:
            dict with keys: unet, vae, text_encoder_one, text_encoder_two,
                           tokenizer_one, tokenizer_two, noise_scheduler
        """
        from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
        from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

        logger.info(f"SDXL 모델 로딩: {model_path}")

        # 노이즈 스케줄러
        noise_scheduler = DDPMScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )

        # 토크나이저 (2개)
        tokenizer_one = CLIPTokenizer.from_pretrained(
            model_path, subfolder="tokenizer"
        )
        tokenizer_two = CLIPTokenizer.from_pretrained(
            model_path, subfolder="tokenizer_2"
        )

        # 텍스트 인코더 (2개)
        logger.info("텍스트 인코더 로딩...")
        text_encoder_one = CLIPTextModel.from_pretrained(
            model_path, subfolder="text_encoder", torch_dtype=dtype
        )
        text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            model_path, subfolder="text_encoder_2", torch_dtype=dtype
        )

        # VAE (NaN 방지)
        logger.info("VAE 로딩...")
        if vae_path:
            vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=dtype)
            logger.info(f"커스텀 VAE 사용: {vae_path}")
        elif use_fp16_vae_fix:
            vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype
            )
            logger.info("fp16-fix VAE 사용")
        else:
            vae = AutoencoderKL.from_pretrained(
                model_path, subfolder="vae", torch_dtype=torch.float32
            )
            logger.info("원본 VAE (float32) 사용")

        # UNet
        logger.info("UNet 로딩...")
        unet = UNet2DConditionModel.from_pretrained(
            model_path, subfolder="unet", torch_dtype=dtype
        )

        logger.info("SDXL 모델 로딩 완료")

        return {
            "unet": unet,
            "vae": vae,
            "text_encoder_one": text_encoder_one,
            "text_encoder_two": text_encoder_two,
            "tokenizer_one": tokenizer_one,
            "tokenizer_two": tokenizer_two,
            "noise_scheduler": noise_scheduler,
        }

    @staticmethod
    def load_pipeline_for_inference(
        model_path: str,
        lora_path: Optional[str] = None,
        lora_weight: float = 1.0,
        dtype: torch.dtype = torch.float16,
    ):
        """추론용 SDXL 파이프라인 로드."""
        from diffusers import StableDiffusionXLPipeline

        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_path, torch_dtype=dtype
        )

        if lora_path:
            pipeline.load_lora_weights(lora_path)
            pipeline.fuse_lora(lora_scale=lora_weight)

        pipeline.to("cuda")
        return pipeline
