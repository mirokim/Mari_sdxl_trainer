import logging
from pathlib import Path

import torch
from safetensors.torch import save_file

logger = logging.getLogger(__name__)


def convert_diffusers_to_single_safetensors(
    model_path: str,
    output_path: str,
    dtype: torch.dtype = torch.float16,
):
    """diffusers 형식 SDXL 모델을 단일 .safetensors로 변환.

    ComfyUI의 CheckpointLoader 호환.

    Args:
        model_path: diffusers save_pretrained() 디렉토리 경로
        output_path: 출력 .safetensors 파일 경로
    """
    from diffusers import StableDiffusionXLPipeline

    logger.info(f"모델 변환 시작: {model_path} → {output_path}")

    # 파이프라인 로드
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        model_path, torch_dtype=dtype
    )

    state_dict = {}

    # UNet → model.diffusion_model.*
    logger.info("UNet 키 변환 중...")
    unet_sd = pipeline.unet.state_dict()
    for key, value in unet_sd.items():
        new_key = f"model.diffusion_model.{key}"
        state_dict[new_key] = value.to(dtype)

    # VAE → first_stage_model.*
    logger.info("VAE 키 변환 중...")
    vae_sd = pipeline.vae.state_dict()
    for key, value in vae_sd.items():
        new_key = f"first_stage_model.{key}"
        state_dict[new_key] = value.to(dtype)

    # 텍스트 인코더 1 → conditioner.embedders.0.transformer.*
    logger.info("텍스트 인코더 1 키 변환 중...")
    te1_sd = pipeline.text_encoder.state_dict()
    for key, value in te1_sd.items():
        new_key = f"conditioner.embedders.0.transformer.{key}"
        state_dict[new_key] = value.to(dtype)

    # 텍스트 인코더 2 → conditioner.embedders.1.model.*
    logger.info("텍스트 인코더 2 키 변환 중...")
    te2_sd = pipeline.text_encoder_2.state_dict()
    for key, value in te2_sd.items():
        new_key = f"conditioner.embedders.1.model.{key}"
        state_dict[new_key] = value.to(dtype)

    # 저장
    save_file(state_dict, output_path)
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    logger.info(f"변환 완료: {output_path} ({size_mb:.1f} MB, {len(state_dict)}개 키)")

    # 메모리 정리
    del pipeline, state_dict
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
