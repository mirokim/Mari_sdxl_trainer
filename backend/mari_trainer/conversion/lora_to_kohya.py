import logging
import re
from typing import Optional

import torch
from safetensors.torch import save_file

logger = logging.getLogger(__name__)


def convert_peft_to_kohya_key(peft_key: str, prefix: str) -> str:
    """PEFT LoRA 키를 Kohya 포맷 키로 변환.

    PEFT: base_model.model.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k.lora_A.weight
    Kohya: lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_k.lora_down.weight
    """
    # base_model.model. 제거
    key = peft_key.replace("base_model.model.", "")

    # lora_A → lora_down, lora_B → lora_up
    key = key.replace(".lora_A.", ".lora_down.")
    key = key.replace(".lora_B.", ".lora_up.")

    # 마지막 두 세그먼트 분리 (예: lora_down.weight)
    parts = key.split(".")
    if len(parts) >= 3:
        module_path = "_".join(parts[:-2])
        lora_suffix = ".".join(parts[-2:])
        key = f"{prefix}{module_path}.{lora_suffix}"
    else:
        key = f"{prefix}{'_'.join(parts)}"

    return key


def convert_and_save_kohya_lora(
    unet,
    text_encoder_one=None,
    text_encoder_two=None,
    output_path: str = "lora_kohya.safetensors",
    dtype: torch.dtype = torch.float16,
    lora_alpha: int = 16,
    adapter_name: str = "default",
):
    """PEFT LoRA 모델을 Kohya 포맷 .safetensors로 변환 저장.

    ComfyUI의 LoraLoader 노드와 호환되는 키 네이밍 사용.
    """
    from peft import get_peft_model_state_dict

    kohya_state_dict = {}

    # --- UNet ---
    try:
        unet_sd = get_peft_model_state_dict(unet, adapter_name=adapter_name)
    except Exception:
        unet_sd = get_peft_model_state_dict(unet)

    for peft_key, weight in unet_sd.items():
        kohya_key = convert_peft_to_kohya_key(peft_key, "lora_unet_")
        kohya_state_dict[kohya_key] = weight.to(dtype)

        # alpha 값 추가
        if ".lora_down." in kohya_key:
            alpha_key = kohya_key.split(".lora_down.")[0] + ".alpha"
            if alpha_key not in kohya_state_dict:
                kohya_state_dict[alpha_key] = torch.tensor(float(lora_alpha))

    logger.info(f"UNet LoRA 키: {len(unet_sd)}개 → Kohya 변환")

    # --- 텍스트 인코더 1 (CLIP ViT-L) ---
    if text_encoder_one is not None:
        try:
            te1_sd = get_peft_model_state_dict(text_encoder_one, adapter_name=adapter_name)
        except Exception:
            te1_sd = get_peft_model_state_dict(text_encoder_one)

        for peft_key, weight in te1_sd.items():
            kohya_key = convert_peft_to_kohya_key(peft_key, "lora_te1_")
            kohya_state_dict[kohya_key] = weight.to(dtype)

            if ".lora_down." in kohya_key:
                alpha_key = kohya_key.split(".lora_down.")[0] + ".alpha"
                if alpha_key not in kohya_state_dict:
                    kohya_state_dict[alpha_key] = torch.tensor(float(lora_alpha))

        logger.info(f"TE1 LoRA 키: {len(te1_sd)}개 → Kohya 변환")

    # --- 텍스트 인코더 2 (OpenCLIP ViT-G) ---
    if text_encoder_two is not None:
        try:
            te2_sd = get_peft_model_state_dict(text_encoder_two, adapter_name=adapter_name)
        except Exception:
            te2_sd = get_peft_model_state_dict(text_encoder_two)

        for peft_key, weight in te2_sd.items():
            kohya_key = convert_peft_to_kohya_key(peft_key, "lora_te2_")
            kohya_state_dict[kohya_key] = weight.to(dtype)

            if ".lora_down." in kohya_key:
                alpha_key = kohya_key.split(".lora_down.")[0] + ".alpha"
                if alpha_key not in kohya_state_dict:
                    kohya_state_dict[alpha_key] = torch.tensor(float(lora_alpha))

        logger.info(f"TE2 LoRA 키: {len(te2_sd)}개 → Kohya 변환")

    # 검증
    _verify_kohya_keys(kohya_state_dict)

    # 저장
    save_file(kohya_state_dict, output_path)
    logger.info(f"Kohya LoRA 저장: {output_path} ({len(kohya_state_dict)}개 키)")


def _verify_kohya_keys(state_dict: dict):
    """Kohya 키 네이밍 검증."""
    valid_pattern = re.compile(r"^lora_(unet|te1|te2)_.+\.(lora_(down|up)\.weight|alpha)$")
    invalid_keys = []

    for key in state_dict.keys():
        if not valid_pattern.match(key):
            invalid_keys.append(key)

    if invalid_keys:
        logger.warning(
            f"유효하지 않은 Kohya 키 {len(invalid_keys)}개 발견: "
            f"{invalid_keys[:5]}..."
        )
    else:
        logger.info("모든 Kohya 키 검증 통과")
