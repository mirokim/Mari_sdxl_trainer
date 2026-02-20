import logging
from typing import List, Optional
from peft import LoraConfig

logger = logging.getLogger(__name__)


def inject_lora_to_unet(
    unet,
    rank: int = 32,
    alpha: int = 16,
    target_modules: Optional[List[str]] = None,
):
    """UNet에 LoRA 어댑터 주입."""
    if target_modules is None:
        target_modules = ["to_k", "to_q", "to_v", "to_out.0"]

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    unet.add_adapter(lora_config)

    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in unet.parameters())
    logger.info(
        f"UNet LoRA 주입 완료: {trainable:,} / {total:,} 파라미터 학습 "
        f"({100 * trainable / total:.2f}%)"
    )
    return unet


def inject_lora_to_text_encoders(
    text_encoder_one,
    text_encoder_two,
    rank: int = 32,
    alpha: int = 16,
):
    """양 텍스트 인코더에 LoRA 어댑터 주입."""
    te_lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        init_lora_weights="gaussian",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    )

    text_encoder_one.add_adapter(te_lora_config)
    text_encoder_two.add_adapter(te_lora_config)

    for name, te in [("TE1 (CLIP ViT-L)", text_encoder_one),
                     ("TE2 (OpenCLIP ViT-G)", text_encoder_two)]:
        trainable = sum(p.numel() for p in te.parameters() if p.requires_grad)
        total = sum(p.numel() for p in te.parameters())
        logger.info(f"{name} LoRA: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    return text_encoder_one, text_encoder_two


def freeze_model(model):
    """모델의 모든 파라미터를 고정."""
    for param in model.parameters():
        param.requires_grad = False


def get_trainable_params(model) -> list:
    """학습 가능한 파라미터만 반환."""
    return [p for p in model.parameters() if p.requires_grad]
