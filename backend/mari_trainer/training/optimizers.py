import logging
import torch
from typing import List

logger = logging.getLogger(__name__)


def create_optimizer(
    optimizer_type: str,
    params: List[dict],
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    **kwargs,
) -> torch.optim.Optimizer:
    """옵티마이저 팩토리.

    Args:
        optimizer_type: AdamW, AdamW8bit, Prodigy, AdaFactor
        params: 파라미터 그룹 리스트
        learning_rate: 학습률
        weight_decay: 가중치 감쇠
    """
    optimizer_type_lower = optimizer_type.lower()

    if optimizer_type_lower == "adamw8bit":
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                params,
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
            )
            logger.info("옵티마이저: AdamW8bit (bitsandbytes)")
        except ImportError:
            logger.warning("bitsandbytes 없음 → 표준 AdamW 사용")
            optimizer = torch.optim.AdamW(
                params,
                lr=learning_rate,
                weight_decay=weight_decay,
            )
    elif optimizer_type_lower == "prodigy":
        try:
            from prodigyopt import Prodigy
            optimizer = Prodigy(
                params,
                lr=1.0,
                decouple=kwargs.get("prodigy_decouple", True),
                weight_decay=weight_decay,
                use_bias_correction=kwargs.get("prodigy_use_bias_correction", False),
                safeguard_warmup=True,
            )
            logger.info("옵티마이저: Prodigy (lr=1.0, adaptive)")
        except ImportError:
            logger.warning("prodigyopt 없음 → 표준 AdamW 사용")
            optimizer = torch.optim.AdamW(
                params,
                lr=learning_rate,
                weight_decay=weight_decay,
            )
    elif optimizer_type_lower == "adafactor":
        from transformers.optimization import Adafactor
        optimizer = Adafactor(
            params,
            lr=learning_rate,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
            weight_decay=weight_decay,
        )
        logger.info("옵티마이저: AdaFactor")
    else:
        optimizer = torch.optim.AdamW(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        logger.info("옵티마이저: AdamW")

    return optimizer


def create_param_groups(
    unet,
    text_encoder_one=None,
    text_encoder_two=None,
    learning_rate: float = 1e-4,
    text_encoder_lr: float = 1e-5,
) -> List[dict]:
    """모델 파라미터 그룹 생성 (학습률 분리)."""
    params = []

    # UNet 파라미터
    unet_params = [p for p in unet.parameters() if p.requires_grad]
    if unet_params:
        params.append({"params": unet_params, "lr": learning_rate})

    # 텍스트 인코더 파라미터
    if text_encoder_one is not None:
        te1_params = [p for p in text_encoder_one.parameters() if p.requires_grad]
        if te1_params:
            params.append({"params": te1_params, "lr": text_encoder_lr})

    if text_encoder_two is not None:
        te2_params = [p for p in text_encoder_two.parameters() if p.requires_grad]
        if te2_params:
            params.append({"params": te2_params, "lr": text_encoder_lr})

    return params
