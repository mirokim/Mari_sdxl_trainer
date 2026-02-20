from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainingConfig:
    """SDXL 학습에 필요한 모든 설정."""

    # 모델
    pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
    vae_path: Optional[str] = None
    use_fp16_vae_fix: bool = True
    training_mode: str = "lora"  # "lora" or "full"

    # LoRA
    lora_rank: int = 32
    lora_alpha: int = 16
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "to_k", "to_q", "to_v", "to_out.0",
    ])
    train_text_encoder: bool = False

    # 옵티마이저
    optimizer_type: str = "AdamW8bit"  # AdamW, AdamW8bit, Prodigy, AdaFactor
    learning_rate: float = 1e-4
    text_encoder_lr: float = 1e-5
    weight_decay: float = 0.01
    lr_scheduler: str = "cosine"  # cosine, cosine_with_restarts, constant, linear
    lr_warmup_steps: int = 100

    # 학습 스케줄
    max_train_steps: int = 1500
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    save_every_n_steps: int = 500
    sample_every_n_steps: int = 250
    max_grad_norm: float = 1.0

    # 메모리 최적화
    mixed_precision: str = "bf16"  # no, fp16, bf16
    gradient_checkpointing: bool = True
    enable_xformers: bool = True
    cache_latents: bool = True
    cache_text_encoder_outputs: bool = True

    # 데이터셋
    dataset_path: str = ""
    resolution: int = 1024
    enable_bucketing: bool = True
    min_bucket_resolution: int = 512
    max_bucket_resolution: int = 2048
    bucket_step: int = 64
    random_flip: bool = True
    caption_dropout_rate: float = 0.05

    # 노이즈
    noise_offset: float = 0.0
    min_snr_gamma: float = 5.0

    # 출력
    output_dir: str = "./outputs"
    run_name: str = ""
    save_kohya_format: bool = True

    # 샘플 생성
    sample_prompts: List[str] = field(default_factory=list)
    sample_negative_prompt: str = "blurry, low quality, deformed, ugly"
    sample_steps: int = 30
    sample_cfg_scale: float = 7.0
    sample_seed: int = 42

    # Prodigy 전용
    prodigy_decouple: bool = True
    prodigy_use_bias_correction: bool = False

    def validate(self) -> List[str]:
        """설정 검증. 경고 목록 반환."""
        warnings = []

        if self.training_mode == "full" and self.lora_rank > 0:
            warnings.append("풀 파인튜닝 모드에서는 LoRA 설정이 무시됩니다.")

        if self.cache_text_encoder_outputs and self.train_text_encoder:
            warnings.append(
                "텍스트 인코더 캐싱과 텍스트 인코더 학습을 동시에 사용할 수 없습니다. "
                "텍스트 인코더 학습이 비활성화됩니다."
            )
            self.train_text_encoder = False

        if self.optimizer_type == "Prodigy" and self.learning_rate != 1.0:
            warnings.append(
                "Prodigy 옵티마이저 사용 시 learning_rate=1.0이 권장됩니다."
            )

        if not self.dataset_path:
            warnings.append("데이터셋 경로가 설정되지 않았습니다.")

        return warnings
