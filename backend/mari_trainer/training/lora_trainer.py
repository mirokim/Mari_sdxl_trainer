import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from .base_trainer import BaseTrainer
from .training_config import TrainingConfig
from .optimizers import create_optimizer, create_param_groups
from .schedulers import create_scheduler
from ..models.sdxl_loader import SDXLModelLoader
from ..models.lora_injection import (
    inject_lora_to_unet,
    inject_lora_to_text_encoders,
    freeze_model,
)
from ..conversion.lora_to_kohya import convert_and_save_kohya_lora
from ..data.preprocessing import compute_sdxl_time_ids
from ..utils.gpu_utils import clear_vram

logger = logging.getLogger(__name__)


class LoRATrainer(BaseTrainer):
    """SDXL LoRA 학습 트레이너."""

    def __init__(self, config: TrainingConfig):
        config.training_mode = "lora"
        super().__init__(config)

    def setup_model(self):
        """SDXL 모델 로드 및 LoRA 주입."""
        self.callbacks.fire_log("SDXL 모델 로딩 중...")

        components = SDXLModelLoader.load_components(
            model_path=self.config.pretrained_model_name_or_path,
            dtype=self.weight_dtype,
            vae_path=self.config.vae_path,
            use_fp16_vae_fix=self.config.use_fp16_vae_fix,
        )

        self.unet = components["unet"]
        self.vae = components["vae"]
        self.text_encoder_one = components["text_encoder_one"]
        self.text_encoder_two = components["text_encoder_two"]
        self.tokenizer_one = components["tokenizer_one"]
        self.tokenizer_two = components["tokenizer_two"]
        self.noise_scheduler = components["noise_scheduler"]

        # 기본 모델 고정
        freeze_model(self.unet)
        freeze_model(self.text_encoder_one)
        freeze_model(self.text_encoder_two)
        if self.vae is not None:
            freeze_model(self.vae)

        # UNet에 LoRA 주입
        self.callbacks.fire_log(
            f"LoRA 주입: rank={self.config.lora_rank}, alpha={self.config.lora_alpha}"
        )
        self.unet = inject_lora_to_unet(
            self.unet,
            rank=self.config.lora_rank,
            alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
        )

        # 텍스트 인코더 LoRA (선택적)
        if self.config.train_text_encoder and not self.config.cache_text_encoder_outputs:
            self.callbacks.fire_log("텍스트 인코더 LoRA 주입...")
            self.text_encoder_one, self.text_encoder_two = inject_lora_to_text_encoders(
                self.text_encoder_one,
                self.text_encoder_two,
                rank=self.config.lora_rank,
                alpha=self.config.lora_alpha,
            )

        # 그래디언트 체크포인팅
        if self.config.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            if self.config.train_text_encoder:
                self.text_encoder_one.gradient_checkpointing_enable()
                self.text_encoder_two.gradient_checkpointing_enable()

        # xformers
        if self.config.enable_xformers:
            try:
                self.unet.enable_xformers_memory_efficient_attention()
                self.callbacks.fire_log("xformers 활성화")
            except Exception:
                self.callbacks.fire_log("xformers 사용 불가, 기본 어텐션 사용")

        # GPU 이동
        self.unet.to(self.device)
        if not self.config.cache_text_encoder_outputs:
            self.text_encoder_one.to(self.device)
            self.text_encoder_two.to(self.device)

        self.callbacks.fire_log("모델 설정 완료")

    def setup_training(self):
        """옵티마이저 및 스케줄러 설정."""
        # 파라미터 그룹
        te1 = self.text_encoder_one if self.config.train_text_encoder else None
        te2 = self.text_encoder_two if self.config.train_text_encoder else None

        param_groups = create_param_groups(
            unet=self.unet,
            text_encoder_one=te1,
            text_encoder_two=te2,
            learning_rate=self.config.learning_rate,
            text_encoder_lr=self.config.text_encoder_lr,
        )

        total_params = sum(p.numel() for group in param_groups for p in group["params"])
        self.callbacks.fire_log(f"학습 파라미터: {total_params:,}개")

        # 옵티마이저
        self.optimizer = create_optimizer(
            optimizer_type=self.config.optimizer_type,
            params=param_groups,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            prodigy_decouple=self.config.prodigy_decouple,
            prodigy_use_bias_correction=self.config.prodigy_use_bias_correction,
        )

        # LR 스케줄러
        self.lr_scheduler = create_scheduler(
            scheduler_type=self.config.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.lr_warmup_steps,
            num_training_steps=self.config.max_train_steps,
        )

    def train_step(self, batch: dict) -> float:
        """단일 학습 스텝."""
        self.unet.train()

        # 1. 잠재벡터
        if "cached_latents" in batch:
            latents = batch["cached_latents"].to(self.device, dtype=self.weight_dtype)
        else:
            with torch.no_grad():
                pixel_values = batch["pixel_values"].to(self.device, dtype=self.weight_dtype)
                latents = self.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor

        # 2. 노이즈 & 타임스텝
        noise = torch.randn_like(latents)
        if self.config.noise_offset > 0:
            noise += self.config.noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
            )

        bsz = latents.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=self.device, dtype=torch.long,
        )
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # 3. 텍스트 임베딩
        if "cached_prompt_embeds" in batch:
            prompt_embeds = batch["cached_prompt_embeds"].to(self.device, dtype=self.weight_dtype)
            pooled_prompt_embeds = batch["cached_pooled_prompt_embeds"].to(
                self.device, dtype=self.weight_dtype
            )
        else:
            prompt_embeds, pooled_prompt_embeds = self.encode_prompt(batch["caption"])

        # 4. SDXL 시간 조건
        if "time_ids" in batch:
            add_time_ids = batch["time_ids"].to(self.device, dtype=self.weight_dtype)
        else:
            add_time_ids = compute_sdxl_time_ids(
                original_size=(self.config.resolution, self.config.resolution),
                crop_coords_top_left=(0, 0),
                target_size=(self.config.resolution, self.config.resolution),
            ).to(self.device, dtype=self.weight_dtype)
            add_time_ids = add_time_ids.repeat(bsz, 1)

        added_cond_kwargs = {
            "text_embeds": pooled_prompt_embeds,
            "time_ids": add_time_ids,
        }

        # 5. 노이즈 예측
        model_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

        # 6. 타겟 결정
        if self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        elif self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        else:
            target = noise

        # 7. 손실 계산
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        # Min-SNR 가중치
        if self.config.min_snr_gamma > 0:
            snr = self._compute_snr(timesteps)
            mse_loss_weights = torch.stack(
                [snr, self.config.min_snr_gamma * torch.ones_like(snr)], dim=1
            ).min(dim=1)[0] / snr
            loss = loss * mse_loss_weights.mean()

        # 8. 역전파
        loss = loss / self.config.gradient_accumulation_steps
        loss.backward()

        if (self.state.current_step + 1) % self.config.gradient_accumulation_steps == 0:
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.unet.parameters() if p.requires_grad],
                    self.config.max_grad_norm,
                )
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

        return loss.item() * self.config.gradient_accumulation_steps

    def _compute_snr(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Signal-to-Noise Ratio 계산 (Min-SNR 감마 가중치용)."""
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(
            device=timesteps.device, dtype=torch.float32
        )
        sqrt_alphas_cumprod = alphas_cumprod ** 0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5
        alpha = sqrt_alphas_cumprod[timesteps]
        sigma = sqrt_one_minus_alphas_cumprod[timesteps]
        snr = (alpha / sigma) ** 2
        return snr

    def save_checkpoint(self, step: int):
        """LoRA 체크포인트 저장."""
        output_dir = Path(self.config.output_dir) / self.config.run_name
        step_dir = output_dir / f"step_{step}"
        step_dir.mkdir(parents=True, exist_ok=True)

        self.callbacks.fire_log(f"체크포인트 저장: step {step}")

        # Kohya 포맷으로 저장 (ComfyUI 호환)
        if self.config.save_kohya_format:
            kohya_path = step_dir / f"{self.config.run_name}_step{step}.safetensors"
            convert_and_save_kohya_lora(
                unet=self.unet,
                text_encoder_one=self.text_encoder_one if self.config.train_text_encoder else None,
                text_encoder_two=self.text_encoder_two if self.config.train_text_encoder else None,
                output_path=str(kohya_path),
                dtype=self.weight_dtype,
                lora_alpha=self.config.lora_alpha,
            )
            self.callbacks.fire_log(f"Kohya LoRA 저장: {kohya_path}")

        # 최종 스텝이면 output_dir 루트에도 저장
        if step == self.config.max_train_steps:
            final_path = output_dir / f"{self.config.run_name}.safetensors"
            convert_and_save_kohya_lora(
                unet=self.unet,
                text_encoder_one=self.text_encoder_one if self.config.train_text_encoder else None,
                text_encoder_two=self.text_encoder_two if self.config.train_text_encoder else None,
                output_path=str(final_path),
                dtype=self.weight_dtype,
                lora_alpha=self.config.lora_alpha,
            )
            self.callbacks.fire_log(f"최종 LoRA 저장: {final_path}")

    def full_setup_and_train(self):
        """전체 학습 파이프라인 실행."""
        try:
            self.setup_model()
            self.setup_dataset()
            self.run_caching()
            self.setup_training()
            self.train()
        finally:
            self.cleanup()
