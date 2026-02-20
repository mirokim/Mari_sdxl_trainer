import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from .base_trainer import BaseTrainer
from .training_config import TrainingConfig
from .optimizers import create_optimizer, create_param_groups
from .schedulers import create_scheduler
from ..models.sdxl_loader import SDXLModelLoader
from ..models.lora_injection import freeze_model
from ..conversion.checkpoint_to_single import convert_diffusers_to_single_safetensors
from ..data.preprocessing import compute_sdxl_time_ids
from ..utils.gpu_utils import clear_vram

logger = logging.getLogger(__name__)


class FullFinetuneTrainer(BaseTrainer):
    """SDXL 풀 체크포인트 파인튜닝 트레이너.

    24GB+ VRAM 필요. 전체 UNet 파라미터 학습.
    """

    def __init__(self, config: TrainingConfig):
        config.training_mode = "full"
        super().__init__(config)

    def setup_model(self):
        """SDXL 모델 로드 (파인튜닝 모드)."""
        self.callbacks.fire_log("SDXL 모델 로딩 (풀 파인튜닝)...")

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

        # VAE는 항상 고정
        freeze_model(self.vae)

        # 텍스트 인코더 고정 (풀 파인튜닝은 UNet만)
        if not self.config.train_text_encoder:
            freeze_model(self.text_encoder_one)
            freeze_model(self.text_encoder_two)

        # UNet은 학습 가능 상태 유지
        self.unet.requires_grad_(True)

        # 그래디언트 체크포인팅
        if self.config.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

        # xformers
        if self.config.enable_xformers:
            try:
                self.unet.enable_xformers_memory_efficient_attention()
                self.callbacks.fire_log("xformers 활성화")
            except Exception:
                self.callbacks.fire_log("xformers 사용 불가")

        # GPU 이동
        self.unet.to(self.device)
        if not self.config.cache_text_encoder_outputs:
            self.text_encoder_one.to(self.device)
            self.text_encoder_two.to(self.device)

        total_params = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
        self.callbacks.fire_log(f"풀 파인튜닝: {total_params:,}개 UNet 파라미터 학습")

    def setup_training(self):
        """옵티마이저 및 스케줄러 설정."""
        te1 = self.text_encoder_one if self.config.train_text_encoder else None
        te2 = self.text_encoder_two if self.config.train_text_encoder else None

        param_groups = create_param_groups(
            unet=self.unet,
            text_encoder_one=te1,
            text_encoder_two=te2,
            learning_rate=self.config.learning_rate,
            text_encoder_lr=self.config.text_encoder_lr,
        )

        self.optimizer = create_optimizer(
            optimizer_type=self.config.optimizer_type,
            params=param_groups,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.lr_scheduler = create_scheduler(
            scheduler_type=self.config.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.lr_warmup_steps,
            num_training_steps=self.config.max_train_steps,
        )

    def train_step(self, batch: dict) -> float:
        """단일 학습 스텝 (LoRA 트레이너와 동일한 로직)."""
        self.unet.train()

        # 잠재벡터
        if "cached_latents" in batch:
            latents = batch["cached_latents"].to(self.device, dtype=self.weight_dtype)
        else:
            with torch.no_grad():
                pixel_values = batch["pixel_values"].to(self.device, dtype=self.weight_dtype)
                latents = self.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor

        # 노이즈
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=self.device, dtype=torch.long,
        )
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # 텍스트 임베딩
        if "cached_prompt_embeds" in batch:
            prompt_embeds = batch["cached_prompt_embeds"].to(self.device, dtype=self.weight_dtype)
            pooled_prompt_embeds = batch["cached_pooled_prompt_embeds"].to(
                self.device, dtype=self.weight_dtype
            )
        else:
            prompt_embeds, pooled_prompt_embeds = self.encode_prompt(batch["caption"])

        # SDXL 시간 조건
        if "time_ids" in batch:
            add_time_ids = batch["time_ids"].to(self.device, dtype=self.weight_dtype)
        else:
            add_time_ids = compute_sdxl_time_ids(
                original_size=(self.config.resolution, self.config.resolution),
                crop_coords_top_left=(0, 0),
                target_size=(self.config.resolution, self.config.resolution),
            ).to(self.device, dtype=self.weight_dtype).repeat(bsz, 1)

        added_cond_kwargs = {
            "text_embeds": pooled_prompt_embeds,
            "time_ids": add_time_ids,
        }

        model_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

        if self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            target = noise

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        loss = loss / self.config.gradient_accumulation_steps
        loss.backward()

        if (self.state.current_step + 1) % self.config.gradient_accumulation_steps == 0:
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.unet.parameters(), self.config.max_grad_norm
                )
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

        return loss.item() * self.config.gradient_accumulation_steps

    def save_checkpoint(self, step: int):
        """풀 체크포인트 저장."""
        from diffusers import StableDiffusionXLPipeline

        output_dir = Path(self.config.output_dir) / self.config.run_name
        step_dir = output_dir / f"step_{step}"
        step_dir.mkdir(parents=True, exist_ok=True)

        self.callbacks.fire_log(f"체크포인트 저장: step {step}")

        # diffusers 형식 저장
        diffusers_dir = step_dir / "diffusers"
        pipeline = StableDiffusionXLPipeline(
            unet=self.unet,
            vae=self.vae,
            text_encoder=self.text_encoder_one,
            text_encoder_2=self.text_encoder_two,
            tokenizer=self.tokenizer_one,
            tokenizer_2=self.tokenizer_two,
            scheduler=self.noise_scheduler,
        )
        pipeline.save_pretrained(str(diffusers_dir))

        # 단일 safetensors 변환
        if step == self.config.max_train_steps:
            single_path = output_dir / f"{self.config.run_name}.safetensors"
            self.callbacks.fire_log("단일 safetensors 변환 중...")
            convert_diffusers_to_single_safetensors(
                model_path=str(diffusers_dir),
                output_path=str(single_path),
                dtype=self.weight_dtype,
            )
            self.callbacks.fire_log(f"최종 체크포인트: {single_path}")

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
