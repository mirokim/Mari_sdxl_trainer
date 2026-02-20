import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .training_config import TrainingConfig
from .callbacks import TrainingState, TrainingCallbackManager
from .optimizers import create_optimizer, create_param_groups
from .schedulers import create_scheduler
from ..data.dataset import SDXLDataset
from ..utils.gpu_utils import clear_vram, get_memory_usage

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """SDXL 학습 베이스 클래스."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.state = TrainingState(total_steps=config.max_train_steps)
        self.callbacks = TrainingCallbackManager()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # dtype 설정
        if config.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16
        elif config.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        else:
            self.weight_dtype = torch.float32

        # 모델 컴포넌트 (서브클래스에서 초기화)
        self.unet = None
        self.vae = None
        self.text_encoder_one = None
        self.text_encoder_two = None
        self.tokenizer_one = None
        self.tokenizer_two = None
        self.noise_scheduler = None
        self.optimizer = None
        self.lr_scheduler = None
        self.dataset = None
        self.dataloader = None

        self._stop_requested = False

    @abstractmethod
    def setup_model(self):
        """모델 로드 및 준비."""
        pass

    @abstractmethod
    def setup_training(self):
        """옵티마이저, 스케줄러 등 학습 준비."""
        pass

    @abstractmethod
    def train_step(self, batch: dict) -> float:
        """단일 학습 스텝. loss 반환."""
        pass

    @abstractmethod
    def save_checkpoint(self, step: int):
        """체크포인트 저장."""
        pass

    def setup_dataset(self):
        """데이터셋 및 데이터로더 설정."""
        cached_latents_dir = None
        cached_text_embeds_dir = None

        output_dir = Path(self.config.output_dir) / self.config.run_name
        if self.config.cache_latents:
            cached_latents_dir = str(output_dir / "cache" / "latents")
        if self.config.cache_text_encoder_outputs:
            cached_text_embeds_dir = str(output_dir / "cache" / "text_embeds")

        self.dataset = SDXLDataset(
            dataset_path=self.config.dataset_path,
            resolution=self.config.resolution,
            enable_bucketing=self.config.enable_bucketing,
            random_flip=self.config.random_flip,
            caption_dropout_rate=self.config.caption_dropout_rate,
            cached_latents_dir=cached_latents_dir,
            cached_text_embeds_dir=cached_text_embeds_dir,
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        stats = self.dataset.get_stats()
        self.callbacks.fire_log(
            f"데이터셋: {stats['unique_images']}개 이미지, "
            f"{stats['total_with_repeats']}개 항목 (반복 포함)"
        )

    def run_caching(self):
        """잠재벡터 / 텍스트 임베딩 사전 캐싱."""
        output_dir = Path(self.config.output_dir) / self.config.run_name

        if self.config.cache_latents and self.vae is not None:
            from ..data.latent_cache import LatentCache

            self.callbacks.fire_log("잠재벡터 캐싱 시작...")
            cache = LatentCache(
                vae=self.vae,
                cache_dir=str(output_dir / "cache" / "latents"),
                dtype=self.weight_dtype,
                device=self.device,
            )
            cache.encode_and_cache(self.dataset.items)

            # VAE VRAM 해제
            self.vae.cpu()
            del self.vae
            self.vae = None
            clear_vram()
            self.callbacks.fire_log("잠재벡터 캐싱 완료, VAE 언로드")

        if self.config.cache_text_encoder_outputs:
            from ..data.text_embed_cache import TextEmbedCache

            self.callbacks.fire_log("텍스트 임베딩 캐싱 시작...")
            cache = TextEmbedCache(
                text_encoder_one=self.text_encoder_one,
                text_encoder_two=self.text_encoder_two,
                tokenizer_one=self.tokenizer_one,
                tokenizer_two=self.tokenizer_two,
                cache_dir=str(output_dir / "cache" / "text_embeds"),
                dtype=self.weight_dtype,
                device=self.device,
            )
            cache.encode_and_cache(self.dataset.items)

            # 텍스트 인코더 VRAM 해제
            self.text_encoder_one.cpu()
            self.text_encoder_two.cpu()
            del self.text_encoder_one, self.text_encoder_two
            self.text_encoder_one = None
            self.text_encoder_two = None
            clear_vram()
            self.callbacks.fire_log("텍스트 임베딩 캐싱 완료, 텍스트 인코더 언로드")

    def encode_prompt(self, captions: list) -> tuple:
        """캡션을 텍스트 인코더로 인코딩 (캐싱 미사용 시)."""
        prompt_embeds_list = []

        # 토큰화
        tokens_one = self.tokenizer_one(
            captions,
            max_length=self.tokenizer_one.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        tokens_two = self.tokenizer_two(
            captions,
            max_length=self.tokenizer_two.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        # 인코딩
        with torch.no_grad():
            output_one = self.text_encoder_one(
                tokens_one.input_ids, output_hidden_states=True
            )
            output_two = self.text_encoder_two(
                tokens_two.input_ids, output_hidden_states=True
            )

        prompt_embeds = torch.cat([
            output_one.hidden_states[-2],
            output_two.hidden_states[-2],
        ], dim=-1)

        pooled_prompt_embeds = output_two[0]

        return prompt_embeds, pooled_prompt_embeds

    def train(self):
        """메인 학습 루프."""
        self.state.is_training = True
        self.state.start_time = time.time()
        self._stop_requested = False

        self.callbacks.fire_log("학습 시작")
        self.callbacks.fire_log(f"모드: {self.config.training_mode}")
        self.callbacks.fire_log(f"총 스텝: {self.config.max_train_steps}")
        self.callbacks.fire_log(f"배치 크기: {self.config.train_batch_size}")
        self.callbacks.fire_log(f"그래디언트 누적: {self.config.gradient_accumulation_steps}")

        try:
            step = 0
            data_iter = iter(self.dataloader)

            while step < self.config.max_train_steps:
                if self._stop_requested:
                    self.callbacks.fire_log("학습 중지 요청")
                    break

                if self.state.is_paused:
                    time.sleep(0.5)
                    continue

                # 데이터 가져오기 (에포크 반복)
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.dataloader)
                    batch = next(data_iter)

                # 학습 스텝
                try:
                    loss = self.train_step(batch)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        clear_vram()
                        self.callbacks.fire_error(
                            f"GPU 메모리 부족! 배치 크기를 줄이거나 메모리 최적화 설정을 확인하세요. "
                            f"현재 사용량: {get_memory_usage()}"
                        )
                        self.state.error = "OOM"
                        break
                    raise

                step += 1
                self.state.current_step = step
                self.state.current_loss = loss
                self.state.loss_history.append(loss)

                if self.lr_scheduler:
                    current_lr = self.lr_scheduler.get_last_lr()[0]
                    self.state.lr_history.append(current_lr)

                self.callbacks.fire_step_end(self.state)

                # 체크포인트 저장
                if step % self.config.save_every_n_steps == 0:
                    self.save_checkpoint(step)
                    self.callbacks.fire_save(self.state, str(
                        Path(self.config.output_dir) / self.config.run_name / f"step_{step}"
                    ))

                # 샘플 생성
                if (self.config.sample_every_n_steps > 0
                        and step % self.config.sample_every_n_steps == 0
                        and self.config.sample_prompts):
                    self._generate_samples(step)

            # 최종 저장
            if not self._stop_requested:
                self.save_checkpoint(step)
                self.callbacks.fire_log("학습 완료!")

        except Exception as e:
            self.state.error = str(e)
            self.callbacks.fire_error(f"학습 오류: {e}")
            logger.exception("학습 중 오류 발생")
        finally:
            self.state.is_training = False
            self.callbacks.fire_complete(self.state)

    def _generate_samples(self, step: int):
        """학습 중 샘플 이미지 생성."""
        # 서브클래스에서 오버라이드 가능
        pass

    def stop(self):
        """학습 중지 요청."""
        self._stop_requested = True
        self.callbacks.fire_log("학습 중지 요청됨...")

    def pause(self):
        """학습 일시정지."""
        self.state.is_paused = True
        self.callbacks.fire_log("학습 일시정지")

    def resume(self):
        """학습 재개."""
        self.state.is_paused = False
        self.callbacks.fire_log("학습 재개")

    def cleanup(self):
        """리소스 정리."""
        for attr in ["unet", "vae", "text_encoder_one", "text_encoder_two"]:
            model = getattr(self, attr, None)
            if model is not None:
                model.cpu()
                setattr(self, attr, None)
        clear_vram()
        self.callbacks.fire_log("리소스 정리 완료")
