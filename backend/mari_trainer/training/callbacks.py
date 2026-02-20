import logging
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """학습 상태 추적."""
    current_step: int = 0
    total_steps: int = 0
    current_loss: float = 0.0
    loss_history: List[float] = field(default_factory=list)
    lr_history: List[float] = field(default_factory=list)
    sample_images: List[str] = field(default_factory=list)
    start_time: float = 0.0
    is_training: bool = False
    is_paused: bool = False
    error: Optional[str] = None

    @property
    def elapsed_seconds(self) -> float:
        if self.start_time == 0:
            return 0
        return time.time() - self.start_time

    @property
    def steps_per_second(self) -> float:
        elapsed = self.elapsed_seconds
        if elapsed == 0 or self.current_step == 0:
            return 0
        return self.current_step / elapsed

    @property
    def eta_seconds(self) -> float:
        sps = self.steps_per_second
        if sps == 0:
            return 0
        remaining = self.total_steps - self.current_step
        return remaining / sps

    @property
    def progress_percent(self) -> float:
        if self.total_steps == 0:
            return 0
        return (self.current_step / self.total_steps) * 100

    def to_dict(self) -> dict:
        return {
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "current_loss": round(self.current_loss, 6),
            "progress_percent": round(self.progress_percent, 1),
            "steps_per_second": round(self.steps_per_second, 2),
            "eta_seconds": round(self.eta_seconds, 1),
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "is_training": self.is_training,
            "is_paused": self.is_paused,
            "error": self.error,
            "loss_history": self.loss_history[-100:],  # 최근 100개만
            "lr_history": self.lr_history[-100:],
            "sample_images": self.sample_images,
        }


class TrainingCallbackManager:
    """학습 콜백 관리."""

    def __init__(self):
        self._on_step_end: List[Callable] = []
        self._on_save: List[Callable] = []
        self._on_sample: List[Callable] = []
        self._on_log: List[Callable] = []
        self._on_error: List[Callable] = []
        self._on_complete: List[Callable] = []

    def on_step_end(self, callback: Callable):
        self._on_step_end.append(callback)

    def on_save(self, callback: Callable):
        self._on_save.append(callback)

    def on_sample(self, callback: Callable):
        self._on_sample.append(callback)

    def on_log(self, callback: Callable):
        self._on_log.append(callback)

    def on_error(self, callback: Callable):
        self._on_error.append(callback)

    def on_complete(self, callback: Callable):
        self._on_complete.append(callback)

    def fire_step_end(self, state: TrainingState):
        for cb in self._on_step_end:
            try:
                cb(state)
            except Exception as e:
                logger.error(f"콜백 오류 (step_end): {e}")

    def fire_save(self, state: TrainingState, path: str):
        for cb in self._on_save:
            try:
                cb(state, path)
            except Exception as e:
                logger.error(f"콜백 오류 (save): {e}")

    def fire_sample(self, state: TrainingState, images: list):
        for cb in self._on_sample:
            try:
                cb(state, images)
            except Exception as e:
                logger.error(f"콜백 오류 (sample): {e}")

    def fire_log(self, message: str):
        for cb in self._on_log:
            try:
                cb(message)
            except Exception as e:
                logger.error(f"콜백 오류 (log): {e}")

    def fire_error(self, error: str):
        for cb in self._on_error:
            try:
                cb(error)
            except Exception as e:
                logger.error(f"콜백 오류 (error): {e}")

    def fire_complete(self, state: TrainingState):
        for cb in self._on_complete:
            try:
                cb(state)
            except Exception as e:
                logger.error(f"콜백 오류 (complete): {e}")
