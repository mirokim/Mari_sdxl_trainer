import logging
from diffusers.optimization import get_scheduler

logger = logging.getLogger(__name__)


def create_scheduler(
    scheduler_type: str,
    optimizer,
    num_warmup_steps: int = 100,
    num_training_steps: int = 1500,
):
    """LR 스케줄러 팩토리.

    Args:
        scheduler_type: cosine, cosine_with_restarts, constant,
                       constant_with_warmup, linear
    """
    scheduler = get_scheduler(
        name=scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    logger.info(
        f"LR 스케줄러: {scheduler_type} "
        f"(warmup={num_warmup_steps}, total={num_training_steps})"
    )
    return scheduler
