from typing import Dict, Any


VRAM_PROFILES: Dict[str, Dict[str, Any]] = {
    "8gb": {
        "description": "8GB VRAM (RTX 3060 8GB, RTX 4060 등)",
        "training_mode": "lora",
        "mixed_precision": "fp16",
        "gradient_checkpointing": True,
        "cache_latents": True,
        "cache_text_encoder_outputs": True,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "enable_xformers": True,
        "lora_rank": 16,
        "lora_alpha": 8,
        "train_text_encoder": False,
        "optimizer_type": "AdamW8bit",
        "resolution": 1024,
        "enable_bucketing": True,
    },
    "12gb": {
        "description": "12GB VRAM (RTX 3060 12GB, RTX 4070 등)",
        "training_mode": "lora",
        "mixed_precision": "bf16",
        "gradient_checkpointing": True,
        "cache_latents": True,
        "cache_text_encoder_outputs": True,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 2,
        "enable_xformers": True,
        "lora_rank": 32,
        "lora_alpha": 16,
        "train_text_encoder": False,
        "optimizer_type": "AdamW8bit",
        "resolution": 1024,
        "enable_bucketing": True,
    },
    "16gb": {
        "description": "16GB VRAM (RTX 4060 Ti 16GB, RTX 4080 등)",
        "training_mode": "lora",
        "mixed_precision": "bf16",
        "gradient_checkpointing": True,
        "cache_latents": True,
        "cache_text_encoder_outputs": False,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "enable_xformers": True,
        "lora_rank": 32,
        "lora_alpha": 16,
        "train_text_encoder": True,
        "optimizer_type": "AdamW8bit",
        "resolution": 1024,
        "enable_bucketing": True,
    },
    "24gb+": {
        "description": "24GB+ VRAM (RTX 3090, RTX 4090, A5000 등)",
        "training_mode": "lora",
        "mixed_precision": "bf16",
        "gradient_checkpointing": False,
        "cache_latents": True,
        "cache_text_encoder_outputs": False,
        "train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "enable_xformers": True,
        "lora_rank": 64,
        "lora_alpha": 32,
        "train_text_encoder": True,
        "optimizer_type": "Prodigy",
        "resolution": 1024,
        "enable_bucketing": True,
    },
}


def get_profile(name: str) -> dict:
    """이름으로 VRAM 프리셋 반환."""
    return VRAM_PROFILES.get(name, VRAM_PROFILES["16gb"]).copy()


def get_profile_names() -> list:
    """사용 가능한 프리셋 이름 목록."""
    return list(VRAM_PROFILES.keys())


def apply_profile_to_config(config, profile_name: str):
    """프리셋을 TrainingConfig에 적용."""
    profile = get_profile(profile_name)
    profile.pop("description", None)
    for key, value in profile.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config
