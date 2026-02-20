import torch
from typing import Optional


def get_gpu_info() -> Optional[dict]:
    """GPU 이름, 총 VRAM, 사용 가능 VRAM 반환."""
    if not torch.cuda.is_available():
        return None

    gpu = torch.cuda.get_device_properties(0)
    total_vram = gpu.total_memory / (1024 ** 3)
    allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
    free_vram = total_vram - allocated

    return {
        "name": gpu.name,
        "total_gb": round(total_vram, 2),
        "allocated_gb": round(allocated, 2),
        "reserved_gb": round(reserved, 2),
        "free_gb": round(free_vram, 2),
        "cuda_version": torch.version.cuda or "N/A",
    }


def suggest_vram_profile(total_vram_gb: float) -> str:
    """GPU VRAM에 따른 프리셋 추천."""
    if total_vram_gb < 10:
        return "8gb"
    elif total_vram_gb < 14:
        return "12gb"
    elif total_vram_gb < 20:
        return "16gb"
    else:
        return "24gb+"


def get_memory_usage() -> dict:
    """현재 GPU 메모리 사용량."""
    if not torch.cuda.is_available():
        return {"allocated_gb": 0, "reserved_gb": 0, "max_allocated_gb": 0}

    return {
        "allocated_gb": round(torch.cuda.memory_allocated(0) / (1024 ** 3), 2),
        "reserved_gb": round(torch.cuda.memory_reserved(0) / (1024 ** 3), 2),
        "max_allocated_gb": round(torch.cuda.max_memory_allocated(0) / (1024 ** 3), 2),
    }


def clear_vram():
    """GPU VRAM 정리."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
