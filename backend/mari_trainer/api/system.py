"""시스템 정보 API."""
from fastapi import APIRouter

from ..utils.gpu_utils import get_gpu_info, suggest_vram_profile, get_memory_usage
from ..training.vram_profiles import VRAM_PROFILES, get_profile_names

router = APIRouter()


@router.get("/gpu")
async def gpu_info():
    """GPU 정보 반환."""
    info = get_gpu_info()
    if info is None:
        return {"available": False, "message": "CUDA GPU를 찾을 수 없습니다."}

    suggested = suggest_vram_profile(info["total_gb"])
    return {
        "available": True,
        **info,
        "suggested_profile": suggested,
    }


@router.get("/memory")
async def memory_usage():
    """현재 GPU 메모리 사용량."""
    return get_memory_usage()


@router.get("/vram-profiles")
async def vram_profiles():
    """사용 가능한 VRAM 프리셋 목록."""
    return {
        "profiles": VRAM_PROFILES,
        "available": get_profile_names(),
    }
