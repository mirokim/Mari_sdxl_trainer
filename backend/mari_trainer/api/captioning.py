"""자동 캡셔닝 API."""
import asyncio
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from PIL import Image

from ..captioning.florence2_captioner import Florence2Captioner
from ..captioning.blip2_captioner import BLIP2Captioner
from ..data.caption_utils import write_caption, add_trigger_word
from ..utils.file_utils import get_image_files

logger = logging.getLogger(__name__)
router = APIRouter()

DATASETS_DIR = Path("../datasets")

# 캡셔닝 상태
_captioning_state = {
    "is_running": False,
    "current": 0,
    "total": 0,
    "current_file": "",
    "error": None,
}


class CaptioningRequest(BaseModel):
    dataset_name: str
    model: str = "florence-2-large"  # florence-2-base, florence-2-large, blip2
    caption_mode: str = "<MORE_DETAILED_CAPTION>"
    trigger_word: str = ""
    trigger_position: str = "prefix"  # prefix or suffix
    overwrite: bool = False


@router.post("/run")
async def run_captioning(request: CaptioningRequest, background_tasks: BackgroundTasks):
    """자동 캡셔닝 실행."""
    if _captioning_state["is_running"]:
        return {"error": "캡셔닝이 이미 실행 중입니다."}

    dataset_path = DATASETS_DIR / request.dataset_name
    if not dataset_path.exists():
        return {"error": "데이터셋 없음"}

    background_tasks.add_task(_run_captioning_task, request)
    return {"status": "started", "dataset": request.dataset_name}


@router.get("/status")
async def captioning_status():
    """캡셔닝 진행 상태."""
    return _captioning_state


@router.get("/models")
async def available_models():
    """사용 가능한 캡셔닝 모델 목록."""
    return {
        "models": [
            {
                "id": "florence-2-large",
                "name": "Florence-2 Large",
                "size": "~1.5GB",
                "description": "상세한 캡션 생성 (권장)",
            },
            {
                "id": "florence-2-base",
                "name": "Florence-2 Base",
                "size": "~500MB",
                "description": "빠른 캡션 생성",
            },
            {
                "id": "blip2",
                "name": "BLIP-2 OPT-2.7B",
                "size": "~5GB",
                "description": "높은 품질, 더 많은 VRAM 필요",
            },
        ]
    }


def _run_captioning_task(request: CaptioningRequest):
    """백그라운드 캡셔닝 작업."""
    global _captioning_state

    _captioning_state["is_running"] = True
    _captioning_state["error"] = None

    try:
        dataset_path = DATASETS_DIR / request.dataset_name
        images = get_image_files(str(dataset_path))
        _captioning_state["total"] = len(images)

        # 캡셔너 선택
        if "florence" in request.model:
            model_name = Florence2Captioner.MODELS.get(
                request.model, "microsoft/Florence-2-large"
            )
            captioner = Florence2Captioner(model_name=model_name)
        else:
            captioner = BLIP2Captioner()

        captioner.load_model()

        for i, img_path in enumerate(images):
            _captioning_state["current"] = i + 1
            _captioning_state["current_file"] = img_path.name

            # 기존 캡션 건너뛰기
            caption_path = img_path.with_suffix(".txt")
            if not request.overwrite and caption_path.exists():
                existing = caption_path.read_text(encoding="utf-8").strip()
                if existing:
                    continue

            # 캡셔닝
            image = Image.open(img_path).convert("RGB")
            caption = captioner.caption_image(image, prompt=request.caption_mode)

            # 트리거 워드 추가
            if request.trigger_word:
                caption = add_trigger_word(
                    caption, request.trigger_word, request.trigger_position
                )

            write_caption(str(img_path), caption)

        captioner.unload_model()
        logger.info(f"캡셔닝 완료: {len(images)}개 이미지")

    except Exception as e:
        logger.exception("캡셔닝 오류")
        _captioning_state["error"] = str(e)
    finally:
        _captioning_state["is_running"] = False
