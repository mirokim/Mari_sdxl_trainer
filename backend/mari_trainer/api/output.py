"""학습 결과물 관리 API."""
import logging
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse

from ..utils.file_utils import get_file_size_mb

logger = logging.getLogger(__name__)
router = APIRouter()

OUTPUTS_DIR = Path("../outputs")


@router.get("/models")
async def list_models():
    """학습된 모델 목록."""
    OUTPUTS_DIR.mkdir(exist_ok=True)
    models = []

    for run_dir in sorted(OUTPUTS_DIR.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue

        safetensors = list(run_dir.glob("*.safetensors"))
        checkpoints = []

        for st_file in safetensors:
            checkpoints.append({
                "filename": st_file.name,
                "path": str(st_file),
                "size_mb": get_file_size_mb(str(st_file)),
            })

        # step 디렉토리 내 체크포인트
        for step_dir in sorted(run_dir.glob("step_*")):
            for st_file in step_dir.glob("*.safetensors"):
                checkpoints.append({
                    "filename": st_file.name,
                    "path": str(st_file),
                    "size_mb": get_file_size_mb(str(st_file)),
                    "step": step_dir.name,
                })

        if checkpoints:
            models.append({
                "name": run_dir.name,
                "path": str(run_dir),
                "checkpoints": checkpoints,
            })

    return {"models": models}


@router.get("/download")
async def download_model(path: str):
    """모델 파일 다운로드."""
    file_path = Path(path)
    if not file_path.exists() or not file_path.suffix == ".safetensors":
        return JSONResponse(status_code=404, content={"error": "파일 없음"})

    return FileResponse(
        path=str(file_path),
        filename=file_path.name,
        media_type="application/octet-stream",
    )


@router.delete("/model")
async def delete_model(path: str):
    """모델 삭제."""
    file_path = Path(path)
    if file_path.exists() and file_path.suffix == ".safetensors":
        file_path.unlink()
        return {"deleted": str(file_path)}
    return JSONResponse(status_code=404, content={"error": "파일 없음"})
