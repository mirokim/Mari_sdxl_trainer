"""데이터셋 관리 API."""
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse

from ..utils.file_utils import (
    get_image_files,
    parse_kohya_folder,
    get_caption_for_image,
    save_caption_for_image,
    ensure_directory,
)
from ..data.caption_utils import validate_captions
from ..data.preprocessing import get_image_dimensions

router = APIRouter()

DATASETS_DIR = Path("../datasets")


@router.get("/list")
async def list_datasets():
    """데이터셋 폴더 목록."""
    DATASETS_DIR.mkdir(exist_ok=True)
    datasets = []
    for d in sorted(DATASETS_DIR.iterdir()):
        if d.is_dir() and not d.name.startswith("."):
            images = get_image_files(str(d))
            datasets.append({
                "name": d.name,
                "path": str(d),
                "image_count": len(images),
                "subfolders": parse_kohya_folder(str(d)),
            })
    return {"datasets": datasets}


@router.post("/create")
async def create_dataset(name: str = Form(...)):
    """새 데이터셋 폴더 생성."""
    dataset_path = DATASETS_DIR / name
    if dataset_path.exists():
        return JSONResponse(
            status_code=400,
            content={"error": f"데이터셋 '{name}'이(가) 이미 존재합니다."},
        )
    dataset_path.mkdir(parents=True)
    return {"path": str(dataset_path), "name": name}


@router.post("/{dataset_name}/upload")
async def upload_images(
    dataset_name: str,
    subfolder: str = Form(""),
    files: list[UploadFile] = File(...),
):
    """이미지 업로드."""
    if subfolder:
        target_dir = DATASETS_DIR / dataset_name / subfolder
    else:
        target_dir = DATASETS_DIR / dataset_name
    target_dir.mkdir(parents=True, exist_ok=True)

    uploaded = []
    for file in files:
        file_path = target_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        uploaded.append(str(file_path))

    return {"uploaded": len(uploaded), "files": uploaded}


@router.get("/{dataset_name}/images")
async def get_images(dataset_name: str):
    """데이터셋의 이미지 목록 및 캡션."""
    dataset_path = DATASETS_DIR / dataset_name
    if not dataset_path.exists():
        return JSONResponse(status_code=404, content={"error": "데이터셋 없음"})

    images = get_image_files(str(dataset_path))
    result = []
    for img_path in images:
        w, h = get_image_dimensions(str(img_path))
        caption = get_caption_for_image(img_path)
        result.append({
            "path": str(img_path),
            "filename": img_path.name,
            "width": w,
            "height": h,
            "caption": caption,
            "has_caption": bool(caption),
            "relative_path": str(img_path.relative_to(DATASETS_DIR)),
        })

    # 통계
    all_paths = [str(img) for img in images]
    stats = validate_captions(all_paths)

    return {"images": result, "stats": stats}


@router.post("/{dataset_name}/caption")
async def update_caption(
    dataset_name: str,
    image_path: str = Form(...),
    caption: str = Form(...),
):
    """개별 이미지 캡션 수정."""
    img = Path(image_path)
    if not img.exists():
        return JSONResponse(status_code=404, content={"error": "이미지 없음"})

    save_caption_for_image(img, caption)
    return {"path": str(img), "caption": caption}


@router.delete("/{dataset_name}/image")
async def delete_image(dataset_name: str, image_path: str):
    """이미지 및 캡션 삭제."""
    img = Path(image_path)
    if img.exists():
        img.unlink()
    caption_path = img.with_suffix(".txt")
    if caption_path.exists():
        caption_path.unlink()
    return {"deleted": str(img)}


@router.get("/{dataset_name}/stats")
async def dataset_stats(dataset_name: str):
    """데이터셋 통계."""
    dataset_path = DATASETS_DIR / dataset_name
    if not dataset_path.exists():
        return JSONResponse(status_code=404, content={"error": "데이터셋 없음"})

    folders = parse_kohya_folder(str(dataset_path))
    images = get_image_files(str(dataset_path))
    all_paths = [str(img) for img in images]

    return {
        "name": dataset_name,
        "folders": folders,
        "total_images": len(images),
        "caption_stats": validate_captions(all_paths),
    }
