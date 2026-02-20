from pathlib import Path
from typing import List, Tuple
import re


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif"}


def get_image_files(directory: str) -> List[Path]:
    """디렉토리에서 이미지 파일 목록 반환."""
    dir_path = Path(directory)
    if not dir_path.exists():
        return []

    images = []
    for f in sorted(dir_path.rglob("*")):
        if f.suffix.lower() in IMAGE_EXTENSIONS and f.is_file():
            images.append(f)
    return images


def get_caption_for_image(image_path: Path) -> str:
    """이미지에 대응하는 .txt 캡션 파일 내용 반환."""
    caption_path = image_path.with_suffix(".txt")
    if caption_path.exists():
        return caption_path.read_text(encoding="utf-8").strip()
    return ""


def save_caption_for_image(image_path: Path, caption: str):
    """이미지에 대응하는 .txt 캡션 파일 저장."""
    caption_path = image_path.with_suffix(".txt")
    caption_path.write_text(caption, encoding="utf-8")


def parse_kohya_folder(directory: str) -> List[dict]:
    """Kohya 스타일 폴더 구조 파싱.

    예: 10_sks_person/ → repeats=10, concept="sks_person"
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return []

    results = []
    for subfolder in sorted(dir_path.iterdir()):
        if not subfolder.is_dir():
            continue
        if subfolder.name.startswith("."):
            continue

        match = re.match(r"^(\d+)_(.+)$", subfolder.name)
        if match:
            repeats = int(match.group(1))
            concept = match.group(2)
        else:
            repeats = 1
            concept = subfolder.name

        images = get_image_files(str(subfolder))
        captioned = sum(1 for img in images if get_caption_for_image(img))

        results.append({
            "path": str(subfolder),
            "name": subfolder.name,
            "repeats": repeats,
            "concept": concept,
            "image_count": len(images),
            "captioned_count": captioned,
            "images": [str(img) for img in images],
        })

    return results


def ensure_directory(path: str) -> Path:
    """디렉토리가 없으면 생성."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_file_size_mb(path: str) -> float:
    """파일 크기를 MB로 반환."""
    p = Path(path)
    if p.exists():
        return round(p.stat().st_size / (1024 * 1024), 2)
    return 0.0
