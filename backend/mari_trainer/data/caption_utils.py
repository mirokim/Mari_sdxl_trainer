import random
from pathlib import Path
from typing import List, Tuple


def read_caption(image_path: str) -> str:
    """이미지에 대응하는 .txt 캡션 파일 읽기."""
    caption_path = Path(image_path).with_suffix(".txt")
    if caption_path.exists():
        return caption_path.read_text(encoding="utf-8").strip()
    return ""


def write_caption(image_path: str, caption: str):
    """이미지에 대응하는 .txt 캡션 파일 쓰기."""
    caption_path = Path(image_path).with_suffix(".txt")
    caption_path.write_text(caption, encoding="utf-8")


def apply_caption_dropout(caption: str, dropout_rate: float = 0.05) -> str:
    """캡션 드롭아웃. 일정 확률로 빈 캡션 반환 (유연성 향상)."""
    if random.random() < dropout_rate:
        return ""
    return caption


def add_trigger_word(caption: str, trigger_word: str, position: str = "prefix") -> str:
    """캡션에 트리거 워드 추가.

    Args:
        position: "prefix" (앞에 추가) or "suffix" (뒤에 추가)
    """
    if not trigger_word:
        return caption

    trigger_word = trigger_word.strip()
    caption = caption.strip()

    if position == "prefix":
        if caption:
            return f"{trigger_word}, {caption}"
        return trigger_word
    else:
        if caption:
            return f"{caption}, {trigger_word}"
        return trigger_word


def get_all_captions(image_paths: List[str]) -> List[Tuple[str, str]]:
    """이미지 경로 목록에서 (경로, 캡션) 쌍 반환."""
    return [(path, read_caption(path)) for path in image_paths]


def validate_captions(image_paths: List[str]) -> dict:
    """캡션 검증 결과 반환."""
    total = len(image_paths)
    captioned = 0
    empty = 0
    missing = 0

    for path in image_paths:
        caption = read_caption(path)
        if caption:
            captioned += 1
        elif Path(path).with_suffix(".txt").exists():
            empty += 1
        else:
            missing += 1

    return {
        "total": total,
        "captioned": captioned,
        "empty": empty,
        "missing": missing,
        "complete": captioned == total,
    }
