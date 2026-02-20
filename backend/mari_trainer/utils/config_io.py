import json
import yaml
from pathlib import Path
from typing import Any


def save_config_yaml(config: dict, path: str):
    """설정을 YAML 파일로 저장."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def load_config_yaml(path: str) -> dict:
    """YAML 설정 파일 로드."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_config_json(config: dict, path: str):
    """설정을 JSON 파일로 저장."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def load_config_json(path: str) -> dict:
    """JSON 설정 파일 로드."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dataclass_to_dict(obj: Any) -> dict:
    """데이터클래스를 딕셔너리로 변환."""
    from dataclasses import asdict
    return asdict(obj)


def dict_to_dataclass(cls, data: dict):
    """딕셔너리를 데이터클래스로 변환 (알 수 없는 필드 무시)."""
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in data.items() if k in field_names}
    return cls(**filtered)
