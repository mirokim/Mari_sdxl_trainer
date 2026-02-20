"""학습 API (시작, 중지, 상태, WebSocket)."""
import asyncio
import json
import logging
import threading
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from ..training.training_config import TrainingConfig
from ..training.lora_trainer import LoRATrainer
from ..training.full_finetune_trainer import FullFinetuneTrainer
from ..training.vram_profiles import apply_profile_to_config
from ..utils.config_io import dataclass_to_dict, dict_to_dataclass

logger = logging.getLogger(__name__)
router = APIRouter()

# 현재 학습 상태
_current_trainer: Optional[object] = None
_training_thread: Optional[threading.Thread] = None
_ws_connections: list = []


class TrainingRequest(BaseModel):
    config: dict
    vram_profile: Optional[str] = None


@router.post("/start")
async def start_training(request: TrainingRequest):
    """학습 시작."""
    global _current_trainer, _training_thread

    if _current_trainer and _current_trainer.state.is_training:
        return {"error": "학습이 이미 진행 중입니다."}

    # 설정 생성
    config = dict_to_dataclass(TrainingConfig, request.config)

    # VRAM 프리셋 적용
    if request.vram_profile:
        config = apply_profile_to_config(config, request.vram_profile)

    # 검증
    warnings = config.validate()
    if warnings:
        logger.warning(f"설정 경고: {warnings}")

    # 트레이너 선택
    if config.training_mode == "lora":
        trainer = LoRATrainer(config)
    else:
        trainer = FullFinetuneTrainer(config)

    _current_trainer = trainer

    # WebSocket 콜백 등록
    def on_step_end(state):
        _broadcast_ws({"type": "step", "data": state.to_dict()})

    def on_log(message):
        _broadcast_ws({"type": "log", "data": {"message": message}})

    def on_error(error):
        _broadcast_ws({"type": "error", "data": {"message": error}})

    def on_save(state, path):
        _broadcast_ws({"type": "save", "data": {"step": state.current_step, "path": path}})

    def on_complete(state):
        _broadcast_ws({"type": "complete", "data": state.to_dict()})

    trainer.callbacks.on_step_end(on_step_end)
    trainer.callbacks.on_log(on_log)
    trainer.callbacks.on_error(on_error)
    trainer.callbacks.on_save(on_save)
    trainer.callbacks.on_complete(on_complete)

    # 백그라운드 스레드에서 학습 실행
    _training_thread = threading.Thread(
        target=trainer.full_setup_and_train,
        daemon=True,
    )
    _training_thread.start()

    return {
        "status": "started",
        "mode": config.training_mode,
        "total_steps": config.max_train_steps,
        "warnings": warnings,
    }


@router.post("/stop")
async def stop_training():
    """학습 중지."""
    if _current_trainer and _current_trainer.state.is_training:
        _current_trainer.stop()
        return {"status": "stopping"}
    return {"status": "not_training"}


@router.post("/pause")
async def pause_training():
    """학습 일시정지."""
    if _current_trainer and _current_trainer.state.is_training:
        _current_trainer.pause()
        return {"status": "paused"}
    return {"status": "not_training"}


@router.post("/resume")
async def resume_training():
    """학습 재개."""
    if _current_trainer and _current_trainer.state.is_paused:
        _current_trainer.resume()
        return {"status": "resumed"}
    return {"status": "not_paused"}


@router.get("/status")
async def training_status():
    """현재 학습 상태."""
    if _current_trainer:
        return _current_trainer.state.to_dict()
    return {"is_training": False}


@router.get("/config/default")
async def default_config():
    """기본 학습 설정."""
    config = TrainingConfig()
    return dataclass_to_dict(config)


# WebSocket 엔드포인트
@router.websocket("/ws")
async def training_websocket(websocket: WebSocket):
    """실시간 학습 업데이트 WebSocket."""
    await websocket.accept()
    _ws_connections.append(websocket)
    logger.info(f"WebSocket 연결: {len(_ws_connections)}개")

    try:
        # 현재 상태 즉시 전송
        if _current_trainer:
            await websocket.send_json({
                "type": "status",
                "data": _current_trainer.state.to_dict(),
            })

        # 연결 유지
        while True:
            data = await websocket.receive_text()
            # 클라이언트에서 ping 등 처리
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in _ws_connections:
            _ws_connections.remove(websocket)
        logger.info(f"WebSocket 연결 해제: {len(_ws_connections)}개 남음")


def _broadcast_ws(message: dict):
    """모든 WebSocket 클라이언트에 메시지 전송."""
    disconnected = []
    for ws in _ws_connections:
        try:
            asyncio.get_event_loop().call_soon_threadsafe(
                asyncio.ensure_future, ws.send_json(message)
            )
        except Exception:
            disconnected.append(ws)

    for ws in disconnected:
        _ws_connections.remove(ws)
