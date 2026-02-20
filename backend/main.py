"""Mari SDXL Trainer - FastAPI 백엔드 서버."""
import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from mari_trainer.api import dataset, captioning, training, output, system

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="Mari SDXL Trainer",
    description="SDXL LoRA & Checkpoint Trainer for ComfyUI",
    version="1.0.0",
)

# CORS (프론트엔드 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 라우터 등록
app.include_router(system.router, prefix="/api/system", tags=["system"])
app.include_router(dataset.router, prefix="/api/dataset", tags=["dataset"])
app.include_router(captioning.router, prefix="/api/captioning", tags=["captioning"])
app.include_router(training.router, prefix="/api/training", tags=["training"])
app.include_router(output.router, prefix="/api/output", tags=["output"])

# 출력 디렉토리 정적 파일 서빙 (학습 결과, 샘플 이미지)
outputs_dir = Path("../outputs")
outputs_dir.mkdir(exist_ok=True)
app.mount("/files/outputs", StaticFiles(directory=str(outputs_dir)), name="outputs")

# 데이터셋 이미지 서빙
datasets_dir = Path("../datasets")
datasets_dir.mkdir(exist_ok=True)
app.mount("/files/datasets", StaticFiles(directory=str(datasets_dir)), name="datasets")


@app.get("/")
async def root():
    return {"message": "Mari SDXL Trainer API", "version": "1.0.0"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
