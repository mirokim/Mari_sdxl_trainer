'use client';
import { useEffect, useState } from 'react';
import Header from '@/components/layout/Header';
import PageContainer from '@/components/layout/PageContainer';
import { systemApi } from '@/lib/api';
import { GpuInfo } from '@/lib/types';

export default function DashboardPage() {
  const [gpu, setGpu] = useState<GpuInfo | null>(null);

  useEffect(() => {
    systemApi.getGpuInfo().then(setGpu).catch(() => {});
  }, []);

  return (
    <PageContainer>
      <Header title="Mari SDXL Trainer" description="ComfyUI 호환 SDXL LoRA & 체크포인트 학습" />

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
        {/* GPU 상태 카드 */}
        <div className="notion-card">
          <h3 className="text-notion-small text-notion-text-secondary font-medium mb-3">
            GPU 상태
          </h3>
          {gpu?.available ? (
            <div className="space-y-2">
              <p className="text-notion-body font-medium">{gpu.name}</p>
              <div className="flex justify-between text-notion-small text-notion-text-secondary">
                <span>VRAM</span>
                <span>{gpu.free_gb}GB / {gpu.total_gb}GB</span>
              </div>
              <div className="w-full h-2 bg-notion-hover rounded-full overflow-hidden">
                <div
                  className="h-full bg-notion-accent rounded-full transition-all"
                  style={{
                    width: `${gpu.total_gb ? ((gpu.total_gb - (gpu.free_gb || 0)) / gpu.total_gb) * 100 : 0}%`,
                  }}
                />
              </div>
              <p className="text-[11px] text-notion-text-secondary">
                CUDA {gpu.cuda_version} · 추천 프리셋: {gpu.suggested_profile}
              </p>
            </div>
          ) : (
            <p className="text-notion-body text-notion-error">GPU를 찾을 수 없습니다</p>
          )}
        </div>

        {/* 빠른 시작 카드 */}
        <div className="notion-card">
          <h3 className="text-notion-small text-notion-text-secondary font-medium mb-3">
            빠른 시작
          </h3>
          <div className="space-y-2">
            <a href="/dataset" className="block p-2 rounded-md hover:bg-notion-hover transition-colors">
              <p className="text-notion-body font-medium">1. 데이터셋 준비</p>
              <p className="text-notion-small text-notion-text-secondary">이미지 업로드 & 캡션 작성</p>
            </a>
            <a href="/training" className="block p-2 rounded-md hover:bg-notion-hover transition-colors">
              <p className="text-notion-body font-medium">2. 학습 시작</p>
              <p className="text-notion-small text-notion-text-secondary">LoRA 또는 풀 파인튜닝</p>
            </a>
            <a href="/output" className="block p-2 rounded-md hover:bg-notion-hover transition-colors">
              <p className="text-notion-body font-medium">3. 결과 확인</p>
              <p className="text-notion-small text-notion-text-secondary">.safetensors → ComfyUI</p>
            </a>
          </div>
        </div>

        {/* 지원 모드 카드 */}
        <div className="notion-card">
          <h3 className="text-notion-small text-notion-text-secondary font-medium mb-3">
            지원 학습 모드
          </h3>
          <div className="space-y-2">
            <div className="flex items-center gap-2 p-2 rounded-md bg-notion-hover">
              <span className="w-2 h-2 rounded-full bg-notion-success" />
              <span className="text-notion-body">LoRA (8-24GB VRAM)</span>
            </div>
            <div className="flex items-center gap-2 p-2 rounded-md bg-notion-hover">
              <span className="w-2 h-2 rounded-full bg-notion-accent" />
              <span className="text-notion-body">Full Fine-tune (24GB+ VRAM)</span>
            </div>
            <div className="flex items-center gap-2 p-2 rounded-md bg-notion-hover">
              <span className="w-2 h-2 rounded-full bg-notion-warning" />
              <span className="text-notion-body">Auto-Captioning (Florence-2 / BLIP-2)</span>
            </div>
          </div>
        </div>
      </div>
    </PageContainer>
  );
}
