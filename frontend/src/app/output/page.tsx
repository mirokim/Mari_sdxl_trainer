'use client';
import { useEffect, useState } from 'react';
import Header from '@/components/layout/Header';
import PageContainer from '@/components/layout/PageContainer';
import { outputApi } from '@/lib/api';
import { ModelOutput } from '@/lib/types';

export default function OutputPage() {
  const [models, setModels] = useState<ModelOutput[]>([]);

  const loadModels = async () => {
    try {
      const res = await outputApi.listModels();
      setModels(res.models);
    } catch {}
  };

  useEffect(() => { loadModels(); }, []);

  return (
    <PageContainer>
      <Header title="결과물" description="학습된 모델 (.safetensors) 관리" />

      {models.length === 0 ? (
        <div className="flex flex-col items-center justify-center h-64 text-notion-text-secondary">
          <p className="text-notion-body">아직 학습된 모델이 없습니다</p>
          <a href="/training" className="mt-2 text-notion-accent hover:underline text-notion-small">
            학습 시작하기 →
          </a>
        </div>
      ) : (
        <div className="space-y-4">
          {models.map((model) => (
            <div key={model.name} className="notion-card">
              <div className="flex items-start justify-between mb-3">
                <div>
                  <h3 className="text-notion-body font-semibold">{model.name}</h3>
                  <p className="text-notion-small text-notion-text-secondary">{model.path}</p>
                </div>
              </div>

              <div className="space-y-2">
                {model.checkpoints.map((cp) => (
                  <div
                    key={cp.path}
                    className="flex items-center justify-between p-3 bg-notion-sidebar rounded-md"
                  >
                    <div className="flex items-center gap-3">
                      <span className="text-lg">◆</span>
                      <div>
                        <p className="text-notion-small font-medium">{cp.filename}</p>
                        <p className="text-[11px] text-notion-text-secondary">
                          {cp.size_mb} MB {cp.step && `· ${cp.step}`}
                        </p>
                      </div>
                    </div>
                    <div className="flex gap-2">
                      <a
                        href={`/api/output/download?path=${encodeURIComponent(cp.path)}`}
                        className="notion-btn-secondary text-notion-small"
                      >
                        다운로드
                      </a>
                      <button
                        onClick={async () => {
                          await outputApi.deleteModel(cp.path);
                          loadModels();
                        }}
                        className="notion-btn-ghost text-notion-small text-notion-error"
                      >
                        삭제
                      </button>
                    </div>
                  </div>
                ))}
              </div>

              {/* ComfyUI 안내 */}
              <div className="mt-3 p-3 bg-blue-50 rounded-md">
                <p className="text-notion-small text-notion-accent">
                  ComfyUI에서 사용하려면 .safetensors 파일을
                  <code className="mx-1 px-1 py-0.5 bg-white rounded text-[11px]">
                    ComfyUI/models/Lora/
                  </code>
                  (LoRA) 또는
                  <code className="mx-1 px-1 py-0.5 bg-white rounded text-[11px]">
                    ComfyUI/models/checkpoints/
                  </code>
                  (풀 체크포인트)에 복사하세요.
                </p>
              </div>
            </div>
          ))}
        </div>
      )}
    </PageContainer>
  );
}
