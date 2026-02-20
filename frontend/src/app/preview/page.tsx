'use client';
import Header from '@/components/layout/Header';
import PageContainer from '@/components/layout/PageContainer';
import { useTrainingStore } from '@/stores/trainingStore';

export default function PreviewPage() {
  const { state } = useTrainingStore();

  return (
    <PageContainer>
      <Header title="미리보기" description="학습 중 생성된 샘플 이미지" />

      {state?.sample_images && state.sample_images.length > 0 ? (
        <div className="space-y-6">
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {state.sample_images.map((img, i) => (
              <div key={i} className="notion-card p-2">
                <div className="aspect-square bg-notion-hover rounded overflow-hidden">
                  <img
                    src={img}
                    alt={`Sample ${i + 1}`}
                    className="w-full h-full object-cover"
                  />
                </div>
                <p className="text-[11px] text-notion-text-secondary mt-1 text-center">
                  Sample #{i + 1}
                </p>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center h-64 text-notion-text-secondary">
          <p className="text-notion-body">샘플 이미지가 아직 없습니다</p>
          <p className="text-notion-small mt-1">
            학습 설정에서 sample_every_n_steps를 설정하면 학습 중 자동으로 샘플이 생성됩니다
          </p>
        </div>
      )}

      {/* 학습 상태 미리보기 */}
      {state?.is_training && (
        <div className="mt-8 notion-card">
          <h3 className="text-notion-body font-medium mb-2">현재 학습 상태</h3>
          <div className="grid grid-cols-4 gap-4 text-notion-small">
            <div>
              <span className="text-notion-text-secondary">Step</span>
              <p className="font-mono">{state.current_step} / {state.total_steps}</p>
            </div>
            <div>
              <span className="text-notion-text-secondary">Loss</span>
              <p className="font-mono">{state.current_loss.toFixed(6)}</p>
            </div>
            <div>
              <span className="text-notion-text-secondary">속도</span>
              <p className="font-mono">{state.steps_per_second.toFixed(2)} it/s</p>
            </div>
            <div>
              <span className="text-notion-text-secondary">진행률</span>
              <p className="font-mono">{state.progress_percent.toFixed(1)}%</p>
            </div>
          </div>
        </div>
      )}
    </PageContainer>
  );
}
