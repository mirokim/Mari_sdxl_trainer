'use client';
import { useEffect, useState } from 'react';
import Header from '@/components/layout/Header';
import PageContainer from '@/components/layout/PageContainer';
import { captioningApi, datasetApi } from '@/lib/api';
import { CaptioningState, DatasetInfo } from '@/lib/types';

export default function CaptioningPage() {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [selectedDataset, setSelectedDataset] = useState('');
  const [model, setModel] = useState('florence-2-large');
  const [captionMode, setCaptionMode] = useState('<MORE_DETAILED_CAPTION>');
  const [triggerWord, setTriggerWord] = useState('');
  const [triggerPosition, setTriggerPosition] = useState('prefix');
  const [overwrite, setOverwrite] = useState(false);
  const [status, setStatus] = useState<CaptioningState | null>(null);

  useEffect(() => {
    datasetApi.list().then((r) => setDatasets(r.datasets)).catch(() => {});
  }, []);

  useEffect(() => {
    if (!status?.is_running) return;
    const interval = setInterval(async () => {
      const s = await captioningApi.getStatus();
      setStatus(s);
      if (!s.is_running) clearInterval(interval);
    }, 1000);
    return () => clearInterval(interval);
  }, [status?.is_running]);

  const handleStart = async () => {
    if (!selectedDataset) return;
    await captioningApi.run({
      dataset_name: selectedDataset,
      model,
      caption_mode: captionMode,
      trigger_word: triggerWord,
      trigger_position: triggerPosition,
      overwrite,
    });
    setStatus({ is_running: true, current: 0, total: 0, current_file: '', error: null });
  };

  return (
    <PageContainer>
      <Header title="자동 캡셔닝" description="AI로 이미지 설명을 자동 생성합니다" />

      <div className="max-w-2xl space-y-6">
        {/* 데이터셋 선택 */}
        <div className="notion-section">
          <label className="notion-label">데이터셋</label>
          <select
            value={selectedDataset}
            onChange={(e) => setSelectedDataset(e.target.value)}
            className="notion-select"
          >
            <option value="">선택하세요</option>
            {datasets.map((ds) => (
              <option key={ds.name} value={ds.name}>
                {ds.name} ({ds.image_count}개 이미지)
              </option>
            ))}
          </select>
        </div>

        {/* 모델 선택 */}
        <div className="notion-section">
          <label className="notion-label">캡셔닝 모델</label>
          <select value={model} onChange={(e) => setModel(e.target.value)} className="notion-select">
            <option value="florence-2-large">Florence-2 Large (~1.5GB, 권장)</option>
            <option value="florence-2-base">Florence-2 Base (~500MB, 빠름)</option>
            <option value="blip2">BLIP-2 OPT-2.7B (~5GB)</option>
          </select>
        </div>

        {/* 캡션 모드 */}
        <div className="notion-section">
          <label className="notion-label">캡션 상세도</label>
          <select
            value={captionMode}
            onChange={(e) => setCaptionMode(e.target.value)}
            className="notion-select"
          >
            <option value="<CAPTION>">간단한 캡션</option>
            <option value="<DETAILED_CAPTION>">상세 캡션</option>
            <option value="<MORE_DETAILED_CAPTION>">매우 상세한 캡션 (권장)</option>
          </select>
        </div>

        {/* 트리거 워드 */}
        <div className="notion-section">
          <label className="notion-label">트리거 워드 (선택)</label>
          <div className="flex gap-2">
            <input
              type="text"
              value={triggerWord}
              onChange={(e) => setTriggerWord(e.target.value)}
              placeholder="예: sks person"
              className="notion-input flex-1"
            />
            <select
              value={triggerPosition}
              onChange={(e) => setTriggerPosition(e.target.value)}
              className="notion-select w-24"
            >
              <option value="prefix">앞에</option>
              <option value="suffix">뒤에</option>
            </select>
          </div>
        </div>

        {/* 덮어쓰기 옵션 */}
        <div className="notion-section">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={overwrite}
              onChange={(e) => setOverwrite(e.target.checked)}
              className="w-4 h-4 rounded border-notion-border"
            />
            <span className="text-notion-body">기존 캡션 덮어쓰기</span>
          </label>
        </div>

        {/* 시작 버튼 */}
        <button
          onClick={handleStart}
          disabled={!selectedDataset || status?.is_running}
          className="notion-btn-primary text-base px-6 py-2"
        >
          {status?.is_running ? '캡셔닝 진행 중...' : '캡셔닝 시작'}
        </button>

        {/* 진행 상태 */}
        {status?.is_running && (
          <div className="notion-card">
            <div className="flex justify-between text-notion-small mb-2">
              <span>진행: {status.current} / {status.total}</span>
              <span>{status.current_file}</span>
            </div>
            <div className="w-full h-2 bg-notion-hover rounded-full overflow-hidden">
              <div
                className="h-full bg-notion-accent rounded-full transition-all"
                style={{
                  width: `${status.total ? (status.current / status.total) * 100 : 0}%`,
                }}
              />
            </div>
          </div>
        )}

        {status?.error && (
          <div className="p-3 bg-red-50 border border-notion-error rounded-md text-notion-small text-notion-error">
            오류: {status.error}
          </div>
        )}
      </div>
    </PageContainer>
  );
}
