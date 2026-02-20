'use client';
import { useEffect, useState } from 'react';
import Header from '@/components/layout/Header';
import PageContainer from '@/components/layout/PageContainer';
import { trainingApi, systemApi, datasetApi } from '@/lib/api';
import { useWebSocket } from '@/hooks/useWebSocket';
import { useTrainingStore } from '@/stores/trainingStore';
import { TrainingConfig, DatasetInfo, VramProfile } from '@/lib/types';

export default function TrainingPage() {
  const { state, setState, logs, addLog } = useTrainingStore();
  const { lastMessage, isConnected } = useWebSocket();
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [profiles, setProfiles] = useState<Record<string, VramProfile>>({});
  const [selectedProfile, setSelectedProfile] = useState('');
  const [config, setConfig] = useState<Partial<TrainingConfig>>({
    pretrained_model_name_or_path: 'stabilityai/stable-diffusion-xl-base-1.0',
    training_mode: 'lora',
    lora_rank: 32,
    lora_alpha: 16,
    optimizer_type: 'AdamW8bit',
    learning_rate: 1e-4,
    lr_scheduler: 'cosine',
    lr_warmup_steps: 100,
    max_train_steps: 1500,
    train_batch_size: 1,
    gradient_accumulation_steps: 4,
    save_every_n_steps: 500,
    sample_every_n_steps: 250,
    mixed_precision: 'bf16',
    gradient_checkpointing: true,
    enable_xformers: true,
    cache_latents: true,
    cache_text_encoder_outputs: true,
    train_text_encoder: false,
    resolution: 1024,
    enable_bucketing: true,
    save_kohya_format: true,
    output_dir: './outputs',
    run_name: '',
    dataset_path: '',
  });

  // WebSocket 메시지 처리
  useEffect(() => {
    if (!lastMessage) return;
    switch (lastMessage.type) {
      case 'step':
      case 'status':
        setState(lastMessage.data);
        break;
      case 'log':
        addLog(lastMessage.data.message);
        break;
      case 'error':
        addLog(`[ERROR] ${lastMessage.data.message}`);
        break;
      case 'complete':
        setState(lastMessage.data);
        addLog('학습 완료!');
        break;
    }
  }, [lastMessage, setState, addLog]);

  useEffect(() => {
    datasetApi.list().then((r) => setDatasets(r.datasets)).catch(() => {});
    systemApi.getVramProfiles().then((r) => setProfiles(r.profiles)).catch(() => {});
  }, []);

  const updateConfig = (key: string, value: any) => {
    setConfig((prev) => ({ ...prev, [key]: value }));
  };

  const applyProfile = (profileName: string) => {
    const profile = profiles[profileName];
    if (!profile) return;
    setSelectedProfile(profileName);
    const profileConfig = { ...profile } as any;
    delete profileConfig.description;
    setConfig((prev) => ({ ...prev, ...profileConfig }));
  };

  const handleStart = async () => {
    try {
      const res = await trainingApi.start(config, selectedProfile || undefined);
      if (res.error) {
        addLog(`[ERROR] ${res.error}`);
      } else {
        addLog(`학습 시작: ${res.mode} 모드, ${res.total_steps} 스텝`);
      }
    } catch (e: any) {
      addLog(`[ERROR] ${e.message}`);
    }
  };

  const isTraining = state?.is_training || false;

  return (
    <PageContainer>
      <Header title="학습" description="SDXL LoRA / 풀 파인튜닝 학습 설정 및 실행" />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 좌측: 설정 */}
        <div className="lg:col-span-2 space-y-6">
          {/* 모델 & 데이터셋 */}
          <section className="notion-card">
            <h2 className="notion-section-title">모델 & 데이터셋</h2>
            <div className="space-y-4">
              <div>
                <label className="notion-label">베이스 모델</label>
                <input
                  type="text"
                  value={config.pretrained_model_name_or_path}
                  onChange={(e) => updateConfig('pretrained_model_name_or_path', e.target.value)}
                  className="notion-input"
                  disabled={isTraining}
                />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="notion-label">데이터셋</label>
                  <select
                    value={config.dataset_path}
                    onChange={(e) => updateConfig('dataset_path', e.target.value)}
                    className="notion-select"
                    disabled={isTraining}
                  >
                    <option value="">선택하세요</option>
                    {datasets.map((ds) => (
                      <option key={ds.name} value={ds.path}>{ds.name} ({ds.image_count}개)</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="notion-label">실행 이름</label>
                  <input
                    type="text"
                    value={config.run_name}
                    onChange={(e) => updateConfig('run_name', e.target.value)}
                    placeholder="my_lora"
                    className="notion-input"
                    disabled={isTraining}
                  />
                </div>
              </div>
              <div>
                <label className="notion-label">학습 모드</label>
                <div className="flex gap-2">
                  {(['lora', 'full'] as const).map((mode) => (
                    <button
                      key={mode}
                      onClick={() => updateConfig('training_mode', mode)}
                      disabled={isTraining}
                      className={`notion-btn ${config.training_mode === mode
                        ? 'bg-notion-accent text-white'
                        : 'bg-notion-hover text-notion-text'
                      }`}
                    >
                      {mode === 'lora' ? 'LoRA' : 'Full Fine-tune'}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </section>

          {/* VRAM 프리셋 */}
          <section className="notion-card">
            <h2 className="notion-section-title">VRAM 프리셋</h2>
            <div className="flex gap-2 flex-wrap">
              {Object.entries(profiles).map(([name, profile]) => (
                <button
                  key={name}
                  onClick={() => applyProfile(name)}
                  disabled={isTraining}
                  className={`notion-btn ${selectedProfile === name
                    ? 'bg-notion-accent text-white'
                    : 'bg-notion-hover text-notion-text'
                  }`}
                >
                  {name.toUpperCase()}
                  <span className="text-[10px] opacity-75 ml-1">
                    ({(profile as any).description?.match(/\(([^)]+)\)/)?.[1] || ''})
                  </span>
                </button>
              ))}
            </div>
          </section>

          {/* LoRA 설정 */}
          {config.training_mode === 'lora' && (
            <section className="notion-card">
              <h2 className="notion-section-title">LoRA 설정</h2>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="notion-label">Rank (차원): {config.lora_rank}</label>
                  <input
                    type="range" min="4" max="128" step="4"
                    value={config.lora_rank}
                    onChange={(e) => updateConfig('lora_rank', Number(e.target.value))}
                    className="w-full" disabled={isTraining}
                  />
                </div>
                <div>
                  <label className="notion-label">Alpha: {config.lora_alpha}</label>
                  <input
                    type="range" min="1" max="128" step="1"
                    value={config.lora_alpha}
                    onChange={(e) => updateConfig('lora_alpha', Number(e.target.value))}
                    className="w-full" disabled={isTraining}
                  />
                </div>
              </div>
            </section>
          )}

          {/* 옵티마이저 & 스케줄 */}
          <section className="notion-card">
            <h2 className="notion-section-title">옵티마이저 & 스케줄</h2>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="notion-label">옵티마이저</label>
                <select
                  value={config.optimizer_type}
                  onChange={(e) => updateConfig('optimizer_type', e.target.value)}
                  className="notion-select" disabled={isTraining}
                >
                  <option value="AdamW8bit">AdamW8bit (권장)</option>
                  <option value="Prodigy">Prodigy (자동 LR)</option>
                  <option value="AdamW">AdamW</option>
                  <option value="AdaFactor">AdaFactor</option>
                </select>
              </div>
              <div>
                <label className="notion-label">학습률</label>
                <input
                  type="number" step="0.00001"
                  value={config.learning_rate}
                  onChange={(e) => updateConfig('learning_rate', Number(e.target.value))}
                  className="notion-input" disabled={isTraining}
                />
              </div>
              <div>
                <label className="notion-label">총 스텝: {config.max_train_steps}</label>
                <input
                  type="range" min="100" max="10000" step="100"
                  value={config.max_train_steps}
                  onChange={(e) => updateConfig('max_train_steps', Number(e.target.value))}
                  className="w-full" disabled={isTraining}
                />
              </div>
              <div>
                <label className="notion-label">배치 크기</label>
                <select
                  value={config.train_batch_size}
                  onChange={(e) => updateConfig('train_batch_size', Number(e.target.value))}
                  className="notion-select" disabled={isTraining}
                >
                  {[1, 2, 4, 8].map((v) => <option key={v} value={v}>{v}</option>)}
                </select>
              </div>
            </div>
          </section>

          {/* 메모리 최적화 */}
          <section className="notion-card">
            <h2 className="notion-section-title">메모리 최적화</h2>
            <div className="grid grid-cols-2 gap-3">
              {[
                { key: 'gradient_checkpointing', label: 'Gradient Checkpointing' },
                { key: 'enable_xformers', label: 'xformers' },
                { key: 'cache_latents', label: '잠재벡터 캐싱' },
                { key: 'cache_text_encoder_outputs', label: '텍스트 임베딩 캐싱' },
                { key: 'train_text_encoder', label: '텍스트 인코더 학습' },
                { key: 'save_kohya_format', label: 'Kohya 포맷 (ComfyUI)' },
              ].map(({ key, label }) => (
                <label key={key} className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={!!(config as any)[key]}
                    onChange={(e) => updateConfig(key, e.target.checked)}
                    className="w-4 h-4 rounded border-notion-border"
                    disabled={isTraining}
                  />
                  <span className="text-notion-small">{label}</span>
                </label>
              ))}
            </div>
          </section>

          {/* 시작/중지 버튼 */}
          <div className="flex gap-3">
            {!isTraining ? (
              <button onClick={handleStart} className="notion-btn-primary text-base px-8 py-2.5">
                학습 시작
              </button>
            ) : (
              <>
                <button onClick={() => trainingApi.stop()} className="notion-btn-danger text-base px-6 py-2.5">
                  중지
                </button>
                <button
                  onClick={() => state?.is_paused ? trainingApi.resume() : trainingApi.pause()}
                  className="notion-btn-secondary text-base px-6 py-2.5"
                >
                  {state?.is_paused ? '재개' : '일시정지'}
                </button>
              </>
            )}
          </div>
        </div>

        {/* 우측: 학습 모니터 */}
        <div className="space-y-4">
          {/* 연결 상태 */}
          <div className="notion-card">
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-notion-success' : 'bg-notion-error'}`} />
              <span className="text-notion-small text-notion-text-secondary">
                {isConnected ? '실시간 연결' : '연결 대기 중...'}
              </span>
            </div>
          </div>

          {/* 진행 상태 */}
          {state && (
            <div className="notion-card">
              <h3 className="text-notion-small font-medium mb-3">학습 진행</h3>
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between text-notion-small mb-1">
                    <span>Step {state.current_step} / {state.total_steps}</span>
                    <span>{state.progress_percent.toFixed(1)}%</span>
                  </div>
                  <div className="w-full h-2 bg-notion-hover rounded-full overflow-hidden">
                    <div
                      className="h-full bg-notion-accent rounded-full transition-all"
                      style={{ width: `${state.progress_percent}%` }}
                    />
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-2 text-notion-small">
                  <div>
                    <span className="text-notion-text-secondary">Loss</span>
                    <p className="font-mono font-medium">{state.current_loss.toFixed(6)}</p>
                  </div>
                  <div>
                    <span className="text-notion-text-secondary">속도</span>
                    <p className="font-mono font-medium">{state.steps_per_second.toFixed(2)} it/s</p>
                  </div>
                  <div>
                    <span className="text-notion-text-secondary">경과</span>
                    <p className="font-mono font-medium">{formatTime(state.elapsed_seconds)}</p>
                  </div>
                  <div>
                    <span className="text-notion-text-secondary">남은 시간</span>
                    <p className="font-mono font-medium">{formatTime(state.eta_seconds)}</p>
                  </div>
                </div>
                {state.error && (
                  <p className="text-notion-small text-notion-error">{state.error}</p>
                )}
              </div>
            </div>
          )}

          {/* 손실 그래프 (간단한 텍스트 기반) */}
          {state && state.loss_history.length > 0 && (
            <div className="notion-card">
              <h3 className="text-notion-small font-medium mb-3">Loss 추이</h3>
              <div className="h-32 flex items-end gap-px">
                {state.loss_history.slice(-50).map((loss, i) => {
                  const maxLoss = Math.max(...state.loss_history.slice(-50));
                  const height = maxLoss > 0 ? (loss / maxLoss) * 100 : 0;
                  return (
                    <div
                      key={i}
                      className="flex-1 bg-notion-accent/60 rounded-t-sm min-h-[2px]"
                      style={{ height: `${height}%` }}
                    />
                  );
                })}
              </div>
            </div>
          )}

          {/* 로그 */}
          <div className="notion-card">
            <h3 className="text-notion-small font-medium mb-3">로그</h3>
            <div className="h-48 overflow-y-auto font-mono text-[11px] text-notion-text-secondary
                          bg-notion-sidebar rounded p-2 space-y-0.5">
              {logs.slice(-50).map((log, i) => (
                <div key={i} className={log.includes('[ERROR]') ? 'text-notion-error' : ''}>
                  {log}
                </div>
              ))}
              {logs.length === 0 && <span className="text-notion-text-secondary/50">로그 대기 중...</span>}
            </div>
          </div>
        </div>
      </div>
    </PageContainer>
  );
}

function formatTime(seconds: number): string {
  if (seconds <= 0) return '--:--';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  if (m >= 60) {
    const h = Math.floor(m / 60);
    return `${h}h ${m % 60}m`;
  }
  return `${m}:${s.toString().padStart(2, '0')}`;
}
