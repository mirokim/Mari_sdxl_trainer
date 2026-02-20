import { create } from 'zustand';
import { TrainingState, TrainingConfig } from '@/lib/types';

interface TrainingStore {
  // 학습 상태
  state: TrainingState | null;
  setState: (state: TrainingState) => void;

  // 로그
  logs: string[];
  addLog: (msg: string) => void;
  clearLogs: () => void;

  // 설정
  config: Partial<TrainingConfig> | null;
  setConfig: (config: Partial<TrainingConfig>) => void;

  // 선택된 데이터셋
  selectedDataset: string;
  setSelectedDataset: (name: string) => void;
}

export const useTrainingStore = create<TrainingStore>((set) => ({
  state: null,
  setState: (state) => set({ state }),

  logs: [],
  addLog: (msg) => set((s) => ({ logs: [...s.logs.slice(-500), msg] })),
  clearLogs: () => set({ logs: [] }),

  config: null,
  setConfig: (config) => set({ config }),

  selectedDataset: '',
  setSelectedDataset: (name) => set({ selectedDataset: name }),
}));
