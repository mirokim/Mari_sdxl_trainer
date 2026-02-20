const API_BASE = '/api';

async function fetchApi<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${url}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || `API error: ${res.status}`);
  }
  return res.json();
}

// System
export const systemApi = {
  getGpuInfo: () => fetchApi<any>('/system/gpu'),
  getMemory: () => fetchApi<any>('/system/memory'),
  getVramProfiles: () => fetchApi<any>('/system/vram-profiles'),
};

// Dataset
export const datasetApi = {
  list: () => fetchApi<any>('/dataset/list'),
  create: (name: string) => {
    const form = new FormData();
    form.append('name', name);
    return fetch(`${API_BASE}/dataset/create`, { method: 'POST', body: form }).then(r => r.json());
  },
  getImages: (name: string) => fetchApi<any>(`/dataset/${name}/images`),
  getStats: (name: string) => fetchApi<any>(`/dataset/${name}/stats`),
  uploadImages: (name: string, files: File[], subfolder = '') => {
    const form = new FormData();
    form.append('subfolder', subfolder);
    files.forEach(f => form.append('files', f));
    return fetch(`${API_BASE}/dataset/${name}/upload`, { method: 'POST', body: form }).then(r => r.json());
  },
  updateCaption: (name: string, imagePath: string, caption: string) => {
    const form = new FormData();
    form.append('image_path', imagePath);
    form.append('caption', caption);
    return fetch(`${API_BASE}/dataset/${name}/caption`, { method: 'POST', body: form }).then(r => r.json());
  },
};

// Captioning
export const captioningApi = {
  run: (params: any) => fetchApi<any>('/captioning/run', {
    method: 'POST',
    body: JSON.stringify(params),
  }),
  getStatus: () => fetchApi<any>('/captioning/status'),
  getModels: () => fetchApi<any>('/captioning/models'),
};

// Training
export const trainingApi = {
  start: (config: any, vramProfile?: string) => fetchApi<any>('/training/start', {
    method: 'POST',
    body: JSON.stringify({ config, vram_profile: vramProfile }),
  }),
  stop: () => fetchApi<any>('/training/stop', { method: 'POST' }),
  pause: () => fetchApi<any>('/training/pause', { method: 'POST' }),
  resume: () => fetchApi<any>('/training/resume', { method: 'POST' }),
  getStatus: () => fetchApi<any>('/training/status'),
  getDefaultConfig: () => fetchApi<any>('/training/config/default'),
};

// Output
export const outputApi = {
  listModels: () => fetchApi<any>('/output/models'),
  deleteModel: (path: string) => fetchApi<any>(`/output/model?path=${encodeURIComponent(path)}`, {
    method: 'DELETE',
  }),
};
