// GPU 정보
export interface GpuInfo {
  available: boolean;
  name?: string;
  total_gb?: number;
  allocated_gb?: number;
  free_gb?: number;
  cuda_version?: string;
  suggested_profile?: string;
}

// 학습 설정
export interface TrainingConfig {
  pretrained_model_name_or_path: string;
  vae_path: string | null;
  use_fp16_vae_fix: boolean;
  training_mode: 'lora' | 'full';
  lora_rank: number;
  lora_alpha: number;
  lora_target_modules: string[];
  train_text_encoder: boolean;
  optimizer_type: string;
  learning_rate: number;
  text_encoder_lr: number;
  weight_decay: number;
  lr_scheduler: string;
  lr_warmup_steps: number;
  max_train_steps: number;
  train_batch_size: number;
  gradient_accumulation_steps: number;
  save_every_n_steps: number;
  sample_every_n_steps: number;
  max_grad_norm: number;
  mixed_precision: string;
  gradient_checkpointing: boolean;
  enable_xformers: boolean;
  cache_latents: boolean;
  cache_text_encoder_outputs: boolean;
  dataset_path: string;
  resolution: number;
  enable_bucketing: boolean;
  random_flip: boolean;
  caption_dropout_rate: number;
  noise_offset: number;
  min_snr_gamma: number;
  output_dir: string;
  run_name: string;
  save_kohya_format: boolean;
  sample_prompts: string[];
  sample_negative_prompt: string;
  sample_steps: number;
  sample_cfg_scale: number;
  sample_seed: number;
}

// 학습 상태
export interface TrainingState {
  current_step: number;
  total_steps: number;
  current_loss: number;
  progress_percent: number;
  steps_per_second: number;
  eta_seconds: number;
  elapsed_seconds: number;
  is_training: boolean;
  is_paused: boolean;
  error: string | null;
  loss_history: number[];
  lr_history: number[];
  sample_images: string[];
}

// 데이터셋
export interface DatasetImage {
  path: string;
  filename: string;
  width: number;
  height: number;
  caption: string;
  has_caption: boolean;
  relative_path: string;
}

export interface DatasetInfo {
  name: string;
  path: string;
  image_count: number;
  subfolders: SubfolderInfo[];
}

export interface SubfolderInfo {
  path: string;
  name: string;
  repeats: number;
  concept: string;
  image_count: number;
  captioned_count: number;
}

// 캡셔닝
export interface CaptioningState {
  is_running: boolean;
  current: number;
  total: number;
  current_file: string;
  error: string | null;
}

// 모델 출력
export interface ModelOutput {
  name: string;
  path: string;
  checkpoints: ModelCheckpoint[];
}

export interface ModelCheckpoint {
  filename: string;
  path: string;
  size_mb: number;
  step?: string;
}

// WebSocket 메시지
export interface WSMessage {
  type: 'step' | 'log' | 'error' | 'save' | 'complete' | 'status';
  data: any;
}

// VRAM 프리셋
export interface VramProfile {
  description: string;
  training_mode: string;
  mixed_precision: string;
  gradient_checkpointing: boolean;
  cache_latents: boolean;
  cache_text_encoder_outputs: boolean;
  train_batch_size: number;
  gradient_accumulation_steps: number;
  enable_xformers: boolean;
  lora_rank: number;
  lora_alpha: number;
  train_text_encoder: boolean;
  optimizer_type: string;
  resolution: number;
}
