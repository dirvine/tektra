export interface ModelStatus {
  isLoaded: boolean;
  modelName: string;
  backend: string;
  isLoading: boolean;
  whisperReady: boolean;
  cameraEnabled: boolean;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
}

export interface AppSettings {
  model_name: string;
  max_tokens: number;
  temperature: number;
  voice_enabled: boolean;
  auto_speech: boolean;
  system_prompt?: string;
  user_prefix?: string;
  assistant_prefix?: string;
}