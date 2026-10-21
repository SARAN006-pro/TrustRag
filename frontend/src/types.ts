export interface Source {
  chunk_id: string;
  doc: string;
  page?: number | null;
  snippet: string;
  score?: number | null;
}

export type MessageStatus = 'idle' | 'retrieving' | 'generating' | 'verifying' | 'done' | 'error';

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  grounded?: boolean | null;
  risk_score?: number | null;
  latency_ms?: number | null;
  status?: MessageStatus;
  statusMessage?: string;
  error?: string | null;
  provider?: string;
  model?: string;
  timestamp: string;
}

export interface LLMProviderInfo {
  id: string;
  label: string;
  default_model: string;
  base_url: string;
  requires_key: boolean;
  has_server_key: boolean;
}

export interface AppSettings {
  provider: string;
  model: string;
  apiKey: string; // Kept only in React state memory, never stored to localStorage or cookies!
  baseUrl: string;
  streamMode: boolean;
}

export interface SystemHealth {
  status: 'ok' | 'degraded' | 'offline';
  ollama_reachable: boolean;
  index_chunks: number;
  model_loaded: boolean;
}
