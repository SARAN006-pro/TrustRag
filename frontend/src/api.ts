import { AppSettings, LLMProviderInfo, Source, SystemHealth } from './types';

export const API_BASE = ''; // Uses Vite proxy or relative path

export async function fetchHealth(): Promise<SystemHealth> {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    return {
      status: 'offline',
      ollama_reachable: false,
      index_chunks: 0,
      model_loaded: false,
    };
  }
}

export async function fetchProviders(): Promise<LLMProviderInfo[]> {
  try {
    const res = await fetch(`${API_BASE}/providers`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.error('Failed to fetch providers', err);
    return [
      {
        id: 'ollama',
        label: 'Ollama (Local)',
        default_model: 'llama3.2:3b',
        base_url: 'http://localhost:11434',
        requires_key: false,
        has_server_key: true,
      },
    ];
  }
}

export interface StreamCallbacks {
  onStatus: (stage: string, message: string) => void;
  onSources: (sources: Source[]) => void;
  onToken: (token: string) => void;
  onDone: (data: { answer: string; sources: Source[]; grounded: boolean; risk_score: number; latency_ms: number }) => void;
  onError: (error: string) => void;
}

export async function askStream(
  question: string,
  settings: AppSettings,
  callbacks: StreamCallbacks,
  signal?: AbortSignal
): Promise<void> {
  const payload: Record<string, any> = {
    question: question.trim(),
    provider: settings.provider,
  };

  if (settings.model) payload.model = settings.model.trim();
  if (settings.baseUrl) payload.base_url = settings.baseUrl.trim();
  if (settings.apiKey) payload.api_key = settings.apiKey.trim();

  try {
    const response = await fetch(`${API_BASE}/ask/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
      signal,
    });

    if (!response.ok) {
      const errText = await response.text();
      callbacks.onError(`Request failed (${response.status}): ${errText}`);
      return;
    }

    if (!response.body) {
      callbacks.onError('Streaming response body is null');
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n\n');
      buffer = lines.pop() || '';

      for (const block of lines) {
        if (!block.trim()) continue;
        const blockLines = block.split('\n');
        let eventType = '';
        let eventData = '';

        for (const line of blockLines) {
          if (line.startsWith('event: ')) {
            eventType = line.slice(7).trim();
          } else if (line.startsWith('data: ')) {
            eventData = line.slice(6).trim();
          }
        }

        if (!eventType || !eventData) continue;

        try {
          const parsed = JSON.parse(eventData);
          if (eventType === 'status') {
            callbacks.onStatus(parsed.stage, parsed.message);
          } else if (eventType === 'sources') {
            callbacks.onSources(parsed.sources || []);
          } else if (eventType === 'token') {
            callbacks.onToken(parsed.token || '');
          } else if (eventType === 'done') {
            callbacks.onDone(parsed);
          } else if (eventType === 'error') {
            callbacks.onError(parsed.error || 'Unknown error occurred');
          }
        } catch (parseErr) {
          console.error('SSE JSON parse error', parseErr, eventData);
        }
      }
    }
  } catch (err: any) {
    if (err.name === 'AbortError') {
      callbacks.onError('Generation stopped by user.');
    } else {
      callbacks.onError(err.message || 'Stream connection failed');
    }
  }
}

export async function askSync(
  question: string,
  settings: AppSettings,
  signal?: AbortSignal
): Promise<{ answer: string; sources: Source[]; grounded: boolean; risk_score: number; latency_ms: number }> {
  const payload: Record<string, any> = {
    question: question.trim(),
    provider: settings.provider,
  };

  if (settings.model) payload.model = settings.model.trim();
  if (settings.baseUrl) payload.base_url = settings.baseUrl.trim();
  if (settings.apiKey) payload.api_key = settings.apiKey.trim();

  const response = await fetch(`${API_BASE}/ask`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
    signal,
  });

  if (!response.ok) {
    const errData = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(errData.detail || `Server error ${response.status}`);
  }

  return await response.json();
}
