import React, { useState, useEffect, useRef } from 'react';
import { Header } from './components/Header';
import { Sidebar } from './components/Sidebar';
import { ChatArea } from './components/ChatArea';
import { EvidencePanel } from './components/EvidencePanel';
import { SettingsModal } from './components/SettingsModal';
import { 
  AppSettings, 
  LLMProviderInfo, 
  Message, 
  Source, 
  SystemHealth 
} from './types';
import { askStream, askSync, fetchHealth, fetchProviders } from './api';

export const App: React.FC = () => {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [providers, setProviders] = useState<LLMProviderInfo[]>([]);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  // App Settings (Ephemeral API keys are session-only in memory)
  const [settings, setSettings] = useState<AppSettings>({
    provider: 'ollama',
    model: 'llama3.2:3b',
    apiKey: '',
    baseUrl: '',
    streamMode: true,
  });

  // Conversation & Evidence
  const [messages, setMessages] = useState<Message[]>([]);
  const [activeEvidence, setActiveEvidence] = useState<Source[]>([]);
  const [activeGrounded, setActiveGrounded] = useState<boolean | null>(null);
  const [activeRiskScore, setActiveRiskScore] = useState<number | null>(null);
  const [activeLatencyMs, setActiveLatencyMs] = useState<number | null>(null);
  const [activeCitationId, setActiveCitationId] = useState<string | null>(null);

  const abortControllerRef = useRef<AbortController | null>(null);

  // Load initial health & provider list
  useEffect(() => {
    fetchHealth().then(setHealth);
    fetchProviders().then((list) => {
      setProviders(list);
      // Auto-select first provider's default model if not configured
      const defaultProvider = list.find((p) => p.id === 'ollama') || list[0];
      if (defaultProvider) {
        setSettings((prev) => ({
          ...prev,
          provider: defaultProvider.id,
          model: defaultProvider.default_model,
          baseUrl: defaultProvider.base_url || '',
        }));
      }
    });

    const interval = setInterval(() => {
      fetchHealth().then(setHealth);
    }, 15000);
    return () => clearInterval(interval);
  }, []);

  const handleSendMessage = async (text: string) => {
    if (!text.trim() || isLoading) return;

    const userMessageId = `msg-user-${Date.now()}`;
    const assistantMessageId = `msg-asst-${Date.now() + 1}`;

    const userMsg: Message = {
      id: userMessageId,
      role: 'user',
      content: text.trim(),
      timestamp: new Date().toLocaleTimeString(),
    };

    const asstMsg: Message = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      status: 'retrieving',
      statusMessage: 'Searching Indian scheme guidelines...',
      provider: settings.provider,
      model: settings.model,
      timestamp: new Date().toLocaleTimeString(),
    };

    setMessages((prev) => [...prev, userMsg, asstMsg]);
    setIsLoading(true);
    setActiveCitationId(null);

    // Abort controller for streaming cancellation
    abortControllerRef.current = new AbortController();

    if (settings.streamMode) {
      await askStream(
        text,
        settings,
        {
          onStatus: (stage, message) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMessageId
                  ? { ...m, status: stage as any, statusMessage: message }
                  : m
              )
            );
          },
          onSources: (sources) => {
            setActiveEvidence(sources);
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMessageId ? { ...m, sources } : m
              )
            );
          },
          onToken: (token) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMessageId
                  ? { ...m, content: m.content + token, status: 'generating' }
                  : m
              )
            );
          },
          onDone: (data) => {
            setActiveEvidence(data.sources || []);
            setActiveGrounded(data.grounded);
            setActiveRiskScore(data.risk_score);
            setActiveLatencyMs(data.latency_ms);

            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMessageId
                  ? {
                      ...m,
                      content: data.answer,
                      sources: data.sources,
                      grounded: data.grounded,
                      risk_score: data.risk_score,
                      latency_ms: data.latency_ms,
                      status: 'done',
                    }
                  : m
              )
            );
            setIsLoading(false);
          },
          onError: (error) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMessageId
                  ? { ...m, error, status: 'error' }
                  : m
              )
            );
            setIsLoading(false);
          },
        },
        abortControllerRef.current.signal
      );
    } else {
      // Non-streaming fallback
      try {
        const res = await askSync(text, settings, abortControllerRef.current.signal);
        setActiveEvidence(res.sources || []);
        setActiveGrounded(res.grounded);
        setActiveRiskScore(res.risk_score);
        setActiveLatencyMs(res.latency_ms);

        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMessageId
              ? {
                  ...m,
                  content: res.answer,
                  sources: res.sources,
                  grounded: res.grounded,
                  risk_score: res.risk_score,
                  latency_ms: res.latency_ms,
                  status: 'done',
                }
              : m
          )
        );
      } catch (err: any) {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMessageId
              ? { ...m, error: err.message || 'Generation failed', status: 'error' }
              : m
          )
        );
      } finally {
        setIsLoading(false);
      }
    }
  };

  const handleCancelStream = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      setIsLoading(false);
    }
  };

  const handleCitationClick = (chunkId: string) => {
    setActiveCitationId(chunkId);
    // Smooth scroll to citation card in Evidence Panel
    const elem = document.getElementById(`citation-${chunkId}`);
    if (elem) {
      elem.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  };

  const handleSelectHistoryMessage = (msgId: string) => {
    const target = messages.find((m) => m.id === msgId);
    if (target && target.role === 'user') {
      const idx = messages.indexOf(target);
      const asstReply = messages[idx + 1];
      if (asstReply && asstReply.sources) {
        setActiveEvidence(asstReply.sources);
        setActiveGrounded(asstReply.grounded ?? null);
        setActiveRiskScore(asstReply.risk_score ?? null);
        setActiveLatencyMs(asstReply.latency_ms ?? null);
      }
    }
  };

  const handleResetChat = () => {
    if (isLoading) handleCancelStream();
    setMessages([]);
    setActiveEvidence([]);
    setActiveGrounded(null);
    setActiveRiskScore(null);
    setActiveLatencyMs(null);
    setActiveCitationId(null);
  };

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-slate-950 text-slate-100">
      {/* Top Navigation */}
      <Header
        health={health}
        settings={settings}
        onOpenSettings={() => setIsSettingsOpen(true)}
        onResetChat={handleResetChat}
      />

      {/* Main Body (Sidebar + ChatArea + EvidencePanel) */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left: Scheme Navigator */}
        <Sidebar
          onSelectQuestion={handleSendMessage}
          messages={messages}
          onSelectHistoryMessage={handleSelectHistoryMessage}
          onClearHistory={handleResetChat}
        />

        {/* Center: Conversation */}
        <ChatArea
          messages={messages}
          isLoading={isLoading}
          onSendMessage={handleSendMessage}
          onCancelStream={handleCancelStream}
          onCitationClick={handleCitationClick}
          onRetry={handleSendMessage}
        />

        {/* Right: Verification & Evidence */}
        <EvidencePanel
          sources={activeEvidence}
          grounded={activeGrounded}
          riskScore={activeRiskScore}
          latencyMs={activeLatencyMs}
          activeCitationId={activeCitationId}
          onSelectCitation={(id) => setActiveCitationId(id)}
        />
      </div>

      {/* Settings Modal */}
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        settings={settings}
        providers={providers}
        onSave={(newSettings) => setSettings(newSettings)}
      />
    </div>
  );
};

export default App;
