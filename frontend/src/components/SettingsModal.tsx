import React, { useState } from 'react';
import { X, ShieldAlert, Key, Globe, Layers, Zap, Eye, EyeOff, Check } from 'lucide-react';
import { AppSettings, LLMProviderInfo } from '../types';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  settings: AppSettings;
  providers: LLMProviderInfo[];
  onSave: (newSettings: AppSettings) => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  settings,
  providers,
  onSave,
}) => {
  const [form, setForm] = useState<AppSettings>(settings);
  const [showKey, setShowKey] = useState(false);

  if (!isOpen) return null;

  const currentProvider = providers.find((p) => p.id === form.provider) || {
    id: form.provider,
    label: form.provider,
    default_model: '',
    base_url: '',
    requires_key: form.provider !== 'ollama',
    has_server_key: false,
  };

  const handleProviderChange = (providerId: string) => {
    const p = providers.find((item) => item.id === providerId);
    setForm({
      ...form,
      provider: providerId,
      model: p ? p.default_model : '',
      baseUrl: p && p.base_url ? p.base_url : '',
    });
  };

  const handleSave = () => {
    onSave(form);
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm animate-fade-in">
      <div className="bg-slate-900 border border-slate-700/80 rounded-2xl w-full max-w-lg shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
        {/* Modal Header */}
        <div className="p-5 border-b border-slate-800 flex items-center justify-between bg-slate-950/40">
          <div className="flex items-center space-x-2.5">
            <div className="p-2 rounded-lg bg-blue-500/10 text-blue-400">
              <Layers className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-white">LLM Provider & Runtime Settings</h2>
              <p className="text-xs text-slate-400">Select inference backend and configure session credentials</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-5 space-y-4 overflow-y-auto">
          {/* Provider Selector */}
          <div>
            <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-1.5">
              Inference Provider
            </label>
            <div className="grid grid-cols-2 gap-2">
              {providers.map((p) => {
                const isSelected = form.provider === p.id;
                return (
                  <button
                    key={p.id}
                    type="button"
                    onClick={() => handleProviderChange(p.id)}
                    className={`flex items-center justify-between p-3 rounded-xl border text-left text-xs font-medium transition ${
                      isSelected
                        ? 'border-blue-500 bg-blue-500/10 text-white font-semibold shadow-sm'
                        : 'border-slate-800 bg-slate-800/40 text-slate-300 hover:bg-slate-800/80'
                    }`}
                  >
                    <span>{p.label}</span>
                    {isSelected && <Check className="w-4 h-4 text-blue-400" />}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Model Name Input */}
          <div>
            <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-1.5">
              Model Name / Identifier
            </label>
            <input
              type="text"
              value={form.model}
              onChange={(e) => setForm({ ...form, model: e.target.value })}
              placeholder={currentProvider.default_model || 'e.g. llama3.2:3b'}
              className="w-full px-3.5 py-2.5 rounded-xl bg-slate-950 border border-slate-700/80 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition font-mono"
            />
            <p className="text-[11px] text-slate-400 mt-1">
              Default for {currentProvider.label}: <span className="text-slate-300 font-mono">{currentProvider.default_model || 'none'}</span>
            </p>
          </div>

          {/* API Key (Ephemeral per-request) */}
          {currentProvider.requires_key && (
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider">
                  Provider API Key (Session-Only)
                </label>
                {currentProvider.has_server_key && (
                  <span className="text-[10px] text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-full border border-emerald-500/20 font-medium">
                    Server Key Configured
                  </span>
                )}
              </div>
              <div className="relative">
                <input
                  type={showKey ? 'text' : 'password'}
                  value={form.apiKey}
                  onChange={(e) => setForm({ ...form, apiKey: e.target.value })}
                  placeholder={currentProvider.has_server_key ? 'Optional override for session' : 'Enter your API key'}
                  className="w-full pl-9 pr-10 py-2.5 rounded-xl bg-slate-950 border border-slate-700/80 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition font-mono"
                />
                <Key className="w-4 h-4 text-slate-500 absolute left-3 top-3" />
                <button
                  type="button"
                  onClick={() => setShowKey(!showKey)}
                  className="absolute right-3 top-3 text-slate-500 hover:text-slate-300 transition"
                >
                  {showKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>

              {/* Security Isolation Notice */}
              <div className="mt-2 p-2.5 rounded-lg bg-blue-950/30 border border-blue-800/40 flex items-start space-x-2 text-[11px] text-blue-300">
                <ShieldAlert className="w-4 h-4 text-blue-400 shrink-0 mt-0.5" />
                <p>
                  <strong className="text-white">Security Isolation:</strong> Your API key is held only in active browser RAM for this tab session. It is never saved to localStorage, browser cookies, disk, or SQLite server logs.
                </p>
              </div>
            </div>
          )}

          {/* Base URL (Optional Override) */}
          <div>
            <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-1.5">
              Custom Endpoint Base URL (Optional)
            </label>
            <div className="relative">
              <input
                type="text"
                value={form.baseUrl}
                onChange={(e) => setForm({ ...form, baseUrl: e.target.value })}
                placeholder={currentProvider.base_url || 'http://localhost:11434'}
                className="w-full pl-9 pr-3.5 py-2.5 rounded-xl bg-slate-950 border border-slate-700/80 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition font-mono"
              />
              <Globe className="w-4 h-4 text-slate-500 absolute left-3 top-3" />
            </div>
          </div>

          {/* Streaming Toggle */}
          <div className="flex items-center justify-between p-3.5 rounded-xl bg-slate-800/40 border border-slate-800">
            <div className="flex items-center space-x-2.5">
              <div className="p-2 rounded-lg bg-amber-500/10 text-amber-400">
                <Zap className="w-4 h-4" />
              </div>
              <div>
                <span className="text-xs font-semibold text-white">Fast Streaming Mode (SSE)</span>
                <p className="text-[11px] text-slate-400">Streams tokens in realtime as generated</p>
              </div>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={form.streamMode}
                onChange={(e) => setForm({ ...form, streamMode: e.target.checked })}
                className="sr-only peer"
              />
              <div className="w-11 h-6 bg-slate-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
            </label>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="p-4 border-t border-slate-800 bg-slate-950/40 flex items-center justify-end space-x-3">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl text-xs font-semibold text-slate-400 hover:text-white hover:bg-slate-800 transition"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            className="px-5 py-2 rounded-xl text-xs font-semibold text-white bg-blue-600 hover:bg-blue-500 shadow-md shadow-blue-500/20 transition"
          >
            Save Settings
          </button>
        </div>
      </div>
    </div>
  );
};
