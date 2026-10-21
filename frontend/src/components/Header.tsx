import React from 'react';
import { ShieldCheck, Server, Settings, RotateCcw, Cpu } from 'lucide-react';
import { AppSettings, SystemHealth } from '../types';

interface HeaderProps {
  health: SystemHealth | null;
  settings: AppSettings;
  onOpenSettings: () => void;
  onResetChat: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  health,
  settings,
  onOpenSettings,
  onResetChat,
}) => {
  const isHealthy = health?.status === 'ok';

  return (
    <header className="h-16 border-b border-slate-800 bg-slate-900/90 backdrop-blur px-4 lg:px-6 flex items-center justify-between z-20">
      {/* Brand & Title */}
      <div className="flex items-center space-x-3">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-blue-600 via-indigo-600 to-amber-500 p-[1.5px] shadow-lg shadow-blue-500/20">
          <div className="w-full h-full bg-slate-950 rounded-[10px] flex items-center justify-center">
            <ShieldCheck className="w-5 h-5 text-blue-400" />
          </div>
        </div>
        <div>
          <div className="flex items-center space-x-2">
            <h1 className="text-base font-bold text-white tracking-tight">TrustRAG</h1>
            <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/20">
              Gov Schemes
            </span>
          </div>
          <p className="text-xs text-slate-400 hidden sm:block">
            Grounded & Verifiable QA over Indian Welfare Schemes
          </p>
        </div>
      </div>

      {/* Status & Actions */}
      <div className="flex items-center space-x-3">
        {/* Backend System Health Pill */}
        <div className="hidden md:flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700/60 text-xs">
          <span className={`w-2 h-2 rounded-full ${isHealthy ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`} />
          <Server className="w-3.5 h-3.5 text-slate-400" />
          <span className="text-slate-300">
            {health?.index_chunks ?? 0} Chunks Indexed
          </span>
          <span className="text-slate-600">|</span>
          <span className={health?.ollama_reachable ? 'text-emerald-400 font-medium' : 'text-slate-400'}>
            Ollama {health?.ollama_reachable ? 'Online' : 'Offline'}
          </span>
        </div>

        {/* Active Provider Pill */}
        <button
          onClick={onOpenSettings}
          className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-blue-500/10 border border-blue-500/30 hover:bg-blue-500/20 text-blue-300 text-xs transition"
          title="Click to change model or provider"
        >
          <Cpu className="w-3.5 h-3.5" />
          <span className="font-semibold capitalize">{settings.provider}</span>
          <span className="text-blue-400/60 hidden sm:inline">({settings.model || 'default'})</span>
        </button>

        {/* Clear Chat */}
        <button
          onClick={onResetChat}
          className="p-2 rounded-lg bg-slate-800/60 hover:bg-slate-800 border border-slate-700/60 text-slate-300 hover:text-white transition"
          title="Start New Conversation"
        >
          <RotateCcw className="w-4 h-4" />
        </button>

        {/* Settings Toggle */}
        <button
          onClick={onOpenSettings}
          className="p-2 rounded-lg bg-slate-800/60 hover:bg-slate-800 border border-slate-700/60 text-slate-300 hover:text-white transition relative"
          title="Configure Providers & Keys"
        >
          <Settings className="w-4 h-4" />
          {settings.apiKey && (
            <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-emerald-400 rounded-full" />
          )}
        </button>
      </div>
    </header>
  );
};
