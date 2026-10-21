import React, { useState } from 'react';
import { 
  Bot, 
  User, 
  ShieldCheck, 
  AlertTriangle, 
  Copy, 
  Check, 
  RotateCcw, 
  Loader2, 
  Cpu
} from 'lucide-react';
import { Message } from '../types';
import { StructuredAnswer } from './StructuredAnswer';

interface MessageItemProps {
  message: Message;
  onCitationClick: (chunkId: string) => void;
  onRetry?: (question: string) => void;
}

export const MessageItem: React.FC<MessageItemProps> = ({
  message,
  onCitationClick,
  onRetry,
}) => {
  const [copied, setCopied] = useState(false);
  const isUser = message.role === 'user';
  const isAssistant = message.role === 'assistant';

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className={`flex gap-3 max-w-4xl mx-auto ${isUser ? 'justify-end' : 'justify-start'} animate-fade-in`}>
      {/* Avatar */}
      {isAssistant && (
        <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-600 flex items-center justify-center shrink-0 shadow-md shadow-blue-500/10 mt-1">
          <Bot className="w-4 h-4 text-white" />
        </div>
      )}

      {/* Bubble Container */}
      <div className={`space-y-2 max-w-[85%] ${isUser ? 'items-end' : 'items-start'}`}>
        <div
          className={`p-4 rounded-2xl ${
            isUser
              ? 'bg-blue-600 text-white rounded-tr-sm shadow-md shadow-blue-600/10'
              : 'bg-slate-900/90 border border-slate-800 text-slate-100 rounded-tl-sm shadow-xl'
          }`}
        >
          {/* User Message */}
          {isUser && (
            <p className="text-sm font-medium leading-relaxed whitespace-pre-wrap">
              {message.content}
            </p>
          )}

          {/* Assistant Message */}
          {isAssistant && (
            <div className="space-y-3">
              {/* Streaming Progress Banner */}
              {message.status && message.status !== 'done' && message.status !== 'idle' && message.status !== 'error' && (
                <div className="flex items-center space-x-2 text-xs text-blue-400 bg-blue-500/10 border border-blue-500/20 px-3 py-1.5 rounded-lg animate-pulse">
                  <Loader2 className="w-3.5 h-3.5 animate-spin shrink-0" />
                  <span>{message.statusMessage || 'Processing query...'}</span>
                </div>
              )}

              {/* Structured Body */}
              {message.content ? (
                <StructuredAnswer
                  content={message.content}
                  onCitationClick={onCitationClick}
                />
              ) : (
                message.status !== 'error' && (
                  <div className="flex items-center space-x-2 text-xs text-slate-400 py-1">
                    <Loader2 className="w-4 h-4 animate-spin text-blue-500" />
                    <span>Analyzing documents...</span>
                  </div>
                )
              )}

              {/* Error Message */}
              {message.error && (
                <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs space-y-2">
                  <div className="flex items-center space-x-2 font-semibold text-rose-400">
                    <AlertTriangle className="w-4 h-4 shrink-0" />
                    <span>Generation Error</span>
                  </div>
                  <p>{message.error}</p>
                  {onRetry && (
                    <button
                      onClick={() => onRetry(message.content || '')}
                      className="inline-flex items-center space-x-1.5 px-3 py-1 rounded-lg bg-rose-500/20 hover:bg-rose-500/30 text-rose-200 text-xs font-semibold transition"
                    >
                      <RotateCcw className="w-3 h-3" />
                      <span>Retry Query</span>
                    </button>
                  )}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Assistant Footer Info (Model, Groundedness Badge, Latency, Actions) */}
        {isAssistant && message.status === 'done' && (
          <div className="flex items-center justify-between px-2 text-[11px] text-slate-400">
            <div className="flex items-center space-x-2">
              {/* Provider Pill */}
              <span className="flex items-center space-x-1 px-2 py-0.5 rounded bg-slate-800/80 border border-slate-700/60 font-mono text-slate-300">
                <Cpu className="w-3 h-3 text-blue-400" />
                <span>{message.provider || 'ollama'}</span>
              </span>

              {/* Verification Status Pill */}
              {message.grounded !== null && message.grounded !== undefined && (
                <span
                  className={`flex items-center space-x-1 px-2 py-0.5 rounded font-medium ${
                    message.grounded
                      ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
                      : 'bg-rose-500/10 text-rose-400 border border-rose-500/30'
                  }`}
                >
                  <ShieldCheck className="w-3 h-3" />
                  <span>{message.grounded ? 'Grounded' : 'Risk'}</span>
                  {message.risk_score !== null && (
                    <span className="font-mono opacity-80">
                      ({((message.risk_score || 0) * 100).toFixed(0)}%)
                    </span>
                  )}
                </span>
              )}

              {/* Latency */}
              {message.latency_ms && (
                <span className="font-mono text-slate-500">
                  {message.latency_ms}ms
                </span>
              )}
            </div>

            {/* Copy Button */}
            <button
              onClick={handleCopy}
              className="p-1 rounded hover:bg-slate-800 text-slate-500 hover:text-slate-300 transition"
              title="Copy Answer"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
            </button>
          </div>
        )}
      </div>

      {/* User Avatar */}
      {isUser && (
        <div className="w-8 h-8 rounded-xl bg-slate-800 border border-slate-700 flex items-center justify-center shrink-0 mt-1">
          <User className="w-4 h-4 text-slate-300" />
        </div>
      )}
    </div>
  );
};
