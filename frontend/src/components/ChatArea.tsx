import React, { useRef, useEffect, useState } from 'react';
import { Send, Square, Sparkles, Shield } from 'lucide-react';
import { Message } from '../types';
import { MessageItem } from './MessageItem';

interface ChatAreaProps {
  messages: Message[];
  isLoading: boolean;
  onSendMessage: (text: string) => void;
  onCancelStream: () => void;
  onCitationClick: (chunkId: string) => void;
  onRetry: (text: string) => void;
}

export const ChatArea: React.FC<ChatAreaProps> = ({
  messages,
  isLoading,
  onSendMessage,
  onCancelStream,
  onCitationClick,
  onRetry,
}) => {
  const [input, setInput] = useState('');
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll on new message or stream tokens
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = () => {
    if (!input.trim() || isLoading) return;
    onSendMessage(input);
    setInput('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    // Auto-adjust height up to 160px
    e.target.style.height = 'auto';
    e.target.style.height = `${Math.min(e.target.scrollHeight, 160)}px`;
  };

  const charCount = input.length;
  const isOverLimit = charCount > 500;

  return (
    <main className="flex-1 flex flex-col h-full bg-slate-950 overflow-hidden relative">
      {/* Messages Scroll Area */}
      <div className="flex-1 overflow-y-auto p-4 lg:p-6 space-y-6">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center max-w-xl mx-auto text-center space-y-4">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-tr from-blue-600/20 via-indigo-600/20 to-amber-500/20 border border-blue-500/30 flex items-center justify-center shadow-xl">
              <Shield className="w-8 h-8 text-blue-400" />
            </div>
            <div className="space-y-1">
              <h2 className="text-xl font-bold text-white tracking-tight">
                TrustRAG Scheme Assistant
              </h2>
              <p className="text-sm text-slate-400 max-w-md mx-auto">
                Ask any question about Indian central welfare schemes. Answers are strictly synthesized from official guidelines and verified by our groundedness classifier.
              </p>
            </div>
            <div className="flex flex-wrap items-center justify-center gap-2 pt-2">
              {[
                'PM-KISAN eligibility',
                'Ayushman Bharat cover',
                'Atal Pension Yojana age limit',
                'PM SVANidhi loan',
              ].map((pill, idx) => (
                <button
                  key={idx}
                  onClick={() => onSendMessage(`What are the key guidelines for ${pill}?`)}
                  className="px-3 py-1.5 rounded-full text-xs font-medium bg-slate-900 border border-slate-800 text-slate-300 hover:text-white hover:border-slate-700 hover:bg-slate-800 transition"
                >
                  {pill}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((msg) => (
            <MessageItem
              key={msg.id}
              message={msg}
              onCitationClick={onCitationClick}
              onRetry={onRetry}
            />
          ))
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input Area */}
      <div className="p-4 border-t border-slate-800 bg-slate-900/80 backdrop-blur">
        <div className="max-w-4xl mx-auto space-y-2">
          <div className="relative rounded-2xl bg-slate-950 border border-slate-700/80 focus-within:border-blue-500 focus-within:ring-1 focus-within:ring-blue-500 shadow-xl transition overflow-hidden">
            <textarea
              ref={textareaRef}
              rows={1}
              value={input}
              onChange={handleInput}
              onKeyDown={handleKeyDown}
              disabled={isLoading}
              placeholder="Ask about PM-KISAN, Ayushman Bharat, PM-SYM, PMAY-G, Mudra..."
              className="w-full px-4 py-3.5 pr-24 bg-transparent text-sm text-white placeholder-slate-500 focus:outline-none resize-none max-h-40 overflow-y-auto disabled:opacity-60"
            />

            {/* Right Action Button (Send or Cancel) */}
            <div className="absolute right-2.5 bottom-2.5 flex items-center space-x-1.5">
              {isLoading ? (
                <button
                  type="button"
                  onClick={onCancelStream}
                  className="px-3 py-1.5 rounded-xl bg-rose-600 hover:bg-rose-500 text-white text-xs font-semibold flex items-center space-x-1.5 shadow-md shadow-rose-600/20 transition"
                >
                  <Square className="w-3.5 h-3.5 fill-current" />
                  <span>Stop</span>
                </button>
              ) : (
                <button
                  type="button"
                  onClick={handleSend}
                  disabled={!input.trim() || isOverLimit}
                  className="p-2 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:hover:bg-blue-600 text-white shadow-md shadow-blue-600/20 transition"
                  title="Send Question (Enter)"
                >
                  <Send className="w-4 h-4" />
                </button>
              )}
            </div>
          </div>

          {/* Sub-bar: Character count & notice */}
          <div className="flex items-center justify-between px-2 text-[11px] text-slate-500">
            <div className="flex items-center space-x-1">
              <Sparkles className="w-3 h-3 text-amber-500/80" />
              <span>Verifiable RAG: Cites [chunk_id] directly from scheme guidelines</span>
            </div>
            <div className={`font-mono ${isOverLimit ? 'text-rose-400 font-bold' : ''}`}>
              {charCount} / 500
            </div>
          </div>
        </div>
      </div>
    </main>
  );
};
