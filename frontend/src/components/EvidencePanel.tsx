import React from 'react';
import { ShieldCheck, AlertOctagon, HelpCircle, FileText, Bookmark } from 'lucide-react';
import { Source } from '../types';

interface EvidencePanelProps {
  sources: Source[];
  grounded: boolean | null;
  riskScore: number | null;
  latencyMs: number | null;
  activeCitationId: string | null;
  onSelectCitation: (chunkId: string) => void;
}

export const EvidencePanel: React.FC<EvidencePanelProps> = ({
  sources,
  grounded,
  riskScore,
  latencyMs,
  activeCitationId,
  onSelectCitation,
}) => {
  // Verification classification logic
  let verificationState: 'verified' | 'unverified' | 'insufficient' | 'none' = 'none';

  if (grounded !== null && riskScore !== null) {
    if (sources.length === 0) {
      verificationState = 'insufficient';
    } else if (grounded && riskScore < 0.5) {
      verificationState = 'verified';
    } else {
      verificationState = 'unverified';
    }
  }

  return (
    <aside className="w-80 lg:w-96 border-l border-slate-800 bg-slate-900/40 backdrop-blur flex flex-col h-full overflow-hidden">
      {/* Panel Header */}
      <div className="p-4 border-b border-slate-800 flex items-center justify-between bg-slate-950/40">
        <div className="flex items-center space-x-2">
          <Bookmark className="w-4 h-4 text-blue-400" />
          <h2 className="text-xs font-bold text-white uppercase tracking-wider">
            Evidence & Verification
          </h2>
        </div>
        {latencyMs !== null && (
          <span className="text-[11px] font-mono text-slate-400 bg-slate-800/80 px-2 py-0.5 rounded">
            {latencyMs}ms
          </span>
        )}
      </div>

      {/* Verification Indicator Banner */}
      <div className="p-4 border-b border-slate-800">
        {verificationState === 'verified' && (
          <div className="p-3.5 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 space-y-1.5 animate-fade-in">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2 font-bold text-xs uppercase tracking-wider text-emerald-400">
                <ShieldCheck className="w-4 h-4 text-emerald-400" />
                <span>Verified Grounded</span>
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/20 font-bold">
                Risk: {((riskScore || 0) * 100).toFixed(1)}%
              </span>
            </div>
            <p className="text-xs text-emerald-200/90 leading-snug">
              Every factual assertion is corroborated by the retrieved government scheme guidelines. Hallucination risk is minimal.
            </p>
          </div>
        )}

        {verificationState === 'unverified' && (
          <div className="p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 space-y-1.5 animate-fade-in">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2 font-bold text-xs uppercase tracking-wider text-rose-400">
                <AlertOctagon className="w-4 h-4 text-rose-400" />
                <span>Verification Risk</span>
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-rose-500/20 font-bold">
                Risk: {((riskScore || 0) * 100).toFixed(1)}%
              </span>
            </div>
            <p className="text-xs text-rose-200/90 leading-snug">
              Classifier flagged potential extrapolation or unsupported claims against the retrieved evidence text.
            </p>
          </div>
        )}

        {verificationState === 'insufficient' && (
          <div className="p-3.5 rounded-xl bg-blue-500/10 border border-blue-500/30 text-blue-300 space-y-1.5 animate-fade-in">
            <div className="flex items-center space-x-2 font-bold text-xs uppercase tracking-wider text-blue-400">
              <HelpCircle className="w-4 h-4 text-blue-400" />
              <span>Insufficient Evidence</span>
            </div>
            <p className="text-xs text-blue-200/90 leading-snug">
              No matching government scheme guidelines were found in the index for this query. The system correctly abstained.
            </p>
          </div>
        )}

        {verificationState === 'none' && (
          <div className="p-4 rounded-xl border border-dashed border-slate-800 text-center text-xs text-slate-500">
            Ask a scheme question to view source passages and verification scores.
          </div>
        )}
      </div>

      {/* Sources List */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
            Retrieved Passages ({sources.length})
          </span>
          <span className="text-[10px] text-slate-500">ChromaDB Top-K</span>
        </div>

        {sources.length === 0 ? (
          <div className="py-12 text-center text-xs text-slate-600">
            No source documents retrieved yet.
          </div>
        ) : (
          sources.map((src) => {
            const isActive = activeCitationId === src.chunk_id;
            return (
              <div
                key={src.chunk_id}
                id={`citation-${src.chunk_id}`}
                onClick={() => onSelectCitation(src.chunk_id)}
                className={`p-3.5 rounded-xl border text-left transition cursor-pointer ${
                  isActive
                    ? 'border-blue-500 bg-blue-500/10 shadow-lg shadow-blue-500/10 ring-1 ring-blue-500'
                    : 'border-slate-800 bg-slate-900/60 hover:border-slate-700 hover:bg-slate-800/40'
                }`}
              >
                <div className="flex items-start justify-between gap-2 mb-1.5">
                  <div className="flex items-center space-x-1.5">
                    <FileText className="w-3.5 h-3.5 text-blue-400 shrink-0" />
                    <span className="text-xs font-bold text-white truncate max-w-[170px]" title={src.doc}>
                      {src.doc}
                    </span>
                  </div>
                  {src.score !== undefined && src.score !== null && (
                    <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 shrink-0 font-medium">
                      {(src.score * 100).toFixed(0)}% match
                    </span>
                  )}
                </div>

                <div className="flex items-center space-x-2 text-[10px] font-mono text-slate-400 mb-2">
                  <span className="px-1.5 py-0.5 rounded bg-slate-800/90 text-blue-300">
                    [{src.chunk_id}]
                  </span>
                  {src.page !== null && src.page !== undefined && (
                    <span>Page {src.page}</span>
                  )}
                </div>

                <p className="text-xs text-slate-300 leading-relaxed font-sans line-clamp-4 hover:line-clamp-none transition-all">
                  "{src.snippet}"
                </p>
              </div>
            );
          })
        )}
      </div>
    </aside>
  );
};
