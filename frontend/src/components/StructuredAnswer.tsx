import React from 'react';
import { CheckCircle2, Award, FileText, ArrowRightCircle, HelpCircle } from 'lucide-react';

interface StructuredAnswerProps {
  content: string;
  onCitationClick?: (chunkId: string) => void;
}

export const StructuredAnswer: React.FC<StructuredAnswerProps> = ({ content, onCitationClick }) => {
  if (!content) return null;

  // Split lines and parse structure
  const lines = content.split('\n');

  // Check if line represents a known section header
  const getSectionType = (line: string): { type: string; title: string } | null => {
    const clean = line.replace(/^[#*\s-]+/, '').trim().toLowerCase();
    if (clean.startsWith('eligibility')) return { type: 'eligibility', title: 'Eligibility Criteria' };
    if (clean.startsWith('benefit')) return { type: 'benefits', title: 'Financial / Scheme Benefits' };
    if (clean.startsWith('required doc') || clean.startsWith('document')) return { type: 'documents', title: 'Required Documents' };
    if (clean.startsWith('application step') || clean.startsWith('how to apply') || clean.startsWith('step')) return { type: 'steps', title: 'Application Steps' };
    if (clean.startsWith('missing') || clean.startsWith('note') || clean.startsWith('disclaimer')) return { type: 'missing', title: 'Important Notes / Missing Details' };
    return null;
  };

  // Helper to render text with clickable citation badges [chunk_id]
  const renderTextWithCitations = (text: string) => {
    const citationRegex = /\[([a-zA-Z0-9_\-.]+#\d+)\]/g;
    const parts = [];
    let lastIndex = 0;
    let match;

    while ((match = citationRegex.exec(text)) !== null) {
      const preceding = text.slice(lastIndex, match.index);
      if (preceding) {
        parts.push(preceding);
      }
      const chunkId = match[1];
      parts.push(
        <button
          key={`cite-${match.index}`}
          type="button"
          onClick={() => onCitationClick && onCitationClick(chunkId)}
          className="inline-flex items-center mx-1 px-1.5 py-0.5 rounded text-[11px] font-mono font-semibold bg-blue-500/20 text-blue-300 hover:bg-blue-500/30 hover:text-blue-100 border border-blue-500/30 transition align-baseline shadow-xs"
          title={`Click to view citation: ${chunkId}`}
        >
          {chunkId}
        </button>
      );
      lastIndex = match.index + match[0].length;
    }

    if (lastIndex < text.length) {
      parts.push(text.slice(lastIndex));
    }

    return parts;
  };

  const sections: { type: string; title: string; lines: string[] }[] = [];
  let currentSection: { type: string; title: string; lines: string[] } = {
    type: 'general',
    title: '',
    lines: [],
  };

  for (const line of lines) {
    const sec = getSectionType(line);
    if (sec) {
      if (currentSection.lines.length > 0 || currentSection.title) {
        sections.push(currentSection);
      }
      currentSection = { type: sec.type, title: sec.title, lines: [] };
    } else {
      currentSection.lines.push(line);
    }
  }
  if (currentSection.lines.length > 0 || currentSection.title) {
    sections.push(currentSection);
  }

  // Icons for structured cards
  const getSectionIcon = (type: string) => {
    switch (type) {
      case 'eligibility':
        return <CheckCircle2 className="w-4 h-4 text-emerald-400" />;
      case 'benefits':
        return <Award className="w-4 h-4 text-amber-400" />;
      case 'documents':
        return <FileText className="w-4 h-4 text-blue-400" />;
      case 'steps':
        return <ArrowRightCircle className="w-4 h-4 text-indigo-400" />;
      case 'missing':
        return <HelpCircle className="w-4 h-4 text-slate-400" />;
      default:
        return null;
    }
  };

  return (
    <div className="space-y-3.5 text-sm leading-relaxed text-slate-200">
      {sections.map((sec, idx) => {
        const textContent = sec.lines.join('\n').trim();
        if (!textContent && !sec.title) return null;

        if (sec.type === 'general' && !sec.title) {
          return (
            <div key={idx} className="whitespace-pre-wrap">
              {renderTextWithCitations(textContent)}
            </div>
          );
        }

        return (
          <div
            key={idx}
            className="rounded-xl border border-slate-800 bg-slate-900/60 p-3.5 space-y-1.5 transition hover:border-slate-700"
          >
            <div className="flex items-center space-x-2 text-xs font-bold uppercase tracking-wider text-slate-300">
              {getSectionIcon(sec.type)}
              <span>{sec.title}</span>
            </div>
            <div className="text-slate-200 text-sm whitespace-pre-wrap pl-6">
              {renderTextWithCitations(textContent)}
            </div>
          </div>
        );
      })}
    </div>
  );
};
