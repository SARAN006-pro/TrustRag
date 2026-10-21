import React from 'react';
import { 
  Compass, 
  History, 
  Sparkles, 
  Wheat, 
  HeartPulse, 
  Coins, 
  Home, 
  Building2, 
  Trash2,
  ChevronRight
} from 'lucide-react';
import { Message } from '../types';

interface SidebarProps {
  onSelectQuestion: (question: string) => void;
  messages: Message[];
  onSelectHistoryMessage: (msgId: string) => void;
  onClearHistory: () => void;
}

interface SchemeCategory {
  title: string;
  icon: React.ReactNode;
  questions: string[];
}

const SCHEME_CATEGORIES: SchemeCategory[] = [
  {
    title: 'Agriculture & Farmers',
    icon: <Wheat className="w-3.5 h-3.5 text-amber-400" />,
    questions: [
      'What is the annual benefit amount in PM-KISAN?',
      'Who is eligible for PM-KISAN income support?',
      'Are institutional landholders eligible for PM-KISAN?',
    ],
  },
  {
    title: 'Healthcare & Insurance',
    icon: <HeartPulse className="w-3.5 h-3.5 text-rose-400" />,
    questions: [
      'What is the health cover limit per family under Ayushman Bharat PM-JAY?',
      'What are the premium rates under Pradhan Mantri Suraksha Bima Yojana?',
      'Who is eligible for PM Jeevan Jyoti Bima Yojana?',
    ],
  },
  {
    title: 'Pensions & Social Security',
    icon: <Coins className="w-3.5 h-3.5 text-blue-400" />,
    questions: [
      'What is the guaranteed monthly pension under PM-SYM?',
      'What is the age limit for Atal Pension Yojana (APY)?',
      'What is the monthly pension under IGNOAPS for senior citizens?',
    ],
  },
  {
    title: 'Livelihood & Small Loans',
    icon: <Building2 className="w-3.5 h-3.5 text-emerald-400" />,
    questions: [
      'What is the initial working capital loan under PM SVANidhi?',
      'What are the loan limits under the 3 categories of Mudra (PMMY)?',
    ],
  },
  {
    title: 'Housing & Savings',
    icon: <Home className="w-3.5 h-3.5 text-indigo-400" />,
    questions: [
      'What is the financial assistance provided under PMAY-G in plain areas?',
      'What is the maximum annual deposit permitted in Sukanya Samriddhi?',
    ],
  },
];

export const Sidebar: React.FC<SidebarProps> = ({
  onSelectQuestion,
  messages,
  onSelectHistoryMessage,
  onClearHistory,
}) => {
  // Extract user questions from messages
  const userQueries = messages.filter((m) => m.role === 'user');

  return (
    <aside className="w-72 lg:w-80 border-r border-slate-800 bg-slate-900/50 backdrop-blur flex flex-col h-full overflow-hidden">
      {/* Tab: Scheme Explorer */}
      <div className="p-4 border-b border-slate-800 flex items-center justify-between bg-slate-950/40">
        <div className="flex items-center space-x-2">
          <Compass className="w-4 h-4 text-blue-400" />
          <h2 className="text-xs font-bold text-white uppercase tracking-wider">
            Scheme Navigator
          </h2>
        </div>
        <span className="text-[10px] font-semibold text-blue-400 bg-blue-500/10 px-2 py-0.5 rounded-full border border-blue-500/20">
          12 Schemes
        </span>
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-4">
        {/* Suggested Questions by Scheme */}
        <div className="space-y-3">
          <div className="flex items-center space-x-1.5 px-2 text-[11px] font-bold text-slate-400 uppercase tracking-wider">
            <Sparkles className="w-3.5 h-3.5 text-amber-400" />
            <span>Suggested Inquiries</span>
          </div>

          <div className="space-y-2">
            {SCHEME_CATEGORIES.map((cat, idx) => (
              <div key={idx} className="rounded-xl border border-slate-800/80 bg-slate-900/40 overflow-hidden">
                <div className="px-3 py-2 bg-slate-800/40 border-b border-slate-800/60 flex items-center space-x-2 text-xs font-semibold text-slate-300">
                  {cat.icon}
                  <span>{cat.title}</span>
                </div>
                <div className="p-1 space-y-0.5">
                  {cat.questions.map((q, qIdx) => (
                    <button
                      key={qIdx}
                      onClick={() => onSelectQuestion(q)}
                      className="w-full text-left p-2 rounded-lg text-xs text-slate-300 hover:text-white hover:bg-slate-800/80 transition flex items-center justify-between group"
                    >
                      <span className="line-clamp-2 leading-snug">{q}</span>
                      <ChevronRight className="w-3 h-3 text-slate-600 group-hover:text-blue-400 shrink-0 ml-1 transition" />
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Session Query History */}
        {userQueries.length > 0 && (
          <div className="pt-2 border-t border-slate-800 space-y-2">
            <div className="flex items-center justify-between px-2">
              <div className="flex items-center space-x-1.5 text-[11px] font-bold text-slate-400 uppercase tracking-wider">
                <History className="w-3.5 h-3.5 text-slate-400" />
                <span>Session Queries</span>
              </div>
              <button
                onClick={onClearHistory}
                className="text-slate-500 hover:text-rose-400 transition"
                title="Clear Session History"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            </div>

            <div className="space-y-1">
              {userQueries.map((q) => (
                <button
                  key={q.id}
                  onClick={() => onSelectHistoryMessage(q.id)}
                  className="w-full text-left px-3 py-2 rounded-lg text-xs text-slate-400 hover:text-slate-200 hover:bg-slate-800/50 truncate transition"
                >
                  {q.content}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </aside>
  );
};
