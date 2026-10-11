import { useState, useMemo } from 'react';
import {
  Search,
  ExternalLink,
  Copy,
  Check,
  FileCheck2,
  Quote,
} from 'lucide-react';
import { EmptyState } from '../common/EmptyState';
import { useToast } from '../../context/ToastContext';

export function EvidenceExplorer({ evidence = [], selectedEvidenceNumber }) {
  const toast = useToast();
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedQuestion, setSelectedQuestion] = useState('all');
  const [copiedIndex, setCopiedIndex] = useState(null);

  // Available questions in the evidence pool
  const questionNumbers = useMemo(() => {
    const set = new Set();
    evidence.forEach((ev) => {
      if (ev.research_question_number) {
        set.add(ev.research_question_number);
      }
    });
    return Array.from(set).sort((a, b) => a - b);
  }, [evidence]);

  // Filtered evidence items
  const filteredEvidence = useMemo(() => {
    return evidence
      .map((item, index) => ({
        ...item,
        evidence_number: index + 1,
      }))
      .filter((item) => {
        const matchesQ =
          selectedQuestion === 'all' ||
          item.research_question_number === Number(selectedQuestion);

        const matchesSearch =
          !searchQuery ||
          item.claim?.toLowerCase().includes(searchQuery.toLowerCase()) ||
          item.supporting_text?.toLowerCase().includes(searchQuery.toLowerCase()) ||
          item.source_url?.toLowerCase().includes(searchQuery.toLowerCase());

        return matchesQ && matchesSearch;
      });
  }, [evidence, selectedQuestion, searchQuery]);

  const handleCopyQuote = async (text, idx) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedIndex(idx);
      toast.success('Supporting quote copied to clipboard');
      setTimeout(() => setCopiedIndex(null), 2000);
    } catch {
      toast.error('Failed to copy');
    }
  };

  if (!evidence || evidence.length === 0) {
    return (
      <EmptyState
        icon={FileCheck2}
        title="No Evidence Extracted Yet"
        description="Verified evidence items will be listed here as the agent reads web pages and deterministically validates claims against source text."
      />
    );
  }

  return (
    <div className="space-y-4">
      {/* Search and Question Filters */}
      <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3">
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search claims or quotes..."
            className="w-full pl-9 pr-3 py-1.5 rounded-xl text-xs sm:text-sm border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 placeholder:text-slate-400 focus:outline-hidden focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500"
          />
        </div>

        {/* Question filter pills */}
        <div className="flex items-center gap-1.5 overflow-x-auto pb-1 sm:pb-0">
          <button
            type="button"
            onClick={() => setSelectedQuestion('all')}
            className={`px-2.5 py-1 rounded-lg text-xs font-medium cursor-pointer transition-colors ${
              selectedQuestion === 'all'
                ? 'bg-indigo-600 text-white dark:bg-indigo-500'
                : 'bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400'
            }`}
          >
            All Evidence ({evidence.length})
          </button>
          {questionNumbers.map((qNum) => (
            <button
              key={qNum}
              type="button"
              onClick={() => setSelectedQuestion(qNum.toString())}
              className={`px-2.5 py-1 rounded-lg text-xs font-medium cursor-pointer transition-colors whitespace-nowrap ${
                selectedQuestion === qNum.toString()
                  ? 'bg-indigo-600 text-white dark:bg-indigo-500'
                  : 'bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400'
              }`}
            >
              Q{qNum}
            </button>
          ))}
        </div>
      </div>

      {/* Evidence Cards Grid */}
      <div className="grid grid-cols-1 gap-3.5">
        {filteredEvidence.map((ev, idx) => {
          const isHighlighted = selectedEvidenceNumber === ev.evidence_number;

          return (
            <div
              key={idx}
              id={`evidence-${ev.evidence_number}`}
              className={`p-4 rounded-xl border transition-all ${
                isHighlighted
                  ? 'border-emerald-500 bg-emerald-50/40 dark:border-emerald-500 dark:bg-emerald-950/30 ring-2 ring-emerald-500/30'
                  : 'border-slate-200 dark:border-slate-800 bg-white/90 dark:bg-slate-900/80'
              }`}
            >
              <div className="flex items-start justify-between gap-3 pb-2 border-b border-slate-100 dark:border-slate-800/80">
                <div className="flex items-center gap-2">
                  <span className="px-2 py-0.5 rounded-md text-xs font-mono font-bold bg-emerald-100 dark:bg-emerald-950 text-emerald-800 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800">
                    Evidence #{ev.evidence_number}
                  </span>
                  <span className="px-2 py-0.5 rounded-md text-xs font-semibold bg-indigo-50 dark:bg-indigo-950/60 text-indigo-700 dark:text-indigo-300">
                    Supports Q{ev.research_question_number}
                  </span>
                </div>

                {ev.source_url && (
                  <a
                    href={ev.source_url}
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex items-center gap-1 text-xs text-indigo-600 dark:text-indigo-400 hover:underline max-w-[200px] truncate"
                    title={ev.source_url}
                  >
                    <span>Source</span>
                    <ExternalLink className="w-3 h-3 shrink-0" />
                  </a>
                )}
              </div>

              {/* Verified Claim */}
              <div className="pt-2 space-y-2">
                <div>
                  <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400 block mb-0.5">
                    Verified Claim
                  </span>
                  <p className="text-sm font-semibold text-slate-900 dark:text-slate-100 leading-snug">
                    {ev.claim}
                  </p>
                </div>

                {/* Verbatim Supporting Text Quote */}
                {ev.supporting_text && (
                  <div className="p-3 rounded-lg bg-slate-50 dark:bg-slate-950/50 border border-slate-200/70 dark:border-slate-800 relative group">
                    <div className="flex items-center justify-between text-[11px] text-slate-400 mb-1">
                      <span className="flex items-center gap-1 font-medium">
                        <Quote className="w-3 h-3 text-slate-400" /> Verbatim Source Quote:
                      </span>
                      <button
                        type="button"
                        onClick={() => handleCopyQuote(ev.supporting_text, idx)}
                        className="p-1 rounded text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 cursor-pointer"
                        title="Copy quote"
                      >
                        {copiedIndex === idx ? (
                          <Check className="w-3 h-3 text-emerald-500" />
                        ) : (
                          <Copy className="w-3 h-3" />
                        )}
                      </button>
                    </div>
                    <p className="text-xs text-slate-600 dark:text-slate-300 italic leading-relaxed">
                      "{ev.supporting_text}"
                    </p>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
