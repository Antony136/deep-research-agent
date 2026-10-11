import { GitBranch, Tag, HelpCircle } from 'lucide-react';
import { EmptyState } from '../common/EmptyState';

export function QuestionsList({ questions = [], currentQuestionIndex = 0, evidence = [] }) {
  if (!questions || questions.length === 0) {
    return (
      <EmptyState
        icon={HelpCircle}
        title="No Research Questions"
        description="The initial plan questions will appear here after the planner node completes question formulation."
      />
    );
  }

  // Count evidence for each question
  const evidenceCountByQuestion = {};
  evidence.forEach((ev) => {
    const qNum = ev.research_question_number;
    if (qNum) {
      evidenceCountByQuestion[qNum] = (evidenceCountByQuestion[qNum] || 0) + 1;
    }
  });

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between pb-1">
        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
          Formulated Questions ({questions.length})
        </h3>
        <span className="text-xs text-slate-400">
          Progress: {currentQuestionIndex} of {questions.length} completed
        </span>
      </div>

      <div className="grid grid-cols-1 gap-3">
        {questions.map((q, idx) => {
          const qNum = idx + 1;
          const isDone = idx < currentQuestionIndex;
          const isActive = idx === currentQuestionIndex;
          const isAdaptive = q.parent_question_number !== null && q.parent_question_number !== undefined;
          const evCount = evidenceCountByQuestion[qNum] || 0;

          return (
            <div
              key={idx}
              className={`p-4 rounded-xl border transition-all ${
                isActive
                  ? 'border-sky-400 dark:border-sky-600 bg-sky-50/40 dark:bg-sky-950/20 ring-2 ring-sky-500/20'
                  : isDone
                  ? 'border-emerald-200 dark:border-emerald-900/60 bg-white/90 dark:bg-slate-900/80'
                  : 'border-slate-200 dark:border-slate-800 bg-white/60 dark:bg-slate-900/40 opacity-80'
              }`}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex items-start gap-2.5">
                  <div
                    className={`w-6 h-6 rounded-md flex items-center justify-center text-xs font-bold shrink-0 mt-0.5 ${
                      isDone
                        ? 'bg-emerald-100 dark:bg-emerald-950 text-emerald-700 dark:text-emerald-300'
                        : isActive
                        ? 'bg-sky-100 dark:bg-sky-950 text-sky-700 dark:text-sky-300 animate-pulse'
                        : 'bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400'
                    }`}
                  >
                    Q{qNum}
                  </div>

                  <div>
                    <div className="flex items-center gap-2">
                      <h4 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                        {q.question}
                      </h4>
                      {isAdaptive && (
                        <span className="inline-flex items-center gap-1 px-1.5 py-0.2 rounded-xs text-[10px] font-medium bg-purple-100 dark:bg-purple-950 text-purple-700 dark:purple-300 border border-purple-200 dark:border-purple-800">
                          <GitBranch className="w-2.5 h-2.5" />
                          Follow-up to Q{q.parent_question_number}
                        </span>
                      )}
                    </div>

                    {/* Search queries list */}
                    {q.search_queries && q.search_queries.length > 0 && (
                      <div className="flex flex-wrap items-center gap-1.5 mt-2">
                        <Tag className="w-3 h-3 text-slate-400" />
                        {q.search_queries.map((query, qIdx) => (
                          <span
                            key={qIdx}
                            className="px-2 py-0.5 rounded-md text-[11px] bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 font-mono"
                          >
                            {query}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>

                {/* Evidence Count Pill */}
                <div className="shrink-0 flex items-center gap-2">
                  <span
                    className={`px-2 py-0.5 rounded-full text-xs font-medium border ${
                      evCount > 0
                        ? 'bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-950/60 dark:text-emerald-300 dark:border-emerald-800'
                        : 'bg-slate-100 text-slate-500 border-slate-200 dark:bg-slate-800 dark:text-slate-400 dark:border-slate-700'
                    }`}
                  >
                    {evCount} evidence
                  </span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
