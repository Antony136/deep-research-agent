import { useState } from 'react';
import {
  Sparkles,
  ShieldAlert,
  ChevronDown,
  ChevronUp,
  Lightbulb,
} from 'lucide-react';
import { Modal } from '../common/Modal';
import { Button } from '../common/Button';
import { api } from '../../api/client';
import { useToast } from '../../context/ToastContext';

const EXAMPLE_QUESTIONS = [
  'What are the most effective approaches to improve retrieval quality in RAG systems, and how do vector search, reranking, and hybrid retrieval compare?',
  'What are the main architectural differences and trade-offs between dense retrieval, BM25, and hybrid search in production?',
  'How do modern agentic workflows handle multi-hop reasoning, and what are the primary failure modes in iterative web research?',
  'What are the current best practices for mitigating hallucinations and verifying claims in LLM synthesis pipelines?',
];

export function CreateSessionModal({ isOpen, onClose, onSessionCreated }) {
  const toast = useToast();
  const [question, setQuestion] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Settings
  const [requireInitialPlanApproval, setRequireInitialPlanApproval] = useState(true);
  const [requireInitialResearchApproval, setRequireInitialResearchApproval] = useState(true);
  const [requireAdaptiveResearchApproval, setRequireAdaptiveResearchApproval] = useState(true);
  const [requireFinalReportApproval, setRequireFinalReportApproval] = useState(true);
  const [maxResearchRounds, setMaxResearchRounds] = useState(3);
  const [maxTotalQuestions, setMaxTotalQuestions] = useState(10);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const cleanQuestion = question.trim();

    if (!cleanQuestion) {
      setError('Please provide a research question.');
      return;
    }
    if (cleanQuestion.length < 5) {
      setError('Research question should be at least 5 characters long.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await api.createSession({
        question: cleanQuestion,
        require_initial_plan_approval: requireInitialPlanApproval,
        require_initial_research_approval: requireInitialResearchApproval,
        require_adaptive_research_approval: requireAdaptiveResearchApproval,
        require_final_report_approval: requireFinalReportApproval,
        max_research_rounds: maxResearchRounds,
        max_total_research_questions: maxTotalQuestions,
      });

      toast.success('Research session initialized and scheduled!', 'Session Created');
      onSessionCreated(result.session_id);
      onClose();
      setQuestion('');
    } catch (err) {
      setError(err.message || 'Failed to initialize research session.');
      toast.error(err.message || 'Session creation failed');
    } finally {
      setLoading(false);
    }
  };

  const handleSelectExample = (q) => {
    setQuestion(q);
    setError(null);
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title="Start New Deep Research"
      subtitle="Configure an autonomous research session with optional human-in-the-loop review checkpoints."
      maxWidth="max-w-2xl"
    >
      <form onSubmit={handleSubmit} className="space-y-5">
        {/* Question Textarea */}
        <div>
          <div className="flex items-center justify-between mb-1.5">
            <label
              htmlFor="question-input"
              className="text-xs font-semibold uppercase tracking-wider text-slate-700 dark:text-slate-300"
            >
              Research Question / Objective <span className="text-rose-500">*</span>
            </label>
            <span className="text-[11px] text-slate-400 font-mono">
              {question.length} / 2000 chars
            </span>
          </div>
          <textarea
            id="question-input"
            rows={4}
            value={question}
            onChange={(e) => {
              setQuestion(e.target.value);
              if (error) setError(null);
            }}
            placeholder="e.g. What are the key architectural trade-offs between reranking and hybrid vector search in production enterprise RAG systems?"
            className="w-full px-3.5 py-2.5 rounded-xl border border-slate-300 bg-slate-50/50 text-slate-900 placeholder:text-slate-400 focus:bg-white focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/20 dark:border-slate-700 dark:bg-slate-950/50 dark:text-slate-100 dark:placeholder:text-slate-500 dark:focus:bg-slate-950 transition-all text-sm leading-relaxed outline-hidden"
          />

          {error && (
            <p className="mt-1.5 text-xs text-rose-600 dark:text-rose-400 font-medium">
              {error}
            </p>
          )}

          {/* Example Suggestions */}
          <div className="mt-2.5">
            <div className="flex items-center gap-1.5 text-xs text-slate-500 dark:text-slate-400 mb-1.5">
              <Lightbulb className="w-3.5 h-3.5 text-amber-500" />
              <span>Prompt suggestions:</span>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {EXAMPLE_QUESTIONS.map((example, idx) => (
                <button
                  key={idx}
                  type="button"
                  onClick={() => handleSelectExample(example)}
                  className="text-left text-xs px-2.5 py-1 rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-100/60 dark:bg-slate-800/60 hover:border-indigo-300 dark:hover:border-indigo-700 text-slate-600 dark:text-slate-400 hover:text-indigo-600 dark:hover:text-indigo-300 transition-all line-clamp-1 max-w-full cursor-pointer"
                >
                  {example}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Collapsible Advanced Governance / HITL Settings */}
        <div className="border border-slate-200 dark:border-slate-800 rounded-xl overflow-hidden">
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="w-full flex items-center justify-between px-4 py-3 bg-slate-50/80 dark:bg-slate-950/60 text-xs font-semibold text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-900/80 transition-colors cursor-pointer"
          >
            <div className="flex items-center gap-2">
              <ShieldAlert className="w-4 h-4 text-indigo-500" />
              <span>Human-in-the-Loop & Adaptive Limits Configuration</span>
            </div>
            {showAdvanced ? (
              <ChevronUp className="w-4 h-4 text-slate-400" />
            ) : (
              <ChevronDown className="w-4 h-4 text-slate-400" />
            )}
          </button>

          {showAdvanced && (
            <div className="p-4 bg-white dark:bg-slate-900/40 space-y-4 text-xs border-t border-slate-200 dark:border-slate-800">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
                {/* Initial Plan Approval Toggle */}
                <label className="flex items-start gap-2.5 p-2.5 rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-950/40 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={requireInitialPlanApproval}
                    onChange={(e) => setRequireInitialPlanApproval(e.target.checked)}
                    className="mt-0.5 rounded-sm border-slate-300 text-indigo-600 focus:ring-indigo-500"
                  />
                  <div>
                    <span className="font-medium text-slate-900 dark:text-slate-200 block">
                      Initial Plan Approval
                    </span>
                    <span className="text-[11px] text-slate-500 dark:text-slate-400">
                      Pause to inspect/revise proposed research questions before web search begins.
                    </span>
                  </div>
                </label>

                {/* Web Research Authorization Toggle */}
                <label className="flex items-start gap-2.5 p-2.5 rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-950/40 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={requireInitialResearchApproval}
                    onChange={(e) => setRequireInitialResearchApproval(e.target.checked)}
                    className="mt-0.5 rounded-sm border-slate-300 text-indigo-600 focus:ring-indigo-500"
                  />
                  <div>
                    <span className="font-medium text-slate-900 dark:text-slate-200 block">
                      Web Research Authorization
                    </span>
                    <span className="text-[11px] text-slate-500 dark:text-slate-400">
                      Request explicit user authorization prior to performing live web scrapes.
                    </span>
                  </div>
                </label>

                {/* Adaptive Proposals Approval Toggle */}
                <label className="flex items-start gap-2.5 p-2.5 rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-950/40 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={requireAdaptiveResearchApproval}
                    onChange={(e) => setRequireAdaptiveResearchApproval(e.target.checked)}
                    className="mt-0.5 rounded-sm border-slate-300 text-indigo-600 focus:ring-indigo-500"
                  />
                  <div>
                    <span className="font-medium text-slate-900 dark:text-slate-200 block">
                      Adaptive Gap Review
                    </span>
                    <span className="text-[11px] text-slate-500 dark:text-slate-400">
                      Pause when sufficiency evaluation proposes follow-up gap-closing questions.
                    </span>
                  </div>
                </label>

                {/* Final Report Approval Toggle */}
                <label className="flex items-start gap-2.5 p-2.5 rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-950/40 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={requireFinalReportApproval}
                    onChange={(e) => setRequireFinalReportApproval(e.target.checked)}
                    className="mt-0.5 rounded-sm border-slate-300 text-indigo-600 focus:ring-indigo-500"
                  />
                  <div>
                    <span className="font-medium text-slate-900 dark:text-slate-200 block">
                      Final Report Review
                    </span>
                    <span className="text-[11px] text-slate-500 dark:text-slate-400">
                      Review report before marking complete; allows requesting additional passes.
                    </span>
                  </div>
                </label>
              </div>

              {/* Sliders for Round and Question Limits */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2 border-t border-slate-200 dark:border-slate-800">
                <div>
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-slate-700 dark:text-slate-300 font-medium">
                      Max Research Rounds
                    </span>
                    <span className="font-mono text-indigo-600 dark:text-indigo-400 font-bold">
                      {maxResearchRounds} rounds
                    </span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="10"
                    value={maxResearchRounds}
                    onChange={(e) => setMaxResearchRounds(Number(e.target.value))}
                    className="w-full accent-indigo-600 cursor-pointer"
                  />
                  <span className="text-[10px] text-slate-400">Range: 1 – 10 rounds</span>
                </div>

                <div>
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-slate-700 dark:text-slate-300 font-medium">
                      Max Total Research Questions
                    </span>
                    <span className="font-mono text-indigo-600 dark:text-indigo-400 font-bold">
                      {maxTotalQuestions} questions
                    </span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="50"
                    value={maxTotalQuestions}
                    onChange={(e) => setMaxTotalQuestions(Number(e.target.value))}
                    className="w-full accent-indigo-600 cursor-pointer"
                  />
                  <span className="text-[10px] text-slate-400">Range: 1 – 50 questions</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Modal Footer Controls */}
        <div className="flex items-center justify-end gap-3 pt-3 border-t border-slate-200 dark:border-slate-800">
          <Button variant="outline" size="md" onClick={onClose} disabled={loading}>
            Cancel
          </Button>
          <Button
            type="submit"
            variant="primary"
            size="md"
            icon={Sparkles}
            isLoading={loading}
          >
            Launch Research Session
          </Button>
        </div>
      </form>
    </Modal>
  );
}
