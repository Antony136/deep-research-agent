import { useState } from 'react';
import {
  ShieldAlert,
  CheckCircle2,
  XCircle,
  Edit3,
  RotateCcw,
  Tag,
  Trash2,
  ArrowRight,
} from 'lucide-react';
import { Button } from '../common/Button';
import { useToast } from '../../context/ToastContext';
import { api } from '../../api/client';

export function ReviewPanel({ sessionId, review, onReviewSubmitted }) {
  const toast = useToast();
  const [submitting, setSubmitting] = useState(false);
  const [activeAction, setActiveAction] = useState(null); // 'approve' | 'revise' | 'reject' | 'research_more'

  // State for 'revise' action
  const [editingQuestionNumber, setEditingQuestionNumber] = useState(null);
  const [revisedQuestionText, setRevisedQuestionText] = useState('');
  const [revisedQueries, setRevisedQueries] = useState([]);
  const [newQueryInput, setNewQueryInput] = useState('');

  // State for 'reject' or 'research_more' reasons
  const [reasonInput, setReasonInput] = useState('');

  if (!review) return null;

  const reviewType = review.type;
  const questionsList = review.proposed_questions || review.research_plan || [];

  // Start revising a specific question
  const handleStartRevise = (q) => {
    setActiveAction('revise');
    setEditingQuestionNumber(q.question_number || 1);
    setRevisedQuestionText(q.question || '');
    setRevisedQueries([...(q.search_queries || [])]);
  };

  const handleAddQuery = () => {
    const q = newQueryInput.trim();
    if (q && !revisedQueries.includes(q)) {
      setRevisedQueries([...revisedQueries, q]);
      setNewQueryInput('');
    }
  };

  const handleRemoveQuery = (index) => {
    setRevisedQueries(revisedQueries.filter((_, idx) => idx !== index));
  };

  // Submit Decision
  const handleDecision = async (action, overrideData = {}) => {
    setSubmitting(true);
    try {
      let payload = { action, ...overrideData };

      if (action === 'revise') {
        if (!revisedQuestionText.trim()) {
          toast.error('Revised question cannot be empty');
          setSubmitting(false);
          return;
        }
        payload = {
          action: 'revise',
          question_number: editingQuestionNumber,
          question: revisedQuestionText.trim(),
          search_queries: revisedQueries,
        };
      } else if (action === 'reject') {
        payload = {
          action: 'reject',
          reason: reasonInput.trim() || 'Rejected by human reviewer.',
        };
      } else if (action === 'research_more') {
        payload = {
          action: 'research_more',
          reason: reasonInput.trim() || 'Reviewer requested additional research passes.',
        };
      }

      await api.submitReview(sessionId, payload);
      toast.success('Review decision submitted. Graph resumption scheduled!', 'Decision Accepted');
      setActiveAction(null);
      setEditingQuestionNumber(null);
      setReasonInput('');
      onReviewSubmitted();
    } catch (err) {
      toast.error(err.message || 'Failed to submit review decision', 'Submission Failed');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="rounded-2xl border-2 border-amber-400 dark:border-amber-600/80 bg-amber-50/70 dark:bg-amber-950/40 p-5 sm:p-6 shadow-xl backdrop-blur-md">
      {/* Header Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-amber-200 dark:border-amber-800/80 gap-3">
        <div className="flex items-start gap-3">
          <div className="p-2.5 rounded-xl bg-amber-500 text-white shadow-md shadow-amber-500/30 shrink-0">
            <ShieldAlert className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold uppercase tracking-wider text-amber-700 dark:text-amber-400">
                Human-in-the-Loop Checkpoint
              </span>
              <span className="px-2 py-0.5 rounded-full text-[10px] font-mono bg-amber-200 dark:bg-amber-900 text-amber-800 dark:text-amber-200">
                {reviewType}
              </span>
            </div>
            <h2 className="text-base sm:text-lg font-bold text-slate-900 dark:text-slate-100 mt-0.5">
              {review.title || 'Review Required'}
            </h2>
            <p className="text-xs sm:text-sm text-slate-600 dark:text-slate-300 mt-1 max-w-2xl leading-relaxed">
              {review.message}
            </p>
          </div>
        </div>
      </div>

      {/* Review Type: INITIAL PLAN REVIEW */}
      {reviewType === 'initial_plan_review' && (
        <div className="mt-5 space-y-4">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-700 dark:text-slate-300">
            Proposed Research Plan ({questionsList.length} Questions)
          </h3>

          <div className="space-y-3">
            {questionsList.map((q, idx) => {
              const qNum = q.question_number || idx + 1;
              const isBeingRevised = editingQuestionNumber === qNum && activeAction === 'revise';

              return (
                <div
                  key={qNum}
                  className={`p-4 rounded-xl border transition-all ${
                    isBeingRevised
                      ? 'border-indigo-400 bg-white dark:border-indigo-600 dark:bg-slate-900 shadow-md ring-2 ring-indigo-500/20'
                      : 'border-slate-200 dark:border-slate-800 bg-white/80 dark:bg-slate-900/60'
                  }`}
                >
                  {isBeingRevised ? (
                    /* Inline Question & Query Editor */
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-bold text-indigo-600 dark:text-indigo-400">
                          Revising Question #{qNum}
                        </span>
                        <button
                          type="button"
                          onClick={() => {
                            setActiveAction(null);
                            setEditingQuestionNumber(null);
                          }}
                          className="text-xs text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"
                        >
                          Cancel
                        </button>
                      </div>

                      <textarea
                        rows={2}
                        value={revisedQuestionText}
                        onChange={(e) => setRevisedQuestionText(e.target.value)}
                        placeholder="Edit question wording..."
                        className="w-full p-2.5 rounded-lg border border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-950 text-xs sm:text-sm text-slate-900 dark:text-slate-100 outline-hidden focus:border-indigo-500"
                      />

                      {/* Queries editor */}
                      <div>
                        <span className="text-[11px] font-semibold text-slate-500 dark:text-slate-400 block mb-1">
                          Search Queries:
                        </span>
                        <div className="flex flex-wrap gap-1.5 mb-2">
                          {revisedQueries.map((query, qIdx) => (
                            <span
                              key={qIdx}
                              className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-xs bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 border border-slate-200 dark:border-slate-700"
                            >
                              <span>{query}</span>
                              <button
                                type="button"
                                onClick={() => handleRemoveQuery(qIdx)}
                                className="text-slate-400 hover:text-rose-500"
                              >
                                <Trash2 className="w-3 h-3" />
                              </button>
                            </span>
                          ))}
                        </div>

                        <div className="flex gap-2">
                          <input
                            type="text"
                            value={newQueryInput}
                            onChange={(e) => setNewQueryInput(e.target.value)}
                            onKeyDown={(e) => {
                              if (e.key === 'Enter') {
                                e.preventDefault();
                                handleAddQuery();
                              }
                            }}
                            placeholder="Add search query tag..."
                            className="flex-1 px-2.5 py-1 text-xs rounded-lg border border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100"
                          />
                          <Button size="sm" variant="secondary" onClick={handleAddQuery}>
                            Add Query
                          </Button>
                        </div>
                      </div>

                      <div className="flex justify-end gap-2 pt-2 border-t border-slate-100 dark:border-slate-800">
                        <Button
                          size="sm"
                          variant="primary"
                          icon={CheckCircle2}
                          isLoading={submitting}
                          onClick={() => handleDecision('revise')}
                        >
                          Save Revision & Update Plan
                        </Button>
                      </div>
                    </div>
                  ) : (
                    /* Display Question Card */
                    <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-3">
                      <div className="space-y-1.5 flex-1">
                        <div className="flex items-center gap-2">
                          <span className="px-2 py-0.5 rounded-md text-xs font-bold bg-indigo-100 dark:bg-indigo-950 text-indigo-700 dark:text-indigo-300">
                            Q{qNum}
                          </span>
                          <span className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                            {q.question}
                          </span>
                        </div>

                        {q.search_queries && q.search_queries.length > 0 && (
                          <div className="flex flex-wrap items-center gap-1.5 pt-1">
                            <Tag className="w-3 h-3 text-slate-400" />
                            {q.search_queries.map((sq, sqIdx) => (
                              <span
                                key={sqIdx}
                                className="px-2 py-0.5 rounded-md text-[11px] bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400"
                              >
                                {sq}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>

                      <Button
                        size="sm"
                        variant="ghost"
                        icon={Edit3}
                        onClick={() => handleStartRevise(q)}
                        disabled={submitting}
                        className="shrink-0 text-slate-600 dark:text-slate-400"
                      >
                        Revise
                      </Button>
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Rejection reason input if active */}
          {activeAction === 'reject' && (
            <div className="p-4 rounded-xl border border-rose-300 dark:border-rose-900 bg-rose-50/60 dark:bg-rose-950/30 space-y-3">
              <span className="text-xs font-semibold text-rose-800 dark:text-rose-300 block">
                Provide Rejection Reason (Optional):
              </span>
              <textarea
                rows={2}
                value={reasonInput}
                onChange={(e) => setReasonInput(e.target.value)}
                placeholder="Reason for rejecting this research plan..."
                className="w-full p-2.5 text-xs rounded-lg border border-rose-200 dark:border-rose-800 bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 outline-hidden"
              />
              <div className="flex justify-end gap-2">
                <Button size="sm" variant="ghost" onClick={() => setActiveAction(null)}>
                  Cancel
                </Button>
                <Button
                  size="sm"
                  variant="danger"
                  icon={XCircle}
                  isLoading={submitting}
                  onClick={() => handleDecision('reject')}
                >
                  Confirm Rejection
                </Button>
              </div>
            </div>
          )}

          {/* Action Bar */}
          {activeAction !== 'reject' && activeAction !== 'revise' && (
            <div className="flex flex-wrap items-center justify-end gap-3 pt-4 border-t border-amber-200 dark:border-amber-800/80">
              <Button
                variant="outline"
                size="md"
                icon={XCircle}
                disabled={submitting}
                onClick={() => setActiveAction('reject')}
                className="border-rose-300 dark:border-rose-800 text-rose-700 dark:text-rose-300 hover:bg-rose-50 dark:hover:bg-rose-950/40"
              >
                Reject Plan
              </Button>

              <Button
                variant="success"
                size="md"
                icon={CheckCircle2}
                isLoading={submitting}
                onClick={() => handleDecision('approve')}
                className="shadow-md shadow-emerald-500/20"
              >
                Approve Plan & Proceed
              </Button>
            </div>
          )}
        </div>
      )}

      {/* Review Type: INITIAL RESEARCH REVIEW */}
      {reviewType === 'initial_research_review' && (
        <div className="mt-5 space-y-4">
          <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-white/80 dark:bg-slate-900/60 text-xs sm:text-sm text-slate-700 dark:text-slate-300 leading-relaxed">
            <p>
              The research plan has been accepted. Authorize the agent to begin connecting to external web search APIs, downloading relevant pages, and extracting verified evidence quotes for each sub-question.
            </p>
            <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
              Note: This approval grants blanket permission for this research session so you will not be prompted for every individual search query or page read.
            </p>
          </div>

          {activeAction === 'reject' && (
            <div className="p-4 rounded-xl border border-rose-300 dark:border-rose-900 bg-rose-50/60 dark:bg-rose-950/30 space-y-3">
              <span className="text-xs font-semibold text-rose-800 dark:text-rose-300 block">
                Provide Rejection Reason:
              </span>
              <textarea
                rows={2}
                value={reasonInput}
                onChange={(e) => setReasonInput(e.target.value)}
                placeholder="Reason for withholding research authorization..."
                className="w-full p-2.5 text-xs rounded-lg border border-rose-200 dark:border-rose-800 bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 outline-hidden"
              />
              <div className="flex justify-end gap-2">
                <Button size="sm" variant="ghost" onClick={() => setActiveAction(null)}>
                  Cancel
                </Button>
                <Button
                  size="sm"
                  variant="danger"
                  isLoading={submitting}
                  onClick={() => handleDecision('reject')}
                >
                  Confirm Rejection
                </Button>
              </div>
            </div>
          )}

          {activeAction !== 'reject' && (
            <div className="flex flex-wrap items-center justify-end gap-3 pt-3 border-t border-amber-200 dark:border-amber-800/80">
              <Button
                variant="outline"
                size="md"
                icon={XCircle}
                disabled={submitting}
                onClick={() => setActiveAction('reject')}
                className="border-rose-300 text-rose-700 dark:border-rose-800 dark:text-rose-300"
              >
                Decline Authorization
              </Button>

              <Button
                variant="success"
                size="md"
                icon={CheckCircle2}
                isLoading={submitting}
                onClick={() => handleDecision('approve')}
                className="shadow-md shadow-emerald-500/20"
              >
                Authorize Web Research Phase
              </Button>
            </div>
          )}
        </div>
      )}

      {/* Review Type: ADAPTIVE RESEARCH REVIEW */}
      {reviewType === 'adaptive_research_review' && (
        <div className="mt-5 space-y-4">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-700 dark:text-slate-300">
            Proposed Follow-up Questions ({questionsList.length} Proposed)
          </h3>

          <div className="space-y-3">
            {questionsList.map((q, idx) => (
              <div
                key={idx}
                className="p-3.5 rounded-xl border border-indigo-200 dark:border-indigo-900/60 bg-white/80 dark:bg-slate-900/60 space-y-1.5"
              >
                <div className="flex items-center gap-2">
                  <span className="px-2 py-0.5 rounded-md text-xs font-bold bg-indigo-100 dark:bg-indigo-950 text-indigo-700 dark:text-indigo-300">
                    Follow-up #{idx + 1}
                  </span>
                  <span className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                    {q.question}
                  </span>
                </div>
                {q.search_queries && q.search_queries.length > 0 && (
                  <div className="flex flex-wrap gap-1.5 pt-1">
                    {q.search_queries.map((sq, sqIdx) => (
                      <span
                        key={sqIdx}
                        className="px-2 py-0.5 rounded-md text-[11px] bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400"
                      >
                        {sq}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>

          <div className="flex flex-wrap items-center justify-end gap-3 pt-3 border-t border-amber-200 dark:border-amber-800/80">
            <Button
              variant="secondary"
              size="md"
              icon={ArrowRight}
              disabled={submitting}
              onClick={() => handleDecision('reject')}
              title="Skip follow-ups and jump directly to final synthesis"
            >
              Skip Follow-ups (Synthesize Report)
            </Button>

            <Button
              variant="success"
              size="md"
              icon={CheckCircle2}
              isLoading={submitting}
              onClick={() => handleDecision('approve')}
            >
              Approve Follow-up Questions
            </Button>
          </div>
        </div>
      )}

      {/* Review Type: FINAL REPORT REVIEW */}
      {reviewType === 'final_report_review' && (
        <div className="mt-5 space-y-4">
          <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-white/80 dark:bg-slate-900/60 space-y-2">
            <h4 className="text-sm font-bold text-slate-900 dark:text-slate-100">
              {review.report?.title || 'Synthesized Research Report'}
            </h4>
            <p className="text-xs sm:text-sm text-slate-600 dark:text-slate-300 leading-relaxed line-clamp-3">
              {review.report?.summary}
            </p>
            <div className="pt-2 flex items-center gap-3 text-xs text-slate-500">
              <span>Verified Evidence Items: {review.verified_evidence_count || 'Yes'}</span>
              <span>•</span>
              <span>Research Round: {review.research_round || 1}</span>
            </div>
          </div>

          {activeAction === 'research_more' && (
            <div className="p-4 rounded-xl border border-indigo-300 dark:border-indigo-900 bg-indigo-50/60 dark:bg-indigo-950/30 space-y-3">
              <span className="text-xs font-semibold text-indigo-900 dark:text-indigo-300 block">
                Additional Research Instructions / Gaps to Address:
              </span>
              <textarea
                rows={2}
                value={reasonInput}
                onChange={(e) => setReasonInput(e.target.value)}
                placeholder="Specify what additional information or aspects to investigate..."
                className="w-full p-2.5 text-xs rounded-lg border border-indigo-200 dark:border-indigo-800 bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 outline-hidden"
              />
              <div className="flex justify-end gap-2">
                <Button size="sm" variant="ghost" onClick={() => setActiveAction(null)}>
                  Cancel
                </Button>
                <Button
                  size="sm"
                  variant="primary"
                  icon={RotateCcw}
                  isLoading={submitting}
                  onClick={() => handleDecision('research_more')}
                >
                  Submit & Run Additional Pass
                </Button>
              </div>
            </div>
          )}

          {activeAction !== 'research_more' && (
            <div className="flex flex-wrap items-center justify-end gap-3 pt-3 border-t border-amber-200 dark:border-amber-800/80">
              <Button
                variant="outline"
                size="md"
                icon={RotateCcw}
                disabled={submitting}
                onClick={() => setActiveAction('research_more')}
              >
                Request Additional Research
              </Button>

              <Button
                variant="success"
                size="md"
                icon={CheckCircle2}
                isLoading={submitting}
                onClick={() => handleDecision('approve')}
                className="shadow-md shadow-emerald-500/20"
              >
                Approve Report & Complete Session
              </Button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
