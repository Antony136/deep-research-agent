import { useState, useEffect, useCallback, useRef } from 'react';
import {
  ArrowLeft,
  RefreshCw,
  RotateCcw,
  Trash2,
  Clock,
  CheckCircle2,
  AlertOctagon,
  FileText,
  FileCheck2,
  HelpCircle,
  Activity,
  Code2,
  Copy,
  Check,
  Radio,
} from 'lucide-react';
import { StatusBadge } from '../common/Badge';
import { Button } from '../common/Button';
import { CardSkeleton } from '../common/Skeleton';
import { WorkflowGraph } from './WorkflowGraph';
import { ReviewPanel } from './ReviewPanel';
import { ReportViewer } from './ReportViewer';
import { EvidenceExplorer } from './EvidenceExplorer';
import { QuestionsList } from './QuestionsList';
import { SessionEventsLog } from './SessionEventsLog';
import { useToast } from '../../context/ToastContext';
import { api } from '../../api/client';

export function SessionWorkbench({
  sessionId,
  onBack,
  onSessionDeleted,
}) {
  const toast = useToast();
  const [session, setSession] = useState(null);
  const [pendingReview, setPendingReview] = useState(null);
  const [events, setEvents] = useState([]);
  const [eventsError, setEventsError] = useState(null);
  const eventsErrorNotified = useRef(false);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [retrying, setRetrying] = useState(false);
  const [activeTab, setActiveTab] = useState('report');
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [copiedId, setCopiedId] = useState(false);
  const [selectedEvidenceNum, setSelectedEvidenceNum] = useState(null);

  // Fetch Session Data
  const loadSessionData = useCallback(async (isPolling = false) => {
    if (!sessionId) return;
    if (!isPolling) setRefreshing(true);

    try {
      const sessionData = await api.getSession(sessionId);
      setSession(sessionData);

      // Check for pending review if session is waiting_for_review
      if (sessionData.status === 'waiting_for_review') {
        const reviewData = await api.getPendingReview(sessionId);
        setPendingReview(reviewData?.review || null);
      } else {
        setPendingReview(null);
      }

      // Load observability events
      try {
        const eventsData = await api.getSessionEvents(sessionId);
        setEvents(eventsData.events || []);
        setEventsError(null);
        eventsErrorNotified.current = false;
      } catch (err) {
        setEventsError(err.message || 'Could not retrieve node events.');
        if (!eventsErrorNotified.current) {
          toast.error(err.message || 'Could not retrieve node events.');
          eventsErrorNotified.current = true;
        }
      }
    } catch (err) {
      if (!isPolling) {
        toast.error(err.message || 'Failed to load session details');
      }
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [sessionId, toast]);

  useEffect(() => {
    loadSessionData();
  }, [loadSessionData]);

  // Polling loop when running or waiting for review
  useEffect(() => {
    if (!autoRefresh) return;
    if (!['created', 'running', 'waiting_for_review'].includes(session?.status)) return;

    const interval = setInterval(() => {
      loadSessionData(true);
    }, 2000);

    return () => clearInterval(interval);
  }, [autoRefresh, session?.status, loadSessionData]);

  const handleCopyId = async () => {
    try {
      await navigator.clipboard.writeText(sessionId);
      setCopiedId(true);
      toast.success('Session ID copied to clipboard');
      setTimeout(() => setCopiedId(false), 2000);
    } catch {
      toast.error('Failed to copy ID');
    }
  };

  const handleRetry = async () => {
    setRetrying(true);
    try {
      await api.retrySession(sessionId);
      toast.success('Session retry scheduled', 'Resumed');
      loadSessionData();
    } catch (err) {
      toast.error(err.message || 'Retry failed');
    } finally {
      setRetrying(false);
    }
  };

  const handleDelete = async () => {
    if (!window.confirm('Delete this research session and all its stored results?')) {
      return;
    }
    try {
      await api.deleteSession(sessionId);
      toast.success('Research session deleted');
      onSessionDeleted();
    } catch (err) {
      toast.error(err.message || 'Failed to delete session');
    }
  };

  if (loading && !session) {
    return (
      <div className="space-y-4 max-w-7xl mx-auto">
        <CardSkeleton />
        <CardSkeleton />
      </div>
    );
  }

  if (!session) {
    return (
      <div className="p-12 text-center max-w-lg mx-auto rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900">
        <AlertOctagon className="w-12 h-12 text-rose-500 mx-auto mb-3" />
        <h3 className="text-lg font-bold text-slate-900 dark:text-slate-100">Session Not Found</h3>
        <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
          The requested research session could not be found or has been deleted.
        </p>
        <div className="mt-5">
          <Button variant="primary" icon={ArrowLeft} onClick={onBack}>
            Back to Sessions
          </Button>
        </div>
      </div>
    );
  }

  const researchState = session.research_state || {};
  const report = session.final_report || researchState.report;
  const questions = researchState.research_questions || [];
  const evidence = researchState.evidence || [];
  const currentQIndex = researchState.current_question_index || 0;

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-12">
      {/* Top Navigation & Toolbar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-200/80 dark:border-slate-800/80">
        <button
          type="button"
          onClick={onBack}
          className="inline-flex items-center gap-1.5 text-xs font-semibold text-slate-500 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-100 transition-colors w-fit cursor-pointer"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Back to All Sessions</span>
        </button>

        <div className="flex items-center gap-2 self-end sm:self-auto">
          {/* Auto-refresh toggle */}
          {(['created', 'running', 'waiting_for_review'].includes(session.status)) && (
            <button
              type="button"
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium border transition-colors cursor-pointer ${
                autoRefresh
                  ? 'bg-sky-50 text-sky-700 border-sky-200 dark:bg-sky-950/60 dark:text-sky-300 dark:border-sky-800'
                  : 'bg-slate-100 text-slate-600 border-slate-200 dark:bg-slate-800 dark:text-slate-400 dark:border-slate-700'
              }`}
              title="Toggle automatic state polling"
            >
              <Radio className={`w-3 h-3 ${autoRefresh ? 'text-sky-500 animate-pulse' : 'text-slate-400'}`} />
              <span>{autoRefresh ? 'Live Sync' : 'Sync Paused'}</span>
            </button>
          )}

          <Button
            size="sm"
            variant="outline"
            icon={RefreshCw}
            isLoading={refreshing}
            onClick={() => loadSessionData(false)}
            title="Refresh session data"
          >
            Refresh
          </Button>

          {session.status === 'failed' && (
            <Button
              size="sm"
              variant="secondary"
              icon={RotateCcw}
              isLoading={retrying}
              onClick={handleRetry}
            >
              Retry
            </Button>
          )}

          <button
            type="button"
            onClick={handleDelete}
            title="Delete session"
            className="p-1.5 rounded-lg text-slate-400 hover:text-rose-600 hover:bg-rose-50 dark:hover:bg-rose-950/40 transition-colors cursor-pointer"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Session Title Header Card */}
      <div className="p-5 sm:p-6 rounded-2xl border border-slate-200/90 dark:border-slate-800 bg-white/90 dark:bg-slate-900/90 shadow-xs backdrop-blur-sm space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
          <div className="flex items-center gap-2">
            <StatusBadge status={session.status} size="sm" pulse />
            <button
              type="button"
              onClick={handleCopyId}
              className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md font-mono text-[11px] bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-400 transition-colors cursor-pointer"
              title="Click to copy full session ID"
            >
              <span>{sessionId}</span>
              {copiedId ? <Check className="w-3 h-3 text-emerald-500" /> : <Copy className="w-3 h-3" />}
            </button>
          </div>

          <div className="flex items-center gap-3 text-slate-500 text-[11px]">
            {session.created_at && (
              <span className="flex items-center gap-1">
                <Clock className="w-3 h-3" /> Started: {new Date(session.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </span>
            )}
            {session.completed_at && (
              <span className="flex items-center gap-1 text-emerald-600 dark:text-emerald-400 font-medium">
                <CheckCircle2 className="w-3 h-3" /> Finished
              </span>
            )}
          </div>
        </div>

        <h1 className="text-lg sm:text-xl font-bold text-slate-900 dark:text-white leading-snug">
          {session.question}
        </h1>

        {/* Failed Error Banner */}
        {session.status === 'failed' && (
          <div className="p-4 rounded-xl border border-rose-300 dark:border-rose-900/60 bg-rose-50/70 dark:bg-rose-950/30 flex items-start justify-between gap-3 text-xs sm:text-sm text-rose-800 dark:text-rose-200">
            <div className="flex items-start gap-2.5">
              <AlertOctagon className="w-5 h-5 text-rose-500 shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold">Workflow Execution Interrupted</p>
                <p className="mt-0.5 text-xs text-rose-700 dark:text-rose-300 font-mono">
                  {session.error_message || 'An unexpected runtime error occurred.'}
                </p>
              </div>
            </div>
            <Button size="sm" variant="danger" icon={RotateCcw} onClick={handleRetry} isLoading={retrying}>
              Retry Workflow
            </Button>
          </div>
        )}
      </div>

      {/* Human Review Checkpoint Action Panel (Shown prominently if waiting_for_review) */}
      {session.status === 'waiting_for_review' && pendingReview && (
        <ReviewPanel
          sessionId={sessionId}
          review={pendingReview}
          onReviewSubmitted={() => loadSessionData(false)}
        />
      )}

      {/* Agent Workflow Pipeline Visualizer */}
      <WorkflowGraph
        session={session}
        events={events}
        pendingReview={pendingReview}
      />

      {/* Workbench Tabbed Section */}
      <div className="space-y-4">
        <div className="border-b border-slate-200 dark:border-slate-800">
          <nav className="flex space-x-1 sm:space-x-4 overflow-x-auto">
            {[
              { id: 'report', label: 'Report & Synthesis', icon: FileText, count: report ? 'Ready' : null },
              { id: 'evidence', label: 'Verified Evidence', icon: FileCheck2, count: evidence.length },
              { id: 'questions', label: 'Research Plan', icon: HelpCircle, count: questions.length },
              { id: 'events', label: 'Node Logs', icon: Activity, count: events.length },
              { id: 'raw', label: 'Raw State', icon: Code2 },
            ].map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 py-3 px-3.5 border-b-2 text-xs sm:text-sm font-semibold transition-colors whitespace-nowrap cursor-pointer ${
                    isActive
                      ? 'border-indigo-600 text-indigo-600 dark:border-indigo-400 dark:text-indigo-400'
                      : 'border-transparent text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200 hover:border-slate-300'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{tab.label}</span>
                  {tab.count !== null && tab.count !== undefined && (
                    <span className="px-1.5 py-0.2 rounded-full text-[10px] bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 font-mono">
                      {tab.count}
                    </span>
                  )}
                </button>
              );
            })}
          </nav>
        </div>

        {/* Tab Content Panes */}
        <div className="pt-2">
          {activeTab === 'report' && (
            <ReportViewer
              report={report}
              evidence={evidence}
              onSelectEvidence={(eNum) => {
                setSelectedEvidenceNum(eNum);
                setActiveTab('evidence');
              }}
            />
          )}

          {activeTab === 'evidence' && (
            <EvidenceExplorer
              evidence={evidence}
              selectedEvidenceNumber={selectedEvidenceNum}
            />
          )}

          {activeTab === 'questions' && (
            <QuestionsList
              questions={questions}
              currentQuestionIndex={currentQIndex}
              evidence={evidence}
            />
          )}

          {activeTab === 'events' && (
            <SessionEventsLog
              events={events}
              error={eventsError}
              loading={refreshing}
              onRefresh={() => loadSessionData(false)}
            />
          )}

          {activeTab === 'raw' && (
            <div className="p-4 rounded-2xl border border-slate-200 dark:border-slate-800 bg-slate-900 text-slate-100 font-mono text-xs overflow-x-auto space-y-2">
              <div className="flex items-center justify-between text-slate-400 text-[11px] pb-2 border-b border-slate-800">
                <span>Complete LangGraph State Snapshot</span>
                <span>{JSON.stringify(session).length} bytes</span>
              </div>
              <pre>{JSON.stringify(session, null, 2)}</pre>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
