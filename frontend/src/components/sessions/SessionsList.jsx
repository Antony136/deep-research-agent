import { useState, useMemo } from 'react';
import {
  Search,
  RefreshCw,
  Clock,
  ArrowRight,
  Trash2,
  RotateCcw,
  AlertCircle,
  FileText,
  CheckCircle2,
  Activity,
  Layers,
} from 'lucide-react';
import { StatusBadge } from '../common/Badge';
import { Button } from '../common/Button';
import { Card } from '../common/Card';
import { CardSkeleton } from '../common/Skeleton';
import { EmptyState } from '../common/EmptyState';
import { useToast } from '../../context/ToastContext';
import { api } from '../../api/client';

export function SessionsList({
  sessions,
  loading,
  onRefresh,
  onSelectSession,
  onOpenNewSession,
}) {
  const toast = useToast();
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [deletingId, setDeletingId] = useState(null);
  const [retryingId, setRetryingId] = useState(null);

  // Status Counts
  const stats = useMemo(() => {
    const total = sessions.length;
    const waiting = sessions.filter((s) => s.status === 'waiting_for_review').length;
    const running = sessions.filter((s) => s.status === 'running').length;
    const completed = sessions.filter((s) => s.status === 'completed').length;
    const failed = sessions.filter((s) => s.status === 'failed').length;
    return { total, waiting, running, completed, failed };
  }, [sessions]);

  // Filtered Sessions
  const filteredSessions = useMemo(() => {
    return sessions.filter((session) => {
      const matchesSearch =
        !searchQuery ||
        session.question?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        session.id?.toLowerCase().includes(searchQuery.toLowerCase());

      const matchesStatus =
        statusFilter === 'all' || session.status === statusFilter;

      return matchesSearch && matchesStatus;
    });
  }, [sessions, searchQuery, statusFilter]);

  const handleDelete = async (e, sessionId) => {
    e.stopPropagation();
    if (!window.confirm('Are you sure you want to delete this research session?')) {
      return;
    }

    setDeletingId(sessionId);
    try {
      await api.deleteSession(sessionId);
      toast.success('Session deleted successfully', 'Deleted');
      onRefresh();
    } catch (err) {
      toast.error(err.message || 'Failed to delete session', 'Delete Failed');
    } finally {
      setDeletingId(null);
    }
  };

  const handleRetry = async (e, sessionId) => {
    e.stopPropagation();
    setRetryingId(sessionId);
    try {
      await api.retrySession(sessionId);
      toast.success('Research session retry scheduled', 'Resumed');
      onRefresh();
    } catch (err) {
      toast.error(err.message || 'Failed to retry session', 'Retry Failed');
    } finally {
      setRetryingId(null);
    }
  };

  const formatDate = (dateStr) => {
    if (!dateStr) return 'N/A';
    try {
      const date = new Date(dateStr);
      return new Intl.DateTimeFormat('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      }).format(date);
    } catch {
      return dateStr;
    }
  };

  const getRelativeTime = (dateStr) => {
    if (!dateStr) return '';
    try {
      const now = new Date();
      const past = new Date(dateStr);
      const diffMs = now - past;
      const diffSec = Math.floor(diffMs / 1000);
      const diffMin = Math.floor(diffSec / 60);
      const diffHour = Math.floor(diffMin / 60);
      const diffDay = Math.floor(diffHour / 24);

      if (diffSec < 60) return 'just now';
      if (diffMin < 60) return `${diffMin}m ago`;
      if (diffHour < 24) return `${diffHour}h ago`;
      if (diffDay < 7) return `${diffDay}d ago`;
      return formatDate(dateStr);
    } catch {
      return '';
    }
  };

  return (
    <div className="space-y-6">
      {/* Overview Stat Metrics */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 sm:gap-4">
        <div className="p-4 rounded-2xl border border-slate-200/80 bg-white/80 dark:border-slate-800/80 dark:bg-slate-900/80 backdrop-blur-sm shadow-xs">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-500 dark:text-slate-400">Total Sessions</span>
            <Layers className="w-4 h-4 text-indigo-500" />
          </div>
          <p className="text-2xl font-bold text-slate-900 dark:text-slate-100 mt-1">{stats.total}</p>
        </div>

        <div className="p-4 rounded-2xl border border-amber-200/70 bg-amber-50/50 dark:border-amber-900/40 dark:bg-amber-950/20 backdrop-blur-sm shadow-xs">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-amber-700 dark:text-amber-400">Needs Review</span>
            <AlertCircle className="w-4 h-4 text-amber-500" />
          </div>
          <p className="text-2xl font-bold text-amber-900 dark:text-amber-200 mt-1">{stats.waiting}</p>
        </div>

        <div className="p-4 rounded-2xl border border-sky-200/70 bg-sky-50/50 dark:border-sky-900/40 dark:bg-sky-950/20 backdrop-blur-sm shadow-xs">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-sky-700 dark:text-sky-400">Active Running</span>
            <Activity className="w-4 h-4 text-sky-500 animate-pulse" />
          </div>
          <p className="text-2xl font-bold text-sky-900 dark:text-sky-200 mt-1">{stats.running}</p>
        </div>

        <div className="p-4 rounded-2xl border border-emerald-200/70 bg-emerald-50/50 dark:border-emerald-900/40 dark:bg-emerald-950/20 backdrop-blur-sm shadow-xs">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-emerald-700 dark:text-emerald-400">Completed</span>
            <CheckCircle2 className="w-4 h-4 text-emerald-500" />
          </div>
          <p className="text-2xl font-bold text-emerald-900 dark:text-emerald-200 mt-1">{stats.completed}</p>
        </div>
      </div>

      {/* Search, Filter Tabs & Toolbar */}
      <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3">
        {/* Search bar */}
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search questions or session IDs..."
            className="w-full pl-10 pr-4 py-2 rounded-xl text-xs sm:text-sm border border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900 text-slate-900 dark:text-slate-100 placeholder:text-slate-400 focus:outline-hidden focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 shadow-xs"
          />
        </div>

        {/* Status filters */}
        <div className="flex items-center gap-1.5 overflow-x-auto pb-1 sm:pb-0">
          {[
            { id: 'all', label: 'All' },
            { id: 'waiting_for_review', label: 'Review Required' },
            { id: 'running', label: 'Running' },
            { id: 'completed', label: 'Completed' },
            { id: 'failed', label: 'Failed' },
          ].map((tab) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => setStatusFilter(tab.id)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap transition-colors cursor-pointer ${
                statusFilter === tab.id
                  ? 'bg-indigo-600 text-white shadow-xs dark:bg-indigo-500'
                  : 'bg-slate-100 dark:bg-slate-800/80 text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
              }`}
            >
              {tab.label}
            </button>
          ))}

          <Button
            variant="outline"
            size="sm"
            icon={RefreshCw}
            isLoading={loading}
            onClick={onRefresh}
            title="Refresh sessions list"
            className="shrink-0"
          >
            <span className="hidden sm:inline">Refresh</span>
          </Button>
        </div>
      </div>

      {/* Sessions Cards Grid / List */}
      {loading && sessions.length === 0 ? (
        <div className="grid grid-cols-1 gap-3.5">
          <CardSkeleton />
          <CardSkeleton />
          <CardSkeleton />
        </div>
      ) : filteredSessions.length === 0 ? (
        <EmptyState
          icon={FileText}
          title={searchQuery || statusFilter !== 'all' ? 'No matching sessions' : 'No research sessions found'}
          description={
            searchQuery || statusFilter !== 'all'
              ? 'Try modifying your search term or clearing the active filter.'
              : 'Start your first autonomous research workflow to explore web sources, extract verified evidence, and generate synthesized reports.'
          }
          actionText={!searchQuery && statusFilter === 'all' ? 'Start New Research' : undefined}
          onAction={!searchQuery && statusFilter === 'all' ? onOpenNewSession : undefined}
        />
      ) : (
        <div className="grid grid-cols-1 gap-3.5">
          {filteredSessions.map((session) => {
            const isWaiting = session.status === 'waiting_for_review';
            const isRunning = session.status === 'running';

            return (
              <Card
                key={session.id}
                hover
                onClick={() => onSelectSession(session.id)}
                className={`relative group transition-all duration-200 ${
                  isWaiting
                    ? 'border-amber-300 dark:border-amber-800/80 bg-gradient-to-r from-amber-500/[0.03] to-transparent'
                    : isRunning
                    ? 'border-sky-300 dark:border-sky-800/80 bg-gradient-to-r from-sky-500/[0.03] to-transparent'
                    : ''
                }`}
              >
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                  <div className="space-y-1.5 flex-1 min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <StatusBadge status={session.status} size="xs" />
                      <span className="text-[11px] font-mono text-slate-400 dark:text-slate-500">
                        {session.id.substring(0, 8)}...
                      </span>
                      <span className="text-slate-300 dark:text-slate-700">•</span>
                      <span className="text-[11px] text-slate-500 dark:text-slate-400 flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {getRelativeTime(session.updated_at || session.created_at)}
                      </span>
                    </div>

                    <h3 className="text-sm sm:text-base font-semibold text-slate-900 dark:text-slate-100 line-clamp-2 group-hover:text-indigo-600 dark:group-hover:text-indigo-400 transition-colors">
                      {session.question}
                    </h3>
                  </div>

                  {/* Actions Bar */}
                  <div className="flex items-center gap-2 shrink-0 self-end sm:self-center">
                    {session.status === 'failed' && (
                      <Button
                        variant="secondary"
                        size="sm"
                        icon={RotateCcw}
                        isLoading={retryingId === session.id}
                        onClick={(e) => handleRetry(e, session.id)}
                        title="Retry session"
                      >
                        Retry
                      </Button>
                    )}

                    <button
                      type="button"
                      onClick={(e) => handleDelete(e, session.id)}
                      disabled={deletingId === session.id}
                      title="Delete session"
                      className="p-1.5 rounded-lg text-slate-400 hover:text-rose-600 hover:bg-rose-50 dark:hover:bg-rose-950/40 transition-colors"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>

                    <Button
                      variant={isWaiting ? 'primary' : 'outline'}
                      size="sm"
                      icon={ArrowRight}
                      onClick={() => onSelectSession(session.id)}
                    >
                      {isWaiting ? 'Review Plan' : 'View Session'}
                    </Button>
                  </div>
                </div>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}
