import { useState } from 'react';
import {
  ChevronDown,
  ChevronUp,
  RefreshCw,
  Terminal,
} from 'lucide-react';
import { Button } from '../common/Button';
import { EmptyState } from '../common/EmptyState';

export function SessionEventsLog({ events = [], error, loading, onRefresh }) {
  const [expandedId, setExpandedId] = useState(null);
  const [levelFilter, setLevelFilter] = useState('all');

  const filteredEvents = events.filter((e) => {
    if (levelFilter === 'all') return true;
    return e.log_level?.toUpperCase() === levelFilter.toUpperCase();
  });

  const formatDuration = (ms) => {
    if (ms === null || ms === undefined) return null;
    if (ms < 1000) return `${Math.round(ms)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  const formatTime = (dateStr) => {
    if (!dateStr) return '';
    try {
      const d = new Date(dateStr);
      return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    } catch {
      return '';
    }
  };

  if (events.length === 0 && !loading && !error) {
    return (
      <EmptyState
        icon={Terminal}
        title="No Observability Events Recorded"
        description="Node execution timings, status transitions, and agent logs will stream here as the research workflow advances."
      />
    );
  }

  return (
    <div className="space-y-4">
      {error && (
        <div
          role="alert"
          className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-xs text-rose-800 dark:border-rose-900/70 dark:bg-rose-950/30 dark:text-rose-200"
        >
          <p className="font-semibold">Could not load node logs</p>
          <p className="mt-1">{error}</p>
        </div>
      )}

      {/* Toolbar */}
      <div className="flex items-center justify-between gap-3 pb-2 border-b border-slate-200 dark:border-slate-800">
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
            Node Event Stream ({events.length})
          </span>
        </div>

        <div className="flex items-center gap-2">
          {/* Level Filter */}
          <div className="flex items-center gap-1">
            {['all', 'INFO', 'WARNING', 'ERROR'].map((lvl) => (
              <button
                key={lvl}
                type="button"
                onClick={() => setLevelFilter(lvl)}
                className={`px-2 py-0.5 rounded-md text-[11px] font-medium transition-colors cursor-pointer ${
                  levelFilter === lvl
                    ? 'bg-indigo-600 text-white dark:bg-indigo-500'
                    : 'bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400'
                }`}
              >
                {lvl}
              </button>
            ))}
          </div>

          <Button
            size="sm"
            variant="outline"
            icon={RefreshCw}
            isLoading={loading}
            onClick={onRefresh}
            title="Refresh events"
          >
            Refresh
          </Button>
        </div>
      </div>

      {/* Events Timeline */}
      <div className="space-y-2 font-mono text-xs">
        {filteredEvents.length === 0 && !loading ? (
          <p className="rounded-xl border border-dashed border-slate-200 px-4 py-8 text-center font-sans text-xs text-slate-500 dark:border-slate-800 dark:text-slate-400">
            {error ? 'Node logs are unavailable. Refresh to try again.' : 'No events match this filter.'}
          </p>
        ) : filteredEvents.map((evt) => {
          const isExpanded = expandedId === evt.id;
          const durationStr = formatDuration(evt.duration_ms);
          const hasDetails = evt.details && Object.keys(evt.details).length > 0;

          const levelColor = {
            INFO: 'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300',
            WARNING: 'bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300',
            ERROR: 'bg-rose-100 text-rose-800 dark:bg-rose-950 dark:text-rose-300',
          }[evt.log_level] || 'bg-slate-100 text-slate-700';

          return (
            <div
              key={evt.id}
              className="p-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-white/90 dark:bg-slate-900/80 shadow-xs space-y-1.5 transition-all"
            >
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-[11px] text-slate-400">
                    {formatTime(evt.created_at)}
                  </span>
                  <span className={`px-1.5 py-0.2 rounded-xs text-[10px] font-bold ${levelColor}`}>
                    {evt.log_level || 'INFO'}
                  </span>
                  {evt.node_name && (
                    <span className="px-2 py-0.5 rounded-md text-[11px] font-semibold bg-indigo-50 text-indigo-700 dark:bg-indigo-950 dark:text-indigo-300 border border-indigo-200 dark:border-indigo-800">
                      [{evt.node_name}]
                    </span>
                  )}
                  <span className="font-semibold text-slate-800 dark:text-slate-200">
                    {evt.event_type}
                  </span>
                </div>

                <div className="flex items-center gap-2 shrink-0">
                  {durationStr && (
                    <span className="px-2 py-0.5 rounded-md text-[11px] font-mono bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400">
                      {durationStr}
                    </span>
                  )}
                  {hasDetails && (
                    <button
                      type="button"
                      onClick={() => setExpandedId(isExpanded ? null : evt.id)}
                      className="p-1 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 cursor-pointer"
                    >
                      {isExpanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                    </button>
                  )}
                </div>
              </div>

              <p className="text-slate-600 dark:text-slate-400 pl-1 text-[11px] font-sans">
                {evt.message}
              </p>

              {/* JSON Details Drawer */}
              {isExpanded && hasDetails && (
                <div className="mt-2 p-2.5 rounded-lg bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 text-[11px] overflow-x-auto text-slate-700 dark:text-slate-300">
                  <pre>{JSON.stringify(evt.details, null, 2)}</pre>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
