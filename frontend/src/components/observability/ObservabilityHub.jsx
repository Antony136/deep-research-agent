import { useState, useEffect, useCallback } from 'react';
import {
  Activity,
  Cpu,
  RefreshCw,
  Server,
  Layers,
  Terminal,
  ExternalLink,
} from 'lucide-react';
import { Button } from '../common/Button';
import { useToast } from '../../context/ToastContext';
import { api } from '../../api/client';

export function ObservabilityHub({ onSelectSession }) {
  const toast = useToast();
  const [metrics, setMetrics] = useState(null);
  const [events, setEvents] = useState([]);
  const [health, setHealth] = useState(null);
  const [refreshing, setRefreshing] = useState(false);
  const [levelFilter, setLevelFilter] = useState('all');

  const loadObservabilityData = useCallback(async () => {
    setRefreshing(true);
    try {
      const [healthData, metricsData, eventsData] = await Promise.allSettled([
        api.healthCheck(),
        api.getMetrics(),
        api.getGlobalEvents(100),
      ]);

      if (healthData.status === 'fulfilled') setHealth(healthData.value);
      if (metricsData.status === 'fulfilled') setMetrics(metricsData.value);
      if (eventsData.status === 'fulfilled') setEvents(eventsData.value.events || []);
    } catch {
      toast.error('Failed to refresh observability data');
    } finally {
      setRefreshing(false);
    }
  }, [toast]);

  useEffect(() => {
    loadObservabilityData();
    const interval = setInterval(loadObservabilityData, 30000);
    return () => clearInterval(interval);
  }, [loadObservabilityData]);

  const nodes = metrics?.nodes || {};
  const nodeNames = Object.keys(nodes);

  const filteredEvents = events.filter((e) => {
    if (levelFilter === 'all') return true;
    return e.log_level?.toUpperCase() === levelFilter.toUpperCase();
  });

  const formatDuration = (ms) => {
    if (ms === null || ms === undefined) return '-';
    if (ms < 1000) return `${Math.round(ms)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-12">
      {/* Top Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-200/80 dark:border-slate-800/80">
        <div>
          <h2 className="text-lg font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
            <Activity className="w-5 h-5 text-indigo-500" />
            <span>Observability & Tracing Hub</span>
          </h2>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
            Real-time execution telemetry, node durations, and PostgreSQL structured log stream.
          </p>
        </div>

        <Button
          size="sm"
          variant="outline"
          icon={RefreshCw}
          isLoading={refreshing}
          onClick={loadObservabilityData}
        >
          Refresh Telemetry
        </Button>
      </div>

      {/* High-Level Server Health Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="p-4 rounded-2xl border border-slate-200/80 bg-white/80 dark:border-slate-800/80 dark:bg-slate-900/80 shadow-xs backdrop-blur-sm">
          <div className="flex items-center justify-between text-xs font-medium text-slate-500">
            <span>API Gateway</span>
            <Server className="w-4 h-4 text-emerald-500" />
          </div>
          <div className="mt-2 flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-base font-bold text-slate-900 dark:text-slate-100">
              {health?.status === 'ok' ? 'Online & Healthy' : 'Checking Status...'}
            </span>
          </div>
          <p className="text-[11px] text-slate-400 mt-1 font-mono">Service: {health?.service || 'deep-research-agent'}</p>
        </div>

        <div className="p-4 rounded-2xl border border-slate-200/80 bg-white/80 dark:border-slate-800/80 dark:bg-slate-900/80 shadow-xs backdrop-blur-sm">
          <div className="flex items-center justify-between text-xs font-medium text-slate-500">
            <span>Graph Nodes Tracked</span>
            <Cpu className="w-4 h-4 text-indigo-500" />
          </div>
          <p className="text-2xl font-bold text-slate-900 dark:text-slate-100 mt-1">
            {nodeNames.length || '11'} Nodes
          </p>
          <p className="text-[11px] text-slate-400 mt-1">Active execution tracing enabled</p>
        </div>

        <div className="p-4 rounded-2xl border border-slate-200/80 bg-white/80 dark:border-slate-800/80 dark:bg-slate-900/80 shadow-xs backdrop-blur-sm">
          <div className="flex items-center justify-between text-xs font-medium text-slate-500">
            <span>Total Recorded Events</span>
            <Layers className="w-4 h-4 text-purple-500" />
          </div>
          <p className="text-2xl font-bold text-slate-900 dark:text-slate-100 mt-1">
            {events.length} Events
          </p>
          <p className="text-[11px] text-slate-400 mt-1">Persisted in PostgreSQL database</p>
        </div>
      </div>

      {/* Node Execution Metrics Table */}
      {nodeNames.length > 0 && (
        <div className="rounded-2xl border border-slate-200/80 bg-white/80 p-5 dark:border-slate-800/80 dark:bg-slate-900/80 shadow-xs backdrop-blur-sm space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400 flex items-center gap-2">
              <Cpu className="w-3.5 h-3.5 text-indigo-500" />
              <span>Node Execution Metrics Snapshot</span>
            </h3>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-sans">
              <thead>
                <tr className="border-b border-slate-100 dark:border-slate-800 text-slate-400 uppercase text-[10px] font-semibold tracking-wider">
                  <th className="py-2.5 px-3">Node Name</th>
                  <th className="py-2.5 px-3">Invocations</th>
                  <th className="py-2.5 px-3">Successes</th>
                  <th className="py-2.5 px-3">Failures</th>
                  <th className="py-2.5 px-3">Total Duration</th>
                  <th className="py-2.5 px-3">Last Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800/60 font-mono">
                {nodeNames.map((nodeName) => {
                  const node = nodes[nodeName];
                  return (
                    <tr key={nodeName} className="hover:bg-slate-50 dark:hover:bg-slate-800/40 transition-colors">
                      <td className="py-2.5 px-3 font-semibold text-slate-900 dark:text-slate-100">
                        {nodeName}
                      </td>
                      <td className="py-2.5 px-3 text-slate-600 dark:text-slate-400">{node.calls}</td>
                      <td className="py-2.5 px-3 text-emerald-600 dark:text-emerald-400">{node.successes}</td>
                      <td className="py-2.5 px-3 text-rose-600 dark:text-rose-400">{node.failures}</td>
                      <td className="py-2.5 px-3 text-slate-600 dark:text-slate-400">{formatDuration(node.duration_ms)}</td>
                      <td className="py-2.5 px-3">
                        <span
                          className={`px-1.5 py-0.5 rounded-xs text-[10px] font-bold ${
                            node.last_status === 'success'
                              ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300'
                              : 'bg-rose-100 text-rose-800 dark:bg-rose-950 dark:text-rose-300'
                          }`}
                        >
                          {node.last_status || 'ok'}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Global Event Stream */}
      <div className="rounded-2xl border border-slate-200/80 bg-white/80 p-5 dark:border-slate-800/80 dark:bg-slate-900/80 shadow-xs backdrop-blur-sm space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-slate-100 dark:border-slate-800 gap-2">
          <div className="flex items-center gap-2">
            <Terminal className="w-4 h-4 text-indigo-500" />
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-300">
              Global Observability Log Stream ({filteredEvents.length})
            </h3>
          </div>

          <div className="flex items-center gap-1.5">
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
        </div>

        <div className="space-y-2 font-mono text-xs">
          {filteredEvents.map((evt) => {
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
                <div className="flex items-center justify-between gap-2 flex-wrap">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-[11px] text-slate-400">
                      {new Date(evt.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                    </span>
                    <span className={`px-1.5 py-0.2 rounded-xs text-[10px] font-bold ${levelColor}`}>
                      {evt.log_level}
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

                  <div className="flex items-center gap-2">
                    {evt.duration_ms && (
                      <span className="px-1.5 py-0.5 rounded-md text-[10px] bg-slate-100 dark:bg-slate-800 text-slate-500">
                        {formatDuration(evt.duration_ms)}
                      </span>
                    )}

                    {evt.session_id && (
                      <button
                        type="button"
                        onClick={() => onSelectSession(evt.session_id)}
                        className="inline-flex items-center gap-1 text-[11px] text-indigo-600 dark:text-indigo-400 hover:underline cursor-pointer"
                        title="Open session workbench"
                      >
                        <span>Session {evt.session_id.substring(0, 8)}...</span>
                        <ExternalLink className="w-2.5 h-2.5" />
                      </button>
                    )}
                  </div>
                </div>

                <p className="text-slate-600 dark:text-slate-400 pl-1 text-[11px] font-sans">
                  {evt.message}
                </p>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
