import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  ArrowUp,
  CheckCircle2,
  Maximize2,
  Minus,
  Plus,
  RotateCcw,
  Workflow,
  XCircle,
} from 'lucide-react';
import { WORKFLOW_NODES } from './workflowNodes';

const NODE_WIDTH = 184;
const NODE_HEIGHT = 76;
const GRAPH_WIDTH = 2300;
const GRAPH_HEIGHT = 1060;

// Explicit routes keep the main path readable and reserve separate lanes
// for loops and review decisions.
const EDGES = [
  { from: '__start__', to: 'planner', d: 'M202 170 H288', label: 'Start' },
  { from: 'planner', to: 'initial_plan_review', d: 'M472 170 H558', label: 'Create plan' },
  { from: 'initial_plan_review', to: 'initial_research_review', d: 'M742 170 H828', label: 'Approve plan' },
  { from: 'initial_plan_review', to: 'initial_plan_review', d: 'M650 132 V105 H690 V132', label: 'Revise', labelX: 690, labelY: 92 },
  { from: 'initial_plan_review', to: '__end__', d: 'M650 208 V840 H1900 V930 H1908', label: 'Reject', labelX: 700, labelY: 824 },
  { from: 'initial_research_review', to: 'researcher', d: 'M1012 170 H1098', label: 'Authorize' },
  { from: 'initial_research_review', to: '__end__', d: 'M920 208 V875 H1900 V930 H1908', label: 'Stop', labelX: 965, labelY: 859 },
  { from: 'researcher', to: 'evidence_extractor', d: 'M1282 170 H1368', label: 'Sources' },
  { from: 'evidence_extractor', to: 'evidence_verifier', d: 'M1552 170 H1638', label: 'Extract claims' },
  { from: 'evidence_verifier', to: 'research_sufficiency', d: 'M1822 170 H1908', label: 'Verified evidence' },
  { from: 'research_sufficiency', to: 'researcher', d: 'M2000 132 V70 H1190 V132', label: 'Next question', labelX: 1580, labelY: 52 },
  { from: 'research_sufficiency', to: 'adaptive_planner', d: 'M2000 208 V402', label: 'Evidence gaps', labelX: 2070, labelY: 300 },
  { from: 'research_sufficiency', to: 'synthesizer', d: 'M2092 170 H2190 V402 H2092', label: 'Sufficient evidence', labelX: 2180, labelY: 290 },
  { from: 'adaptive_planner', to: 'adaptive_review', d: 'M1908 440 H1830 V642', label: 'Request review', labelX: 1840, labelY: 540 },
  { from: 'adaptive_planner', to: 'auto_accept_adaptive', d: 'M2000 478 V642', label: 'Auto-approve', labelX: 2070, labelY: 560 },
  { from: 'adaptive_review', to: 'researcher', d: 'M1638 680 H1570 V790 H1190 V208', label: 'Approved questions', labelX: 1390, labelY: 772 },
  { from: 'adaptive_review', to: 'synthesizer', d: 'M1822 680 H1870 V550 H2000 V478', label: 'Decline follow-ups', labelX: 1900, labelY: 530 },
  { from: 'auto_accept_adaptive', to: 'researcher', d: 'M2000 718 V825 H1190 V208', label: 'Continue research', labelX: 1570, labelY: 807 },
  { from: 'auto_accept_adaptive', to: 'synthesizer', d: 'M2092 680 H2190 V478 H2092', label: 'No follow-ups', labelX: 2190, labelY: 580 },
  { from: 'synthesizer', to: 'final_report_review', d: 'M2000 402 V208', label: 'Draft report', labelX: 2070, labelY: 310 },
  { from: 'final_report_review', to: 'adaptive_planner', d: 'M1908 170 H1870 V350 H2000 V402', label: 'Research more', labelX: 1880, labelY: 330 },
  { from: 'final_report_review', to: '__end__', d: 'M2092 208 H2220 V930 H2092', label: 'Approve report', labelX: 2180, labelY: 860 },
];

const STATUS_CLASSES = {
  completed: 'border-emerald-400 bg-emerald-50 text-emerald-950 dark:border-emerald-700 dark:bg-emerald-950 dark:text-emerald-100',
  active: 'border-sky-500 bg-sky-50 text-sky-950 ring-2 ring-sky-400/40 shadow-lg dark:border-sky-500 dark:bg-sky-950 dark:text-sky-100',
  waiting: 'border-amber-400 bg-amber-50 text-amber-950 ring-2 ring-amber-400/30 dark:border-amber-700 dark:bg-amber-950 dark:text-amber-100',
  failed: 'border-rose-400 bg-rose-50 text-rose-950 dark:border-rose-700 dark:bg-rose-950 dark:text-rose-100',
  pending: 'border-slate-200 bg-white text-slate-700 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300',
};

function getCurrentNode(events, status) {
  if (status !== 'running') return null;

  const latestStart = [...events].reverse().find(
    (event) => event.node_name && event.event_type === 'node_started',
  );

  if (!latestStart) return null;

  const finished = events.some(
    (event) =>
      event.node_name === latestStart.node_name &&
      event.id > latestStart.id &&
      ['node_completed', 'node_failed', 'node_interrupted'].includes(event.event_type),
  );

  return finished ? null : latestStart.node_name;
}

function IconButton({ label, onClick, children }) {
  return (
    <button
      type="button"
      title={label}
      aria-label={label}
      onClick={onClick}
      className="inline-flex h-9 w-9 items-center justify-center rounded-lg border border-slate-200 bg-white text-slate-600 hover:bg-slate-100 focus:outline-none focus:ring-2 focus:ring-indigo-500 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300 dark:hover:bg-slate-800"
    >
      {children}
    </button>
  );
}

export function WorkflowGraph({ session, pendingReview, events = [] }) {
  const viewportRef = useRef(null);
  const dragRef = useRef(null);
  const [zoom, setZoom] = useState(0.5);
  const [pan, setPan] = useState({ x: 16, y: 16 });
  const [dragging, setDragging] = useState(false);

  const status = session?.status;
  const researchState = useMemo(() => session?.research_state || {}, [session?.research_state]);
  const questions = researchState.research_questions || [];
  const reviewType = pendingReview?.type;
  const currentNode = getCurrentNode(events, status);

  const graphNodes = useMemo(() => WORKFLOW_NODES, []);

  const getNodeStatus = useCallback((node) => {
    if (node.id === '__start__') {
      return events.length || ['running', 'waiting_for_review', 'completed'].includes(status)
        ? 'completed'
        : 'pending';
    }

    if (node.id === '__end__') {
      return status === 'completed' ? 'completed' : 'pending';
    }

    const reviewNode = reviewType === 'adaptive_research_review'
      ? 'adaptive_review'
      : reviewType;

    if (status === 'waiting_for_review' && node.id === reviewNode) {
      return 'waiting';
    }

    const nodeEvents = events.filter((event) => event.node_name === node.id);
    const lastEvent = nodeEvents.at(-1)?.event_type;

    if (lastEvent === 'node_failed') return 'failed';
    if (lastEvent === 'node_interrupted') return 'waiting';
    if (currentNode === node.id) return 'active';
    if (lastEvent === 'node_completed') return 'completed';

    if (node.id === 'planner' && questions.length) return 'completed';
    if (node.id === 'initial_plan_review' && researchState.initial_plan_approved) return 'completed';
    if (node.id === 'initial_research_review' && researchState.research_authorized) return 'completed';
    if (node.id === 'evidence_verifier' && (researchState.evidence || []).length) return 'completed';
    if (node.id === 'synthesizer' && (session?.final_report || researchState.report)) return 'completed';
    if (node.id === 'final_report_review' && researchState.final_report_approved) return 'completed';

    return 'pending';
  }, [currentNode, events, questions.length, researchState, reviewType, session, status]);

  const zoomBy = (amount) => {
    setZoom((value) => Math.min(1.25, Math.max(0.25, Number((value + amount).toFixed(2)))));
  };

  const panBy = (dx, dy) => {
    setPan((value) => ({ x: value.x + dx, y: value.y + dy }));
  };

  const fitGraph = useCallback(() => {
    const viewport = viewportRef.current;
    if (!viewport) return;

    const padding = 48;
    const nextZoom = Math.min(
      1.25,
      Math.max(0.25, Math.min(
        (viewport.clientWidth - padding) / GRAPH_WIDTH,
        (viewport.clientHeight - padding) / GRAPH_HEIGHT,
      )),
    );

    setZoom(nextZoom);
    setPan({
      x: Math.max(12, (viewport.clientWidth - GRAPH_WIDTH * nextZoom) / 2),
      y: Math.max(12, (viewport.clientHeight - GRAPH_HEIGHT * nextZoom) / 2),
    });
  }, []);

  useEffect(() => {
    fitGraph();
    const viewport = viewportRef.current;
    if (!viewport || typeof ResizeObserver === 'undefined') return undefined;

    const observer = new ResizeObserver(() => fitGraph());
    observer.observe(viewport);

    return () => observer.disconnect();
  }, [fitGraph]);

  const onPointerDown = (event) => {
    if (event.button !== 0 || event.target.closest('button')) return;
    dragRef.current = { x: event.clientX, y: event.clientY, panX: pan.x, panY: pan.y };
    setDragging(true);
    event.currentTarget.setPointerCapture?.(event.pointerId);
  };

  const onPointerMove = (event) => {
    if (!dragRef.current) return;
    setPan({
      x: dragRef.current.panX + event.clientX - dragRef.current.x,
      y: dragRef.current.panY + event.clientY - dragRef.current.y,
    });
  };

  const stopDragging = () => {
    dragRef.current = null;
    setDragging(false);
  };

  const fitToView = () => fitGraph();

  return (
    <div className="min-w-0 space-y-5">
      <section className="min-w-0 overflow-hidden rounded-2xl border border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900">
        <header className="flex flex-col justify-between gap-4 border-b border-slate-200 p-4 dark:border-slate-800 lg:flex-row lg:items-center">
          <div>
            <h2 className="flex items-center gap-2 text-sm font-semibold text-slate-900 dark:text-slate-100">
              <Workflow className="h-4 w-4 text-indigo-500" />
              LangGraph Execution
            </h2>
            <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
              Explore the research flow, approval checkpoints, and adaptive research routes.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <IconButton label="Zoom out" onClick={() => zoomBy(-0.1)}>
              <Minus className="h-4 w-4" />
            </IconButton>
            <span className="w-12 text-center text-xs font-semibold tabular-nums text-slate-600 dark:text-slate-300">
              {Math.round(zoom * 100)}%
            </span>
            <IconButton label="Zoom in" onClick={() => zoomBy(0.1)}>
              <Plus className="h-4 w-4" />
            </IconButton>
            <IconButton label="Fit graph to viewport" onClick={fitToView}>
              <Maximize2 className="h-4 w-4" />
            </IconButton>
            <IconButton label="Reset graph" onClick={() => { setZoom(0.5); setPan({ x: 16, y: 16 }); }}>
              <RotateCcw className="h-4 w-4" />
            </IconButton>
          </div>
        </header>

        <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-slate-100 bg-slate-50 px-4 py-3 text-[11px] text-slate-500 dark:border-slate-800 dark:bg-slate-950/40 dark:text-slate-400">
          {[
            ['bg-emerald-500', 'Completed'],
            ['bg-sky-500', 'Running'],
            ['bg-amber-500', 'Awaiting review'],
            ['bg-rose-500', 'Failed'],
            ['bg-slate-400', 'Upcoming'],
          ].map(([color, label]) => (
            <span key={label} className="inline-flex items-center gap-1.5">
              <span className={`h-2 w-2 rounded-full ${color}`} />
              {label}
            </span>
          ))}
        </div>

        <div className="flex items-center justify-between gap-3 border-b border-slate-100 px-4 py-2 dark:border-slate-800">
          <span className="text-[11px] text-slate-500 dark:text-slate-400">Move graph</span>
          <div className="flex items-center gap-1">
            <IconButton label="Move left" onClick={() => panBy(60, 0)}><ArrowLeft className="h-4 w-4" /></IconButton>
            <IconButton label="Move up" onClick={() => panBy(0, 60)}><ArrowUp className="h-4 w-4" /></IconButton>
            <IconButton label="Move down" onClick={() => panBy(0, -60)}><ArrowDown className="h-4 w-4" /></IconButton>
            <IconButton label="Move right" onClick={() => panBy(-60, 0)}><ArrowRight className="h-4 w-4" /></IconButton>
          </div>
        </div>

        <div
          ref={viewportRef}
          className={`relative h-[560px] w-full overflow-hidden ${
            dragging ? 'cursor-grabbing' : 'cursor-grab'
          }`}
          style={{
            backgroundColor: '#f8fafc',
            backgroundImage: 'radial-gradient(circle, rgba(148,163,184,.28) 1px, transparent 1px)',
            backgroundSize: '22px 22px',
            touchAction: 'none',
          }}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={stopDragging}
          onPointerCancel={stopDragging}
          onWheel={(event) => {
            if (event.ctrlKey || event.metaKey) {
              event.preventDefault();
              zoomBy(event.deltaY < 0 ? 0.05 : -0.05);
            }
          }}
          tabIndex={0}
          aria-label="Interactive research workflow. Drag to pan and hold Ctrl or Cmd while scrolling to zoom."
        >
          <div
            className="absolute left-0 top-0"
            style={{
              width: GRAPH_WIDTH,
              height: GRAPH_HEIGHT,
              transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
              transformOrigin: 'top left',
              transition: dragging ? 'none' : 'transform 100ms ease-out',
            }}
          >
            <svg
              className="pointer-events-none absolute left-0 top-0"
              width={GRAPH_WIDTH}
              height={GRAPH_HEIGHT}
              viewBox={`0 0 ${GRAPH_WIDTH} ${GRAPH_HEIGHT}`}
              aria-hidden="true"
            >
              <defs>
                <marker
                  id="workflow-arrow"
                  markerWidth="9"
                  markerHeight="9"
                  refX="7"
                  refY="4.5"
                  orient="auto"
                >
                  <path d="M1 1 L8 4.5 L1 8" fill="none" stroke="#94a3b8" strokeWidth="1.5" />
                </marker>
              </defs>

              {EDGES.map((edge, index) => (
                <g key={`${edge.from}-${edge.to}-${index}`}>
                  <path
                    d={edge.d}
                    fill="none"
                    stroke="#94a3b8"
                    strokeWidth="2"
                    strokeLinejoin="round"
                    strokeLinecap="round"
                    markerEnd="url(#workflow-arrow)"
                  />
                  {edge.label && (
                    <g transform={`translate(${edge.labelX ?? ((WORKFLOW_NODES.find(n => n.id === edge.from)?.x + WORKFLOW_NODES.find(n => n.id === edge.to)?.x) / 2)}, ${edge.labelY ?? ((WORKFLOW_NODES.find(n => n.id === edge.from)?.y + WORKFLOW_NODES.find(n => n.id === edge.to)?.y) / 2 - 16)})`}>
                      <rect
                        x={-Math.max(34, edge.label.length * 3.3)}
                        y="-10"
                        width={Math.max(68, edge.label.length * 6.6)}
                        height="20"
                        rx="5"
                        fill="#f8fafc"
                        stroke="#e2e8f0"
                      />
                      <text
                        textAnchor="middle"
                        y="3.5"
                        fontSize="10"
                        fontWeight="600"
                        fill="#475569"
                      >
                        {edge.label}
                      </text>
                    </g>
                  )}
                </g>
              ))}
            </svg>

            {graphNodes.map((node) => {
              const nodeStatus = getNodeStatus(node);
              const Icon = nodeStatus === 'completed'
                ? CheckCircle2
                : nodeStatus === 'failed'
                  ? XCircle
                  : node.icon;

              return (
                <div
                  key={node.id}
                  className={`absolute box-border flex items-center gap-3 rounded-xl border px-3 py-3 shadow-sm ${STATUS_CLASSES[nodeStatus]}`}
                  style={{
                    left: node.x - NODE_WIDTH / 2,
                    top: node.y - NODE_HEIGHT / 2,
                    width: NODE_WIDTH,
                    height: NODE_HEIGHT,
                  }}
                  title={node.description}
                  aria-label={`${node.label}: ${nodeStatus}`}
                >
                  <Icon className="h-5 w-5 shrink-0" />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-1.5">
                      <span className="break-words text-xs font-semibold leading-tight">{node.label}</span>
                      {node.hitl && (
                        <span className="shrink-0 rounded bg-indigo-100 px-1 py-0.5 text-[8px] font-bold text-indigo-700 dark:bg-indigo-950 dark:text-indigo-300">
                          HITL
                        </span>
                      )}
                    </div>
                    <span className="mt-1 block text-[10px] capitalize opacity-70">
                      {nodeStatus === 'active' ? 'Running now' : nodeStatus === 'waiting' ? 'Awaiting review' : nodeStatus}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <footer className="border-t border-slate-200 px-4 py-3 text-[11px] text-slate-500 dark:border-slate-800 dark:text-slate-400">
          Drag to pan. Use the directional controls to move in 60 px steps. Hold Ctrl/Cmd while scrolling to zoom.
        </footer>
      </section>

      <section className="min-w-0 rounded-2xl border border-slate-200 bg-white p-5 dark:border-slate-800 dark:bg-slate-900">
        <div className="mb-4 border-b border-slate-100 pb-4 dark:border-slate-800">
          <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">Workflow Node Guide</h2>
          <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
            A reference guide to each node and its responsibility. Execution status is shown only in the graph above.
          </p>
        </div>

      </section>
    </div>
  );
}

export default WorkflowGraph;
