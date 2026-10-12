import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  ArrowUp,
  Maximize2,
  Minus,
  Plus,
  RotateCcw,
  Workflow,
} from 'lucide-react';

import { useTheme } from '../../context/ThemeContext';
import {
  GRAPH_CONTENT_BOUNDS,
  GRAPH_EDGES,
  GRAPH_HEIGHT,
  GRAPH_NODES,
  GRAPH_WIDTH,
  MAX_ZOOM,
  MIN_ZOOM,
  NODE_HEIGHT,
  NODE_WIDTH,
  getActiveNode,
  getEdgePoints,
} from './workflowGraphModel';

const ZOOM_STEP = 0.1;
const PAN_MARGIN = 24;
const FIT_PADDING = 42;

function constrainPan(pan, zoom, viewport) {
  const clampAxis = (offset, viewportSize, start, end) => {
    const scaledStart = start * zoom;
    const scaledSize = (end - start) * zoom;

    if (scaledSize <= viewportSize - PAN_MARGIN * 2) {
      return (viewportSize - scaledSize) / 2 - scaledStart;
    }

    return Math.min(
      PAN_MARGIN - scaledStart,
      Math.max(viewportSize - end * zoom - PAN_MARGIN, offset),
    );
  };

  return {
    x: clampAxis(
      pan.x,
      viewport.clientWidth,
      GRAPH_CONTENT_BOUNDS.left,
      GRAPH_CONTENT_BOUNDS.right,
    ),
    y: clampAxis(
      pan.y,
      viewport.clientHeight,
      GRAPH_CONTENT_BOUNDS.top,
      GRAPH_CONTENT_BOUNDS.bottom,
    ),
  };
}

function IconButton({ label, onClick, children, disabled = false }) {
  return (
    <button
      type="button"
      title={label}
      aria-label={label}
      onClick={onClick}
      disabled={disabled}
      className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-slate-200 bg-white text-slate-600 transition hover:bg-slate-100 focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:cursor-not-allowed disabled:opacity-40 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300 dark:hover:bg-slate-800"
    >
      {children}
    </button>
  );
}

function edgePath(edge, nodesById) {
  return getEdgePoints(edge, nodesById)
    .map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x} ${point.y}`)
    .join(' ');
}

function getFitZoom(viewport) {
  const availableWidth = Math.max(1, viewport.clientWidth - FIT_PADDING * 2);
  const availableHeight = Math.max(1, viewport.clientHeight - FIT_PADDING * 2);
  const contentWidth = GRAPH_CONTENT_BOUNDS.right - GRAPH_CONTENT_BOUNDS.left;
  const contentHeight = GRAPH_CONTENT_BOUNDS.bottom - GRAPH_CONTENT_BOUNDS.top;

  return Math.min(
    MAX_ZOOM,
    Math.max(MIN_ZOOM, Math.min(availableWidth / contentWidth, availableHeight / contentHeight)),
  );
}

export function WorkflowGraph({ session, events = [], pendingReview = null }) {
  const viewportRef = useRef(null);
  const dragRef = useRef(null);
  const hasNavigatedRef = useRef(false);
  const zoomRef = useRef(MIN_ZOOM);
  const minZoomRef = useRef(MIN_ZOOM);
  const { resolvedTheme, reduceMotion } = useTheme();

  const [zoom, setZoom] = useState(MIN_ZOOM);
  const [minZoom, setMinZoom] = useState(MIN_ZOOM);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    zoomRef.current = zoom;
  }, [zoom]);

  const nodesById = useMemo(
    () => Object.fromEntries(GRAPH_NODES.map((node) => [node.id, node])),
    [],
  );
  const activeNode = getActiveNode(events, session?.status, pendingReview);
  const isDark = resolvedTheme === 'dark';

  const fitGraph = useCallback(() => {
    const viewport = viewportRef.current;
    if (!viewport) return;

    const nextZoom = getFitZoom(viewport);

    minZoomRef.current = nextZoom;
    setMinZoom(nextZoom);
    zoomRef.current = nextZoom;
    setZoom(nextZoom);
    setPan(constrainPan({ x: 0, y: 0 }, nextZoom, viewport));
  }, []);

  useEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport) return undefined;

    const initialFrame = requestAnimationFrame(fitGraph);
    const observer = new ResizeObserver(() => {
      if (!hasNavigatedRef.current) {
        fitGraph();
      } else {
        const nextMinZoom = getFitZoom(viewport);
        const nextZoom = Math.max(nextMinZoom, zoomRef.current);
        minZoomRef.current = nextMinZoom;
        setMinZoom(nextMinZoom);
        zoomRef.current = nextZoom;
        setZoom(nextZoom);
        setPan((currentPan) =>
          constrainPan(currentPan, nextZoom, viewport),
        );
      }
    });

    observer.observe(viewport);
    return () => {
      cancelAnimationFrame(initialFrame);
      observer.disconnect();
    };
  }, [fitGraph]);

  const markNavigated = () => {
    hasNavigatedRef.current = true;
  };

  const zoomAt = useCallback((nextZoom, point) => {
    const viewport = viewportRef.current;
    if (!viewport) return;

    const clampedZoom = Math.min(
      MAX_ZOOM,
      Math.max(minZoomRef.current, nextZoom),
    );
    if (clampedZoom === zoom) return;

    markNavigated();
    const ratio = clampedZoom / zoom;
    setPan((currentPan) =>
      constrainPan(
        {
          x: point.x - (point.x - currentPan.x) * ratio,
          y: point.y - (point.y - currentPan.y) * ratio,
        },
        clampedZoom,
        viewport,
      ),
    );
    zoomRef.current = clampedZoom;
    setZoom(clampedZoom);
  }, [zoom]);

  const zoomBy = useCallback((amount) => {
    const viewport = viewportRef.current;
    if (!viewport) return;
    zoomAt(zoom + amount, {
      x: viewport.clientWidth / 2,
      y: viewport.clientHeight / 2,
    });
  }, [zoom, zoomAt]);

  const panBy = (dx, dy) => {
    const viewport = viewportRef.current;
    if (!viewport) return;
    markNavigated();
    setPan((currentPan) =>
      constrainPan(
        { x: currentPan.x + dx, y: currentPan.y + dy },
        zoom,
        viewport,
      ),
    );
  };

  const resetGraph = () => {
    hasNavigatedRef.current = false;
    fitGraph();
  };

  const onPointerDown = (event) => {
    if (event.button !== 0 || event.target.closest('button')) return;

    dragRef.current = {
      pointerX: event.clientX,
      pointerY: event.clientY,
      panX: pan.x,
      panY: pan.y,
    };

    markNavigated();
    setDragging(true);
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const onPointerMove = (event) => {
    if (!dragRef.current) return;
    setPan(
      constrainPan(
        {
          x: dragRef.current.panX + event.clientX - dragRef.current.pointerX,
          y: dragRef.current.panY + event.clientY - dragRef.current.pointerY,
        },
        zoom,
        event.currentTarget,
      ),
    );
  };

  const stopDragging = () => {
    dragRef.current = null;
    setDragging(false);
  };

  const onKeyDown = (event) => {
    if (event.target !== event.currentTarget) return;

    const panKeys = {
      ArrowLeft: [60, 0],
      ArrowRight: [-60, 0],
      ArrowUp: [0, 60],
      ArrowDown: [0, -60],
    };

    if (panKeys[event.key]) {
      event.preventDefault();
      panBy(...panKeys[event.key]);
    } else if (event.key === '+' || event.key === '=') {
      event.preventDefault();
      zoomBy(ZOOM_STEP);
    } else if (event.key === '-') {
      event.preventDefault();
      zoomBy(-ZOOM_STEP);
    } else if (event.key === 'Home') {
      event.preventDefault();
      resetGraph();
    }
  };

  const zoomAtPointer = (event) => {
    if (!event.ctrlKey && !event.metaKey) return;
    event.preventDefault();
    const bounds = event.currentTarget.getBoundingClientRect();
    zoomAt(zoom + (event.deltaY < 0 ? 0.05 : -0.05), {
      x: event.clientX - bounds.left,
      y: event.clientY - bounds.top,
    });
  };

  const edgeColor = isDark ? '#64748b' : '#94a3b8';
  const labelColor = isDark ? '#cbd5e1' : '#475569';
  const labelFill = isDark ? '#111827' : '#ffffff';
  const labelBorder = isDark ? '#334155' : '#cbd5e1';

  return (
    <section className="min-w-0 overflow-hidden rounded-2xl border border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900">
      <header className="flex flex-col justify-between gap-3 border-b border-slate-200 p-4 dark:border-slate-800 sm:flex-row sm:items-center">
        <div>
          <h2 className="flex items-center gap-2 text-sm font-semibold text-slate-900 dark:text-slate-100">
            <Workflow className="h-4 w-4 text-indigo-500" />
            LangGraph Execution
          </h2>
          <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
            The executing node is highlighted as the workflow progresses.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <IconButton
            label="Zoom out"
            onClick={() => zoomBy(-ZOOM_STEP)}
            disabled={zoom <= minZoom}
          >
            <Minus className="h-4 w-4" />
          </IconButton>
          <span className="w-12 text-center text-xs font-semibold tabular-nums text-slate-600 dark:text-slate-300">
            {Math.round(zoom * 100)}%
          </span>
          <IconButton
            label="Zoom in"
            onClick={() => zoomBy(ZOOM_STEP)}
            disabled={zoom >= MAX_ZOOM}
          >
            <Plus className="h-4 w-4" />
          </IconButton>
          <IconButton label="Fit graph to screen" onClick={resetGraph}>
            <Maximize2 className="h-4 w-4" />
          </IconButton>
          <IconButton label="Reset graph view" onClick={resetGraph}>
            <RotateCcw className="h-4 w-4" />
          </IconButton>
        </div>
      </header>

      <div className="flex items-center justify-end gap-1 border-b border-slate-100 px-4 py-2 dark:border-slate-800">
        <span className="mr-auto text-[11px] text-slate-500 dark:text-slate-400">
          Move workflow
        </span>
        <IconButton label="Move left" onClick={() => panBy(60, 0)}>
          <ArrowLeft className="h-4 w-4" />
        </IconButton>
        <IconButton label="Move up" onClick={() => panBy(0, 60)}>
          <ArrowUp className="h-4 w-4" />
        </IconButton>
        <IconButton label="Move down" onClick={() => panBy(0, -60)}>
          <ArrowDown className="h-4 w-4" />
        </IconButton>
        <IconButton label="Move right" onClick={() => panBy(-60, 0)}>
          <ArrowRight className="h-4 w-4" />
        </IconButton>
      </div>

      <div
        ref={viewportRef}
        className={`relative h-[min(78vh,820px)] min-h-[540px] w-full overflow-hidden select-none ${
          dragging ? 'cursor-grabbing' : 'cursor-grab'
        }`}
        style={{
          backgroundColor: isDark ? '#0b1120' : '#f8fafc',
          backgroundImage: isDark
            ? 'radial-gradient(circle, rgba(148,163,184,.13) 1px, transparent 1px)'
            : 'radial-gradient(circle, rgba(100,116,139,.18) 1px, transparent 1px)',
          backgroundSize: '24px 24px',
          touchAction: 'none',
        }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={stopDragging}
        onPointerCancel={stopDragging}
        onLostPointerCapture={stopDragging}
        onWheel={zoomAtPointer}
        onKeyDown={onKeyDown}
        tabIndex={0}
        role="application"
        aria-label="Interactive LangGraph workflow. Drag to move. Use arrow keys to pan, plus and minus to zoom, or Home to fit."
      >
        <div
          className="absolute left-0 top-0"
          style={{
            width: GRAPH_WIDTH,
            height: GRAPH_HEIGHT,
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
            transformOrigin: 'top left',
            transition: 'none',
            willChange: 'transform',
          }}
        >
          <svg
            className="pointer-events-none absolute left-0 top-0 overflow-visible"
            width={GRAPH_WIDTH}
            height={GRAPH_HEIGHT}
            viewBox={`0 0 ${GRAPH_WIDTH} ${GRAPH_HEIGHT}`}
            role="img"
            aria-label="Workflow connections"
          >
            <defs>
              <marker
                id="workflow-arrow"
                markerWidth="9"
                markerHeight="9"
                refX="7"
                refY="4.5"
                orient="auto"
                markerUnits="userSpaceOnUse"
              >
                <path
                  d="M1 1 L8 4.5 L1 8"
                  fill="none"
                  stroke={edgeColor}
                  strokeWidth="1.5"
                />
              </marker>
            </defs>

            {GRAPH_EDGES.map((edge, index) => (
              <g key={`${edge.from}-${edge.to}-${index}`}>
                <path
                  d={edgePath(edge, nodesById)}
                  fill="none"
                  stroke={edgeColor}
                  strokeWidth="2"
                  strokeLinejoin="round"
                  strokeLinecap="round"
                  markerEnd="url(#workflow-arrow)"
                />
                <g transform={`translate(${edge.labelX}, ${edge.labelY})`}>
                  <rect
                    x={-Math.max(28, edge.label.length * 3.3)}
                    y="-9"
                    width={Math.max(56, edge.label.length * 6.6)}
                    height="18"
                    rx="5"
                    fill={labelFill}
                    stroke={labelBorder}
                    strokeWidth="1"
                  />
                  <text
                    textAnchor="middle"
                    y="3.5"
                    fontSize="10"
                    fontWeight="600"
                    fill={labelColor}
                  >
                    {edge.label}
                  </text>
                </g>
              </g>
            ))}
          </svg>

          {GRAPH_NODES.map((node) => {
            const Icon = node.icon;
            const isActive = node.id === activeNode;
            const isAwaitingApproval =
              isActive && session?.status === 'waiting_for_review';

            return (
              <div
                key={node.id}
                className={`absolute box-border flex items-center gap-3 rounded-xl border px-4 py-3 ${
                  isActive
                    ? 'z-10 border-cyan-400 bg-cyan-100 text-slate-950 shadow-[0_0_0_4px_rgba(34,211,238,0.18),0_0_28px_rgba(34,211,238,0.28)] dark:border-cyan-300 dark:bg-cyan-950 dark:text-cyan-50 dark:shadow-[0_0_0_4px_rgba(34,211,238,0.2),0_0_32px_rgba(34,211,238,0.25)]'
                    : 'z-0 border-slate-200 bg-white text-slate-700 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200'
                }`}
                style={{
                  left: node.x - NODE_WIDTH / 2,
                  top: node.y - NODE_HEIGHT / 2,
                  width: NODE_WIDTH,
                  height: NODE_HEIGHT,
                  transition: reduceMotion ? 'none' : 'background-color 160ms ease, border-color 160ms ease, box-shadow 160ms ease',
                }}
                aria-label={
                  isActive
                    ? isAwaitingApproval
                      ? `${node.label}, awaiting approval`
                      : `${node.label}, currently executing`
                    : node.label
                }
                role="img"
              >
                <Icon
                  className={`h-5 w-5 shrink-0 ${
                    isActive
                      ? 'text-cyan-700 dark:text-cyan-300'
                      : 'text-slate-400 dark:text-slate-500'
                  }`}
                  aria-hidden="true"
                />
                <span className="min-w-0 text-sm font-semibold leading-tight">
                  {node.label}
                </span>
                {isActive && (
                  <span
                    className="absolute right-3 top-3 h-2.5 w-2.5 rounded-full bg-cyan-500 shadow-[0_0_12px_rgba(6,182,212,0.8)] motion-safe:animate-pulse"
                    aria-hidden="true"
                  />
                )}
              </div>
            );
          })}
        </div>
      </div>

      <footer className="border-t border-slate-200 px-4 py-3 text-[11px] text-slate-500 dark:border-slate-800 dark:text-slate-400">
        Drag to pan. Use the arrow controls or keyboard arrows to move. Hold
        Ctrl/Cmd while scrolling to zoom.
      </footer>
    </section>
  );
}

export default WorkflowGraph;
