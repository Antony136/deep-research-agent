import { WORKFLOW_NODES } from './workflowNodes.js';

export const NODE_WIDTH = 212;
export const NODE_HEIGHT = 78;
export const GRAPH_WIDTH = 1530;
export const GRAPH_HEIGHT = 900;
export const MIN_ZOOM = 0.1;
export const MAX_ZOOM = 1.5;

const NODE_POSITIONS = {
  __start__: { x: 120, y: 155 },
  planner: { x: 430, y: 155 },
  initial_plan_review: { x: 740, y: 155 },
  initial_research_review: { x: 1050, y: 155 },
  researcher: { x: 1050, y: 395 },
  evidence_extractor: { x: 740, y: 395 },
  evidence_verifier: { x: 430, y: 395 },
  research_sufficiency: { x: 120, y: 395 },
  adaptive_planner: { x: 430, y: 605 },
  adaptive_review: { x: 120, y: 605 },
  auto_accept_adaptive: { x: 740, y: 605 },
  synthesizer: { x: 1050, y: 605 },
  final_report_review: { x: 1050, y: 795 },
  __end__: { x: 1360, y: 795 },
};

export const GRAPH_NODES = WORKFLOW_NODES.map((node) => ({
  ...node,
  ...(NODE_POSITIONS[node.id] || {}),
}));

export const GRAPH_EDGES = [
  { from: '__start__', to: 'planner', label: 'Begin', fromSide: 'right', toSide: 'left', labelX: 275, labelY: 137 },
  { from: 'planner', to: 'initial_plan_review', label: 'Proposed plan', fromSide: 'right', toSide: 'left', labelX: 585, labelY: 137 },
  {
    from: 'initial_plan_review',
    to: 'initial_plan_review',
    label: 'Revise',
    fromSide: 'left',
    toSide: 'top',
    fromOffset: -18,
    toOffset: -18,
    via: [{ x: 600, y: 137 }, { x: 600, y: 86 }, { x: 722, y: 86 }],
    labelX: 661,
    labelY: 72,
  },
  { from: 'initial_plan_review', to: 'initial_research_review', label: 'Approve plan', fromSide: 'right', toSide: 'left', labelX: 895, labelY: 137 },
  {
    from: 'initial_plan_review',
    to: '__end__',
    label: 'Reject',
    fromSide: 'top',
    toSide: 'right',
    toOffset: -18,
    via: [{ x: 740, y: 40 }, { x: 1480, y: 40 }, { x: 1480, y: 777 }],
    labelX: 1240,
    labelY: 38,
  },
  { from: 'initial_research_review', to: 'researcher', label: 'Authorize', fromSide: 'bottom', toSide: 'top', labelX: 1125, labelY: 275 },
  {
    from: 'initial_research_review',
    to: '__end__',
    label: 'Stop',
    fromSide: 'top',
    toSide: 'right',
    toOffset: 18,
    via: [{ x: 1050, y: 62 }, { x: 1500, y: 62 }, { x: 1500, y: 813 }],
    labelX: 1280,
    labelY: 60,
  },
  { from: 'researcher', to: 'evidence_extractor', label: 'Sources', fromSide: 'left', toSide: 'right', labelX: 895, labelY: 377 },
  { from: 'evidence_extractor', to: 'evidence_verifier', label: 'Extract claims', fromSide: 'left', toSide: 'right', labelX: 585, labelY: 377 },
  { from: 'evidence_verifier', to: 'research_sufficiency', label: 'Verify', fromSide: 'left', toSide: 'right', labelX: 275, labelY: 377 },
  {
    from: 'research_sufficiency',
    to: 'researcher',
    label: 'Next question',
    fromSide: 'top',
    toSide: 'top',
    via: [{ x: 120, y: 280 }, { x: 1050, y: 280 }],
    labelX: 585,
    labelY: 260,
  },
  {
    from: 'research_sufficiency',
    to: 'adaptive_planner',
    label: 'Evidence gaps',
    fromSide: 'bottom',
    toSide: 'top',
    via: [{ x: 120, y: 490 }, { x: 430, y: 490 }],
    labelX: 275,
    labelY: 483,
  },
  {
    from: 'research_sufficiency',
    to: 'synthesizer',
    label: 'Sufficient evidence',
    fromSide: 'right',
    toSide: 'top',
    via: [{ x: 300, y: 395 }, { x: 300, y: 540 }, { x: 1050, y: 540 }],
    labelX: 675,
    labelY: 525,
  },
  {
    from: 'adaptive_planner',
    to: 'synthesizer',
    label: 'No usable questions',
    fromSide: 'bottom',
    toSide: 'bottom',
    via: [{ x: 430, y: 675 }, { x: 1050, y: 675 }],
    labelX: 740,
    labelY: 660,
  },
  { from: 'adaptive_planner', to: 'adaptive_review', label: 'Request review', fromSide: 'left', toSide: 'right', labelX: 275, labelY: 587 },
  { from: 'adaptive_planner', to: 'auto_accept_adaptive', label: 'Auto-approve', fromSide: 'right', toSide: 'left', labelX: 585, labelY: 587 },
  {
    from: 'adaptive_review',
    to: 'researcher',
    label: 'Approved questions',
    fromSide: 'bottom',
    toSide: 'right',
    toOffset: -18,
    via: [{ x: 120, y: 700 }, { x: 1460, y: 700 }, { x: 1460, y: 377 }],
    labelX: 820,
    labelY: 687,
  },
  {
    from: 'adaptive_review',
    to: 'synthesizer',
    label: 'Decline follow-ups',
    fromSide: 'bottom',
    toSide: 'bottom',
    via: [{ x: 120, y: 730 }, { x: 1050, y: 730 }],
    labelX: 585,
    labelY: 717,
  },
  {
    from: 'auto_accept_adaptive',
    to: 'researcher',
    label: 'Continue research',
    fromSide: 'bottom',
    toSide: 'right',
    toOffset: 18,
    via: [{ x: 740, y: 750 }, { x: 1500, y: 750 }, { x: 1500, y: 413 }],
    labelX: 1120,
    labelY: 737,
  },
  { from: 'auto_accept_adaptive', to: 'synthesizer', label: 'No follow-ups', fromSide: 'right', toSide: 'left', labelX: 895, labelY: 587 },
  { from: 'synthesizer', to: 'final_report_review', label: 'Draft report', fromSide: 'bottom', toSide: 'top', labelX: 1120, labelY: 700 },
  {
    from: 'final_report_review',
    to: 'adaptive_planner',
    label: 'Research more',
    fromSide: 'left',
    toSide: 'bottom',
    via: [{ x: 900, y: 795 }, { x: 900, y: 850 }, { x: 430, y: 850 }],
    labelX: 665,
    labelY: 837,
  },
  { from: 'final_report_review', to: '__end__', label: 'Approve report', fromSide: 'right', toSide: 'left', labelX: 1205, labelY: 777 },
];

const REVIEW_NODE_BY_TYPE = {
  initial_plan_review: 'initial_plan_review',
  initial_research_review: 'initial_research_review',
  adaptive_research_review: 'adaptive_review',
  final_report_review: 'final_report_review',
};

export function getNodeAnchor(node, side, offset = 0) {
  switch (side) {
    case 'left':
      return { x: node.x - NODE_WIDTH / 2, y: node.y + offset };
    case 'right':
      return { x: node.x + NODE_WIDTH / 2, y: node.y + offset };
    case 'top':
      return { x: node.x + offset, y: node.y - NODE_HEIGHT / 2 };
    case 'bottom':
      return { x: node.x + offset, y: node.y + NODE_HEIGHT / 2 };
    default:
      throw new Error(`Unsupported workflow edge side: ${side}`);
  }
}

export function getEdgePoints(edge, nodesById) {
  const source = nodesById[edge.from];
  const target = nodesById[edge.to];
  if (!source || !target) {
    throw new Error(`Workflow edge references an unknown node: ${edge.from} -> ${edge.to}`);
  }

  return [
    getNodeAnchor(source, edge.fromSide, edge.fromOffset),
    ...(edge.via || []),
    getNodeAnchor(target, edge.toSide, edge.toOffset),
  ];
}

export function getGraphBounds(nodes, edges) {
  const nodesById = Object.fromEntries(nodes.map((node) => [node.id, node]));
  const points = [];

  for (const node of nodes) {
    points.push(
      { x: node.x - NODE_WIDTH / 2, y: node.y - NODE_HEIGHT / 2 },
      { x: node.x + NODE_WIDTH / 2, y: node.y + NODE_HEIGHT / 2 },
    );
  }

  for (const edge of edges) {
    points.push(...getEdgePoints(edge, nodesById));
    if (edge.label) {
      const labelHalfWidth = Math.max(28, edge.label.length * 3.3) + 8;
      points.push(
        { x: edge.labelX - labelHalfWidth, y: edge.labelY - 14 },
        { x: edge.labelX + labelHalfWidth, y: edge.labelY + 14 },
      );
    }
  }

  return {
    left: Math.min(...points.map((point) => point.x)) - 8,
    right: Math.max(...points.map((point) => point.x)) + 8,
    top: Math.min(...points.map((point) => point.y)) - 8,
    bottom: Math.max(...points.map((point) => point.y)) + 8,
  };
}

export function getActiveNode(
  events = [],
  sessionStatus,
  pendingReview = null,
  jobStatus = null,
) {
  if (sessionStatus === 'waiting_for_review') {
    return REVIEW_NODE_BY_TYPE[pendingReview?.type] || null;
  }

  if (
    sessionStatus !== 'running' ||
    jobStatus === 'retry_wait' ||
    !Array.isArray(events)
  ) {
    return null;
  }

  const knownNodes = new Set(GRAPH_NODES.map((node) => node.id));
  let activeNode = null;

  for (const event of events) {
    if (!knownNodes.has(event?.node_name)) {
      continue;
    }

    if (event.event_type === 'node_started') {
      activeNode = event.node_name;
    } else if (
      event.node_name === activeNode &&
      ['node_completed', 'node_failed', 'node_interrupted'].includes(event.event_type)
    ) {
      activeNode = null;
    }
  }

  return activeNode;
}

export const GRAPH_CONTENT_BOUNDS = getGraphBounds(GRAPH_NODES, GRAPH_EDGES);
