import {
  Brain,
  CheckCircle2,
  FileSearch,
  FileCheck,
  Search,
  Scale,
  Sparkles,
  GitMerge,
  BookOpen,
  ShieldCheck,
} from 'lucide-react';

const WORKFLOW_NODES = [
  {
    id: 'planner',
    label: 'Planner',
    shortDesc: 'Decomposes question into core sub-questions',
    icon: Brain,
    isHumanCheckpoint: false,
  },
  {
    id: 'initial_plan_review',
    label: 'Plan Review',
    shortDesc: 'HITL Checkpoint: Approve or revise questions',
    icon: ShieldCheck,
    isHumanCheckpoint: true,
  },
  {
    id: 'initial_research_review',
    label: 'Research Auth',
    shortDesc: 'HITL Checkpoint: Authorize web research phase',
    icon: ShieldCheck,
    isHumanCheckpoint: true,
  },
  {
    id: 'researcher',
    label: 'Web Researcher',
    shortDesc: 'Performs search queries and reads web sources',
    icon: Search,
    isHumanCheckpoint: false,
  },
  {
    id: 'evidence_extractor',
    label: 'Evidence Extractor',
    shortDesc: 'Extracts claims and supporting text quotes',
    icon: FileSearch,
    isHumanCheckpoint: false,
  },
  {
    id: 'evidence_verifier',
    label: 'Evidence Verifier',
    shortDesc: 'Deterministically verifies claims and citations',
    icon: FileCheck,
    isHumanCheckpoint: false,
  },
  {
    id: 'research_sufficiency',
    label: 'Sufficiency Evaluator',
    shortDesc: 'Assesses coverage against research questions',
    icon: Scale,
    isHumanCheckpoint: false,
  },
  {
    id: 'adaptive_planner',
    label: 'Adaptive Planner',
    shortDesc: 'Formulates targeted gap-closing questions',
    icon: GitMerge,
    isHumanCheckpoint: false,
  },
  {
    id: 'adaptive_review',
    label: 'Adaptive Review',
    shortDesc: 'HITL Checkpoint: Approve proposed follow-ups',
    icon: ShieldCheck,
    isHumanCheckpoint: true,
  },
  {
    id: 'synthesizer',
    label: 'Report Synthesizer',
    shortDesc: 'Drafts evidence-backed findings and citations',
    icon: Sparkles,
    isHumanCheckpoint: false,
  },
  {
    id: 'final_report_review',
    label: 'Report Review',
    shortDesc: 'HITL Checkpoint: Approve report or request more',
    icon: BookOpen,
    isHumanCheckpoint: true,
  },
];

export function WorkflowGraph({ session, pendingReview }) {
  const status = session?.status;
  const researchState = session?.research_state || {};
  const reviewType = pendingReview?.type;

  // Determine current active or paused node
  const getNodeStatus = (node) => {
    if (status === 'completed') {
      return 'completed';
    }

    if (status === 'waiting_for_review') {
      if (node.id === reviewType) {
        return 'waiting';
      }
    }

    // Check if node is in completed state based on session research_state
    const hasReport = !!session?.final_report || !!researchState.report;
    const hasEvidence = (researchState.evidence || []).length > 0;
    const hasQuestions = (researchState.research_questions || []).length > 0;
    const initialPlanApproved = researchState.initial_plan_approved;
    const researchAuthorized = researchState.research_authorized;

    if (node.id === 'planner') {
      return hasQuestions ? 'completed' : status === 'running' ? 'active' : 'pending';
    }
    if (node.id === 'initial_plan_review') {
      if (initialPlanApproved) return 'completed';
      if (reviewType === 'initial_plan_review') return 'waiting';
      return hasQuestions ? 'active' : 'pending';
    }
    if (node.id === 'initial_research_review') {
      if (researchAuthorized) return 'completed';
      if (reviewType === 'initial_research_review') return 'waiting';
      return initialPlanApproved ? 'active' : 'pending';
    }
    if (node.id === 'researcher' || node.id === 'evidence_extractor' || node.id === 'evidence_verifier') {
      if (hasEvidence) return 'completed';
      if (researchAuthorized && status === 'running') return 'active';
      return 'pending';
    }
    if (node.id === 'research_sufficiency' || node.id === 'adaptive_planner') {
      if (hasEvidence && (researchState.research_round > 0 || hasReport)) return 'completed';
      if (hasEvidence && status === 'running') return 'active';
      return 'pending';
    }
    if (node.id === 'adaptive_review') {
      if (reviewType === 'adaptive_research_review') return 'waiting';
      if (researchState.adaptive_review_decision) return 'completed';
      return 'pending';
    }
    if (node.id === 'synthesizer') {
      if (hasReport) return 'completed';
      if (researchState.research_sufficient && status === 'running') return 'active';
      return 'pending';
    }
    if (node.id === 'final_report_review') {
      if (researchState.final_report_approved) return 'completed';
      if (reviewType === 'final_report_review') return 'waiting';
      if (hasReport && status === 'running') return 'active';
      return 'pending';
    }

    return 'pending';
  };

  return (
    <div className="rounded-2xl border border-slate-200/80 bg-white/80 p-5 dark:border-slate-800/80 dark:bg-slate-900/80 backdrop-blur-sm">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 mb-4 border-b border-slate-100 dark:border-slate-800/80 gap-2">
        <div>
          <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100 flex items-center gap-2">
            <span>Agent Workflow Pipeline</span>
            {status === 'waiting_for_review' && (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-semibold bg-amber-100 text-amber-800 dark:bg-amber-950/80 dark:text-amber-300 border border-amber-300 dark:border-amber-700">
                <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-ping" />
                Paused at Checkpoint
              </span>
            )}
            {status === 'running' && (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-semibold bg-sky-100 text-sky-800 dark:bg-sky-950/80 dark:text-sky-300 border border-sky-300 dark:border-sky-700">
                <span className="w-1.5 h-1.5 rounded-full bg-sky-500 animate-pulse" />
                Live Execution
              </span>
            )}
          </h3>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
            Deterministic LangGraph state machine with verified evidence loops and human governance checkpoints.
          </p>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-3 text-[11px] text-slate-500 dark:text-slate-400">
          <div className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-emerald-500" />
            <span>Done</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-sky-500 animate-pulse" />
            <span>Active</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-amber-500 animate-ping" />
            <span>Review</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-slate-300 dark:bg-slate-700" />
            <span>Pending</span>
          </div>
        </div>
      </div>

      {/* Pipeline Node Flow */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
        {WORKFLOW_NODES.map((node) => {
          const nodeStatus = getNodeStatus(node);
          const Icon = node.icon;

          const statusStyles = {
            completed:
              'border-emerald-300/80 bg-emerald-50/40 dark:border-emerald-900/60 dark:bg-emerald-950/20 text-emerald-900 dark:text-emerald-100',
            active:
              'border-sky-400 bg-sky-50/60 dark:border-sky-700 dark:bg-sky-950/40 text-sky-900 dark:text-sky-100 ring-2 ring-sky-500/20 shadow-md',
            waiting:
              'border-amber-400 bg-amber-50/70 dark:border-amber-700 dark:bg-amber-950/50 text-amber-900 dark:text-amber-100 ring-2 ring-amber-500/30 shadow-lg animate-pulse-subtle',
            pending:
              'border-slate-200/70 bg-slate-50/40 dark:border-slate-800/60 dark:bg-slate-900/40 text-slate-600 dark:text-slate-400 opacity-75',
          }[nodeStatus];

          const iconColor = {
            completed: 'text-emerald-500 dark:text-emerald-400',
            active: 'text-sky-500 dark:text-sky-400 animate-pulse',
            waiting: 'text-amber-500 dark:text-amber-400',
            pending: 'text-slate-400 dark:text-slate-600',
          }[nodeStatus];

          return (
            <div
              key={node.id}
              className={`relative flex items-start gap-3 p-3.5 rounded-xl border transition-all ${statusStyles}`}
            >
              <div
                className={`p-2 rounded-lg shrink-0 ${
                  nodeStatus === 'waiting'
                    ? 'bg-amber-100 dark:bg-amber-900/60'
                    : nodeStatus === 'active'
                    ? 'bg-sky-100 dark:bg-sky-900/60'
                    : nodeStatus === 'completed'
                    ? 'bg-emerald-100 dark:bg-emerald-900/60'
                    : 'bg-slate-200/60 dark:bg-slate-800/60'
                }`}
              >
                <Icon className={`w-4 h-4 ${iconColor}`} />
              </div>

              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between gap-1">
                  <div className="flex items-center gap-1.5">
                    <span className="text-xs font-semibold">{node.label}</span>
                    {node.isHumanCheckpoint && (
                      <span className="px-1.5 py-0.2 rounded-xs text-[9px] font-bold uppercase tracking-wider bg-indigo-100 dark:bg-indigo-950 text-indigo-700 dark:text-indigo-300 border border-indigo-200 dark:border-indigo-800">
                        HITL
                      </span>
                    )}
                  </div>
                  {nodeStatus === 'completed' && (
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500 shrink-0" />
                  )}
                  {nodeStatus === 'waiting' && (
                    <span className="w-2 h-2 rounded-full bg-amber-500 shrink-0 animate-ping" />
                  )}
                </div>
                <p className="text-[11px] leading-snug mt-1 opacity-80 line-clamp-2">
                  {node.shortDesc}
                </p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
