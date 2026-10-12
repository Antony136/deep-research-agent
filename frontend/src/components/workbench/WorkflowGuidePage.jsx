import { useMemo, useState } from 'react';
import {
  ArrowDown,
  ArrowRight,
  CheckCircle2,
  CircleHelp,
  GitBranch,
  Search,
  ShieldCheck,
  Workflow,
  X,
} from 'lucide-react';
import { WORKFLOW_NODES } from './workflowNodes';

const NODE_DETAILS = {
  __start__: {
    group: '01 · Entry & approvals',
    role: 'Workflow entry',
    purpose: 'Marks the entry point of a new LangGraph research run and hands control to the planner.',
    receives: 'The initial research request and session configuration.',
    produces: 'No research output; routes directly to Planner.',
    routing: 'Always starts at Planner.',
  },
  planner: {
    group: '01 · Entry & approvals',
    role: 'Planning',
    purpose: 'Turns the user request into a structured plan with focused research questions and web search queries.',
    receives: 'The research topic, planning instructions, and configured research limits.',
    produces: 'The initial question list and search queries used to guide research.',
    routing: 'Always continues to Initial Plan Review.',
  },
  initial_plan_review: {
    group: '01 · Entry & approvals',
    role: 'Human review',
    purpose: 'Presents the proposed plan so a person can approve it, request revisions, or reject the research.',
    receives: 'The plan produced by Planner and the reviewer’s decision.',
    produces: 'An approved plan, a revised plan awaiting review, or a rejected session.',
    routing: 'Approval advances to Research Approval. A revision loops back to this review. Rejection ends the workflow.',
    hitl: true,
  },
  initial_research_review: {
    group: '01 · Entry & approvals',
    role: 'Human review',
    purpose: 'Pauses before web research begins and obtains explicit authorization to use the approved plan.',
    receives: 'The approved research plan and the authorization decision.',
    produces: 'Authorization to begin research, or a stop decision.',
    routing: 'Authorized research advances to Researcher. A stop or denial ends the workflow.',
    hitl: true,
  },
  researcher: {
    group: '02 · Question research',
    role: 'Web research',
    purpose: 'Works through the active research question, searches the web, selects relevant sources, and gathers source content.',
    receives: 'The current question, its search queries, accepted follow-up questions, and prior research state.',
    produces: 'Collected source material and updated question progress for evidence processing.',
    routing: 'Always passes the collected material to Evidence Extractor.',
  },
  evidence_extractor: {
    group: '02 · Question research',
    role: 'Evidence extraction',
    purpose: 'Finds factual claims and supporting passages in the collected source material for the active question.',
    receives: 'The active research question and pages or source excerpts gathered by Researcher.',
    produces: 'Candidate claims paired with their source passages and attribution.',
    routing: 'Always sends candidate evidence to Evidence Verifier.',
  },
  evidence_verifier: {
    group: '02 · Question research',
    role: 'Evidence validation',
    purpose: 'Checks candidate claims against their supporting source text and separates verified evidence from unsupported claims.',
    receives: 'Extracted claims, source passages, and source metadata.',
    produces: 'Verified evidence and verification results for the current question.',
    routing: 'Always passes the accumulated research state to Research Sufficiency.',
  },
  research_sufficiency: {
    group: '02 · Question research',
    role: 'Coverage decision',
    purpose: 'Assesses progress against the research questions, evidence coverage, and configured question and round limits.',
    receives: 'The current question position, completed questions, verified evidence, and research budgets.',
    produces: 'A decision to continue the planned questions, add targeted follow-ups, or synthesize the findings.',
    routing: 'More planned questions return to Researcher. Evidence gaps within the remaining budget go to Adaptive Planner. Sufficient coverage or an exhausted budget advances to Report Synthesizer.',
  },
  adaptive_planner: {
    group: '03 · Adaptive research',
    role: 'Follow-up planning',
    purpose: 'Creates focused follow-up questions when the evidence reveals unanswered parts of the original request.',
    receives: 'The research findings, identified coverage gaps, and remaining research budgets.',
    produces: 'Proposed follow-up questions, or a decision that no useful questions can be added.',
    routing: 'Valid proposals go to Adaptive Review when human approval is enabled, or Automatic Approval otherwise. If no usable follow-ups remain, the workflow proceeds to Report Synthesizer.',
  },
  adaptive_review: {
    group: '03 · Adaptive research',
    role: 'Human review',
    purpose: 'Lets a reviewer approve or decline the proposed follow-up questions before another research round.',
    receives: 'Proposed follow-up questions and the reviewer’s decision.',
    produces: 'Accepted follow-up questions, or a decision to stop expanding research.',
    routing: 'Approved questions return to Researcher. Declined follow-ups proceed to Report Synthesizer.',
    hitl: true,
  },
  auto_accept_adaptive: {
    group: '03 · Adaptive research',
    role: 'Automatic decision',
    purpose: 'Handles proposed follow-up questions when manual approval is disabled.',
    receives: 'Proposed follow-up questions and the current question and round limits.',
    produces: 'Accepted follow-up questions when valid, or a decision to stop adding questions.',
    routing: 'Accepted questions return to Researcher. If the proposals cannot continue, the workflow proceeds to Report Synthesizer.',
  },
  synthesizer: {
    group: '04 · Reporting',
    role: 'Report generation',
    purpose: 'Combines verified evidence into a structured report that addresses the original request.',
    receives: 'Verified evidence, the research plan, source details, and coverage context.',
    produces: 'A synthesized research report with evidence and source references.',
    routing: 'Always continues to Final Report Review.',
  },
  final_report_review: {
    group: '04 · Reporting',
    role: 'Human review',
    purpose: 'Presents the report for final approval and allows the reviewer to request another research round.',
    receives: 'The synthesized report and the reviewer’s decision.',
    produces: 'An approved report, a request for more research, or a decision to finish.',
    routing: 'Approval or a finish decision ends the workflow. A request for more research returns to Adaptive Planner if question and round budgets remain; otherwise the workflow ends.',
    hitl: true,
  },
  __end__: {
    group: '04 · Reporting',
    role: 'Workflow exit',
    purpose: 'Marks a terminal state: research was approved, rejected, stopped, or finished within its configured limits.',
    receives: 'A terminal decision from an approval checkpoint or the report review.',
    produces: 'A completed workflow state for the research session.',
    routing: 'No further nodes run after this point.',
  },
};

const GROUPS = [
  '01 · Entry & approvals',
  '02 · Question research',
  '03 · Adaptive research',
  '04 · Reporting',
];

function DetailField({ label, children, icon: Icon }) {
  return (
    <div className="flex gap-3">
      <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-slate-100 text-slate-500 dark:bg-slate-800 dark:text-slate-400">
        <Icon className="h-3.5 w-3.5" />
      </span>
      <div className="min-w-0">
        <h4 className="text-[10px] font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">{label}</h4>
        <p className="mt-1 text-xs leading-relaxed text-slate-600 dark:text-slate-300">{children}</p>
      </div>
    </div>
  );
}

export function WorkflowGuidePage() {
  const [query, setQuery] = useState('');
  const normalizedQuery = query.trim().toLowerCase();
  const filteredNodes = useMemo(
    () => WORKFLOW_NODES.filter((node) => {
      const details = NODE_DETAILS[node.id];
      const searchable = [
        node.id,
        node.label,
        node.description,
        details.role,
        details.purpose,
        details.receives,
        details.produces,
        details.routing,
        details.group,
      ].join(' ').toLowerCase();
      return searchable.includes(normalizedQuery);
    }),
    [normalizedQuery],
  );

  return (
    <div className="mx-auto max-w-6xl space-y-8 pb-12">
      <section className="relative overflow-hidden rounded-3xl border border-indigo-200/70 bg-gradient-to-br from-indigo-50 via-white to-violet-50 p-6 dark:border-indigo-900/60 dark:from-indigo-950/50 dark:via-slate-900 dark:to-violet-950/30 sm:p-9">
        <div className="pointer-events-none absolute -right-14 -top-20 h-64 w-64 rounded-full bg-indigo-300/20 blur-3xl dark:bg-indigo-500/10" />
        <div className="relative grid gap-7 lg:grid-cols-[1fr_auto] lg:items-end">
          <div className="max-w-2xl">
            <span className="inline-flex items-center gap-2 rounded-full border border-indigo-200 bg-white/80 px-3 py-1.5 text-[11px] font-semibold uppercase tracking-wider text-indigo-700 dark:border-indigo-800 dark:bg-slate-900/70 dark:text-indigo-300">
              <Workflow className="h-3.5 w-3.5" />
              Workflow reference
            </span>
            <h1 className="mt-4 text-3xl font-bold tracking-tight text-slate-950 dark:text-white sm:text-4xl">
              How research moves
              <span className="block text-indigo-600 dark:text-indigo-400">from question to report</span>
            </h1>
            <p className="mt-4 max-w-xl text-sm leading-relaxed text-slate-600 dark:text-slate-300">
              Explore what every LangGraph node does, what information it handles, and how its decisions route the research workflow.
            </p>
          </div>

          <div className="grid grid-cols-3 gap-2 sm:gap-3">
            {[
              { value: WORKFLOW_NODES.length - 2, label: 'Processing nodes' },
              { value: WORKFLOW_NODES.filter((node) => NODE_DETAILS[node.id].hitl).length, label: 'Review points' },
              { value: GROUPS.length, label: 'Workflow stages' },
            ].map((stat) => (
              <div key={stat.label} className="min-w-0 rounded-2xl border border-white/80 bg-white/75 px-3 py-3 text-center shadow-sm dark:border-slate-700/70 dark:bg-slate-900/70 sm:px-5">
                <div className="text-xl font-bold tabular-nums text-slate-900 dark:text-white">{stat.value}</div>
                <div className="mt-1 text-[10px] font-medium leading-tight text-slate-500 dark:text-slate-400">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="relative mt-7 flex flex-col gap-3 border-t border-indigo-200/70 pt-5 dark:border-indigo-900/60 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-2 text-xs text-slate-600 dark:text-slate-300">
            <GitBranch className="h-4 w-4 shrink-0 text-indigo-500" />
            The workflow loops through each question and can branch into follow-up research.
          </div>
          <label className="relative block w-full sm:max-w-xs">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
            <input
              type="search"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Find a node or decision..."
              aria-label="Search workflow nodes"
              className="w-full rounded-xl border border-slate-200 bg-white py-2.5 pl-9 pr-9 text-xs text-slate-800 outline-none transition focus:border-indigo-400 focus:ring-2 focus:ring-indigo-500/20 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100 dark:focus:border-indigo-500"
            />
            {query && (
              <button
                type="button"
                onClick={() => setQuery('')}
                aria-label="Clear node search"
                className="absolute right-2 top-1/2 -translate-y-1/2 rounded-md p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-700 dark:hover:bg-slate-800 dark:hover:text-slate-200"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            )}
          </label>
        </div>
      </section>

      {GROUPS.map((group, groupIndex) => {
        const nodes = filteredNodes.filter((node) => NODE_DETAILS[node.id].group === group);
        if (!nodes.length) return null;

        return (
          <section key={group} aria-labelledby={`workflow-group-${groupIndex}`}>
            <div className="mb-4 flex items-end justify-between gap-4 border-b border-slate-200 pb-3 dark:border-slate-800">
              <div>
                <p className="text-[10px] font-bold uppercase tracking-[0.18em] text-indigo-600 dark:text-indigo-400">
                  Stage {String(groupIndex + 1).padStart(2, '0')}
                </p>
                <h2 id={`workflow-group-${groupIndex}`} className="mt-1 text-lg font-semibold text-slate-900 dark:text-slate-100">
                  {group.slice(5)}
                </h2>
              </div>
              <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[10px] font-semibold tabular-nums text-slate-600 dark:bg-slate-800 dark:text-slate-300">
                {nodes.length} {nodes.length === 1 ? 'node' : 'nodes'}
              </span>
            </div>

            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              {nodes.map((node) => {
                const Icon = node.icon;
                const details = NODE_DETAILS[node.id];
                const isEndpoint = node.id === '__start__' || node.id === '__end__';

                return (
                  <article
                    key={node.id}
                    className={`group overflow-hidden rounded-2xl border bg-white shadow-sm transition hover:-translate-y-0.5 hover:shadow-md dark:bg-slate-900 ${
                      details.hitl
                        ? 'border-violet-200 hover:border-violet-300 dark:border-violet-900/70 dark:hover:border-violet-700'
                        : 'border-slate-200 hover:border-indigo-200 dark:border-slate-800 dark:hover:border-indigo-900'
                    }`}
                  >
                    <div className="flex items-start gap-3.5 p-4 sm:p-5">
                      <span className={`flex h-11 w-11 shrink-0 items-center justify-center rounded-xl ${
                        isEndpoint
                          ? 'bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300'
                          : details.hitl
                            ? 'bg-violet-50 text-violet-600 dark:bg-violet-950/60 dark:text-violet-300'
                            : 'bg-indigo-50 text-indigo-600 dark:bg-indigo-950/60 dark:text-indigo-300'
                      }`}>
                        <Icon className="h-5 w-5" />
                      </span>
                      <div className="min-w-0 flex-1">
                        <div className="flex flex-wrap items-center gap-2">
                          <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">{node.label}</h3>
                          {details.hitl && (
                            <span className="inline-flex items-center gap-1 rounded-full bg-violet-100 px-2 py-0.5 text-[9px] font-bold uppercase tracking-wide text-violet-700 dark:bg-violet-950 dark:text-violet-300">
                              <ShieldCheck className="h-3 w-3" />
                              Human review
                            </span>
                          )}
                          {isEndpoint && (
                            <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[9px] font-bold uppercase tracking-wide text-slate-600 dark:bg-slate-800 dark:text-slate-300">
                              {node.id === '__start__' ? 'Entry' : 'Terminal'}
                            </span>
                          )}
                        </div>
                        <p className="mt-1 font-mono text-[10px] text-slate-400 dark:text-slate-500">{node.id}</p>
                        <p className="mt-2 text-[11px] font-semibold uppercase tracking-wider text-indigo-600 dark:text-indigo-400">
                          {details.role}
                        </p>
                      </div>
                    </div>

                    <div className="space-y-4 border-t border-slate-100 bg-slate-50/70 p-4 dark:border-slate-800 dark:bg-slate-950/30 sm:p-5">
                      <p className="text-xs leading-relaxed text-slate-700 dark:text-slate-300">{details.purpose}</p>
                      <div className="grid gap-4 sm:grid-cols-2">
                        <DetailField label="Receives" icon={ArrowDown}>{details.receives}</DetailField>
                        <DetailField label="Produces" icon={CheckCircle2}>{details.produces}</DetailField>
                      </div>
                      <div className="flex gap-3 rounded-xl border border-indigo-100 bg-indigo-50/70 p-3 dark:border-indigo-900/60 dark:bg-indigo-950/30">
                        <ArrowRight className="mt-0.5 h-4 w-4 shrink-0 text-indigo-500" />
                        <div>
                          <h4 className="text-[10px] font-bold uppercase tracking-wider text-indigo-700 dark:text-indigo-300">Next / routing</h4>
                          <p className="mt-1 text-xs leading-relaxed text-indigo-950/80 dark:text-indigo-100/80">{details.routing}</p>
                        </div>
                      </div>
                    </div>
                  </article>
                );
              })}
            </div>
          </section>
        );
      })}

      {filteredNodes.length === 0 && (
        <div className="rounded-2xl border border-dashed border-slate-300 bg-white/70 px-6 py-14 text-center dark:border-slate-700 dark:bg-slate-900/50">
          <CircleHelp className="mx-auto h-8 w-8 text-slate-400" />
          <h2 className="mt-3 text-sm font-semibold text-slate-900 dark:text-slate-100">No matching workflow nodes</h2>
          <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">Try searching by node name, responsibility, or routing decision.</p>
          <button
            type="button"
            onClick={() => setQuery('')}
            className="mt-4 rounded-lg bg-indigo-600 px-3 py-2 text-xs font-semibold text-white hover:bg-indigo-500"
          >
            Clear search
          </button>
        </div>
      )}

      <p className="text-center text-[11px] text-slate-500 dark:text-slate-400">
        Node behavior and routing are based on the configured research workflow. Human review checkpoints pause execution until a decision is submitted.
      </p>
    </div>
  );
}
