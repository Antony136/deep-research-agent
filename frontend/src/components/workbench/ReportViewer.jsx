import { useState } from 'react';
import {
  FileText,
  Copy,
  Check,
  Download,
  ExternalLink,
  ShieldCheck,
  AlertTriangle,
  Layers,
  ChevronDown,
  ChevronUp,
  Bookmark,
  Printer,
  Sparkles,
} from 'lucide-react';
import { CoverageBadge } from '../common/Badge';
import { Button } from '../common/Button';
import { useToast } from '../../context/ToastContext';

export function ReportViewer({ report, evidence = [], onSelectEvidence }) {
  const toast = useToast();
  const [copied, setCopied] = useState(false);
  const [expandedFinding, setExpandedFinding] = useState(null);

  if (!report) {
    return (
      <div className="p-12 text-center rounded-2xl border border-dashed border-slate-300 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/30">
        <FileText className="w-10 h-10 text-slate-400 mx-auto mb-3" />
        <h3 className="text-base font-semibold text-slate-800 dark:text-slate-200">
          Report In Progress
        </h3>
        <p className="text-xs text-slate-500 dark:text-slate-400 max-w-md mx-auto mt-1">
          The synthesized research report will be generated and displayed here once evidence extraction, verification, and sufficiency evaluations complete.
        </p>
      </div>
    );
  }

  const {
    title = 'Research Report',
    summary = '',
    findings = [],
    coverage = [],
    research_gaps = [],
    sources = [],
  } = report;

  // Format Report as clean Markdown for export
  const generateMarkdown = () => {
    let md = `# ${title}\n\n`;
    md += `## Executive Summary\n\n${summary}\n\n`;

    if (findings.length > 0) {
      md += `## Key Findings\n\n`;
      findings.forEach((f, idx) => {
        const evRefs = (f.evidence_numbers || []).map((e) => `[E#${e}]`).join(' ');
        const qRefs = (f.research_question_numbers || []).map((q) => `Q${q}`).join(', ');
        md += `### Finding ${idx + 1} (${qRefs})\n${f.text} ${evRefs}\n\n`;
      });
    }

    if (coverage.length > 0) {
      md += `## Question Coverage\n\n`;
      coverage.forEach((c) => {
        md += `### Q${c.research_question_number}: ${c.question}\n`;
        md += `- **Status**: ${c.status}\n`;
        md += `- **Assessment**: ${c.explanation}\n\n`;
      });
    }

    if (research_gaps.length > 0) {
      md += `## Research Gaps & Limitations\n\n`;
      research_gaps.forEach((gap) => {
        md += `- ${gap}\n`;
      });
      md += `\n`;
    }

    if (sources.length > 0) {
      md += `## Verified Sources\n\n`;
      sources.forEach((s, idx) => {
        md += `${idx + 1}. ${s}\n`;
      });
    }

    return md;
  };

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(generateMarkdown());
      setCopied(true);
      toast.success('Report copied to clipboard in Markdown format');
      setTimeout(() => setCopied(false), 2500);
    } catch {
      toast.error('Failed to copy report');
    }
  };

  const handleDownloadMd = () => {
    const md = generateMarkdown();
    const blob = new Blob([md], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${title.toLowerCase().replace(/[^a-z0-9]+/g, '-').slice(0, 40)}.md`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success('Downloaded report as Markdown file');
  };

  const handleDownloadJson = () => {
    const blob = new Blob([JSON.stringify(report, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${title.toLowerCase().replace(/[^a-z0-9]+/g, '-').slice(0, 40)}.json`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success('Downloaded report as JSON file');
  };

  const handlePrint = () => {
    window.print();
  };

  return (
    <div className="space-y-6">
      {/* Top Action Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-200 dark:border-slate-800 gap-3">
        <div className="flex items-center gap-2">
          <div className="p-2 rounded-xl bg-indigo-50 dark:bg-indigo-950/80 text-indigo-600 dark:text-indigo-400 border border-indigo-200 dark:border-indigo-800">
            <Sparkles className="w-5 h-5" />
          </div>
          <div>
            <span className="text-xs font-bold uppercase tracking-wider text-indigo-600 dark:text-indigo-400">
              Verified Synthesis
            </span>
            <h2 className="text-lg font-bold text-slate-900 dark:text-slate-100">
              Research Report
            </h2>
          </div>
        </div>

        {/* Export & Action Buttons */}
        <div className="flex flex-wrap items-center gap-2">
          <Button
            size="sm"
            variant="outline"
            icon={copied ? Check : Copy}
            onClick={handleCopy}
            title="Copy entire report in Markdown"
          >
            {copied ? 'Copied' : 'Copy MD'}
          </Button>

          <Button
            size="sm"
            variant="outline"
            icon={Download}
            onClick={handleDownloadMd}
            title="Download report as .md"
          >
            Markdown
          </Button>

          <Button
            size="sm"
            variant="outline"
            icon={Download}
            onClick={handleDownloadJson}
            title="Download raw report as JSON"
          >
            JSON
          </Button>

          <Button
            size="sm"
            variant="ghost"
            icon={Printer}
            onClick={handlePrint}
            title="Print or save as PDF"
            className="hidden sm:inline-flex"
          >
            Print
          </Button>
        </div>
      </div>

      {/* Title & Executive Summary Card */}
      <div className="p-6 rounded-2xl border border-slate-200/90 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-xs space-y-3">
        <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-slate-900 dark:text-white leading-snug">
          {title}
        </h1>

        <div className="pt-2">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-1.5">
            Executive Summary
          </h3>
          <p className="text-sm sm:text-base text-slate-700 dark:text-slate-300 leading-relaxed">
            {summary}
          </p>
        </div>
      </div>

      {/* Key Findings Section */}
      {findings.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400 flex items-center gap-2">
              <Bookmark className="w-3.5 h-3.5 text-indigo-500" />
              <span>Evidence-Backed Findings ({findings.length})</span>
            </h3>
            <span className="text-[11px] text-slate-400">
              Citations link to deterministic verified quotes
            </span>
          </div>

          <div className="grid grid-cols-1 gap-3">
            {findings.map((finding, idx) => {
              const isExpanded = expandedFinding === idx;
              const findingEvidence = (finding.evidence_numbers || [])
                .map((eNum) => ({
                  number: eNum,
                  data: evidence[eNum - 1],
                }))
                .filter((item) => item.data);

              return (
                <div
                  key={idx}
                  className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-white/90 dark:bg-slate-900/80 shadow-xs space-y-2.5 transition-all"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex items-start gap-2.5">
                      <span className="shrink-0 w-6 h-6 rounded-full bg-indigo-100 dark:bg-indigo-950/80 text-indigo-700 dark:text-indigo-300 text-xs font-bold flex items-center justify-center mt-0.5">
                        {idx + 1}
                      </span>
                      <p className="text-sm font-medium text-slate-800 dark:text-slate-200 leading-relaxed">
                        {finding.text}
                      </p>
                    </div>

                    {findingEvidence.length > 0 && (
                      <button
                        type="button"
                        onClick={() => setExpandedFinding(isExpanded ? null : idx)}
                        className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 p-1 rounded-lg shrink-0 cursor-pointer"
                        title="Toggle evidence quotes"
                      >
                        {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                      </button>
                    )}
                  </div>

                  {/* Metadata & Citation Pills */}
                  <div className="flex flex-wrap items-center gap-2 pt-1 border-t border-slate-100 dark:border-slate-800/80 text-xs">
                    {/* Questions */}
                    {(finding.research_question_numbers || []).map((qNum) => (
                      <span
                        key={qNum}
                        className="px-2 py-0.5 rounded-md text-[11px] font-semibold bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400"
                      >
                        Question #{qNum}
                      </span>
                    ))}

                    {/* Evidence Citations */}
                    {(finding.evidence_numbers || []).map((eNum) => (
                      <button
                        key={eNum}
                        type="button"
                        onClick={() => {
                          if (onSelectEvidence) onSelectEvidence(eNum);
                          setExpandedFinding(idx);
                        }}
                        className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[11px] font-mono font-medium bg-emerald-50 dark:bg-emerald-950/60 text-emerald-700 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800 hover:bg-emerald-100 transition-colors cursor-pointer"
                        title="View verified evidence claim & quote"
                      >
                        <ShieldCheck className="w-3 h-3 text-emerald-500" />
                        <span>Evidence #{eNum}</span>
                      </button>
                    ))}
                  </div>

                  {/* Expanded Quote Drawer */}
                  {isExpanded && findingEvidence.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-slate-100 dark:border-slate-800 space-y-2">
                      <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
                        Supporting Evidence Claims:
                      </span>
                      {findingEvidence.map(({ number, data }) => (
                        <div
                          key={number}
                          className="p-3 rounded-lg bg-slate-50 dark:bg-slate-950/60 border border-slate-200/80 dark:border-slate-800 text-xs space-y-1.5"
                        >
                          <div className="flex items-center justify-between text-[11px]">
                            <span className="font-semibold text-emerald-600 dark:text-emerald-400">
                              Claim #{number}: {data.claim}
                            </span>
                            {data.source_url && (
                              <a
                                href={data.source_url}
                                target="_blank"
                                rel="noreferrer"
                                className="text-indigo-600 dark:text-indigo-400 hover:underline flex items-center gap-0.5"
                              >
                                Source <ExternalLink className="w-2.5 h-2.5" />
                              </a>
                            )}
                          </div>
                          {data.supporting_text && (
                            <p className="text-slate-600 dark:text-slate-400 italic line-clamp-3">
                              "{data.supporting_text}"
                            </p>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Question Coverage Matrix */}
      {coverage.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400 flex items-center gap-2">
            <Layers className="w-3.5 h-3.5 text-indigo-500" />
            <span>Research Question Coverage Assessment</span>
          </h3>

          <div className="grid grid-cols-1 gap-3">
            {coverage.map((c) => (
              <div
                key={c.research_question_number}
                className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-white/90 dark:bg-slate-900/80 space-y-2"
              >
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <span className="px-2 py-0.5 rounded-md text-xs font-bold bg-indigo-100 dark:bg-indigo-950 text-indigo-700 dark:text-indigo-300">
                      Q{c.research_question_number}
                    </span>
                    <span className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                      {c.question}
                    </span>
                  </div>
                  <CoverageBadge status={c.status} />
                </div>

                <p className="text-xs sm:text-sm text-slate-600 dark:text-slate-300 leading-relaxed pl-1">
                  {c.explanation}
                </p>

                {c.finding_numbers && c.finding_numbers.length > 0 && (
                  <div className="flex items-center gap-1.5 text-xs text-slate-400 pt-1">
                    <span>Addresses Findings:</span>
                    {c.finding_numbers.map((fNum) => (
                      <span
                        key={fNum}
                        className="px-1.5 py-0.2 rounded-sm bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 font-mono text-[11px]"
                      >
                        #{fNum}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Research Gaps / Limitations */}
      {research_gaps.length > 0 && (
        <div className="p-5 rounded-2xl border border-amber-200 dark:border-amber-900/60 bg-amber-50/40 dark:bg-amber-950/20 space-y-3">
          <h3 className="text-xs font-bold uppercase tracking-wider text-amber-800 dark:text-amber-300 flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-amber-500" />
            <span>Identified Research Gaps & Unresolved Limitations</span>
          </h3>
          <ul className="space-y-2 text-xs sm:text-sm text-slate-700 dark:text-slate-300">
            {research_gaps.map((gap, gIdx) => (
              <li key={gIdx} className="flex items-start gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-amber-500 shrink-0 mt-2" />
                <span className="leading-relaxed">{gap}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Verified Web Sources */}
      {sources.length > 0 && (
        <div className="p-5 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white/80 dark:bg-slate-900/80 space-y-3">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
            Deterministically Verified Sources ({sources.length})
          </h3>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
            {sources.map((sourceUrl, sIdx) => {
              let domain;
              try {
                domain = new URL(sourceUrl).hostname.replace('www.', '');
              } catch {
                domain = 'web';
              }

              return (
                <a
                  key={sIdx}
                  href={sourceUrl}
                  target="_blank"
                  rel="noreferrer"
                  className="flex items-center justify-between p-2.5 rounded-xl border border-slate-200 dark:border-slate-800 hover:border-indigo-300 dark:hover:border-indigo-700 bg-slate-50/50 dark:bg-slate-950/40 text-slate-700 dark:text-slate-300 hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors group cursor-pointer"
                >
                  <div className="flex items-center gap-2 min-w-0">
                    <span className="px-1.5 py-0.5 rounded-xs text-[10px] font-mono bg-slate-200 dark:bg-slate-800 shrink-0">
                      {domain}
                    </span>
                    <span className="truncate text-xs">{sourceUrl}</span>
                  </div>
                  <ExternalLink className="w-3.5 h-3.5 opacity-50 group-hover:opacity-100 shrink-0 ml-1.5" />
                </a>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
