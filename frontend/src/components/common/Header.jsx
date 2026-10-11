import { useState, useEffect, useCallback } from 'react';
import {
  Sparkles,
  Sun,
  Moon,
  Laptop,
  Activity,
  Plus,
  RefreshCw,
  FolderGit2,
  Cpu,
  Menu,
  X,
  Wifi,
  WifiOff,
} from 'lucide-react';
import { useTheme } from '../../context/ThemeContext';
import { Button } from './Button';
import { api } from '../../api/client';

export function Header({
  activeTab,
  setActiveTab,
  onOpenNewSession,
  activeSessionId,
  sessionCount = 0,
}) {
  const { theme, toggleTheme, resolvedTheme } = useTheme();
  const [apiOnline, setApiOnline] = useState(null);
  const [checkingApi, setCheckingApi] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const checkHealth = useCallback(async () => {
    setCheckingApi(true);
    try {
      await api.healthCheck();
      setApiOnline(true);
    } catch {
      setApiOnline(false);
    } finally {
      setCheckingApi(false);
    }
  }, []);

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, [checkHealth]);

  const navItems = [
    {
      id: 'sessions',
      label: 'Sessions',
      icon: FolderGit2,
      count: sessionCount,
    },
    ...(activeSessionId
      ? [
          {
            id: 'workbench',
            label: 'Active Session',
            icon: Cpu,
            highlight: true,
          },
        ]
      : []),
    {
      id: 'observability',
      label: 'Observability',
      icon: Activity,
    },
  ];

  return (
    <header className="sticky top-0 z-40 w-full border-b border-slate-200/80 bg-white/80 backdrop-blur-md dark:border-slate-800/80 dark:bg-slate-950/80 transition-colors">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-4">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div
            onClick={() => setActiveTab('sessions')}
            className="flex items-center gap-2.5 cursor-pointer group"
          >
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-violet-600 to-purple-500 flex items-center justify-center text-white shadow-md shadow-indigo-500/25 group-hover:scale-105 transition-transform">
              <Sparkles className="w-5 h-5 animate-pulse-subtle" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-base tracking-tight text-slate-900 dark:text-white">
                  Deep Research
                </span>
                <span className="text-xs px-2 py-0.5 rounded-full font-semibold uppercase tracking-wider bg-indigo-100 text-indigo-700 dark:bg-indigo-950 dark:text-indigo-300 border border-indigo-200 dark:border-indigo-800">
                  Agent
                </span>
              </div>
              <p className="text-[11px] text-slate-500 dark:text-slate-400 font-medium hidden sm:block">
                LangGraph Autonomous Research & HITL Synthesis
              </p>
            </div>
          </div>
        </div>

        {/* Desktop Navigation */}
        <nav className="hidden md:flex items-center gap-1.5 bg-slate-100/80 dark:bg-slate-900/80 p-1 rounded-xl border border-slate-200/60 dark:border-slate-800/60">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => setActiveTab(item.id)}
                className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all cursor-pointer ${
                  isActive
                    ? 'bg-white dark:bg-slate-800 text-indigo-600 dark:text-indigo-400 shadow-xs font-semibold'
                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                <span>{item.label}</span>
                {item.count !== undefined && item.count > 0 && (
                  <span className="px-1.5 py-0.2 rounded-full text-[10px] bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300 font-mono">
                    {item.count}
                  </span>
                )}
                {item.highlight && (
                  <span className="w-1.5 h-1.5 rounded-full bg-indigo-500 animate-pulse" />
                )}
              </button>
            );
          })}
        </nav>

        {/* Actions Toolbar */}
        <div className="flex items-center gap-2 sm:gap-3">
          {/* Health indicator badge */}
          <button
            type="button"
            onClick={checkHealth}
            title={apiOnline ? 'Backend API Connected. Click to recheck.' : 'Backend API Disconnected. Click to retry.'}
            className={`hidden sm:inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border transition-colors ${
              apiOnline === true
                ? 'bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-950/40 dark:text-emerald-400 dark:border-emerald-800/60'
                : apiOnline === false
                ? 'bg-rose-50 text-rose-700 border-rose-200 dark:bg-rose-950/40 dark:text-rose-400 dark:border-rose-800/60 animate-pulse'
                : 'bg-slate-100 text-slate-600 border-slate-200 dark:bg-slate-800 dark:text-slate-400 dark:border-slate-700'
            }`}
          >
            {checkingApi ? (
              <RefreshCw className="w-3 h-3 animate-spin" />
            ) : apiOnline ? (
              <Wifi className="w-3 h-3 text-emerald-500" />
            ) : (
              <WifiOff className="w-3 h-3 text-rose-500" />
            )}
            <span>{apiOnline ? 'API Ready' : apiOnline === false ? 'API Offline' : 'Connecting...'}</span>
          </button>

          {/* Theme switcher */}
          <button
            type="button"
            onClick={toggleTheme}
            aria-label="Toggle theme"
            className="p-2 rounded-xl text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors border border-transparent hover:border-slate-200 dark:hover:border-slate-700 cursor-pointer"
            title={`Current: ${theme} (resolved: ${resolvedTheme})`}
          >
            {theme === 'system' ? (
              <Laptop className="w-4 h-4" />
            ) : resolvedTheme === 'dark' ? (
              <Moon className="w-4 h-4 text-indigo-400" />
            ) : (
              <Sun className="w-4 h-4 text-amber-500" />
            )}
          </button>

          {/* New Research CTA */}
          <Button
            variant="primary"
            size="sm"
            icon={Plus}
            onClick={onOpenNewSession}
            className="shadow-md shadow-indigo-600/20"
          >
            <span className="hidden sm:inline">New Research</span>
            <span className="sm:hidden">New</span>
          </Button>

          {/* Mobile menu toggle */}
          <button
            type="button"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden p-2 rounded-xl text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800"
          >
            {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>
      </div>

      {/* Mobile dropdown menu */}
      {mobileMenuOpen && (
        <div className="md:hidden px-4 pt-2 pb-4 border-t border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => {
                  setActiveTab(item.id);
                  setMobileMenuOpen(false);
                }}
                className={`w-full flex items-center justify-between px-3 py-2 rounded-xl text-sm font-medium ${
                  isActive
                    ? 'bg-indigo-50 dark:bg-indigo-950/60 text-indigo-600 dark:text-indigo-400'
                    : 'text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-900'
                }`}
              >
                <div className="flex items-center gap-2.5">
                  <Icon className="w-4 h-4" />
                  <span>{item.label}</span>
                </div>
                {item.count !== undefined && item.count > 0 && (
                  <span className="px-2 py-0.5 rounded-full text-xs bg-slate-200 dark:bg-slate-800">
                    {item.count}
                  </span>
                )}
              </button>
            );
          })}
          <div className="pt-2 flex items-center justify-between px-3 text-xs text-slate-500">
            <span>Server status: {apiOnline ? 'Online' : 'Offline'}</span>
            <button
              onClick={checkHealth}
              className="text-indigo-600 dark:text-indigo-400 underline"
            >
              Check ping
            </button>
          </div>
        </div>
      )}
    </header>
  );
}
