import { useState, useEffect, useCallback } from 'react';
import { ThemeProvider } from './context/ThemeContext';
import { ToastProvider, useToast } from './context/ToastContext';
import { ErrorBoundary } from './components/common/ErrorBoundary';
import { Header } from './components/common/Header';
import { SessionsList } from './components/sessions/SessionsList';
import { CreateSessionModal } from './components/sessions/CreateSessionModal';
import { SessionWorkbench } from './components/workbench/SessionWorkbench';
import { ObservabilityHub } from './components/observability/ObservabilityHub';
import { SettingsPage } from './components/settings/SettingsPage';
import { WorkflowGuidePage } from './components/workbench/WorkflowGuidePage';
import { api } from './api/client';
import { WifiOff } from 'lucide-react';

function AppContent() {
  const toast = useToast();
  const [sessions, setSessions] = useState([]);
  const [loadingSessions, setLoadingSessions] = useState(true);
  const [activeSessionId, setActiveSessionId] = useState(() => {
    const hash = window.location.hash;
    if (hash && hash.startsWith('#session-')) {
      return hash.replace('#session-', '');
    }
    return null;
  });
  const [activeTab, setActiveTab] = useState(() => {
    const hash = window.location.hash;
    return hash && hash.startsWith('#session-') ? 'workbench' : 'sessions';
  });
  const [isNewModalOpen, setIsNewModalOpen] = useState(false);
  const [isOffline, setIsOffline] = useState(false);

  // Sync hash with active session
  useEffect(() => {
    if (activeSessionId) {
      window.location.hash = `session-${activeSessionId}`;
    } else {
      if (window.location.hash.startsWith('#session-')) {
        history.replaceState(null, '', window.location.pathname);
      }
    }
  }, [activeSessionId]);

  // Handle browser back/forward buttons
  useEffect(() => {
    const handleHashChange = () => {
      const hash = window.location.hash;
      if (hash && hash.startsWith('#session-')) {
        const id = hash.replace('#session-', '');
        setActiveSessionId(id);
        setActiveTab('workbench');
      } else if (!hash) {
        setActiveSessionId(null);
        setActiveTab('sessions');
      }
    };
    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  // Fetch recent sessions
  const fetchSessions = useCallback(async () => {
    setLoadingSessions(true);
    try {
      const data = await api.listSessions(50);
      setSessions(data.sessions || []);
      setIsOffline(false);
    } catch {
      setIsOffline(true);
      toast.error('Unable to reach research server. Please verify backend is running.');
    } finally {
      setLoadingSessions(false);
    }
  }, [toast]);

  useEffect(() => {
    fetchSessions();
  }, [fetchSessions]);

  // Navigate to workbench for a specific session
  const handleSelectSession = (sessionId) => {
    setActiveSessionId(sessionId);
    setActiveTab('workbench');
  };

  // Called when a new session is scheduled
  const handleSessionCreated = (newSessionId) => {
    fetchSessions();
    setActiveSessionId(newSessionId);
    setActiveTab('workbench');
  };

  // Called when a session is deleted
  const handleSessionDeleted = () => {
    fetchSessions();
    setActiveSessionId(null);
    setActiveTab('sessions');
  };

  return (
    <div className="app-shell min-h-screen flex flex-col transition-colors selection:bg-indigo-500/20 selection:text-indigo-600 dark:selection:text-indigo-300">
      {/* Offline Alert Banner */}
      {isOffline && (
        <div className="bg-rose-600 text-white px-4 py-2 text-xs font-medium flex items-center justify-center gap-2 shadow-sm">
          <WifiOff className="w-4 h-4 shrink-0" />
          <span>Cannot reach backend API at {import.meta.env.VITE_API_BASE_URL || 'localhost:8000'}. Make sure the server is running.</span>
          <button
            onClick={fetchSessions}
            className="underline hover:text-rose-100 ml-2 font-semibold cursor-pointer"
          >
            Retry Connection
          </button>
        </div>
      )}

      {/* Main Top Header */}
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        onOpenNewSession={() => setIsNewModalOpen(true)}
        activeSessionId={activeSessionId}
        sessionCount={sessions.length}
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {activeTab === 'sessions' && (
          <SessionsList
            sessions={sessions}
            loading={loadingSessions}
            onRefresh={fetchSessions}
            onSelectSession={handleSelectSession}
            onOpenNewSession={() => setIsNewModalOpen(true)}
          />
        )}

        {activeTab === 'workbench' && (
          <SessionWorkbench
            sessionId={activeSessionId}
            onBack={() => {
              setActiveSessionId(null);
              setActiveTab('sessions');
            }}
            onSessionDeleted={handleSessionDeleted}
          />
        )}

        {activeTab === 'observability' && (
          <ObservabilityHub
            onSelectSession={handleSelectSession}
          />
        )}

        {activeTab === 'workflow' && <WorkflowGuidePage />}

        {activeTab === 'settings' && <SettingsPage />}
      </main>

      {/* New Research Session Modal */}
      <CreateSessionModal
        isOpen={isNewModalOpen}
        onClose={() => setIsNewModalOpen(false)}
        onSessionCreated={handleSessionCreated}
      />
    </div>
  );
}

export default function App() {
  return (
    <ErrorBoundary>
      <ThemeProvider>
        <ToastProvider>
          <AppContent />
        </ToastProvider>
      </ThemeProvider>
    </ErrorBoundary>
  );
}
