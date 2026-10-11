import { createContext, useContext, useState, useCallback } from 'react';
import { CheckCircle2, AlertCircle, Info, AlertTriangle, X } from 'lucide-react';

const ToastContext = createContext();

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([]);

  const addToast = useCallback(({ message, title, type = 'info', duration = 4000 }) => {
    const id = Date.now() + Math.random().toString(36).substring(2, 7);
    setToasts((prev) => [...prev, { id, message, title, type }]);

    if (duration > 0) {
      setTimeout(() => {
        setToasts((prev) => prev.filter((t) => t.id !== id));
      }, duration);
    }
  }, []);

  const removeToast = useCallback((id) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const toast = {
    success: (message, title = 'Success') => addToast({ message, title, type: 'success' }),
    error: (message, title = 'Error') => addToast({ message, title, type: 'error', duration: 6000 }),
    info: (message, title = 'Notice') => addToast({ message, title, type: 'info' }),
    warning: (message, title = 'Warning') => addToast({ message, title, type: 'warning', duration: 5000 }),
  };

  return (
    <ToastContext.Provider value={toast}>
      {children}
      {/* Toast Render Area */}
      <div className="fixed bottom-5 right-5 z-50 flex flex-col gap-2 max-w-sm w-full pointer-events-none">
        {toasts.map((t) => {
          const typeStyles = {
            success: 'bg-emerald-950/90 border-emerald-500/40 text-emerald-100 dark:bg-emerald-950/95',
            error: 'bg-rose-950/90 border-rose-500/40 text-rose-100 dark:bg-rose-950/95',
            warning: 'bg-amber-950/90 border-amber-500/40 text-amber-100 dark:bg-amber-950/95',
            info: 'bg-slate-900/90 border-indigo-500/40 text-slate-100 dark:bg-slate-900/95',
          }[t.type];

          const IconComponent = {
            success: CheckCircle2,
            error: AlertCircle,
            warning: AlertTriangle,
            info: Info,
          }[t.type];

          const iconColor = {
            success: 'text-emerald-400',
            error: 'text-rose-400',
            warning: 'text-amber-400',
            info: 'text-indigo-400',
          }[t.type];

          return (
            <div
              key={t.id}
              className={`pointer-events-auto flex items-start gap-3 p-3.5 rounded-xl border shadow-xl backdrop-blur-md transition-all duration-300 animate-in fade-in slide-in-from-bottom-2 ${typeStyles}`}
            >
              <IconComponent className={`w-5 h-5 shrink-0 mt-0.5 ${iconColor}`} />
              <div className="flex-1 min-w-0">
                {t.title && <p className="text-xs font-semibold uppercase tracking-wider">{t.title}</p>}
                <p className="text-sm leading-snug mt-0.5 break-words">{t.message}</p>
              </div>
              <button
                type="button"
                onClick={() => removeToast(t.id)}
                className="text-slate-400 hover:text-white p-0.5 rounded-lg transition-colors"
                aria-label="Dismiss toast"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          );
        })}
      </div>
    </ToastContext.Provider>
  );
}

// eslint-disable-next-line react-refresh/only-export-components
export function useToast() {
  const context = useContext(ToastContext);
  if (!context) {
    throw new Error('useToast must be used within a ToastProvider');
  }
  return context;
}
