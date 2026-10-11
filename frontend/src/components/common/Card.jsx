export function Card({ children, className = '', hover = false, onClick }) {
  return (
    <div
      onClick={onClick}
      className={`rounded-2xl border border-slate-200/80 bg-white/90 p-5 shadow-xs backdrop-blur-md transition-all dark:border-slate-800/80 dark:bg-slate-900/80 ${
        hover ? 'hover:border-slate-300 hover:shadow-md dark:hover:border-slate-700 cursor-pointer' : ''
      } ${className}`}
    >
      {children}
    </div>
  );
}

export function CardHeader({ title, subtitle, badge, action, className = '' }) {
  return (
    <div className={`flex items-start justify-between gap-4 mb-4 ${className}`}>
      <div>
        <div className="flex items-center gap-2.5">
          {typeof title === 'string' ? (
            <h3 className="text-base font-semibold text-slate-900 dark:text-slate-100">{title}</h3>
          ) : (
            title
          )}
          {badge}
        </div>
        {subtitle && <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">{subtitle}</p>}
      </div>
      {action && <div className="shrink-0">{action}</div>}
    </div>
  );
}
