import { Loader2 } from 'lucide-react';

export function Button({
  children,
  variant = 'primary',
  size = 'md',
  isLoading = false,
  disabled = false,
  icon: Icon,
  className = '',
  type = 'button',
  onClick,
  ...props
}) {
  const baseClasses =
    'inline-flex items-center justify-center font-medium transition-all duration-150 rounded-xl focus:outline-hidden focus:ring-2 focus:ring-offset-2 dark:focus:ring-offset-slate-900 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed select-none active:scale-[0.98] disabled:active:scale-100';

  const sizeClasses = {
    sm: 'text-xs px-2.5 py-1.5 gap-1.5',
    md: 'text-sm px-4 py-2 gap-2',
    lg: 'text-base px-5 py-2.5 gap-2.5',
  }[size] || 'text-sm px-4 py-2 gap-2';

  const variantClasses = {
    primary:
      'bg-indigo-600 hover:bg-indigo-500 text-white shadow-sm shadow-indigo-500/20 focus:ring-indigo-500 dark:bg-indigo-500 dark:hover:bg-indigo-400',
    secondary:
      'bg-slate-200/80 hover:bg-slate-300/80 text-slate-800 focus:ring-slate-400 dark:bg-slate-800 dark:hover:bg-slate-700 dark:text-slate-200',
    outline:
      'border border-slate-300 dark:border-slate-700 bg-transparent hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-300 focus:ring-slate-400',
    danger:
      'bg-rose-600 hover:bg-rose-500 text-white shadow-sm shadow-rose-500/20 focus:ring-rose-500 dark:bg-rose-600 dark:hover:bg-rose-500',
    success:
      'bg-emerald-600 hover:bg-emerald-500 text-white shadow-sm shadow-emerald-500/20 focus:ring-emerald-500 dark:bg-emerald-600 dark:hover:bg-emerald-500',
    ghost:
      'bg-transparent hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-400 focus:ring-slate-400',
  }[variant] || 'bg-indigo-600 text-white';

  return (
    <button
      type={type}
      disabled={disabled || isLoading}
      onClick={onClick}
      className={`${baseClasses} ${sizeClasses} ${variantClasses} ${className}`}
      {...props}
    >
      {isLoading ? (
        <Loader2 className="w-4 h-4 animate-spin shrink-0" />
      ) : Icon ? (
        <Icon className="w-4 h-4 shrink-0" />
      ) : null}
      <span>{children}</span>
    </button>
  );
}
