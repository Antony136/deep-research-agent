import {
  Check,
  Monitor,
  Moon,
  RotateCcw,
  Sun,
  Type,
  TextCursorInput,
  Sparkles,
} from 'lucide-react';
import { useTheme } from '../../context/ThemeContext';

const themes = [
  {
    id: 'light',
    name: 'Light',
    description: 'Bright and clean',
    icon: Sun,
    swatches: ['#f8fafc', '#ffffff', '#4f46e5'],
  },
  {
    id: 'dark',
    name: 'Dark',
    description: 'Easy on the eyes',
    icon: Moon,
    swatches: ['#020617', '#0f172a', '#818cf8'],
  },
  {
    id: 'warm',
    name: 'Warm Light',
    description: 'Soft ivory with amber accents',
    icon: Sun,
    swatches: ['#f4efe6', '#fffdf8', '#b45309'],
  },
  {
    id: 'amethyst',
    name: 'Amethyst',
    description: 'Rich plum with lavender accents',
    icon: Moon,
    swatches: ['#171321', '#241d30', '#c4a7ff'],
  },
  {
    id: 'system',
    name: 'System',
    description: 'Follow your device settings',
    icon: Monitor,
    swatches: ['#f8fafc', '#0f172a', '#818cf8'],
  },
];

const fontFamilies = [
  { id: 'sans', name: 'Sans serif', sample: 'Aa', description: 'Inter' },
  { id: 'serif', name: 'Serif', sample: 'Aa', description: 'Georgia' },
  { id: 'mono', name: 'Monospace', sample: 'Aa', description: 'JetBrains Mono' },
];

const fontSizes = [
  { id: 'small', name: 'Small', sampleSize: 'text-sm' },
  { id: 'medium', name: 'Medium', sampleSize: 'text-base' },
  { id: 'large', name: 'Large', sampleSize: 'text-lg' },
];

function SettingsSection({ icon: Icon, title, description, children }) {
  return (
    <section className="rounded-2xl border border-slate-200/80 bg-white/90 p-5 shadow-xs dark:border-slate-800/80 dark:bg-slate-900/80 sm:p-6">
      <div className="mb-5 flex items-start gap-3">
        <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-indigo-50 text-indigo-600 dark:bg-indigo-950/60 dark:text-indigo-300">
          <Icon className="h-4 w-4" />
        </span>
        <div>
          <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">{title}</h2>
          <p className="mt-1 text-xs leading-relaxed text-slate-500 dark:text-slate-400">{description}</p>
        </div>
      </div>
      {children}
    </section>
  );
}

export function SettingsPage() {
  const {
    theme,
    setTheme,
    fontFamily,
    setFontFamily,
    fontSize,
    setFontSize,
    reduceMotion,
    setReduceMotion,
    resetPreferences,
  } = useTheme();

  return (
    <div className="mx-auto max-w-5xl space-y-6 pb-12">
      <div className="flex flex-col gap-4 border-b border-slate-200/80 pb-5 dark:border-slate-800/80 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-indigo-600 dark:text-indigo-400">
            <Sparkles className="h-3.5 w-3.5" />
            Your workspace
          </p>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-slate-100">Settings</h1>
          <p className="mt-1.5 text-sm text-slate-500 dark:text-slate-400">
            Make your research workspace feel just right. Changes are saved automatically.
          </p>
        </div>
        <button
          type="button"
          onClick={resetPreferences}
          className="inline-flex items-center gap-2 self-start rounded-xl border border-slate-200 px-3 py-2 text-xs font-medium text-slate-600 transition-colors hover:bg-slate-50 focus:outline-hidden focus:ring-2 focus:ring-indigo-500 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800 sm:self-auto"
        >
          <RotateCcw className="h-3.5 w-3.5" />
          Reset to defaults
        </button>
      </div>

      <SettingsSection
        icon={Monitor}
        title="Appearance"
        description="Choose a color theme for your workspace."
      >
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {themes.map((option) => {
            const Icon = option.icon;
            const selected = theme === option.id;
            return (
              <button
                key={option.id}
                type="button"
                onClick={() => setTheme(option.id)}
                aria-pressed={selected}
                className={`group rounded-xl border p-3 text-left transition-all focus:outline-hidden focus:ring-2 focus:ring-indigo-500 ${
                  selected
                    ? 'border-indigo-500 bg-indigo-50/70 ring-1 ring-indigo-500/20 dark:border-indigo-400 dark:bg-indigo-950/30'
                    : 'border-slate-200 hover:border-slate-300 hover:bg-slate-50 dark:border-slate-700 dark:hover:border-slate-600 dark:hover:bg-slate-800/60'
                }`}
              >
                <span
                  className="mb-3 flex h-16 items-end gap-1.5 overflow-hidden rounded-lg border border-slate-200/70 p-2 dark:border-slate-700/70"
                  style={{ backgroundColor: option.swatches[0] }}
                >
                  {option.swatches.map((swatch, index) => (
                    <span
                      key={swatch}
                      className="h-7 flex-1 rounded-t-md border border-white/20"
                      style={{ backgroundColor: swatch, opacity: index === 0 ? 0.5 : 1 }}
                    />
                  ))}
                  <span
                    className="ml-1 flex h-7 w-9 items-center justify-center rounded-md"
                    style={{ backgroundColor: option.swatches[2] }}
                  >
                    <Icon className="h-3.5 w-3.5 text-white" />
                  </span>
                </span>
                <span className="flex items-center justify-between gap-2">
                  <span>
                    <span className="block text-xs font-semibold text-slate-800 dark:text-slate-100">{option.name}</span>
                    <span className="mt-0.5 block text-[11px] text-slate-500 dark:text-slate-400">{option.description}</span>
                  </span>
                  {selected && <Check className="h-4 w-4 shrink-0 text-indigo-600 dark:text-indigo-400" />}
                </span>
              </button>
            );
          })}
        </div>
      </SettingsSection>

      <div className="grid gap-6 lg:grid-cols-2">
        <SettingsSection
          icon={Type}
          title="Font style"
          description="Pick the typeface used throughout the interface."
        >
          <div className="space-y-2">
            {fontFamilies.map((option) => {
              const selected = fontFamily === option.id;
              return (
                <button
                  key={option.id}
                  type="button"
                  onClick={() => setFontFamily(option.id)}
                  aria-pressed={selected}
                  className={`flex w-full items-center justify-between rounded-xl border px-3.5 py-3 text-left transition-colors focus:outline-hidden focus:ring-2 focus:ring-indigo-500 ${
                    selected
                      ? 'border-indigo-500 bg-indigo-50/70 dark:border-indigo-400 dark:bg-indigo-950/30'
                      : 'border-slate-200 hover:bg-slate-50 dark:border-slate-700 dark:hover:bg-slate-800/60'
                  }`}
                >
                  <span className="flex items-center gap-3">
                    <span className="text-xl text-slate-800 dark:text-slate-100" style={{ fontFamily: option.id === 'serif' ? 'Georgia, serif' : option.id === 'mono' ? 'monospace' : 'Inter, sans-serif' }}>
                      {option.sample}
                    </span>
                    <span>
                      <span className="block text-xs font-medium text-slate-800 dark:text-slate-100">{option.name}</span>
                      <span className="mt-0.5 block text-[11px] text-slate-500 dark:text-slate-400">{option.description}</span>
                    </span>
                  </span>
                  {selected && <Check className="h-4 w-4 text-indigo-600 dark:text-indigo-400" />}
                </button>
              );
            })}
          </div>
        </SettingsSection>

        <SettingsSection
          icon={TextCursorInput}
          title="Text size"
          description="Adjust the overall scale of text and interface elements."
        >
          <div className="grid grid-cols-3 gap-2">
            {fontSizes.map((option) => {
              const selected = fontSize === option.id;
              return (
                <button
                  key={option.id}
                  type="button"
                  onClick={() => setFontSize(option.id)}
                  aria-pressed={selected}
                  className={`flex min-h-24 flex-col items-center justify-center gap-2 rounded-xl border px-2 py-3 transition-colors focus:outline-hidden focus:ring-2 focus:ring-indigo-500 ${
                    selected
                      ? 'border-indigo-500 bg-indigo-50/70 text-indigo-700 dark:border-indigo-400 dark:bg-indigo-950/30 dark:text-indigo-300'
                      : 'border-slate-200 text-slate-600 hover:bg-slate-50 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800/60'
                  }`}
                >
                  <span className={`${option.sampleSize} font-semibold`}>Aa</span>
                  <span className="text-[11px] font-medium">{option.name}</span>
                </button>
              );
            })}
          </div>
          <p className="mt-4 rounded-lg bg-slate-50 px-3 py-2 text-xs leading-relaxed text-slate-600 dark:bg-slate-800/60 dark:text-slate-300">
            Preview: Your workspace is ready for focused research.
          </p>
        </SettingsSection>
      </div>

      <SettingsSection
        icon={Sparkles}
        title="Motion"
        description="Reduce animations and transitions across the interface."
      >
        <label className="flex cursor-pointer items-center justify-between gap-4 rounded-xl border border-slate-200 px-4 py-3 dark:border-slate-700">
          <span>
            <span className="block text-xs font-medium text-slate-800 dark:text-slate-100">Reduce motion</span>
            <span className="mt-0.5 block text-[11px] text-slate-500 dark:text-slate-400">
              Helpful if you prefer fewer visual effects.
            </span>
          </span>
          <input
            type="checkbox"
            checked={reduceMotion}
            onChange={(event) => setReduceMotion(event.target.checked)}
            className="peer sr-only"
          />
          <span
            aria-hidden="true"
            className={`relative h-6 w-11 shrink-0 rounded-full transition-colors peer-focus-visible:ring-2 peer-focus-visible:ring-indigo-500 peer-focus-visible:ring-offset-2 dark:peer-focus-visible:ring-offset-slate-900 ${
              reduceMotion ? 'bg-indigo-600' : 'bg-slate-300 dark:bg-slate-600'
            }`}
          >
            <span
              className={`absolute left-0.5 top-0.5 h-5 w-5 rounded-full bg-white shadow-sm transition-transform ${
                reduceMotion ? 'translate-x-5' : ''
              }`}
            />
          </span>
        </label>
      </SettingsSection>
    </div>
  );
}
