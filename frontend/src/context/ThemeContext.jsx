import { createContext, useContext, useEffect, useState } from 'react';

const THEME_OPTIONS = ['system', 'light', 'dark', 'warm', 'amethyst'];
const FONT_FAMILY_OPTIONS = ['sans', 'serif', 'mono'];
const FONT_SIZE_OPTIONS = ['small', 'medium', 'large'];
const DEFAULT_PREFERENCES = {
  theme: 'system',
  fontFamily: 'sans',
  fontSize: 'medium',
  reduceMotion: false,
};
const FONT_FAMILIES = {
  sans: "'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
  serif: "Georgia, 'Times New Roman', serif",
  mono: "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
};
const DARK_THEMES = ['dark', 'amethyst'];
const STORAGE_KEYS = {
  theme: 'dra_theme',
  fontFamily: 'dra_font_family',
  fontSize: 'dra_font_size',
  reduceMotion: 'dra_reduce_motion',
};
const ThemeContext = createContext(null);

export function ThemeProvider({ children }) {
  const [theme, setTheme] = useState(() => {
    const storedTheme = localStorage.getItem(STORAGE_KEYS.theme);
    return THEME_OPTIONS.includes(storedTheme) ? storedTheme : DEFAULT_PREFERENCES.theme;
  });
  const [fontFamily, setFontFamily] = useState(() => {
    const storedFontFamily = localStorage.getItem(STORAGE_KEYS.fontFamily);
    return FONT_FAMILY_OPTIONS.includes(storedFontFamily) ? storedFontFamily : DEFAULT_PREFERENCES.fontFamily;
  });
  const [fontSize, setFontSize] = useState(() => {
    const storedFontSize = localStorage.getItem(STORAGE_KEYS.fontSize);
    return FONT_SIZE_OPTIONS.includes(storedFontSize) ? storedFontSize : DEFAULT_PREFERENCES.fontSize;
  });
  const [reduceMotion, setReduceMotion] = useState(() => {
    return localStorage.getItem(STORAGE_KEYS.reduceMotion) === 'true';
  });
  const [resolvedTheme, setResolvedTheme] = useState(() => {
    if (theme === 'system') {
      return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }
    return DARK_THEMES.includes(theme) ? 'dark' : 'light';
  });

  useEffect(() => {
    const root = document.documentElement;

    function applyTheme() {
      const isSystemTheme = theme === 'system';
      const isDark = isSystemTheme
        ? window.matchMedia('(prefers-color-scheme: dark)').matches
        : DARK_THEMES.includes(theme);
      const nextResolvedTheme = isDark ? 'dark' : 'light';

      setResolvedTheme(nextResolvedTheme);
      root.dataset.theme = theme === 'system' ? nextResolvedTheme : theme;
      root.style.colorScheme = nextResolvedTheme;
      if (isDark) {
        root.classList.add('dark');
      } else {
        root.classList.remove('dark');
      }
    }

    applyTheme();
    localStorage.setItem(STORAGE_KEYS.theme, theme);

    if (theme === 'system') {
      const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
      const listener = () => applyTheme();
      mediaQuery.addEventListener('change', listener);
      return () => mediaQuery.removeEventListener('change', listener);
    }
  }, [theme]);

  useEffect(() => {
    const root = document.documentElement;
    root.style.setProperty('--font-sans', FONT_FAMILIES[fontFamily]);
    root.dataset.fontSize = fontSize;
    root.dataset.reduceMotion = String(reduceMotion);
    localStorage.setItem(STORAGE_KEYS.fontFamily, fontFamily);
    localStorage.setItem(STORAGE_KEYS.fontSize, fontSize);
    localStorage.setItem(STORAGE_KEYS.reduceMotion, String(reduceMotion));
  }, [fontFamily, fontSize, reduceMotion]);

  const toggleTheme = () => setTheme(resolvedTheme === 'dark' ? 'light' : 'dark');

  const resetPreferences = () => {
    setTheme(DEFAULT_PREFERENCES.theme);
    setFontFamily(DEFAULT_PREFERENCES.fontFamily);
    setFontSize(DEFAULT_PREFERENCES.fontSize);
    setReduceMotion(DEFAULT_PREFERENCES.reduceMotion);
  };

  return (
    <ThemeContext.Provider
      value={{
        theme,
        setTheme,
        resolvedTheme,
        toggleTheme,
        fontFamily,
        setFontFamily,
        fontSize,
        setFontSize,
        reduceMotion,
        setReduceMotion,
        resetPreferences,
      }}
    >
      {children}
    </ThemeContext.Provider>
  );
}

// eslint-disable-next-line react-refresh/only-export-components
export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}
