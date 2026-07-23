import * as React from "react";
import { usePreferencesStore, type ThemePreference } from "@/stores/preferences";

const DARK_QUERY = "(prefers-color-scheme: dark)";

function systemPrefersDark() {
  return typeof window !== "undefined" && window.matchMedia(DARK_QUERY).matches;
}

function resolve(theme: ThemePreference, systemDark: boolean): "light" | "dark" {
  if (theme === "system") return systemDark ? "dark" : "light";
  return theme;
}

/** The theme actually rendered right now, following the OS when set to system. */
export function useResolvedTheme(): "light" | "dark" {
  const theme = usePreferencesStore((state) => state.theme);
  const [systemDark, setSystemDark] = React.useState(systemPrefersDark);

  React.useEffect(() => {
    const media = window.matchMedia(DARK_QUERY);
    const onChange = (event: MediaQueryListEvent) => setSystemDark(event.matches);
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, []);

  return resolve(theme, systemDark);
}

/**
 * Keeps `<html>` in sync with the stored preference. Mount once, at the root.
 */
export function useApplyTheme() {
  const resolved = useResolvedTheme();

  React.useEffect(() => {
    const root = document.documentElement;
    root.classList.toggle("dark", resolved === "dark");
    root.style.colorScheme = resolved;
  }, [resolved]);

  return resolved;
}

export function useThemeToggle() {
  const theme = usePreferencesStore((state) => state.theme);
  const setTheme = usePreferencesStore((state) => state.setTheme);
  const resolved = useResolvedTheme();

  const toggle = React.useCallback(() => {
    setTheme(resolved === "dark" ? "light" : "dark");
  }, [resolved, setTheme]);

  return { theme, resolved, setTheme, toggle };
}
