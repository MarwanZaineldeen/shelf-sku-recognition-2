import { create } from "zustand";
import { persist } from "zustand/middleware";

export type ThemePreference = "light" | "dark" | "system";

interface PreferencesState {
  theme: ThemePreference;
  sidebarCollapsed: boolean;
  reducedMotion: boolean;
  /** Attributed to every HITL review submitted from this browser. */
  reviewerId: string;
  setTheme: (theme: ThemePreference) => void;
  toggleSidebar: () => void;
  setSidebarCollapsed: (collapsed: boolean) => void;
  setReducedMotion: (value: boolean) => void;
  setReviewerId: (value: string) => void;
}

/**
 * Durable UI preferences.
 *
 * The storage key and shape are mirrored by the inline script in `index.html`,
 * which resolves the theme before React mounts to avoid a flash of wrong theme.
 * Changing either one means changing both.
 */
export const usePreferencesStore = create<PreferencesState>()(
  persist(
    (set) => ({
      theme: "dark",
      sidebarCollapsed: false,
      reducedMotion: false,
      reviewerId: "merchandiser_user",
      setTheme: (theme) => set({ theme }),
      toggleSidebar: () => set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
      setSidebarCollapsed: (sidebarCollapsed) => set({ sidebarCollapsed }),
      setReducedMotion: (reducedMotion) => set({ reducedMotion }),
      setReviewerId: (reviewerId) => set({ reviewerId }),
    }),
    { name: "retail-ai:preferences", version: 1 },
  ),
);
