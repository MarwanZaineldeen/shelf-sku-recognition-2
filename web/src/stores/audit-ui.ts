import { create } from "zustand";
import type { FacingStatus } from "@/lib/audit";

export type FacingFilter = "all" | FacingStatus;

interface AuditUiState {
  filter: FacingFilter;
  /** `Facing.key` of the crop open in the inspector, or null. */
  selectedKey: string | null;
  /** Canvas zoom multiplier, 1 = fit to container. */
  zoom: number;
  /**
   * Force every box label on. Off by default: a dense shelf carries 150+
   * facings, and showing every label at once buries the image. Labels always
   * appear for the hovered, focused and selected boxes regardless.
   */
  showLabels: boolean;
  setFilter: (filter: FacingFilter) => void;
  select: (key: string | null) => void;
  setZoom: (zoom: number) => void;
  toggleLabels: () => void;
  reset: () => void;
}

/** Ephemeral view state for the audit workspace — never persisted. */
export const useAuditUiStore = create<AuditUiState>()((set) => ({
  filter: "all",
  selectedKey: null,
  zoom: 1,
  showLabels: false,
  setFilter: (filter) => set({ filter }),
  select: (selectedKey) => set({ selectedKey }),
  setZoom: (zoom) => set({ zoom: Math.min(6, Math.max(0.5, zoom)) }),
  toggleLabels: () => set((state) => ({ showLabels: !state.showLabels })),
  reset: () => set({ filter: "all", selectedKey: null, zoom: 1 }),
}));
