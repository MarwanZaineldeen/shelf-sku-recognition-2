import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";

/**
 * Gate for the Continual Learning workbench.
 *
 * This is the same client-side developer gate the previous UI used — it keeps
 * destructive curation controls out of a merchandiser's way, it is not a
 * security boundary. Session-scoped on purpose: closing the tab re-locks it.
 */
const DEVELOPER_PASSCODE = "0000";

interface AdminState {
  unlocked: boolean;
  unlock: (passcode: string) => boolean;
  lock: () => void;
}

export const useAdminStore = create<AdminState>()(
  persist(
    (set) => ({
      unlocked: false,
      unlock: (passcode) => {
        const ok = passcode.trim() === DEVELOPER_PASSCODE;
        if (ok) set({ unlocked: true });
        return ok;
      },
      lock: () => set({ unlocked: false }),
    }),
    {
      name: "retail-ai:admin",
      storage: createJSONStorage(() => sessionStorage),
    },
  ),
);
