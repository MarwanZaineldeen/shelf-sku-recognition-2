import { create } from "zustand";
import type { OnboardPayload, OnboardResponse } from "@/types/api";
import * as api from "@/lib/api/endpoints";
import { queryClient } from "@/lib/api/query-client";
import { queryKeys } from "@/lib/api/queries";
import { toast } from "sonner";
import { formatInteger } from "@/lib/format";

interface OnboardingState {
  isPending: boolean;
  result: OnboardResponse | null;
  error: Error | null;
  showSuccessModal: boolean;
  onboardingClassId: number | null;
  startOnboarding: (payload: OnboardPayload, fallbackClassId?: number) => Promise<void>;
  dismissModal: () => void;
  reset: () => void;
}

/** Global store for SKU onboarding — background process survives page navigation. */
export const useOnboardingStore = create<OnboardingState>((set) => ({
  isPending: false,
  result: null,
  error: null,
  showSuccessModal: false,
  onboardingClassId: null,

  startOnboarding: async (payload, fallbackClassId) => {
    const classId = payload.class_id ?? fallbackClassId ?? null;
    set({ isPending: true, error: null, onboardingClassId: classId });

    toast.info("SKU onboarding started in background", {
      description: "You can navigate to other pages while crops are being processed.",
    });

    try {
      const data = await api.onboardSku(payload);
      set({
        isPending: false,
        result: data,
        showSuccessModal: true,
        error: null,
        onboardingClassId: null,
      });

      void queryClient.invalidateQueries({ queryKey: queryKeys.catalog });
      void queryClient.invalidateQueries({ queryKey: queryKeys.nextClassId });

      const assignedId = data.class_id ?? classId;
      toast.success(`SKU Class ${assignedId ?? ""} Onboarded Successfully!`, {
        description: `${formatInteger(data.crops_added)} crops embedded into gallery v${data.version}.`,
        duration: 8000,
      });
    } catch (err) {
      const errorObj = err instanceof Error ? err : new Error("Onboarding failed");
      set({ isPending: false, error: errorObj, onboardingClassId: null });
      toast.error("Onboarding failed", {
        description: errorObj.message,
      });
    }
  },

  dismissModal: () => set({ showSuccessModal: false }),
  reset: () =>
    set({
      isPending: false,
      result: null,
      error: null,
      showSuccessModal: false,
      onboardingClassId: null,
    }),
}));
