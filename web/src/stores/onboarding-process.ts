import { create } from "zustand";
import { toast } from "sonner";
import { onboardSku } from "@/lib/api/endpoints";
import { queryKeys } from "@/lib/api/queries";
import { formatInteger } from "@/lib/format";
import type { OnboardPayload, OnboardResponse } from "@/types/api";
import type { QueryClient } from "@tanstack/react-query";

interface OnboardingProcessState {
  isProcessing: boolean;
  skuTitle: string | null;
  lastResult: OnboardResponse | null;
  error: string | null;
  runOnboardJob: (payload: OnboardPayload, queryClient: QueryClient) => Promise<void>;
  clearResult: () => void;
}

export const useOnboardingProcessStore = create<OnboardingProcessState>((set) => ({
  isProcessing: false,
  skuTitle: null,
  lastResult: null,
  error: null,

  runOnboardJob: async (payload, queryClient) => {
    const skuName = payload.display_name || `${payload.brand} ${payload.product_name}`.trim();
    set({
      isProcessing: true,
      skuTitle: skuName,
      error: null,
    });

    const toastId = toast.loading(`Registering SKU "${skuName}"...`, {
      description: "Background processing in progress. You can safely navigate to other pages.",
    });

    try {
      const data = await onboardSku(payload);

      set({
        isProcessing: false,
        lastResult: data,
        error: null,
      });

      void queryClient.invalidateQueries({ queryKey: queryKeys.catalog });
      void queryClient.invalidateQueries({ queryKey: queryKeys.nextClassId });

      toast.success(`SKU "${skuName}" added successfully!`, {
        id: toastId,
        description: `Class ${data.class_id ?? payload.class_id} registered with ${formatInteger(
          data.crops_added,
        )} crops embedded (gallery v${data.version}).`,
        duration: 6000,
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown error during onboarding";
      set({
        isProcessing: false,
        error: msg,
      });

      toast.error(`Onboarding failed for "${skuName}"`, {
        id: toastId,
        description: msg,
        duration: 6000,
      });
    }
  },

  clearResult: () => set({ lastResult: null, error: null }),
}));
