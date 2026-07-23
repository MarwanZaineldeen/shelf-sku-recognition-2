/**
 * TanStack Query bindings. Components consume these hooks and never call
 * `endpoints.ts` directly, so caching and invalidation stay in one place.
 */
import {
  useMutation,
  useQuery,
  useQueryClient,
  type UseQueryOptions,
} from "@tanstack/react-query";
import * as api from "./endpoints";
import type {
  ActiveLearningStatus,
  AuditResponse,
  CatalogClass,
  CatalogResponse,
  HealthResponse,
  HitlReviewPayload,
  NextClassIdResponse,
  OnboardPayload,
} from "@/types/api";

export const queryKeys = {
  catalog: ["catalog"] as const,
  nextClassId: ["catalog", "next-class-id"] as const,
  health: ["system", "health"] as const,
  activeLearning: ["active-learning", "status"] as const,
  /** Result of the most recent shelf audit, shared across routes. */
  currentAudit: ["audit", "current"] as const,
};

/* --------------------------------- Catalog --------------------------------- */

/** A catalogue entry flattened for list rendering and select options. */
export interface CatalogEntry {
  classId: number;
  displayName: string;
  brand: string;
  productName: string;
  variant: string;
  packCount: string;
  packType: string;
  status: string;
  instanceCount: number;
  notes: string;
}

export interface CatalogData {
  byId: Map<number, CatalogEntry>;
  entries: CatalogEntry[];
  brands: string[];
}

function normalizeCatalog(response: CatalogResponse): CatalogData {
  const entries = Object.entries(response.classes ?? {})
    .map(([key, info]: [string, CatalogClass]): CatalogEntry => {
      const classId = info.training_class_id ?? Number.parseInt(key, 10);
      return {
        classId,
        displayName: info.display_name?.trim() || `SKU Class ${classId}`,
        brand: info.brand?.trim() || "Unbranded",
        productName: info.product_name ?? "",
        variant: info.variant ?? "",
        packCount: info.pack_count ?? "",
        packType: info.pack_type ?? "",
        status: info.status ?? "proposed",
        instanceCount: info.instance_count ?? 0,
        notes: info.notes ?? "",
      };
    })
    .filter((entry) => Number.isFinite(entry.classId))
    .sort((a, b) => a.classId - b.classId);

  return {
    byId: new Map(entries.map((entry) => [entry.classId, entry])),
    entries,
    brands: [...new Set(entries.map((entry) => entry.brand))].sort((a, b) => a.localeCompare(b)),
  };
}

export function useCatalog() {
  return useQuery({
    queryKey: queryKeys.catalog,
    queryFn: ({ signal }) => api.fetchCatalog(signal),
    select: normalizeCatalog,
    staleTime: 60_000,
  });
}

export function useNextClassId(options?: Partial<UseQueryOptions<NextClassIdResponse>>) {
  return useQuery({
    queryKey: queryKeys.nextClassId,
    queryFn: ({ signal }) => api.fetchNextClassId(signal),
    staleTime: 0,
    ...options,
  });
}

export function useDeleteSkus() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (classIds: number[]) => api.deleteSkus(classIds),
    onSuccess: (result) => {
      // The server already told us the new next-id; seed it rather than refetch.
      client.setQueryData<NextClassIdResponse>(queryKeys.nextClassId, {
        next_class_id: result.next_class_id,
      });
      void client.invalidateQueries({ queryKey: queryKeys.catalog });
    },
  });
}

/* ---------------------------------- Audit ---------------------------------- */

/** Reads the audit currently held in cache without triggering a fetch. */
export function useCurrentAudit() {
  return useQuery({
    queryKey: queryKeys.currentAudit,
    queryFn: () => null as AuditResponse | null,
    enabled: false,
    staleTime: Infinity,
    gcTime: Infinity,
    initialData: null,
  });
}

export function useRunAudit() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (input: File | "sample") =>
      input === "sample" ? api.auditSample() : api.auditShelf(input),
    onSuccess: (data) => {
      client.setQueryData<AuditResponse>(queryKeys.currentAudit, data);
    },
  });
}

/* ----------------------------------- HITL ---------------------------------- */

export function useSaveHitlReview() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (payload: HitlReviewPayload) => api.saveHitlReview(payload),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: queryKeys.activeLearning });
    },
  });
}

/* ----------------------------- Active learning ----------------------------- */

export function useActiveLearningStatus(enabled: boolean) {
  return useQuery<ActiveLearningStatus>({
    queryKey: queryKeys.activeLearning,
    queryFn: ({ signal }) => api.fetchActiveLearningStatus(signal),
    enabled,
    staleTime: 15_000,
  });
}

export function useRunCuration() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: () => api.runCuration(),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: queryKeys.activeLearning });
    },
  });
}

/* -------------------------------- Onboarding ------------------------------- */

export function useOnboardSku() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (payload: OnboardPayload) => api.onboardSku(payload),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: queryKeys.catalog });
      void client.invalidateQueries({ queryKey: queryKeys.nextClassId });
    },
  });
}

/* ---------------------------------- System --------------------------------- */

export function useHealth() {
  return useQuery<HealthResponse>({
    queryKey: queryKeys.health,
    queryFn: ({ signal }) => api.fetchHealth(signal),
    staleTime: 60_000,
    refetchInterval: 120_000,
    retry: false,
  });
}
