/**
 * One function per backend route. Nothing above this file constructs URLs.
 *
 * Route contracts are unchanged from the previous vanilla-JS client — the same
 * paths, verbs, and multipart field names are used.
 */
import { request, toFormData } from "./client";
import type {
  ActiveLearningStatus,
  AuditResponse,
  CatalogResponse,
  CurationResponse,
  DeleteSkusResponse,
  HealthResponse,
  HitlReviewPayload,
  HitlReviewResponse,
  NextClassIdResponse,
  OnboardPayload,
  OnboardResponse,
} from "@/types/api";

/* ----------------------------------- Audit --------------------------------- */

export function auditShelf(file: File, signal?: AbortSignal) {
  const form = new FormData();
  form.append("file", file);
  return request<AuditResponse>("/v1/audit/shelf", { method: "POST", body: form, signal });
}

export function auditSample(signal?: AbortSignal) {
  return request<AuditResponse>("/v1/audit/sample", { signal });
}

/* ---------------------------------- Catalog -------------------------------- */

export function fetchCatalog(signal?: AbortSignal) {
  return request<CatalogResponse>("/api/catalog", { signal });
}

export function deleteSkus(classIds: number[]) {
  return request<DeleteSkusResponse>("/v1/catalog/delete", {
    method: "POST",
    body: { class_ids: classIds, confirmation: "DELETE" },
  });
}

export function fetchNextClassId(signal?: AbortSignal) {
  return request<NextClassIdResponse>("/v1/next-class-id", { signal });
}

/** Exemplar thumbnail for a catalogue class. Served directly as an image. */
export function exemplarUrl(classId: number) {
  return `/v1/exemplars/${classId}`;
}

/* ------------------------------------ HITL --------------------------------- */

export function saveHitlReview(payload: HitlReviewPayload) {
  return request<HitlReviewResponse>("/v1/hitl/review", {
    method: "POST",
    body: toFormData({
      hitl_id: payload.hitl_id,
      crop_id: payload.crop_id,
      parent_image_name: payload.parent_image_name,
      assigned_class_id: payload.assigned_class_id,
      reviewer_id: payload.reviewer_id,
      predicted_class_id: payload.predicted_class_id,
      top1_similarity: payload.top1_similarity,
    }),
  });
}

/* ------------------------------ Active learning ---------------------------- */

export function fetchActiveLearningStatus(signal?: AbortSignal) {
  return request<ActiveLearningStatus>("/v1/active-learning/status", { signal });
}

export function runCuration() {
  return request<CurationResponse>("/v1/active-learning/curate", {
    method: "POST",
    body: { apply: true, confirmation: "CURATE" },
  });
}

/* -------------------------------- Onboarding ------------------------------- */

export function onboardSku(payload: OnboardPayload) {
  const form = toFormData({
    class_id: payload.class_id,
    // The backend reads both keys; keep them in sync.
    old_class_id: payload.class_id,
    brand: payload.brand,
    product_name: payload.product_name,
    variant: payload.variant,
    size: payload.size,
    pack_type: payload.pack_type,
    display_name: payload.display_name,
    notes: payload.notes,
  });

  if (payload.folderPath) {
    form.append("folder_path", payload.folderPath);
  } else if (payload.referenceImages?.length) {
    form.append("pack_count", `${payload.referenceImages.length} crops`);
    for (const file of payload.referenceImages) form.append("reference_images", file);
  }

  if (payload.validationShelfImage) {
    form.append("validation_shelf_image", payload.validationShelfImage);
  }

  return request<OnboardResponse>("/v1/onboard/sku", { method: "POST", body: form });
}

/* ---------------------------------- System --------------------------------- */

export function fetchHealth(signal?: AbortSignal) {
  return request<HealthResponse>("/healthz", { signal });
}
