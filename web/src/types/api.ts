/**
 * Wire types for the FastAPI service (`server/schemas.py` + ad-hoc dict routes).
 *
 * These mirror the backend exactly — nothing here is reshaped. Presentation
 * models live in `@/lib/audit` and are derived from these.
 */

/** Sentinel the pipeline uses for "not in catalogue". */
export const UNKNOWN_CLASS_ID = -1;

export interface BBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  confidence: number;
}

export interface CommercialSku {
  project_sku_id: string;
  display_name: string;
  brand: string;
  product_name: string;
  variant: string;
  pack_count: string;
  pack_type: string;
}

export interface Candidate {
  class_id: number;
  display_name: string;
  similarity: number;
  vlm_selected?: boolean | null;
  s_fused?: number | null;
  exemplar_url?: string | null;
}

export interface Annotation {
  crop_id: string;
  bbox: BBox;
  class_id: number;
  confidence: number;
  crop_data_url?: string | null;
  parent_image_name?: string | null;
  ocr_text?: string | null;
  vlm_verified?: boolean | null;
  vlm_reason?: string | null;
  commercial_sku?: CommercialSku | null;
  /** Present on annotations produced by newer pipeline builds. */
  top5_candidates?: Candidate[] | null;
}

export interface HitlRecord {
  hitl_id: string;
  crop_id: string;
  bbox: BBox;
  class_id?: number | null;
  confidence: number;
  reject_reason: string;
  crop_data_url?: string | null;
  parent_image_name?: string | null;
  vlm_verified?: boolean | null;
  vlm_reason?: string | null;
  commercial_sku?: CommercialSku | null;
  top5_candidates?: Candidate[] | null;
  predicted_class_id?: number | null;
  top1_similarity?: number | null;
}

export interface AuditResponse {
  image_name: string;
  parent_image_data_url?: string | null;
  processing_time_ms: number;
  annotations: Annotation[];
  hitl_queue: HitlRecord[];
}

export interface HealthResponse {
  status: string;
  loaded_models: string[];
  db_version: number;
}

/** A row of `configs/sku_mapping_v2.json`. */
export interface CatalogClass {
  raw_class_id?: string;
  training_class_id?: number;
  project_sku_id?: string;
  brand?: string;
  product_name?: string;
  variant?: string;
  size?: string;
  pack_count?: string;
  pack_type?: string;
  display_name?: string;
  status?: string;
  identity_confidence?: string;
  instance_count?: number;
  source_image_count?: number;
  evidence?: string;
  notes?: string;
}

export interface CatalogResponse {
  classes: Record<string, CatalogClass>;
}

export interface NextClassIdResponse {
  next_class_id: number;
}

export interface DeleteSkusResponse {
  status: string;
  deleted_class_ids: number[];
  deleted_vectors_count: number;
  next_class_id: number;
}

export interface HitlReviewResponse {
  status: string;
  hitl_id: string;
  assigned_class_id: number;
  display_name: string;
  review_id: string | null;
  embedding_captured: boolean;
}

export type ReviewDecision = "APPROVED" | "CORRECTED" | "NOT_IN_CATALOG" | string;

export interface RecentReview {
  review_id: string;
  parent_image: string;
  crop_id: string;
  decision: ReviewDecision;
  true_class_id: number | null;
  predicted_class_id: number | null;
  embedding_captured: boolean;
}

export interface ActiveLearningStatus {
  total_reviews: number;
  embeddings_captured: number;
  corrected_count: number;
  approved_count: number;
  gallery_size: number;
  recent_reviews: RecentReview[];
}

export interface CurationResponse {
  status: string;
  pruned_count: number;
  new_gallery_size: number;
}

export interface OnboardValidationAudit {
  facings_detected: number;
  mean_similarity: number;
  recommendation: string;
  pass_validation: boolean;
}

export interface OnboardResponse {
  status: string;
  version: number;
  crops_added: number;
  class_id?: number | null;
  message?: string | null;
  metadata?: Record<string, string> | null;
  validation_audit?: OnboardValidationAudit | null;
}

/** Multipart payload accepted by `POST /v1/hitl/review`. */
export interface HitlReviewPayload {
  hitl_id: string;
  crop_id: string;
  parent_image_name: string;
  assigned_class_id: number;
  reviewer_id: string;
  predicted_class_id: number;
  top1_similarity: number;
}

/** Multipart payload accepted by `POST /v1/onboard/sku`. */
export interface OnboardPayload {
  class_id: number;
  brand: string;
  product_name: string;
  variant: string;
  size: string;
  pack_type: string;
  display_name: string;
  notes: string;
  /** Mode A — client-side files. Mode B — a server-side folder path. */
  referenceImages?: File[];
  folderPath?: string;
  validationShelfImage?: File | null;
}
