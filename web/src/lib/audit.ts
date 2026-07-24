/**
 * Domain model for a shelf audit.
 *
 * The backend returns two parallel lists — confidently matched `annotations`
 * and a `hitl_queue` needing human review. The UI works with a single flat
 * `Facing[]`, because a facing is a facing regardless of which bucket it landed
 * in; the distinction is carried by `status`.
 */
import { UNKNOWN_CLASS_ID, type Annotation, type AuditResponse, type BBox, type Candidate, type CommercialSku, type HitlRecord } from "@/types/api";
import { formatPercent } from "./format";

export type FacingStatus = "automated" | "review" | "unknown";

export interface Facing {
  /** Stable identity across re-renders and optimistic updates. */
  key: string;
  index: number;
  cropId: string;
  hitlId: string | null;
  bbox: BBox;
  classId: number;
  confidence: number;
  status: FacingStatus;
  /**
   * True when the pipeline routed this facing to `hitl_queue`. Distinct from
   * `status`: the annotation list also contains open-set rejections, which read
   * as `unknown` on the canvas but were never queued for a human decision.
   */
  queued: boolean;
  title: string;
  brand: string;
  packCount: string;
  cropDataUrl: string | null;
  parentImageName: string | null;
  commercialSku: CommercialSku | null;
  candidates: Candidate[];
  vlmVerified: boolean;
  vlmReason: string | null;
  rejectReason: string | null;
  /** Model's original top-1, preserved so a review records what was corrected. */
  predictedClassId: number;
  topSimilarity: number;
  inferenceMode: string | null;
}

export const FACING_STATUS_META: Record<
  FacingStatus,
  { label: string; description: string; hex: string; token: string }
> = {
  automated: {
    label: "Automated",
    description: "High-confidence match accepted without review",
    hex: "#10b981",
    token: "success",
  },
  review: {
    label: "Needs review",
    description: "Routed to the human-in-the-loop queue",
    hex: "#f59e0b",
    token: "warning",
  },
  unknown: {
    label: "Unknown class",
    description: "Rejected by the open-set gate — not in the catalogue",
    hex: "#f43f5e",
    token: "destructive",
  },
};

/** Never render `Class null` or `Class -1`; the pipeline uses -1 for open set. */
export function formatClassTitle(classId: number | null | undefined, sku?: CommercialSku | null) {
  if (classId === null || classId === undefined || classId === UNKNOWN_CLASS_ID) {
    return "Class Unknown";
  }
  const name = sku?.display_name;
  if (name && name !== "null") return name;
  return `Class ${classId}`;
}

export function isUnknownClass(classId: number | null | undefined, sku?: CommercialSku | null) {
  return (
    classId === null ||
    classId === undefined ||
    classId === UNKNOWN_CLASS_ID ||
    sku?.display_name === "Class Unknown"
  );
}

function topSimilarityOf(candidates: Candidate[] | null | undefined, fallback: number) {
  return candidates?.length ? candidates[0].similarity : fallback;
}

function toFacing(
  source: Annotation | HitlRecord,
  index: number,
  fromQueue: boolean,
  imageName: string,
): Facing {
  const record = source as Annotation & HitlRecord;
  const classId = record.class_id ?? UNKNOWN_CLASS_ID;
  const unknown = isUnknownClass(classId, record.commercial_sku);
  const candidates = record.top5_candidates ?? [];

  return {
    key: record.hitl_id ?? `${record.crop_id}#${index}`,
    index,
    cropId: record.crop_id,
    hitlId: record.hitl_id ?? null,
    bbox: record.bbox,
    classId,
    confidence: record.confidence ?? 0,
    status: unknown ? "unknown" : fromQueue ? "review" : "automated",
    queued: fromQueue,
    title: formatClassTitle(classId, record.commercial_sku),
    brand: record.commercial_sku?.brand || "Unbranded",
    packCount: record.commercial_sku?.pack_count || "—",
    cropDataUrl: record.crop_data_url ?? null,
    parentImageName: record.parent_image_name ?? imageName,
    commercialSku: record.commercial_sku ?? null,
    candidates,
    vlmVerified: Boolean(record.vlm_verified) || Boolean(record.ocr_text?.includes("VLM")),
    vlmReason: record.vlm_reason ?? null,
    rejectReason: record.reject_reason ?? null,
    predictedClassId: record.predicted_class_id ?? classId,
    topSimilarity: record.top1_similarity ?? topSimilarityOf(candidates, record.confidence ?? 0),
    inferenceMode: record.inference_mode ?? null,
  };
}

export interface AuditModel {
  imageName: string;
  imageDataUrl: string | null;
  processingTimeMs: number;
  facings: Facing[];
  automatedCount: number;
  reviewCount: number;
  unknownCount: number;
  /** Facings the pipeline explicitly queued for a human decision. */
  queuedCount: number;
  total: number;
  automationRate: number;
  reviewRate: number;
  perFacingMs: number;
  meanConfidence: number;
  imageWidth: number;
  imageHeight: number;
}

/** Flatten an API response into the model every audit view renders from. */
export function buildAuditModel(audit: AuditResponse | null | undefined): AuditModel | null {
  if (!audit) return null;

  const imageName = audit.image_name || "shelf.jpg";
  const facings: Facing[] = [
    ...(audit.annotations ?? []).map((a, i) => toFacing(a, i, false, imageName)),
    ...(audit.hitl_queue ?? []).map((h, i) =>
      toFacing(h, (audit.annotations?.length ?? 0) + i, true, imageName),
    ),
  ];

  const automatedCount = facings.filter((f) => f.status === "automated").length;
  const reviewCount = facings.filter((f) => f.status === "review").length;
  const unknownCount = facings.filter((f) => f.status === "unknown").length;
  const queuedCount = facings.filter((f) => f.queued).length;
  const total = facings.length;
  const confidenceSum = facings.reduce((sum, f) => sum + f.confidence, 0);

  return {
    imageName,
    imageDataUrl: audit.parent_image_data_url ?? null,
    processingTimeMs: audit.processing_time_ms ?? 0,
    facings,
    automatedCount,
    reviewCount,
    unknownCount,
    queuedCount,
    total,
    automationRate: total ? automatedCount / total : 0,
    reviewRate: total ? (reviewCount + unknownCount) / total : 0,
    perFacingMs: total ? (audit.processing_time_ms ?? 0) / total : 0,
    meanConfidence: total ? confidenceSum / total : 0,
    imageWidth: audit.image_width ?? 0,
    imageHeight: audit.image_height ?? 0,
  };
}

/** Final YOLO labels: automated + HITL-reviewed known classes, never pending/Unknown. */
export function buildYoloAnnotations(model: AuditModel): string {
  if (model.imageWidth <= 0 || model.imageHeight <= 0) {
    throw new Error("The audit response does not include valid source-image dimensions.");
  }
  return model.facings
    .filter((facing) => !facing.queued && facing.classId !== UNKNOWN_CLASS_ID)
    .map((facing) => {
      const { x1, y1, x2, y2 } = facing.bbox;
      const centerX = ((x1 + x2) / 2) / model.imageWidth;
      const centerY = ((y1 + y2) / 2) / model.imageHeight;
      const width = (x2 - x1) / model.imageWidth;
      const height = (y2 - y1) / model.imageHeight;
      const clamp = (value: number) => Math.max(0, Math.min(1, value)).toFixed(6);
      return `${facing.classId} ${clamp(centerX)} ${clamp(centerY)} ${clamp(width)} ${clamp(height)}`;
    })
    .join("\n");
}

/** Plain-text audit report, kept byte-compatible in spirit with the old export. */
export function buildAuditReport(model: AuditModel): string {
  const rule = "=".repeat(70);
  const thinRule = "-".repeat(70);
  const lines: string[] = [
    rule,
    "RETAIL SKU RECOGNITION PLATFORM — AUTOMATED SHELF AUDIT REPORT",
    rule,
    `Shelf Image File: ${model.imageName}`,
    `Timestamp:        ${new Date().toISOString().replace("T", " ").slice(0, 19)}`,
    `Total Facings:    ${model.total}`,
    `Automated:        ${model.automatedCount}`,
    `Needs Review:     ${model.reviewCount}`,
    `Unknown Class:    ${model.unknownCount}`,
    `Automation Rate:  ${formatPercent(model.automationRate)}`,
    `Processing Time:  ${model.processingTimeMs.toFixed(1)} ms`,
    rule,
    "",
  ];

  const section = (title: string, facings: Facing[]) => {
    lines.push(thinRule, title, thinRule, "");
    if (facings.length === 0) {
      lines.push("  (none)", "");
      return;
    }
    facings.forEach((facing, index) => {
      const { bbox } = facing;
      lines.push(
        `[${index + 1}] Crop ID:           ${facing.cropId}`,
        `    Class Target ID:   ${facing.classId === UNKNOWN_CLASS_ID ? "Unknown (-1)" : `Class ${facing.classId}`}`,
        `    SKU Product Title: ${facing.title}`,
        `    Brand:             ${facing.brand}`,
        `    Bounding Box:      [x1=${Math.round(bbox.x1)}, y1=${Math.round(bbox.y1)}, x2=${Math.round(bbox.x2)}, y2=${Math.round(bbox.y2)}]`,
        `    Confidence Prob:   ${formatPercent(facing.confidence)}`,
        `    Verification:      ${facing.vlmVerified ? "VLM Verified" : "DINOv3 Direct Visual Match"}`,
      );
      if (facing.rejectReason) lines.push(`    Reject Reason:     ${facing.rejectReason}`);
      lines.push("");
    });
  };

  section(
    `AUTOMATED ANNOTATIONS (${model.automatedCount} Verified Facings)`,
    model.facings.filter((f) => f.status === "automated"),
  );
  section(
    `HUMAN REVIEW QUEUE (${model.reviewCount} Facings)`,
    model.facings.filter((f) => f.status === "review"),
  );
  section(
    `UNKNOWN / OUT-OF-CATALOGUE (${model.unknownCount} Facings)`,
    model.facings.filter((f) => f.status === "unknown"),
  );

  return lines.join("\n");
}
