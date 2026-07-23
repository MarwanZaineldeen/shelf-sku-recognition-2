import * as React from "react";
import { useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { queryKeys, useCatalog, useSaveHitlReview, type CatalogEntry } from "@/lib/api/queries";
import { formatClassTitle, type Facing } from "@/lib/audit";
import { usePreferencesStore } from "@/stores/preferences";
import { UNKNOWN_CLASS_ID, type Annotation, type AuditResponse, type CommercialSku } from "@/types/api";

function toCommercialSku(entry: CatalogEntry | undefined, classId: number): CommercialSku | null {
  if (classId === UNKNOWN_CLASS_ID) {
    return {
      project_sku_id: "UNKNOWN",
      display_name: "Class Unknown",
      brand: "Unknown",
      product_name: "",
      variant: "",
      pack_count: "",
      pack_type: "",
    };
  }
  if (!entry) return null;
  return {
    project_sku_id: String(entry.classId),
    display_name: entry.displayName,
    brand: entry.brand,
    product_name: entry.productName,
    variant: entry.variant,
    pack_count: entry.packCount,
    pack_type: entry.packType,
  };
}

/**
 * Submits a human verdict for one facing and reconciles the cached audit so the
 * canvas, queue and metrics all update without a second pipeline run.
 *
 * The cache write is optimistic in effect but applied after the server confirms
 * — the request is cheap and a rollback UI would be more confusing than a
 * half-second wait on a decision the reviewer just made deliberately.
 */
export function useFacingReview() {
  const client = useQueryClient();
  const { data: catalog } = useCatalog();
  const reviewerId = usePreferencesStore((state) => state.reviewerId);
  const mutation = useSaveHitlReview();

  const submit = React.useCallback(
    async (facing: Facing, assignedClassId: number) => {
      const entry = catalog?.byId.get(assignedClassId);
      const sku = toCommercialSku(entry, assignedClassId);
      const title = formatClassTitle(assignedClassId, sku);

      const result = await mutation.mutateAsync({
        hitl_id: facing.hitlId ?? facing.cropId,
        crop_id: facing.cropId,
        parent_image_name: facing.parentImageName ?? "shelf.jpg",
        assigned_class_id: assignedClassId,
        reviewer_id: reviewerId,
        predicted_class_id: facing.predictedClassId ?? UNKNOWN_CLASS_ID,
        top1_similarity: facing.topSimilarity ?? 0,
      });

      client.setQueryData<AuditResponse | null>(queryKeys.currentAudit, (previous) => {
        if (!previous) return previous;

        // A reviewed facing is human-verified: it leaves the queue and joins the
        // annotation list at full confidence.
        const reviewed: Annotation = {
          crop_id: facing.cropId,
          bbox: facing.bbox,
          class_id: assignedClassId,
          confidence: 1,
          crop_data_url: facing.cropDataUrl,
          parent_image_name: facing.parentImageName,
          vlm_verified: facing.vlmVerified,
          vlm_reason: facing.vlmReason,
          commercial_sku: sku,
          top5_candidates: facing.candidates,
        };

        return {
          ...previous,
          annotations: [
            ...previous.annotations.filter((item) => item.crop_id !== facing.cropId),
            reviewed,
          ],
          hitl_queue: previous.hitl_queue.filter((item) => item.crop_id !== facing.cropId),
        };
      });

      toast.success(`Recorded “${title}”`, {
        description: result.embedding_captured
          ? "768-D DINOv3 vector captured for continual learning."
          : "Review persisted to reviews.db.",
      });

      return result;
    },
    [catalog, client, mutation, reviewerId],
  );

  return { submit, isPending: mutation.isPending };
}
