import * as React from "react";
import { Check, MousePointerClick, PenLine, Sparkles } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { EmptyState } from "@/components/common/states";
import { CandidateTable } from "./candidate-table";
import { CropThumb } from "./crop-thumb";
import { FacingStatusBadge } from "./status-badge";
import { SkuPicker } from "./sku-picker";
import { formatBBox, formatPercent, formatScore } from "@/lib/format";
import type { Facing } from "@/lib/audit";
import { useFacingReview } from "@/hooks/use-facing-review";

interface FacingInspectorProps {
  facing: Facing | null;
}

/**
 * Detail view for one facing: crop, verdict, score gauges, retrieval slate, and
 * the two actions a reviewer needs — accept the prediction, or reassign it.
 */
export function FacingInspector({ facing }: FacingInspectorProps) {
  const { submit, isPending } = useFacingReview();
  const [correcting, setCorrecting] = React.useState(false);
  const [draftClassId, setDraftClassId] = React.useState<number>(facing?.classId ?? -1);
  const pickerId = React.useId();

  // Reset the correction affordance whenever a different facing is selected.
  React.useEffect(() => {
    setCorrecting(false);
    setDraftClassId(facing?.classId ?? -1);
  }, [facing?.key, facing?.classId]);

  if (!facing) {
    return (
      <EmptyState
        icon={MousePointerClick}
        title="Select a facing"
        description="Click any bounding box on the shelf image to inspect its prediction, retrieval candidates and score breakdown."
      />
    );
  }

  const confidencePct = facing.confidence * 100;
  const visualPct = facing.topSimilarity * 100;

  const handleApprove = () => {
    void submit(facing, facing.classId);
  };

  const handleSaveCorrection = () => {
    void submit(facing, draftClassId).then(() => setCorrecting(false));
  };

  return (
    <div className="flex flex-col">
      {/* -------------------------------- Summary --------------------------- */}
      <div className="flex gap-3 p-4">
        <CropThumb
          src={facing.cropDataUrl}
          alt={`Cropped facing ${facing.cropId}`}
          className="size-24 shrink-0"
        />
        <div className="min-w-0 flex-1 space-y-2">
          <div>
            <p className="text-muted-foreground text-2xs font-semibold tracking-wide uppercase">
              Predicted SKU
            </p>
            <h3 className="text-sm leading-snug font-semibold text-balance">{facing.title}</h3>
          </div>
          <div className="flex flex-wrap items-center gap-1.5">
            <FacingStatusBadge status={facing.status} />
            <Badge variant="secondary">{facing.brand}</Badge>
            {facing.packCount !== "—" && <Badge variant="outline">{facing.packCount}</Badge>}
            {facing.vlmVerified && (
              <Badge variant="warning">
                <Sparkles aria-hidden />
                Qwen2-VL verified
              </Badge>
            )}
          </div>
          <dl className="text-muted-foreground grid grid-cols-[auto_1fr] gap-x-2 gap-y-0.5 font-mono text-2xs">
            <dt>crop</dt>
            <dd className="truncate">{facing.cropId}</dd>
            <dt>bbox</dt>
            <dd className="truncate">{formatBBox(facing.bbox)}</dd>
            {facing.rejectReason && (
              <>
                <dt>reason</dt>
                <dd className="text-destructive truncate">{facing.rejectReason}</dd>
              </>
            )}
          </dl>
        </div>
      </div>

      <Separator />

      {/* --------------------------------- Scores --------------------------- */}
      <div className="grid grid-cols-2 gap-4 p-4">
        <Gauge
          label="Decision confidence"
          value={formatPercent(facing.confidence)}
          percent={confidencePct}
          tone={facing.status === "automated" ? "success" : "warning"}
        />
        <Gauge
          label="DINOv3 visual score"
          value={formatScore(facing.topSimilarity)}
          percent={visualPct}
          tone="info"
        />
      </div>

      {facing.vlmReason && (
        <p className="text-muted-foreground border-warning/40 bg-warning-subtle/40 mx-4 mb-4 rounded-md border-l-2 px-3 py-2 text-xs leading-relaxed">
          <span className="text-foreground font-semibold">VLM rationale — </span>
          {facing.vlmReason}
        </p>
      )}

      <Separator />

      {/* ------------------------------- Candidates ------------------------- */}
      <section className="p-4" aria-labelledby={`${facing.key}-candidates`}>
        <h4
          id={`${facing.key}-candidates`}
          className="text-muted-foreground mb-2 text-2xs font-semibold tracking-wide uppercase"
        >
          Top class-unique candidates
        </h4>
        <div className="border-border overflow-hidden rounded-lg border">
          <CandidateTable candidates={facing.candidates} />
        </div>
      </section>

      <Separator />

      {/* --------------------------------- Actions -------------------------- */}
      <div className="bg-muted/40 space-y-3 p-4">
        {correcting ? (
          <div className="space-y-2">
            <Label htmlFor={pickerId}>Reassign to commercial SKU class</Label>
            <SkuPicker id={pickerId} value={draftClassId} onChange={setDraftClassId} />
            <div className="flex gap-2">
              <Button
                variant="ghost"
                size="sm"
                className="flex-1"
                onClick={() => setCorrecting(false)}
                disabled={isPending}
              >
                Cancel
              </Button>
              <Button
                size="sm"
                className="flex-1"
                onClick={handleSaveCorrection}
                loading={isPending}
              >
                Save correction
              </Button>
            </div>
          </div>
        ) : (
          <div className="flex flex-col gap-2 sm:flex-row">
            <Button
              variant="success"
              className="flex-1"
              onClick={handleApprove}
              loading={isPending}
            >
              <Check aria-hidden />
              Approve prediction
            </Button>
            <Button variant="outline" className="flex-1" onClick={() => setCorrecting(true)}>
              <PenLine aria-hidden />
              Correct class
            </Button>
          </div>
        )}
        <p className="text-muted-foreground text-2xs leading-relaxed">
          Every verdict is persisted to <code className="font-mono">reviews.db</code> with the
          audit-time 768-D embedding, feeding the continual-learning loop.
        </p>
      </div>
    </div>
  );
}

function Gauge({
  label,
  value,
  percent,
  tone,
}: {
  label: string;
  value: string;
  percent: number;
  tone: "success" | "warning" | "info";
}) {
  return (
    <div className="space-y-1.5">
      <p className="text-muted-foreground text-2xs font-medium">{label}</p>
      <p className="tabular text-lg leading-none font-bold">{value}</p>
      <Progress value={Math.min(100, Math.max(0, percent))} tone={tone} aria-label={label} />
    </div>
  );
}
