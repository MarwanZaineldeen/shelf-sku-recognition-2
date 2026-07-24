import * as React from "react";
import { toast } from "sonner";
import {
  Boxes,
  CheckCircle2,
  CircleHelp,
  Clock,
  Download,
  Gauge,
  ScanSearch,
  TriangleAlert,
  Upload,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Sheet, SheetBody, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { PageHeader } from "@/components/common/page-header";
import { StatCard } from "@/components/common/stat-card";
import { EmptyState, ErrorState } from "@/components/common/states";
import { ShelfCanvas } from "@/components/audit/shelf-canvas";
import { ShelfDropzone } from "@/components/audit/shelf-dropzone";
import { FacingInspector } from "@/components/audit/facing-inspector";
import { buildAuditModel, buildYoloAnnotations, FACING_STATUS_META, type FacingStatus } from "@/lib/audit";
import { downloadTextFile, formatDuration, formatInteger, formatPercent, slugifyFilename } from "@/lib/format";
import { useCurrentAudit, useRunAudit } from "@/lib/api/queries";
import { useAuditUiStore, type FacingFilter } from "@/stores/audit-ui";
import { useIsDesktop } from "@/hooks/use-media-query";

/**
 * Shelf Audit workspace — the primary workflow.
 *
 * Upload → detect → inspect → decide, all on one screen: the overlay and the
 * inspector sit side by side on desktop so a reviewer never loses shelf context
 * while judging a facing.
 */
export default function AuditPage() {
  const isDesktop = useIsDesktop();
  const { data: audit } = useCurrentAudit();
  const runAudit = useRunAudit();

  const filter = useAuditUiStore((state) => state.filter);
  const setFilter = useAuditUiStore((state) => state.setFilter);
  const selectedKey = useAuditUiStore((state) => state.selectedKey);
  const select = useAuditUiStore((state) => state.select);
  const resetUi = useAuditUiStore((state) => state.reset);

  const model = React.useMemo(() => buildAuditModel(audit), [audit]);

  const visibleFacings = React.useMemo(() => {
    if (!model) return [];
    return filter === "all" ? model.facings : model.facings.filter((f) => f.status === filter);
  }, [model, filter]);

  const visibleKeys = React.useMemo(
    () => new Set(visibleFacings.map((facing) => facing.key)),
    [visibleFacings],
  );

  const selectedFacing = React.useMemo(
    () => model?.facings.find((facing) => facing.key === selectedKey) ?? null,
    [model, selectedKey],
  );

  const start = React.useCallback(
    (input: File | "sample") => {
      resetUi();
      runAudit.mutate(input, {
        onSuccess: (data) => {
          const count = (data.annotations?.length ?? 0) + (data.hitl_queue?.length ?? 0);
          toast.success(`Audit complete — ${formatInteger(count)} facings detected`, {
            description: data.image_name,
          });
        },
        onError: (error) => {
          toast.error("Audit failed", {
            description: error instanceof Error ? error.message : "Unknown error",
          });
        },
      });
    },
    [runAudit, resetUi],
  );

  const handleExport = () => {
    if (!model) return;
    try {
      const baseName = slugifyFilename(model.imageName.replace(/\.[^.]+$/, ""));
      downloadTextFile(`${baseName}.txt`, `${buildYoloAnnotations(model)}\n`);
      toast.success("Verified YOLO annotations exported", {
        description: "Includes auto-approved and HITL-reviewed known crops; Unknown crops are excluded.",
      });
    } catch (error) {
      toast.error("Annotation export failed", {
        description: error instanceof Error ? error.message : "Unknown error",
      });
    }
  };

  const busy = runAudit.isPending;

  return (
    <div className="mx-auto w-full max-w-[1800px] space-y-6 p-4 sm:p-6">
      <PageHeader
        title="Shelf Audit"
        description="Localise every product facing, match it against the commercial catalogue, and route anything the model is unsure about to human review."
        breadcrumbs={[{ label: "Workflow" }, { label: "Shelf Audit" }]}
        actions={
          <>
            <UploadButton onFile={(file) => start(file)} disabled={busy} />
            <Button variant="secondary" onClick={handleExport} disabled={!model || busy}>
              <Download aria-hidden />
              Export YOLO annotations
            </Button>
          </>
        }
      />

      {/* --------------------------------- Metrics --------------------------- */}
      <section aria-label="Audit summary" className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
        <StatCard
          icon={Boxes}
          tone="primary"
          label="Facings detected"
          value={formatInteger(model?.total ?? 0)}
          hint="YOLOv8l · SKU110K"
          loading={busy}
        />
        <StatCard
          icon={CheckCircle2}
          tone="success"
          label="Automated matches"
          value={formatInteger(model?.automatedCount ?? 0)}
          hint={`${formatPercent(model?.automationRate ?? 0)} automation rate`}
          loading={busy}
        />
        <StatCard
          icon={TriangleAlert}
          tone="warning"
          label="Queued for review"
          value={formatInteger(model?.queuedCount ?? 0)}
          hint={`${formatInteger(model?.unknownCount ?? 0)} out-of-catalogue facings`}
          loading={busy}
        />
        <StatCard
          icon={Gauge}
          tone="info"
          label="Time per Face"
          value={model ? formatDuration(model.perFacingMs) : "—"}
          hint={model ? `${formatDuration(model.processingTimeMs)} total · CPU` : "CPU optimised"}
          loading={busy}
        />
        <StatCard
          icon={Clock}
          tone="info"
          label="Total Processing Time"
          value={model ? formatDuration(model.processingTimeMs) : "—"}
          hint={model ? "Total pipeline latency" : "Pipeline latency"}
          loading={busy}
        />
      </section>

      {/* -------------------------------- Workspace -------------------------- */}
      <div
        className={cn(
          "grid min-h-0 items-stretch gap-4",
          isDesktop && model ? "xl:grid-cols-[minmax(0,1fr)_480px]" : "grid-cols-1",
        )}
      >
        <Card className="flex flex-col h-full min-h-[780px] xl:h-[calc(100dvh-10rem)] overflow-hidden">
          <CardHeader className="shrink-0 gap-2">
            <div className="min-w-0">
              <CardTitle className="flex items-center gap-2">
                <ScanSearch className="text-primary size-4 shrink-0" aria-hidden />
                Bounding box overlay
              </CardTitle>
              {model && (
                <p className="text-muted-foreground mt-1 truncate font-mono text-2xs">
                  {model.imageName}
                </p>
              )}
            </div>
            {model && (
              <FilterBar
                value={filter}
                onChange={setFilter}
                counts={{
                  all: model.total,
                  automated: model.automatedCount,
                  review: model.reviewCount,
                  unknown: model.unknownCount,
                }}
              />
            )}
          </CardHeader>

          <CardContent className="flex min-h-0 flex-1 flex-col p-0 overflow-hidden">
            {runAudit.isError && !busy ? (
              <div className="p-4">
                <ErrorState
                  title="Audit could not be completed"
                  error={runAudit.error}
                  onRetry={() => runAudit.reset()}
                />
              </div>
            ) : !model || busy ? (
              <ShelfDropzone
                onFile={(file) => start(file)}
                busy={busy}
                busyLabel="Running the recognition pipeline…"
              />
            ) : !model.imageDataUrl ? (
              <EmptyState
                icon={ScanSearch}
                title="Shelf image unavailable"
                description="The audit completed but the service did not return a rendered shelf image."
              />
            ) : (
              <div className="flex h-full min-h-0 flex-1 flex-col">
                <ShelfCanvas
                  imageSrc={model.imageDataUrl}
                  imageAlt={`Shelf scan ${model.imageName}`}
                  facings={model.facings}
                  visibleKeys={visibleKeys}
                  selectedKey={selectedKey}
                  onSelect={select}
                />
              </div>
            )}
          </CardContent>
        </Card>

        {/* Inspector: docked panel on desktop, drawer on smaller screens. */}
        {isDesktop && model && (
          <Card className="flex flex-col h-full min-h-[780px] xl:h-[calc(100dvh-10rem)] overflow-hidden">
            <CardHeader className="shrink-0">
              <CardTitle>Facing inspector</CardTitle>
            </CardHeader>
            <CardContent className="min-h-0 flex-1 overflow-y-auto p-0">
              <FacingInspector facing={selectedFacing} />
            </CardContent>
          </Card>
        )}
      </div>

      {!isDesktop && (
        <Sheet
          open={Boolean(selectedFacing)}
          onOpenChange={(open) => {
            if (!open) select(null);
          }}
        >
          <SheetContent side="bottom" className="max-h-[88dvh]">
            <SheetHeader>
              <SheetTitle>Facing inspector</SheetTitle>
            </SheetHeader>
            <SheetBody className="p-0">
              <FacingInspector facing={selectedFacing} />
            </SheetBody>
          </SheetContent>
        </Sheet>
      )}
    </div>
  );
}

/* --------------------------------- Helpers -------------------------------- */

function UploadButton({ onFile, disabled }: { onFile: (file: File) => void; disabled?: boolean }) {
  const inputRef = React.useRef<HTMLInputElement>(null);
  return (
    <>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="sr-only"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) onFile(file);
          event.target.value = "";
        }}
      />
      <Button onClick={() => inputRef.current?.click()} disabled={disabled}>
        <Upload aria-hidden />
        Upload shelf
      </Button>
    </>
  );
}

const FILTER_ICONS: Record<FacingStatus, typeof CheckCircle2> = {
  automated: CheckCircle2,
  review: TriangleAlert,
  unknown: CircleHelp,
};

const FILTER_ACTIVE: Record<FacingFilter, string> = {
  all: "bg-secondary text-secondary-foreground",
  automated: "bg-success-subtle text-success",
  review: "bg-warning-subtle text-warning",
  unknown: "bg-destructive-subtle text-destructive",
};

function FilterBar({
  value,
  onChange,
  counts,
}: {
  value: FacingFilter;
  onChange: (filter: FacingFilter) => void;
  counts: Record<FacingFilter, number>;
}) {
  const options: FacingFilter[] = ["all", "automated", "review", "unknown"];

  return (
    <div
      role="group"
      aria-label="Filter facings by status"
      className="bg-muted flex w-full flex-wrap gap-0.5 rounded-lg p-0.5 sm:w-auto"
    >
      {options.map((option) => {
        const Icon = option === "all" ? Boxes : FILTER_ICONS[option];
        const label = option === "all" ? "All" : FACING_STATUS_META[option].label;
        const active = value === option;

        return (
          <button
            key={option}
            type="button"
            onClick={() => onChange(option)}
            aria-pressed={active}
            className={cn(
              "flex flex-1 cursor-pointer items-center justify-center gap-1.5 rounded-md px-2.5 py-1.5",
              "text-2xs font-semibold whitespace-nowrap transition-colors sm:flex-none",
              "focus-visible:ring-ring focus-visible:ring-2 focus-visible:outline-none",
              active
                ? cn(FILTER_ACTIVE[option], "shadow-xs")
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            <Icon className="size-3.5" aria-hidden />
            {label}
            <span className="tabular opacity-70">{counts[option]}</span>
          </button>
        );
      })}
    </div>
  );
}
