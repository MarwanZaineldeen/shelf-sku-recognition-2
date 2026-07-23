import * as React from "react";
import { Link } from "react-router-dom";
import { CheckCheck, Inbox, PartyPopper, ScanSearch, Search } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { PageHeader } from "@/components/common/page-header";
import { EmptyState } from "@/components/common/states";
import { CropThumb } from "@/components/audit/crop-thumb";
import { FacingStatusBadge } from "@/components/audit/status-badge";
import { SkuPicker } from "@/components/audit/sku-picker";
import { buildAuditModel, type Facing } from "@/lib/audit";
import { formatBBox, formatPercent, pluralize } from "@/lib/format";
import { useCurrentAudit } from "@/lib/api/queries";
import { useFacingReview } from "@/hooks/use-facing-review";

/**
 * Human-in-the-loop queue.
 *
 * Optimised for throughput: one row per facing, the class picker pre-filled
 * with the model's guess, and a single confirm action. Rows leave the queue as
 * soon as the verdict lands.
 */
export default function ReviewPage() {
  const { data: audit } = useCurrentAudit();
  const [query, setQuery] = React.useState("");

  const model = React.useMemo(() => buildAuditModel(audit), [audit]);

  // Only facings the pipeline actually queued. Open-set rejections sitting in
  // the annotation list are triaged from the shelf canvas, not here.
  const pending = React.useMemo(
    () => model?.facings.filter((facing) => facing.queued) ?? [],
    [model],
  );

  const filtered = React.useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return pending;
    return pending.filter((facing) =>
      [facing.title, facing.brand, facing.cropId, facing.rejectReason ?? ""]
        .join(" ")
        .toLowerCase()
        .includes(needle),
    );
  }, [pending, query]);

  return (
    <div className="mx-auto w-full max-w-[1600px] space-y-6 p-4 sm:p-6">
      <PageHeader
        title="Review Queue"
        description="Confirm or reassign the facings the pipeline could not resolve on its own. Each verdict is persisted with its audit-time embedding and feeds the continual-learning loop."
        breadcrumbs={[{ label: "Workflow" }, { label: "Review Queue" }]}
        actions={
          <Button variant="outline" asChild>
            <Link to="/audit">
              <ScanSearch aria-hidden />
              Back to shelf
            </Link>
          </Button>
        }
      />

      <Card>
        <CardHeader>
          <div className="min-w-0">
            <CardTitle className="flex items-center gap-2">
              Pending decisions
              {pending.length > 0 && <Badge variant="warning">{pending.length}</Badge>}
            </CardTitle>
            {model && (
              <p className="text-muted-foreground mt-1 text-xs">
                {pluralize(model.total, "facing")} audited from{" "}
                <span className="font-mono">{model.imageName}</span>
              </p>
            )}
          </div>
          {pending.length > 0 && (
            <div className="relative w-full sm:w-64">
              <Search
                className="text-muted-foreground pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2"
                aria-hidden
              />
              <Input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Filter by SKU, crop or reason…"
                aria-label="Filter review queue"
                className="pl-8"
              />
            </div>
          )}
        </CardHeader>

        <CardContent className="p-0">
          {!model ? (
            <EmptyState
              icon={Inbox}
              title="No audit loaded"
              description="Run a shelf audit first — anything the model is unsure about will land here."
              action={
                <Button asChild>
                  <Link to="/audit">
                    <ScanSearch aria-hidden />
                    Go to Shelf Audit
                  </Link>
                </Button>
              }
            />
          ) : pending.length === 0 ? (
            <EmptyState
              icon={PartyPopper}
              title="Queue is clear"
              description={
                model.unknownCount > 0
                  ? `Nothing from ${model.imageName} was routed for review. ${pluralize(model.unknownCount, "facing")} were rejected as out-of-catalogue — triage those from the shelf canvas.`
                  : `All ${model.total} facings on ${model.imageName} were matched confidently. Nothing needs a human decision.`
              }
              action={
                <Button variant="outline" asChild>
                  <Link to="/audit">
                    <ScanSearch aria-hidden />
                    Back to shelf canvas
                  </Link>
                </Button>
              }
            />
          ) : filtered.length === 0 ? (
            <EmptyState
              icon={Search}
              title="No matches"
              description={`Nothing in the queue matches “${query}”.`}
              action={
                <Button variant="outline" size="sm" onClick={() => setQuery("")}>
                  Clear filter
                </Button>
              }
            />
          ) : (
            // Fixed layout: long SKU titles would otherwise stretch the table
            // past its container and push the confirm action out of reach.
            <Table className="table-fixed min-w-[60rem]">
              <TableHeader>
                <TableRow>
                  <TableHead className="w-16">Crop</TableHead>
                  <TableHead className="w-44">Facing</TableHead>
                  <TableHead>Model prediction</TableHead>
                  <TableHead className="w-[34rem]">Assign class</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filtered.map((facing) => (
                  <ReviewRow key={facing.key} facing={facing} />
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

/** One queue row. Owns its own draft class so rows never re-render each other. */
const ReviewRow = React.memo(function ReviewRow({ facing }: { facing: Facing }) {
  const { submit, isPending } = useFacingReview();
  const [assigned, setAssigned] = React.useState(facing.classId);
  const pickerId = React.useId();

  return (
    <TableRow>
      <TableCell>
        <CropThumb
          src={facing.cropDataUrl}
          alt={`Crop ${facing.cropId}`}
          className="size-12"
        />
      </TableCell>

      <TableCell className="min-w-0">
        <p className="truncate font-mono text-xs font-semibold">{facing.cropId}</p>
        <p className="text-muted-foreground truncate font-mono text-2xs">
          {formatBBox(facing.bbox)}
        </p>
        <div className="mt-1">
          <FacingStatusBadge status={facing.status} />
        </div>
      </TableCell>

      <TableCell className="min-w-0">
        <p className="truncate text-sm font-medium">{facing.title}</p>
        <p className="text-muted-foreground truncate text-2xs">{facing.brand}</p>
        <div className="mt-1 flex flex-wrap items-center gap-1.5">
          <Badge variant="secondary" className="tabular font-mono">
            {formatPercent(facing.confidence)}
          </Badge>
          <Badge variant="destructive" className="max-w-48 truncate font-mono">
            {facing.rejectReason ?? "LOW_CONFIDENCE"}
          </Badge>
        </div>
      </TableCell>

      <TableCell className="min-w-[340px]">
        <div className="flex items-center gap-2">
          <label htmlFor={pickerId} className="sr-only">
            Assign a commercial SKU class to crop {facing.cropId}
          </label>
          <SkuPicker
            id={pickerId}
            value={assigned}
            onChange={setAssigned}
            className="w-full min-w-[220px] flex-1"
          />
          <Button
            size="sm"
            variant="success"
            loading={isPending}
            onClick={() => void submit(facing, assigned)}
            className="shrink-0"
          >
            <CheckCheck aria-hidden />
            Confirm
          </Button>
        </div>
      </TableCell>
    </TableRow>
  );
});
