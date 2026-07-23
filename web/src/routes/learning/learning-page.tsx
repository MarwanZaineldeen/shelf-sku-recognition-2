import * as React from "react";
import { toast } from "sonner";
import {
  Binary,
  Brain,
  Database,
  Filter,
  Lock,
  PenLine,
  RefreshCw,
  ShieldCheck,
} from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { PageHeader } from "@/components/common/page-header";
import { StatCard } from "@/components/common/stat-card";
import { EmptyState, ErrorState, TableSkeleton } from "@/components/common/states";
import { ConfirmDialog } from "@/components/common/confirm-dialog";
import { useActiveLearningStatus, useRunCuration } from "@/lib/api/queries";
import { formatInteger } from "@/lib/format";
import { useAdminStore } from "@/stores/admin";
import { UNKNOWN_CLASS_ID, type RecentReview } from "@/types/api";

/**
 * Pipeline 3 — continual learning workbench.
 *
 * Reached only through the developer gate. Shows what human review has
 * accumulated in `reviews.db` and runs the fast-loop curation that prunes
 * redundant gallery vectors and promotes verified crops.
 */
export default function LearningPage() {
  const lock = useAdminStore((state) => state.lock);
  const { data, isLoading, isError, error, refetch, isFetching } = useActiveLearningStatus(true);
  const curate = useRunCuration();
  const [confirmOpen, setConfirmOpen] = React.useState(false);

  const runCuration = () => {
    curate.mutate(undefined, {
      onSuccess: (result) => {
        setConfirmOpen(false);
        toast.success("Gallery curation complete", {
          description: `Pruned ${formatInteger(result.pruned_count)} redundant vectors — gallery now holds ${formatInteger(result.new_gallery_size)}.`,
        });
      },
      onError: (err) => {
        setConfirmOpen(false);
        toast.error("Curation did not complete", {
          description: err instanceof Error ? err.message : "Unknown error",
        });
      },
    });
  };

  return (
    <div className="mx-auto w-full max-w-[1600px] space-y-6 p-4 sm:p-6">
      <PageHeader
        title="Continual Learning"
        description="Monitors how human review accumulates in reviews.db, tracks 768-D embedding capture, and runs fast-loop curation over the production gallery."
        breadcrumbs={[{ label: "Operations" }, { label: "Continual Learning" }]}
        actions={
          <>
            <Badge variant="admin">
              <ShieldCheck aria-hidden />
              Developer session
            </Badge>
            <Button variant="outline" onClick={() => void refetch()} loading={isFetching}>
              <RefreshCw aria-hidden />
              Refresh
            </Button>
            <Button
              variant="ghost"
              className="text-destructive"
              onClick={() => {
                lock();
                toast.info("Developer session locked");
              }}
            >
              <Lock aria-hidden />
              Lock
            </Button>
          </>
        }
      />

      <section aria-label="Review store metrics" className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          icon={Brain}
          tone="admin"
          label="Reviews logged"
          value={formatInteger(data?.total_reviews ?? 0)}
          hint={`${formatInteger(data?.approved_count ?? 0)} approved`}
          loading={isLoading}
        />
        <StatCard
          icon={Binary}
          tone="success"
          label="768-D embeddings captured"
          value={formatInteger(data?.embeddings_captured ?? 0)}
          hint="Reused from audit time"
          loading={isLoading}
        />
        <StatCard
          icon={PenLine}
          tone="destructive"
          label="Corrected reviews"
          value={formatInteger(data?.corrected_count ?? 0)}
          hint="Highest-value training signal"
          loading={isLoading}
        />
        <StatCard
          icon={Database}
          tone="info"
          label="Active gallery size"
          value={formatInteger(data?.gallery_size ?? 0)}
          hint="Vectors in the production index"
          loading={isLoading}
        />
      </section>

      <Card className="border-admin/30 bg-admin-subtle/25">
        <CardHeader>
          <div className="min-w-0">
            <CardTitle className="flex items-center gap-2">
              <Filter className="text-admin size-4 shrink-0" aria-hidden />
              Fast-loop gallery curation
            </CardTitle>
            <CardDescription>
              Runs k-center greedy coverage and per-class vector capping (max 500 vectors per SKU)
              to prune near-duplicate embeddings, then promotes human-verified review crops into the
              production gallery.
            </CardDescription>
          </div>
        </CardHeader>
        <CardContent>
          <Button variant="admin" onClick={() => setConfirmOpen(true)} loading={curate.isPending}>
            <Filter aria-hidden />
            Run curation &amp; promotion
          </Button>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <div className="min-w-0">
            <CardTitle>Review persistence log</CardTitle>
            <CardDescription>
              The ten most recent verdicts written to <code className="font-mono">reviews.db</code>.
            </CardDescription>
          </div>
        </CardHeader>
        <CardContent className="p-0">
          {isLoading ? (
            <TableSkeleton rows={5} columns={5} />
          ) : isError ? (
            <div className="p-4">
              <ErrorState
                title="Could not read the review store"
                error={error}
                onRetry={() => void refetch()}
              />
            </div>
          ) : (data?.recent_reviews.length ?? 0) === 0 ? (
            <EmptyState
              icon={Database}
              title="No reviews logged yet"
              description="Confirm or correct a facing in the review queue — it will appear here with its captured embedding."
            />
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Source image</TableHead>
                  <TableHead>Review id</TableHead>
                  <TableHead>Decision</TableHead>
                  <TableHead>Predicted</TableHead>
                  <TableHead>Assigned</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {data?.recent_reviews.map((review) => (
                  <ReviewLogRow key={review.review_id} review={review} />
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      <Alert variant="admin">
        <ShieldCheck aria-hidden />
        <AlertTitle>Why this is gated</AlertTitle>
        <AlertDescription>
          Curation rewrites the vector store that every shelf audit retrieves against. The passcode
          gate keeps it out of a merchandiser's normal workflow; it is not an authentication
          boundary.
        </AlertDescription>
      </Alert>

      <ConfirmDialog
        open={confirmOpen}
        onOpenChange={setConfirmOpen}
        title="Run gallery curation?"
        confirmLabel="Run curation"
        loading={curate.isPending}
        description="Near-duplicate embeddings will be pruned and human-verified review crops promoted into the production gallery. Retrieval results for existing SKUs may shift after this completes."
        onConfirm={runCuration}
      />
    </div>
  );
}

const DECISION_VARIANTS: Record<string, "success" | "destructive" | "admin" | "secondary"> = {
  APPROVED: "success",
  CORRECTED: "destructive",
  NOT_IN_CATALOG: "admin",
};

function classLabel(classId: number | null) {
  if (classId === null || classId === UNKNOWN_CLASS_ID) return "Unknown";
  return `Class ${classId}`;
}

const ReviewLogRow = React.memo(function ReviewLogRow({ review }: { review: RecentReview }) {
  return (
    <TableRow>
      <TableCell className="max-w-56 truncate font-medium">{review.parent_image}</TableCell>
      <TableCell className="text-muted-foreground max-w-40 truncate font-mono text-2xs">
        {review.crop_id}
      </TableCell>
      <TableCell>
        <Badge variant={DECISION_VARIANTS[review.decision] ?? "secondary"}>{review.decision}</Badge>
      </TableCell>
      <TableCell className="font-mono text-xs">{classLabel(review.predicted_class_id)}</TableCell>
      <TableCell className="text-success font-mono text-xs font-semibold">
        {classLabel(review.true_class_id)}
      </TableCell>
    </TableRow>
  );
});
