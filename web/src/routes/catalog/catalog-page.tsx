import * as React from "react";
import { Link, useSearchParams } from "react-router-dom";
import { toast } from "sonner";
import {
  Boxes,
  CheckSquare,
  PackagePlus,
  PackageSearch,
  Search,
  Trash2,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { TooltipTip } from "@/components/ui/tooltip";
import { PageHeader } from "@/components/common/page-header";
import { CardGridSkeleton, EmptyState, ErrorState } from "@/components/common/states";
import { ConfirmDialog } from "@/components/common/confirm-dialog";
import { CropThumb } from "@/components/audit/crop-thumb";
import { exemplarUrl } from "@/lib/api/endpoints";
import { useCatalog, useDeleteSkus, type CatalogEntry } from "@/lib/api/queries";
import { formatInteger, pluralize } from "@/lib/format";

const ALL_BRANDS = "__all__";

/**
 * Commercial catalogue explorer.
 *
 * Search and brand filtering live in the URL so a filtered view is shareable
 * and survives a reload; the command palette deep-links here with `?q=`.
 */
export default function CatalogPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const { data: catalog, isLoading, isError, error, refetch } = useCatalog();
  const deleteSkus = useDeleteSkus();

  const query = searchParams.get("q") ?? "";
  const brand = searchParams.get("brand") ?? ALL_BRANDS;

  const [selection, setSelection] = React.useState<Set<number>>(new Set());
  const [selectMode, setSelectMode] = React.useState(false);
  const [pendingDelete, setPendingDelete] = React.useState<number[] | null>(null);

  const setParam = React.useCallback(
    (key: string, value: string) => {
      setSearchParams(
        (params) => {
          const next = new URLSearchParams(params);
          if (!value || value === ALL_BRANDS) next.delete(key);
          else next.set(key, value);
          return next;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  const filtered = React.useMemo(() => {
    const entries = catalog?.entries ?? [];
    const needle = query.trim().toLowerCase();
    return entries.filter((entry) => {
      if (brand !== ALL_BRANDS && entry.brand !== brand) return false;
      if (!needle) return true;
      return (
        entry.displayName.toLowerCase().includes(needle) ||
        entry.brand.toLowerCase().includes(needle) ||
        String(entry.classId).includes(needle)
      );
    });
  }, [catalog, query, brand]);

  const toggle = (classId: number) => {
    setSelection((current) => {
      const next = new Set(current);
      if (next.has(classId)) next.delete(classId);
      else next.add(classId);
      return next;
    });
  };

  const exitSelectMode = () => {
    setSelectMode(false);
    setSelection(new Set());
  };

  const confirmDelete = () => {
    const ids = pendingDelete;
    if (!ids?.length) return;
    deleteSkus.mutate(ids, {
      onSuccess: (result) => {
        toast.success(
          `Removed ${pluralize(ids.length, "SKU class", "SKU classes")}`,
          {
            description: `${formatInteger(result.deleted_vectors_count)} vector embeddings purged. Next class id is ${result.next_class_id}.`,
          },
        );
        setPendingDelete(null);
        exitSelectMode();
      },
      onError: (err) => {
        toast.error("Delete failed", {
          description: err instanceof Error ? err.message : "Unknown error",
        });
        setPendingDelete(null);
      },
    });
  };

  return (
    <div className="mx-auto w-full max-w-[1800px] space-y-6 p-4 sm:p-6">
      <PageHeader
        title="SKU Catalogue"
        description="Every product class registered in the SQLite vector gallery. Retiring a class purges its embeddings, catalogue entry and exemplar thumbnails."
        breadcrumbs={[{ label: "Catalogue" }, { label: "SKU Catalogue" }]}
        actions={
          <>
            <Button
              variant={selectMode ? "secondary" : "outline"}
              onClick={() => (selectMode ? exitSelectMode() : setSelectMode(true))}
              aria-pressed={selectMode}
            >
              <CheckSquare aria-hidden />
              {selectMode ? "Exit selection" : "Select"}
            </Button>
            <Button asChild>
              <Link to="/onboarding">
                <PackagePlus aria-hidden />
                Add new SKU
              </Link>
            </Button>
          </>
        }
      />

      <Card>
        <CardHeader className="gap-3">
          <div className="min-w-0">
            <CardTitle className="flex items-center gap-2">
              <Boxes className="text-primary size-4 shrink-0" aria-hidden />
              Registered classes
              <Badge variant="secondary">{formatInteger(catalog?.entries.length ?? 0)}</Badge>
            </CardTitle>
          </div>

          <div className="flex w-full flex-col gap-2 sm:w-auto sm:flex-row">
            <div className="relative sm:w-72">
              <Search
                className="text-muted-foreground pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2"
                aria-hidden
              />
              <Input
                value={query}
                onChange={(event) => setParam("q", event.target.value)}
                placeholder="Search title, brand or class id…"
                aria-label="Search catalogue"
                className="pl-8"
              />
              {query && (
                <button
                  type="button"
                  onClick={() => setParam("q", "")}
                  aria-label="Clear search"
                  className="text-muted-foreground hover:text-foreground absolute top-1/2 right-2 -translate-y-1/2 cursor-pointer rounded p-0.5"
                >
                  <X className="size-3.5" aria-hidden />
                </button>
              )}
            </div>

            <Select value={brand} onValueChange={(value) => setParam("brand", value)}>
              <SelectTrigger className="sm:w-48" aria-label="Filter by brand">
                <SelectValue placeholder="All brands" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value={ALL_BRANDS}>All brands</SelectItem>
                {catalog?.brands.map((name) => (
                  <SelectItem key={name} value={name}>
                    {name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </CardHeader>

        <CardContent>
          {isLoading ? (
            <CardGridSkeleton count={10} />
          ) : isError ? (
            <ErrorState title="Could not load the catalogue" error={error} onRetry={() => void refetch()} />
          ) : filtered.length === 0 ? (
            <EmptyState
              icon={PackageSearch}
              title={query || brand !== ALL_BRANDS ? "No matching SKUs" : "Catalogue is empty"}
              description={
                query || brand !== ALL_BRANDS
                  ? "Try a different search term or clear the brand filter."
                  : "Onboard a product with 10–50 reference crops to register its first class."
              }
              action={
                query || brand !== ALL_BRANDS ? (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setSearchParams(new URLSearchParams(), { replace: true })}
                  >
                    Clear filters
                  </Button>
                ) : (
                  <Button asChild size="sm">
                    <Link to="/onboarding">
                      <PackagePlus aria-hidden />
                      Add new SKU
                    </Link>
                  </Button>
                )
              }
            />
          ) : (
            <ul className="grid grid-cols-[repeat(auto-fill,minmax(190px,1fr))] gap-4">
              {filtered.map((entry) => (
                <li key={entry.classId}>
                  <SkuCard
                    entry={entry}
                    selectMode={selectMode}
                    selected={selection.has(entry.classId)}
                    onToggle={() => toggle(entry.classId)}
                    onDelete={() => setPendingDelete([entry.classId])}
                  />
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>

      {/* Bulk action bar — only present when there is something to act on. */}
      {selectMode && (
        <div className="pointer-events-none sticky bottom-4 z-10 flex justify-center">
          <div className="border-border bg-popover pointer-events-auto flex flex-wrap items-center gap-2 rounded-full border px-3 py-2 shadow-lg">
            <span className="px-2 text-xs font-semibold">
              {pluralize(selection.size, "SKU")} selected
            </span>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setSelection(new Set(filtered.map((entry) => entry.classId)))}
            >
              Select all shown
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setSelection(new Set())}
              disabled={selection.size === 0}
            >
              Clear
            </Button>
            <Button
              variant="destructive"
              size="sm"
              disabled={selection.size === 0}
              onClick={() => setPendingDelete([...selection])}
            >
              <Trash2 aria-hidden />
              Delete
            </Button>
          </div>
        </div>
      )}

      <ConfirmDialog
        open={Boolean(pendingDelete)}
        onOpenChange={(open) => !open && setPendingDelete(null)}
        destructive
        loading={deleteSkus.isPending}
        title={`Delete ${pluralize(pendingDelete?.length ?? 0, "SKU class", "SKU classes")}?`}
        confirmLabel="Delete permanently"
        description={
          <>
            This purges every vector embedding for{" "}
            <span className="font-mono">
              {pendingDelete?.slice(0, 8).join(", ")}
              {(pendingDelete?.length ?? 0) > 8 ? ` +${(pendingDelete?.length ?? 0) - 8} more` : ""}
            </span>{" "}
            from the SQLite gallery, removes their catalogue entries and deletes their exemplar
            thumbnails. This cannot be undone.
          </>
        }
        onConfirm={confirmDelete}
      />
    </div>
  );
}

interface SkuCardProps {
  entry: CatalogEntry;
  selectMode: boolean;
  selected: boolean;
  onToggle: () => void;
  onDelete: () => void;
}

const SkuCard = React.memo(function SkuCard({
  entry,
  selectMode,
  selected,
  onToggle,
  onDelete,
}: SkuCardProps) {
  const body = (
    <>
      <CropThumb
        src={exemplarUrl(entry.classId)}
        alt={`Reference exemplar for ${entry.displayName}`}
        className="aspect-square w-full"
      />
      <div className="mt-2.5 min-w-0 space-y-1">
        <p className="line-clamp-2 text-xs leading-snug font-semibold">{entry.displayName}</p>
        <p className="text-muted-foreground truncate text-2xs">{entry.brand}</p>
        <div className="flex flex-wrap items-center gap-1 pt-0.5">
          <Badge variant="outline" className="font-mono">
            #{entry.classId}
          </Badge>
          {entry.status === "verified" && <Badge variant="success">Verified</Badge>}
          {entry.instanceCount > 0 && (
            <Badge variant="secondary">{formatInteger(entry.instanceCount)} crops</Badge>
          )}
        </div>
      </div>
    </>
  );

  const shell = cn(
    "group border-border bg-card relative block w-full rounded-xl border p-3 text-left transition-colors",
    "hover:border-primary/50 hover:bg-accent/40",
    selected && "border-primary bg-primary-muted/40",
  );

  if (selectMode) {
    return (
      <button
        type="button"
        onClick={onToggle}
        aria-pressed={selected}
        className={cn(shell, "cursor-pointer focus-visible:ring-ring focus-visible:ring-2 focus-visible:outline-none")}
      >
        <span className="absolute top-4 left-4 z-10">
          <Checkbox checked={selected} tabIndex={-1} aria-hidden className="pointer-events-none" />
        </span>
        {body}
      </button>
    );
  }

  return (
    <div className={shell}>
      <TooltipTip label={`Delete class ${entry.classId}`}>
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={onDelete}
          aria-label={`Delete ${entry.displayName}`}
          className="text-muted-foreground hover:text-destructive hover:bg-destructive-subtle absolute top-2 right-2 z-10 opacity-0 transition-opacity group-hover:opacity-100 focus-visible:opacity-100"
        >
          <Trash2 aria-hidden />
        </Button>
      </TooltipTip>
      {body}
    </div>
  );
});
