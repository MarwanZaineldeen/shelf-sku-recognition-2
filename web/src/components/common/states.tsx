import * as React from "react";
import { AlertTriangle, RefreshCw, type LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";

/* --------------------------------- Empty ---------------------------------- */

interface EmptyStateProps {
  icon: LucideIcon;
  title: string;
  description?: React.ReactNode;
  action?: React.ReactNode;
  className?: string;
  /** `sm` for inline panels, `default` for a whole page region. */
  size?: "sm" | "default";
}

/** Never show blank space — say what's missing and what to do about it. */
export function EmptyState({
  icon: Icon,
  title,
  description,
  action,
  className,
  size = "default",
}: EmptyStateProps) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center text-center",
        size === "default" ? "gap-3 px-6 py-14" : "gap-2 px-4 py-8",
        className,
      )}
    >
      <span
        className={cn(
          "bg-muted text-muted-foreground flex items-center justify-center rounded-full",
          size === "default" ? "size-12" : "size-10",
        )}
      >
        <Icon className={size === "default" ? "size-5" : "size-4"} aria-hidden />
      </span>
      <div className="space-y-1">
        <p className={cn("font-semibold", size === "default" ? "text-base" : "text-sm")}>{title}</p>
        {description && (
          <p className="text-muted-foreground mx-auto max-w-sm text-sm leading-relaxed text-balance">
            {description}
          </p>
        )}
      </div>
      {action && <div className="mt-1">{action}</div>}
    </div>
  );
}

/* --------------------------------- Error ---------------------------------- */

interface ErrorStateProps {
  title?: string;
  error?: unknown;
  onRetry?: () => void;
  className?: string;
}

function messageOf(error: unknown): string {
  if (error instanceof Error) return error.message;
  if (typeof error === "string") return error;
  return "An unexpected error occurred.";
}

export function ErrorState({
  title = "Something went wrong",
  error,
  onRetry,
  className,
}: ErrorStateProps) {
  return (
    <div
      role="alert"
      className={cn(
        "border-destructive/30 bg-destructive-subtle/50 flex flex-col items-center gap-3 rounded-lg border px-6 py-10 text-center",
        className,
      )}
    >
      <span className="bg-destructive/15 text-destructive flex size-11 items-center justify-center rounded-full">
        <AlertTriangle className="size-5" aria-hidden />
      </span>
      <div className="space-y-1">
        <p className="text-base font-semibold">{title}</p>
        <p className="text-muted-foreground mx-auto max-w-md text-sm leading-relaxed">
          {messageOf(error)}
        </p>
      </div>
      {onRetry && (
        <Button variant="outline" size="sm" onClick={onRetry}>
          <RefreshCw aria-hidden />
          Try again
        </Button>
      )}
    </div>
  );
}

/* -------------------------------- Loading --------------------------------- */

/** Table skeleton sized to the real row height so nothing jumps on load. */
export function TableSkeleton({ rows = 5, columns = 5 }: { rows?: number; columns?: number }) {
  return (
    <div className="space-y-2 p-4" aria-hidden>
      {Array.from({ length: rows }, (_, rowIndex) => (
        <div key={rowIndex} className="flex items-center gap-3">
          {Array.from({ length: columns }, (_, colIndex) => (
            <Skeleton
              key={colIndex}
              className={cn("h-8", colIndex === 0 ? "w-12 shrink-0" : "flex-1")}
            />
          ))}
        </div>
      ))}
    </div>
  );
}

export function CardGridSkeleton({ count = 8 }: { count?: number }) {
  return (
    <div
      className="grid grid-cols-[repeat(auto-fill,minmax(190px,1fr))] gap-4"
      aria-hidden
    >
      {Array.from({ length: count }, (_, index) => (
        <div key={index} className="border-border space-y-3 rounded-xl border p-3">
          <Skeleton className="aspect-square w-full" />
          <Skeleton className="h-3.5 w-4/5" />
          <Skeleton className="h-3 w-1/2" />
        </div>
      ))}
    </div>
  );
}

/** Full-route fallback used by `React.Suspense` while a chunk downloads. */
export function RouteFallback() {
  return (
    <div className="space-y-6 p-6" aria-busy aria-label="Loading page">
      <div className="space-y-2">
        <Skeleton className="h-7 w-64" />
        <Skeleton className="h-4 w-96 max-w-full" />
      </div>
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {Array.from({ length: 4 }, (_, index) => (
          <Skeleton key={index} className="h-24" />
        ))}
      </div>
      <Skeleton className="h-80 w-full" />
    </div>
  );
}
