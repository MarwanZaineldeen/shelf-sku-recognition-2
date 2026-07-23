import { CircleAlert, CircleDot } from "lucide-react";
import { cn } from "@/lib/utils";
import { TooltipTip } from "@/components/ui/tooltip";
import { useHealth } from "@/lib/api/queries";

/**
 * Live service indicator. Colour is backed by an icon and text so it is never
 * the only channel carrying the state.
 */
export function SystemStatus({ compact = false }: { compact?: boolean }) {
  const { data, isLoading, isError } = useHealth();

  const state = isLoading
    ? { label: "Connecting…", tone: "text-muted-foreground", dot: "bg-muted-foreground" }
    : isError
      ? { label: "Service offline", tone: "text-destructive", dot: "bg-destructive" }
      : { label: "Pipeline online", tone: "text-success", dot: "bg-success" };

  const detail = data
    ? `${data.loaded_models.join(" · ")} — gallery v${data.db_version}`
    : "Waiting for /healthz";

  return (
    <TooltipTip label={detail}>
      <div
        className={cn(
          "border-border bg-card flex items-center gap-2 rounded-full border px-2.5 py-1 text-2xs font-semibold",
          state.tone,
        )}
        role="status"
      >
        <span className="relative flex size-2 shrink-0">
          {!isError && !isLoading && (
            <span className="bg-success absolute inline-flex size-full animate-ping rounded-full opacity-60" />
          )}
          <span className={cn("relative inline-flex size-2 rounded-full", state.dot)} />
        </span>
        {isError ? (
          <CircleAlert className="size-3 shrink-0" aria-hidden />
        ) : (
          <CircleDot className="size-3 shrink-0" aria-hidden />
        )}
        {!compact && <span className="whitespace-nowrap">{state.label}</span>}
        <span className="sr-only">{detail}</span>
      </div>
    </TooltipTip>
  );
}
