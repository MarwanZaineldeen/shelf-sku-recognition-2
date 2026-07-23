import { Layers3 } from "lucide-react";
import { cn } from "@/lib/utils";

/** Product mark. `compact` drops the wordmark for the collapsed rail. */
export function Brand({ compact = false, className }: { compact?: boolean; className?: string }) {
  return (
    <div className={cn("flex min-w-0 items-center gap-2.5", className)}>
      <span className="bg-primary text-primary-foreground flex size-8 shrink-0 items-center justify-center rounded-lg shadow-xs">
        <Layers3 className="size-4.5" aria-hidden />
      </span>
      {!compact && (
        <span className="flex min-w-0 flex-col leading-none">
          <span className="truncate text-sm font-bold tracking-tight">
            Retail<span className="text-primary">AI</span>
          </span>
          <span className="text-muted-foreground mt-1 truncate text-2xs">Shelf SKU Audit Suite</span>
        </span>
      )}
    </div>
  );
}
