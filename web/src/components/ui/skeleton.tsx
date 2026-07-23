import { cn } from "@/lib/utils";

/**
 * Placeholder block sized like the content it stands in for, so async loads
 * never shift the layout.
 */
function Skeleton({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="skeleton"
      aria-hidden
      className={cn("bg-muted animate-pulse rounded-md", className)}
      {...props}
    />
  );
}

export { Skeleton };
