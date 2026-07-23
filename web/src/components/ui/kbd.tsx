import { cn } from "@/lib/utils";

/** Small keycap used in tooltips, menus and the command palette. */
function Kbd({ className, ...props }: React.ComponentProps<"kbd">) {
  return (
    <kbd
      className={cn(
        "bg-muted text-muted-foreground border-border inline-flex h-5 min-w-5 items-center justify-center",
        "rounded border px-1.5 font-mono text-2xs font-medium",
        className,
      )}
      {...props}
    />
  );
}

export { Kbd };
