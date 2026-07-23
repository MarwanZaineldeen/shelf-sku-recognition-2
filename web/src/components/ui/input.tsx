import * as React from "react";
import { cn } from "@/lib/utils";

const fieldBase = [
  "border-input bg-card w-full rounded-md border text-sm shadow-xs transition-[color,box-shadow,border-color]",
  "placeholder:text-muted-foreground/70",
  "focus-visible:border-ring focus-visible:ring-ring/35 focus-visible:ring-[3px] focus-visible:outline-none",
  "disabled:cursor-not-allowed disabled:opacity-60",
  "aria-invalid:border-destructive aria-invalid:ring-destructive/25",
];

function Input({ className, type = "text", ...props }: React.ComponentProps<"input">) {
  return (
    <input
      type={type}
      data-slot="input"
      className={cn(
        fieldBase,
        "h-9 px-3 py-1",
        "file:text-foreground file:mr-3 file:h-7 file:cursor-pointer file:rounded file:border-0 file:bg-secondary file:px-3 file:text-xs file:font-semibold",
        className,
      )}
      {...props}
    />
  );
}

function Textarea({ className, ...props }: React.ComponentProps<"textarea">) {
  return (
    <textarea
      data-slot="textarea"
      className={cn(fieldBase, "field-sizing-content min-h-20 px-3 py-2", className)}
      {...props}
    />
  );
}

export { Input, Textarea, fieldBase };
