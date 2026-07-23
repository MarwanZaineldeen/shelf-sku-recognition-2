import * as React from "react";
import { motion } from "framer-motion";
import type { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import { Skeleton } from "@/components/ui/skeleton";

const tones = {
  primary: { icon: "bg-primary-muted text-primary", rail: "bg-primary" },
  success: { icon: "bg-success-subtle text-success", rail: "bg-success" },
  warning: { icon: "bg-warning-subtle text-warning", rail: "bg-warning" },
  destructive: { icon: "bg-destructive-subtle text-destructive", rail: "bg-destructive" },
  info: { icon: "bg-info-subtle text-info", rail: "bg-info" },
  admin: { icon: "bg-admin-subtle text-admin", rail: "bg-admin" },
} as const;

export type StatTone = keyof typeof tones;

interface StatCardProps {
  label: string;
  value: React.ReactNode;
  hint?: React.ReactNode;
  icon: LucideIcon;
  tone?: StatTone;
  loading?: boolean;
  className?: string;
}

/**
 * Metric tile. Value is the loudest element; the label reads as a sentence
 * fragment, and the hint carries the qualifier (rate, model, unit).
 */
export const StatCard = React.memo(function StatCard({
  label,
  value,
  hint,
  icon: Icon,
  tone = "primary",
  loading = false,
  className,
}: StatCardProps) {
  const palette = tones[tone];

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, ease: [0.22, 1, 0.36, 1] }}
      className={cn(
        "bg-card border-border relative flex items-start gap-3 overflow-hidden rounded-xl border p-4 shadow-xs",
        className,
      )}
    >
      <span className={cn("absolute inset-y-0 left-0 w-0.5", palette.rail)} aria-hidden />
      <span
        className={cn("flex size-9 shrink-0 items-center justify-center rounded-lg", palette.icon)}
        aria-hidden
      >
        <Icon className="size-4" />
      </span>
      <div className="min-w-0 flex-1 space-y-0.5">
        <p className="text-muted-foreground text-xs font-medium">{label}</p>
        {loading ? (
          <Skeleton className="mt-1 h-7 w-20" />
        ) : (
          <p className="tabular truncate text-2xl leading-tight font-bold tracking-tight">{value}</p>
        )}
        {hint && !loading && <p className="text-muted-foreground truncate text-2xs">{hint}</p>}
      </div>
    </motion.div>
  );
});
