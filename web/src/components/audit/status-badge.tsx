import { CheckCircle2, CircleHelp, TriangleAlert } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { FACING_STATUS_META, type FacingStatus } from "@/lib/audit";

const ICONS = {
  automated: CheckCircle2,
  review: TriangleAlert,
  unknown: CircleHelp,
} as const;

const VARIANTS = {
  automated: "success",
  review: "warning",
  unknown: "destructive",
} as const;

/** Status pill carrying both an icon and a word — never colour alone. */
export function FacingStatusBadge({ status }: { status: FacingStatus }) {
  const Icon = ICONS[status];
  return (
    <Badge variant={VARIANTS[status]}>
      <Icon aria-hidden />
      {FACING_STATUS_META[status].label}
    </Badge>
  );
}
