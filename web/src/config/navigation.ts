import {
  Boxes,
  Brain,
  Gauge,
  PackagePlus,
  ScanSearch,
  UserRoundCheck,
  type LucideIcon,
} from "lucide-react";

export interface NavItem {
  to: string;
  label: string;
  /** Shown in the command palette and page header description. */
  description: string;
  icon: LucideIcon;
  group: "Workflow" | "Catalogue" | "Operations";
  /** Requires the developer gate to be unlocked. */
  admin?: boolean;
  /** Key into the live badge map rendered by the sidebar. */
  badgeKey?: "review";
}

/**
 * Single source of truth for navigation: sidebar, mobile drawer, command
 * palette and route metadata all read from here.
 */
export const NAV_ITEMS: NavItem[] = [
  {
    to: "/audit",
    label: "Shelf Audit",
    description: "Run detection, retrieval and VLM reranking over a shelf photo",
    icon: ScanSearch,
    group: "Workflow",
  },
  {
    to: "/review",
    label: "Review Queue",
    description: "Confirm or correct low-confidence facings",
    icon: UserRoundCheck,
    group: "Workflow",
    badgeKey: "review",
  },
  {
    to: "/catalog",
    label: "SKU Catalogue",
    description: "Browse, search and retire registered product classes",
    icon: Boxes,
    group: "Catalogue",
  },
  {
    to: "/onboarding",
    label: "Add New SKU",
    description: "Few-shot onboarding from 10–50 reference crops",
    icon: PackagePlus,
    group: "Catalogue",
  },
  {
    to: "/performance",
    label: "Performance",
    description: "Per-stage latency and pipeline architecture",
    icon: Gauge,
    group: "Operations",
  },
  {
    to: "/learning",
    label: "Continual Learning",
    description: "Review persistence and gallery vector curation",
    icon: Brain,
    group: "Operations",
    admin: true,
  },
];

export const NAV_GROUPS = ["Workflow", "Catalogue", "Operations"] as const;

export function findNavItem(pathname: string): NavItem | undefined {
  return NAV_ITEMS.find((item) => pathname.startsWith(item.to));
}
