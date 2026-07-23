import * as React from "react";
import { NavLink } from "react-router-dom";
import { Lock } from "lucide-react";
import { cn } from "@/lib/utils";
import { NAV_GROUPS, NAV_ITEMS, type NavItem } from "@/config/navigation";
import { Badge } from "@/components/ui/badge";
import { TooltipTip } from "@/components/ui/tooltip";
import { useAdminStore } from "@/stores/admin";

interface SidebarNavProps {
  collapsed?: boolean;
  badges?: Partial<Record<NonNullable<NavItem["badgeKey"]>, number>>;
  onNavigate?: () => void;
}

/** The navigation list, shared by the desktop rail and the mobile drawer. */
export function SidebarNav({ collapsed = false, badges, onNavigate }: SidebarNavProps) {
  const unlocked = useAdminStore((state) => state.unlocked);

  return (
    <nav aria-label="Primary" className="flex flex-col gap-5">
      {NAV_GROUPS.map((group) => {
        const items = NAV_ITEMS.filter((item) => item.group === group);
        if (items.length === 0) return null;

        return (
          <div key={group} className="flex flex-col gap-1">
            {!collapsed && (
              <p className="text-muted-foreground px-3 pb-1 text-2xs font-semibold tracking-wider uppercase">
                {group}
              </p>
            )}
            <ul className="flex flex-col gap-0.5">
              {items.map((item) => (
                <li key={item.to}>
                  <NavItemLink
                    item={item}
                    collapsed={collapsed}
                    locked={Boolean(item.admin) && !unlocked}
                    count={item.badgeKey ? badges?.[item.badgeKey] : undefined}
                    onNavigate={onNavigate}
                  />
                </li>
              ))}
            </ul>
          </div>
        );
      })}
    </nav>
  );
}

const NavItemLink = React.memo(function NavItemLink({
  item,
  collapsed,
  locked,
  count,
  onNavigate,
}: {
  item: NavItem;
  collapsed: boolean;
  locked: boolean;
  count?: number;
  onNavigate?: () => void;
}) {
  const Icon = item.icon;

  const link = (
    <NavLink
      to={item.to}
      onClick={onNavigate}
      className={({ isActive }) =>
        cn(
          "group relative flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
          "focus-visible:ring-ring focus-visible:ring-2 focus-visible:outline-none",
          collapsed && "justify-center px-0",
          isActive
            ? "bg-primary-muted text-primary"
            : "text-muted-foreground hover:bg-accent hover:text-foreground",
        )
      }
    >
      {({ isActive }) => (
        <>
          {/* Active rail — a second, non-colour cue for the current page. */}
          <span
            aria-hidden
            className={cn(
              "bg-primary absolute left-0 h-5 w-0.5 rounded-r-full transition-opacity",
              isActive ? "opacity-100" : "opacity-0",
            )}
          />
          <Icon className="size-4 shrink-0" aria-hidden />
          {!collapsed && <span className="min-w-0 flex-1 truncate">{item.label}</span>}
          {!collapsed && locked && (
            <Lock className="text-muted-foreground size-3 shrink-0" aria-label="Requires developer access" />
          )}
          {!collapsed && !locked && count !== undefined && count > 0 && (
            <Badge variant="warning" className="shrink-0">
              {count}
            </Badge>
          )}
          {collapsed && count !== undefined && count > 0 && (
            <span className="bg-warning absolute top-1.5 right-2 size-1.5 rounded-full" aria-hidden />
          )}
        </>
      )}
    </NavLink>
  );

  if (!collapsed) return link;

  return (
    <TooltipTip
      side="right"
      label={
        <span className="flex items-center gap-2">
          {item.label}
          {count ? `· ${count} queued` : null}
        </span>
      }
    >
      {link}
    </TooltipTip>
  );
});
