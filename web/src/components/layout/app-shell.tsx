import * as React from "react";
import { Link, Outlet, useLocation } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import {
  ChevronsLeft,
  ChevronsRight,
  Lock,
  LogOut,
  Menu,
  Search,
  ShieldCheck,
} from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Kbd } from "@/components/ui/kbd";
import { Separator } from "@/components/ui/separator";
import { Sheet, SheetContent, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { TooltipTip } from "@/components/ui/tooltip";
import { ErrorBoundary } from "@/components/common/error-boundary";
import { RouteFallback } from "@/components/common/states";
import { Brand } from "./brand";
import { SidebarNav } from "./sidebar-nav";
import { ThemeToggle } from "./theme-toggle";
import { CommandMenu } from "./command-menu";
import { AdminGateDialog } from "./admin-gate";
import { findNavItem } from "@/config/navigation";
import { useAdminStore } from "@/stores/admin";
import { usePreferencesStore } from "@/stores/preferences";
import { useHotkey } from "@/hooks/use-hotkey";
import { useIsDesktop } from "@/hooks/use-media-query";
import { useCurrentAudit } from "@/lib/api/queries";
import { buildAuditModel } from "@/lib/audit";

/**
 * Application chrome: persistent rail on desktop, drawer on mobile, plus the
 * command palette and the developer gate that any route can summon.
 */
export function AppShell() {
  const location = useLocation();
  const isDesktop = useIsDesktop();
  const collapsed = usePreferencesStore((state) => state.sidebarCollapsed);
  const toggleSidebar = usePreferencesStore((state) => state.toggleSidebar);
  const unlocked = useAdminStore((state) => state.unlocked);
  const lock = useAdminStore((state) => state.lock);

  const [commandOpen, setCommandOpen] = React.useState(false);
  const [adminOpen, setAdminOpen] = React.useState(false);
  const [mobileNavOpen, setMobileNavOpen] = React.useState(false);

  useHotkey("k", () => setCommandOpen((open) => !open), { meta: true, allowInInput: true });
  useHotkey("/", () => setCommandOpen(true));

  // The queue badge tracks whatever audit is currently loaded.
  const { data: audit } = useCurrentAudit();
  const reviewCount = React.useMemo(() => buildAuditModel(audit)?.queuedCount ?? 0, [audit]);

  const railWidth = collapsed ? "lg:w-16" : "lg:w-60";
  const currentPage = findNavItem(location.pathname);

  return (
    <div className="bg-background min-h-dvh">
      <a href="#main-content" className="skip-link">
        Skip to main content
      </a>

      {/* ------------------------------ Desktop rail ------------------------ */}
      <aside
        className={cn(
          "border-border bg-card fixed inset-y-0 left-0 z-10 hidden shrink-0 flex-col border-r lg:flex",
          "transition-[width] duration-300 ease-[var(--ease-out-quint)]",
          railWidth,
        )}
      >
        <div
          className={cn(
            "flex h-14 shrink-0 items-center px-3",
            collapsed && "justify-center px-0",
          )}
        >
          <Link
            to="/audit"
            className="focus-visible:ring-ring min-w-0 rounded-md focus-visible:ring-2 focus-visible:outline-none"
          >
            <Brand compact={collapsed} />
          </Link>
        </div>

        <Separator />

        <div className="min-h-0 flex-1 overflow-y-auto px-2 py-4">
          <SidebarNav collapsed={collapsed} badges={{ review: reviewCount }} />
        </div>

        <div className={cn("space-y-2 border-t p-2", collapsed && "flex flex-col items-center")}>
          {unlocked && (
            <TooltipTip side="right" label="Lock the developer workbench">
              <Button
                variant="ghost"
                size={collapsed ? "icon" : "sm"}
                className={cn("text-admin", !collapsed && "w-full justify-start")}
                onClick={() => {
                  lock();
                  toast.info("Developer session locked");
                }}
              >
                <ShieldCheck aria-hidden />
                {!collapsed && "Lock admin session"}
              </Button>
            </TooltipTip>
          )}
          <TooltipTip side="right" label={collapsed ? "Expand sidebar" : "Collapse sidebar"}>
            <Button
              variant="ghost"
              size={collapsed ? "icon" : "sm"}
              className={cn("text-muted-foreground", !collapsed && "w-full justify-start")}
              onClick={toggleSidebar}
              aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            >
              {collapsed ? <ChevronsRight aria-hidden /> : <ChevronsLeft aria-hidden />}
              {!collapsed && "Collapse"}
            </Button>
          </TooltipTip>
        </div>
      </aside>

      {/* --------------------------------- Main ----------------------------- */}
      <div
        className={cn(
          "flex min-h-dvh flex-col transition-[padding] duration-300 ease-[var(--ease-out-quint)]",
          collapsed ? "lg:pl-16" : "lg:pl-60",
        )}
      >
        <header className="border-border bg-background/85 sticky top-0 z-10 flex h-14 shrink-0 items-center gap-2 border-b px-3 backdrop-blur-md sm:px-5">
          {/* Mobile drawer trigger */}
          <Sheet open={mobileNavOpen} onOpenChange={setMobileNavOpen}>
            <SheetTrigger asChild>
              <Button variant="ghost" size="icon" className="lg:hidden" aria-label="Open navigation">
                <Menu aria-hidden />
              </Button>
            </SheetTrigger>
            <SheetContent side="left" className="p-0">
              <SheetTitle className="sr-only">Navigation</SheetTitle>
              <div className="flex h-14 items-center px-4">
                <Brand />
              </div>
              <Separator />
              <div className="min-h-0 flex-1 overflow-y-auto px-2 py-4">
                <SidebarNav
                  badges={{ review: reviewCount }}
                  onNavigate={() => setMobileNavOpen(false)}
                />
              </div>
            </SheetContent>
          </Sheet>

          <Link to="/audit" className="lg:hidden" aria-label="Retail AI home">
            <Brand compact />
          </Link>

          <h2 className="hidden min-w-0 truncate text-sm font-semibold lg:block">
            {currentPage?.label ?? "Retail AI"}
          </h2>

          <div className="flex-1" />

          {/* Search / command palette entry point */}
          <Button
            variant="outline"
            size="sm"
            onClick={() => setCommandOpen(true)}
            className="text-muted-foreground hidden w-56 justify-start gap-2 font-normal sm:flex xl:w-72"
          >
            <Search className="size-3.5" aria-hidden />
            <span className="flex-1 text-left">Search…</span>
            <Kbd>⌘K</Kbd>
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="sm:hidden"
            onClick={() => setCommandOpen(true)}
            aria-label="Open search"
          >
            <Search aria-hidden />
          </Button>

          {!unlocked && (
            <TooltipTip label="Unlock the developer workbench">
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setAdminOpen(true)}
                aria-label="Unlock developer workbench"
              >
                <Lock aria-hidden />
              </Button>
            </TooltipTip>
          )}
          {unlocked && (
            <TooltipTip label="Lock the developer workbench">
              <Button
                variant="ghost"
                size="icon"
                className="text-admin lg:hidden"
                onClick={() => {
                  lock();
                  toast.info("Developer session locked");
                }}
                aria-label="Lock developer workbench"
              >
                <LogOut aria-hidden />
              </Button>
            </TooltipTip>
          )}

          <ThemeToggle />
        </header>

        <main id="main-content" className="min-h-0 flex-1">
          <ErrorBoundary resetKey={location.pathname}>
            <React.Suspense fallback={<RouteFallback />}>
              <AnimatePresence mode="wait" initial={false}>
                <motion.div
                  key={location.pathname}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -4 }}
                  transition={{ duration: 0.18, ease: [0.22, 1, 0.36, 1] }}
                  className="h-full"
                >
                  <Outlet />
                </motion.div>
              </AnimatePresence>
            </React.Suspense>
          </ErrorBoundary>
        </main>
      </div>

      <CommandMenu
        open={commandOpen}
        onOpenChange={setCommandOpen}
        onRequestAdmin={() => setAdminOpen(true)}
      />
      <AdminGateDialog open={adminOpen} onOpenChange={setAdminOpen} redirectTo="/learning" />

      {/* Keeps the collapse preference sensible when resizing to a small screen. */}
      <ResizeSync isDesktop={isDesktop} />
    </div>
  );
}

/** Expands the rail again when returning to desktop from a narrow viewport. */
function ResizeSync({ isDesktop }: { isDesktop: boolean }) {
  const setCollapsed = usePreferencesStore((state) => state.setSidebarCollapsed);
  React.useEffect(() => {
    if (!isDesktop) setCollapsed(false);
  }, [isDesktop, setCollapsed]);
  return null;
}
