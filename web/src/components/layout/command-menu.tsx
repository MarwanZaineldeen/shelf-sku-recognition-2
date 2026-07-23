import * as React from "react";
import { useNavigate } from "react-router-dom";
import { Lock, Moon, Package, Sun } from "lucide-react";
import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
} from "@/components/ui/command";
import { Badge } from "@/components/ui/badge";
import { NAV_GROUPS, NAV_ITEMS } from "@/config/navigation";
import { useCatalog } from "@/lib/api/queries";
import { useAdminStore } from "@/stores/admin";
import { useThemeToggle } from "@/hooks/use-theme";

interface CommandMenuProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onRequestAdmin: () => void;
}

/**
 * ⌘K palette. Navigation is always available; catalogue SKUs are searchable
 * once the catalogue query has resolved, and jump straight to the filtered
 * catalogue view.
 */
export function CommandMenu({ open, onOpenChange, onRequestAdmin }: CommandMenuProps) {
  const navigate = useNavigate();
  const unlocked = useAdminStore((state) => state.unlocked);
  const { resolved, setTheme } = useThemeToggle();
  // Only fetch the catalogue for the palette once the user actually opens it.
  const { data: catalog } = useCatalog();

  const run = React.useCallback(
    (action: () => void) => {
      onOpenChange(false);
      // Defer so the dialog's close animation is not interrupted by navigation.
      requestAnimationFrame(action);
    },
    [onOpenChange],
  );

  const skuResults = React.useMemo(() => catalog?.entries.slice(0, 200) ?? [], [catalog]);

  return (
    <CommandDialog open={open} onOpenChange={onOpenChange}>
      <CommandInput placeholder="Search pages, SKUs and actions…" />
      <CommandList>
        <CommandEmpty>No matches found.</CommandEmpty>

        {NAV_GROUPS.map((group) => {
          const items = NAV_ITEMS.filter((item) => item.group === group);
          if (items.length === 0) return null;
          return (
            <CommandGroup key={group} heading={group}>
              {items.map((item) => {
                const locked = Boolean(item.admin) && !unlocked;
                return (
                  <CommandItem
                    key={item.to}
                    value={`${item.label} ${item.description}`}
                    onSelect={() => run(() => (locked ? onRequestAdmin() : navigate(item.to)))}
                  >
                    <item.icon aria-hidden />
                    <span className="flex-1 truncate">{item.label}</span>
                    {locked ? (
                      <Lock className="text-muted-foreground size-3" aria-label="Locked" />
                    ) : (
                      <span className="text-muted-foreground truncate text-2xs">
                        {item.description}
                      </span>
                    )}
                  </CommandItem>
                );
              })}
            </CommandGroup>
          );
        })}

        <CommandSeparator />

        <CommandGroup heading="Appearance">
          <CommandItem
            value="toggle theme dark light appearance"
            onSelect={() => run(() => setTheme(resolved === "dark" ? "light" : "dark"))}
          >
            {resolved === "dark" ? <Sun aria-hidden /> : <Moon aria-hidden />}
            Switch to {resolved === "dark" ? "light" : "dark"} theme
          </CommandItem>
        </CommandGroup>

        {skuResults.length > 0 && (
          <>
            <CommandSeparator />
            <CommandGroup heading="Catalogue SKUs">
              {skuResults.map((entry) => (
                <CommandItem
                  key={entry.classId}
                  value={`${entry.classId} ${entry.displayName} ${entry.brand}`}
                  onSelect={() =>
                    run(() => navigate(`/catalog?q=${encodeURIComponent(entry.displayName)}`))
                  }
                >
                  <Package aria-hidden />
                  <span className="flex-1 truncate">{entry.displayName}</span>
                  <Badge variant="outline" className="font-mono">
                    {entry.classId}
                  </Badge>
                </CommandItem>
              ))}
            </CommandGroup>
          </>
        )}
      </CommandList>
    </CommandDialog>
  );
}
