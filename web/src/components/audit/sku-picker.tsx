import * as React from "react";
import { Check, ChevronsUpDown, CircleHelp } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { useCatalog } from "@/lib/api/queries";
import { UNKNOWN_CLASS_ID } from "@/types/api";

interface SkuPickerProps {
  value: number;
  onChange: (classId: number) => void;
  id?: string;
  disabled?: boolean;
  className?: string;
  "aria-describedby"?: string;
}

/**
 * Searchable class picker.
 *
 * A native `<select>` over ~70 classes forces the reviewer to scan a long list;
 * type-ahead over brand, title and class id gets them to the right SKU in a
 * couple of keystrokes. "Class Unknown" is pinned first because open-set
 * rejection is a first-class verdict, not an edge case.
 */
export function SkuPicker({
  value,
  onChange,
  id,
  disabled,
  className,
  ...aria
}: SkuPickerProps) {
  const [open, setOpen] = React.useState(false);
  const { data: catalog, isLoading } = useCatalog();

  const selected = value === UNKNOWN_CLASS_ID ? null : catalog?.byId.get(value);
  const label =
    value === UNKNOWN_CLASS_ID
      ? "Class Unknown / out of catalogue"
      : (selected?.displayName ?? `Class ${value}`);

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          id={id}
          type="button"
          variant="outline"
          role="combobox"
          aria-expanded={open}
          disabled={disabled || isLoading}
          className={cn("w-full justify-between gap-2 font-normal", className)}
          {...aria}
        >
          <span className="flex min-w-0 items-center gap-2">
            {value === UNKNOWN_CLASS_ID && (
              <CircleHelp className="text-destructive size-3.5 shrink-0" aria-hidden />
            )}
            <span className="truncate">{isLoading ? "Loading catalogue…" : label}</span>
          </span>
          <ChevronsUpDown className="text-muted-foreground size-3.5 shrink-0" aria-hidden />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[var(--radix-popover-trigger-width)] min-w-[420px] p-0" align="start">
        <Command
          filter={(itemValue, search) =>
            itemValue.toLowerCase().includes(search.toLowerCase()) ? 1 : 0
          }
        >
          <CommandInput placeholder="Search brand, title or class id…" />
          <CommandList>
            <CommandEmpty>No matching SKU class.</CommandEmpty>
            <CommandGroup heading="Open set">
              <CommandItem
                value="unknown class out of catalogue -1"
                onSelect={() => {
                  onChange(UNKNOWN_CLASS_ID);
                  setOpen(false);
                }}
              >
                <CircleHelp className="text-destructive" aria-hidden />
                <span className="flex-1">Class Unknown / out of catalogue</span>
                {value === UNKNOWN_CLASS_ID && <Check className="text-primary size-4" aria-hidden />}
              </CommandItem>
            </CommandGroup>
            <CommandGroup heading="Catalogue">
              {catalog?.entries.map((entry) => (
                <CommandItem
                  key={entry.classId}
                  value={`${entry.classId} ${entry.brand} ${entry.displayName}`}
                  onSelect={() => {
                    onChange(entry.classId);
                    setOpen(false);
                  }}
                >
                  <span className="text-muted-foreground w-8 shrink-0 font-mono text-2xs">
                    {entry.classId}
                  </span>
                  <span className="flex-1 truncate">{entry.displayName}</span>
                  {value === entry.classId && <Check className="text-primary size-4" aria-hidden />}
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}
