import * as React from "react";
import { Maximize2, Minus, Plus, Tag } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { TooltipTip } from "@/components/ui/tooltip";
import { FACING_STATUS_META, type Facing } from "@/lib/audit";
import { formatPercent } from "@/lib/format";
import { useAuditUiStore } from "@/stores/audit-ui";

interface ShelfCanvasProps {
  imageSrc: string;
  imageAlt: string;
  facings: Facing[];
  visibleKeys: Set<string>;
  selectedKey: string | null;
  onSelect: (key: string) => void;
}

const STATUS_CLASSES: Record<Facing["status"], string> = {
  automated: "border-[#10b981] bg-[#10b981]/15 hover:bg-[#10b981]/25",
  review: "border-[#f59e0b] bg-[#f59e0b]/15 hover:bg-[#f59e0b]/25",
  unknown: "border-[#f43f5e] bg-[#f43f5e]/20 hover:bg-[#f43f5e]/30",
};

const LABEL_CLASSES: Record<Facing["status"], string> = {
  automated: "bg-[#10b981] text-[#04231a]",
  review: "bg-[#f59e0b] text-[#2a1a02]",
  unknown: "bg-[#f43f5e] text-white",
};

/**
 * Bounding-box overlay.
 *
 * Boxes are real focusable `<button>` elements positioned in percentages over
 * the shelf photo, rather than shapes painted into a `<canvas>`. That keeps the
 * overlay keyboard-navigable and screen-reader readable, keeps label text crisp
 * at every zoom level, and removes the manual hit-testing the old canvas
 * implementation needed.
 */
export function ShelfCanvas({
  imageSrc,
  imageAlt,
  facings,
  visibleKeys,
  selectedKey,
  onSelect,
}: ShelfCanvasProps) {
  const zoom = useAuditUiStore((state) => state.zoom);
  const setZoom = useAuditUiStore((state) => state.setZoom);
  const showLabels = useAuditUiStore((state) => state.showLabels);
  const toggleLabels = useAuditUiStore((state) => state.toggleLabels);

  const [natural, setNatural] = React.useState({ width: 0, height: 0 });
  const scrollRef = React.useRef<HTMLDivElement>(null);
  const selectedRef = React.useRef<HTMLButtonElement>(null);

  // Keep the active facing in view when selection changes from the queue/table.
  React.useEffect(() => {
    selectedRef.current?.scrollIntoView({ block: "center", inline: "center" });
  }, [selectedKey]);

  const percent = React.useCallback(
    (facing: Facing) => {
      if (!natural.width || !natural.height) return null;
      const { bbox } = facing;
      return {
        left: `${(bbox.x1 / natural.width) * 100}%`,
        top: `${(bbox.y1 / natural.height) * 100}%`,
        width: `${((bbox.x2 - bbox.x1) / natural.width) * 100}%`,
        height: `${((bbox.y2 - bbox.y1) / natural.height) * 100}%`,
      };
    },
    [natural],
  );

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {/* ------------------------------- Toolbar ------------------------------ */}
      <div className="border-border bg-muted/40 flex flex-wrap items-center gap-2 border-b px-3 py-2">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5">
          {(Object.keys(FACING_STATUS_META) as Facing["status"][]).map((status) => (
            <span key={status} className="flex items-center gap-1.5 text-2xs font-medium">
              <span
                className="size-2.5 rounded-[3px]"
                style={{ backgroundColor: FACING_STATUS_META[status].hex }}
                aria-hidden
              />
              {FACING_STATUS_META[status].label}
            </span>
          ))}
        </div>

        <div className="ml-auto flex items-center gap-1">
          <TooltipTip
            label={showLabels ? "Show labels on hover only" : "Show every box label"}
          >
            <Button
              variant={showLabels ? "secondary" : "ghost"}
              size="icon-sm"
              onClick={toggleLabels}
              aria-pressed={showLabels}
              aria-label="Show every box label"
            >
              <Tag aria-hidden />
            </Button>
          </TooltipTip>
          <Separator orientation="vertical" className="mx-1 h-5" />
          <TooltipTip label="Zoom out">
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={() => setZoom(zoom - 0.25)}
              disabled={zoom <= 0.5}
              aria-label="Zoom out"
            >
              <Minus aria-hidden />
            </Button>
          </TooltipTip>
          <span className="tabular text-muted-foreground w-11 text-center text-2xs font-semibold">
            {Math.round(zoom * 100)}%
          </span>
          <TooltipTip label="Zoom in">
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={() => setZoom(zoom + 0.25)}
              disabled={zoom >= 6}
              aria-label="Zoom in"
            >
              <Plus aria-hidden />
            </Button>
          </TooltipTip>
          <TooltipTip label="Fit to panel">
            <Button variant="ghost" size="icon-sm" onClick={() => setZoom(1)} aria-label="Fit image to panel">
              <Maximize2 aria-hidden />
            </Button>
          </TooltipTip>
        </div>
      </div>

      {/* ------------------------------- Viewport ----------------------------- */}
      <div
        ref={scrollRef}
        className="bg-muted/30 flex min-h-0 flex-1 flex-col items-center justify-start overflow-auto p-3"
        style={{ overscrollBehavior: "contain" }}
      >
        <div
          className="relative mx-auto w-full"
          style={{ width: `${zoom * 100}%`, maxWidth: zoom <= 1 ? "100%" : "none" }}
        >
          <img
            src={imageSrc}
            alt={imageAlt}
            decoding="async"
            className="block h-auto w-full rounded-lg select-none"
            onLoad={(event) => {
              const img = event.currentTarget;
              setNatural({ width: img.naturalWidth, height: img.naturalHeight });
            }}
          />

          <ul
            className="absolute inset-0 m-0 list-none p-0"
            aria-label={`${facings.length} detected product facings`}
          >
            {facings.map((facing) => {
              if (!visibleKeys.has(facing.key)) return null;
              const box = percent(facing);
              if (!box) return null;

              const isSelected = facing.key === selectedKey;
              const meta = FACING_STATUS_META[facing.status];

              return (
                <li key={facing.key} className="absolute" style={box}>
                  <button
                    ref={isSelected ? selectedRef : undefined}
                    type="button"
                    onClick={() => onSelect(facing.key)}
                    aria-pressed={isSelected}
                    aria-label={`${facing.title}. ${meta.label}. Confidence ${formatPercent(facing.confidence, 0)}. Crop ${facing.cropId}`}
                    className={cn(
                      "group absolute inset-0 cursor-pointer rounded-[3px] border-2 transition-colors duration-150",
                      "focus-visible:ring-ring focus-visible:ring-2 focus-visible:ring-offset-1 focus-visible:outline-none",
                      STATUS_CLASSES[facing.status],
                      isSelected && "border-info ring-info z-10 bg-[#06b6d4]/30 ring-2",
                    )}
                  >
                    {/*
                      Labels are revealed on hover/focus and for the selected
                      box; "show all" is an explicit opt-in. On a dense shelf
                      every-label-always is unreadable.
                    */}
                    <span
                      className={cn(
                        "pointer-events-none absolute bottom-full left-0 mb-0.5 max-w-[240px] truncate",
                        "rounded-[3px] px-1 py-px font-mono text-[10px] leading-tight font-semibold",
                        "transition-opacity duration-150",
                        isSelected ? "bg-info text-info-foreground" : LABEL_CLASSES[facing.status],
                        showLabels || isSelected
                          ? "opacity-100"
                          : "z-20 opacity-0 group-hover:opacity-100 group-focus-visible:opacity-100",
                      )}
                    >
                      {facing.status === "unknown"
                        ? "Unknown"
                        : `${facing.title} ${formatPercent(facing.confidence, 0)}`}
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        </div>
      </div>
    </div>
  );
}
