import * as React from "react";
import { CloudUpload, ImagePlus, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

interface ShelfDropzoneProps {
  onFile: (file: File) => void;
  onSample: () => void;
  busy?: boolean;
  busyLabel?: string;
}

/**
 * First-run surface for the audit workspace: drag-and-drop, click-to-browse,
 * or run the bundled sample shelf. Keyboard users get the same affordance
 * because the drop area is a real button.
 */
export function ShelfDropzone({ onFile, onSample, busy, busyLabel }: ShelfDropzoneProps) {
  const inputRef = React.useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = React.useState(false);

  const accept = (files: FileList | null) => {
    const file = files?.[0];
    if (file?.type.startsWith("image/")) onFile(file);
  };

  if (busy) {
    return (
      <div className="flex flex-col items-center justify-center gap-3 px-6 py-20 text-center">
        <Loader2 className="text-primary size-8 animate-spin" aria-hidden />
        <div className="space-y-1">
          <p className="text-sm font-semibold">{busyLabel ?? "Running pipeline…"}</p>
          <p className="text-muted-foreground max-w-sm text-xs leading-relaxed">
            YOLOv8 localisation → DINOv3 embedding → vector retrieval → Qwen2-VL rerank. This runs
            on CPU and can take a few seconds per shelf.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-4">
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="sr-only"
        onChange={(event) => {
          accept(event.target.files);
          event.target.value = "";
        }}
      />
      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          accept(event.dataTransfer.files);
        }}
        className={cn(
          "flex w-full cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed",
          "px-6 py-14 text-center transition-colors duration-200",
          "focus-visible:ring-ring focus-visible:ring-2 focus-visible:outline-none",
          dragging
            ? "border-primary bg-primary-muted/60"
            : "border-border hover:border-primary/60 hover:bg-accent/50",
        )}
      >
        <span className="bg-primary-muted text-primary flex size-14 items-center justify-center rounded-full">
          <CloudUpload className="size-6" aria-hidden />
        </span>
        <span className="space-y-1">
          <span className="block text-base font-semibold">Drop a shelf photo to audit</span>
          <span className="text-muted-foreground mx-auto block max-w-md text-sm leading-relaxed">
            A high-resolution scan (JPG or PNG) gives the detector the most facings to work with.
            Click to browse, or drag a file anywhere onto this panel.
          </span>
        </span>
        <span className="text-muted-foreground text-2xs">JPG · PNG · up to ~4000 px wide</span>
      </button>

      <div className="mt-4 flex flex-col items-center gap-2">
        <p className="text-muted-foreground text-xs">Nothing to hand?</p>
        <Button variant="outline" size="sm" onClick={onSample}>
          <ImagePlus aria-hidden />
          Run the sample shelf
        </Button>
      </div>
    </div>
  );
}
