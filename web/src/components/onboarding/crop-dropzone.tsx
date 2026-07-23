import * as React from "react";
import { FolderOpen, ImageUp, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

export const MIN_CROPS = 10;
export const MAX_CROPS = 50;

interface CropDropzoneProps {
  files: File[];
  onChange: (files: File[]) => void;
  mode?: "files" | "folder";
  disabled?: boolean;
  "aria-describedby"?: string;
  id?: string;
}

/**
 * Multi-image & folder picker for few-shot onboarding.
 *
 * Previews use object URLs keyed by file identity and are revoked on unmount,
 * so selecting 50 crops does not leak blobs across submissions.
 */
export function CropDropzone({
  files,
  onChange,
  mode = "files",
  disabled,
  id,
  ...aria
}: CropDropzoneProps) {
  const inputRef = React.useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = React.useState(false);

  const previews = React.useMemo(
    () => files.map((file) => ({ file, url: URL.createObjectURL(file) })),
    [files],
  );

  React.useEffect(() => {
    return () => previews.forEach(({ url }) => URL.revokeObjectURL(url));
  }, [previews]);

  const addFiles = (incoming: FileList | null) => {
    const images = Array.from(incoming ?? []).filter((file) => file.type.startsWith("image/"));
    if (images.length === 0) return;
    // De-duplicate by name+size so re-dropping the same folder is harmless.
    const seen = new Set(files.map((file) => `${file.name}:${file.size}`));
    const merged = [...files];
    for (const file of images) {
      const key = `${file.name}:${file.size}`;
      if (!seen.has(key)) {
        seen.add(key);
        merged.push(file);
      }
    }
    onChange(merged.slice(0, MAX_CROPS));
  };

  const generatedId = React.useId();
  const inputId = id ?? generatedId;
  const inRange = files.length >= MIN_CROPS && files.length <= MAX_CROPS;
  const isFolder = mode === "folder";
  const Icon = isFolder ? FolderOpen : ImageUp;

  return (
    <div className="space-y-3">
      <input
        ref={inputRef}
        id={inputId}
        type="file"
        accept="image/*"
        multiple
        {...(isFolder ? { webkitdirectory: "true", directory: "true" } : {})}
        disabled={disabled}
        className="hidden"
        onChange={(event) => {
          addFiles(event.target.files);
          event.target.value = "";
        }}
        {...aria}
      />

      <label
        htmlFor={inputId}
        onDragOver={(event) => {
          event.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          if (!disabled) addFiles(event.dataTransfer.files);
        }}
        className={cn(
          "flex w-full cursor-pointer flex-col items-center gap-2 rounded-lg border-2 border-dashed px-6 py-8 text-center",
          "transition-colors duration-200 focus-within:ring-ring focus-within:ring-2 focus-within:outline-none",
          dragging ? "border-primary bg-primary-muted/60" : "border-border hover:border-primary/60 hover:bg-accent/50",
          disabled && "cursor-not-allowed opacity-60 pointer-events-none",
        )}
      >
        <span className="bg-primary-muted text-primary flex size-10 items-center justify-center rounded-full pointer-events-none">
          <Icon className="size-5" aria-hidden />
        </span>
        <span className="text-sm font-semibold pointer-events-none">
          {isFolder ? "Drop folder or click to upload folder" : "Drop files or click to upload files"}
        </span>
        <span className="text-muted-foreground max-w-sm text-xs leading-relaxed pointer-events-none">
          {isFolder
            ? `Select or drop a folder containing between ${MIN_CROPS} and ${MAX_CROPS} reference product crops.`
            : `Tightly cropped images of this product only — no shelf context. Between ${MIN_CROPS} and ${MAX_CROPS} views covering different angles and lighting.`}
        </span>
      </label>

      <div className="flex flex-wrap items-center gap-2">
        <Badge variant={files.length === 0 ? "secondary" : inRange ? "success" : "warning"}>
          {files.length} / {MAX_CROPS} selected
        </Badge>
        <span className="text-muted-foreground text-2xs">
          {files.length === 0
            ? `${MIN_CROPS}–${MAX_CROPS} crops required`
            : inRange
              ? "Ready to embed"
              : files.length < MIN_CROPS
                ? `${MIN_CROPS - files.length} more needed`
                : "Too many — trim the selection"}
        </span>
        {files.length > 0 && (
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="ml-auto"
            onClick={(e) => {
              e.stopPropagation();
              onChange([]);
            }}
            disabled={disabled}
          >
            Clear all
          </Button>
        )}
      </div>

      {previews.length > 0 && (
        <ul className="grid grid-cols-[repeat(auto-fill,minmax(64px,1fr))] gap-2">
          {previews.map(({ file, url }, index) => (
            <li key={`${file.name}-${file.size}-${index}`} className="group relative">
              <img
                src={url}
                alt={`Reference crop ${index + 1}: ${file.name}`}
                loading="lazy"
                decoding="async"
                className="border-border bg-muted aspect-square w-full rounded-md border object-contain"
              />
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  onChange(files.filter((_, i) => i !== index));
                }}
                aria-label={`Remove ${file.name}`}
                className={cn(
                  "bg-destructive text-destructive-foreground absolute -top-1.5 -right-1.5 cursor-pointer rounded-full p-0.5",
                  "opacity-0 transition-opacity group-hover:opacity-100 focus-visible:opacity-100",
                  "focus-visible:ring-ring focus-visible:ring-2 focus-visible:outline-none",
                )}
              >
                <X className="size-3" aria-hidden />
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
