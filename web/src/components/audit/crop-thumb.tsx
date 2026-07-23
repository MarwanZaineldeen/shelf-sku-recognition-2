import { ImageOff } from "lucide-react";
import { cn } from "@/lib/utils";

interface CropThumbProps {
  src?: string | null;
  alt: string;
  className?: string;
  /** Falls back to a neutral placeholder when the image cannot be decoded. */
  onErrorHide?: boolean;
}

/**
 * Fixed-ratio crop preview. The wrapper reserves the box before the image
 * decodes so grids and tables never reflow mid-load.
 */
export function CropThumb({ src, alt, className, onErrorHide = true }: CropThumbProps) {
  return (
    <span
      className={cn(
        "bg-muted border-border relative flex shrink-0 items-center justify-center overflow-hidden rounded-md border",
        className,
      )}
    >
      {src ? (
        <img
          src={src}
          alt={alt}
          loading="lazy"
          decoding="async"
          className="size-full object-contain"
          onError={
            onErrorHide
              ? (event) => {
                  event.currentTarget.style.visibility = "hidden";
                }
              : undefined
          }
        />
      ) : (
        <ImageOff className="text-muted-foreground size-4" aria-hidden />
      )}
    </span>
  );
}
