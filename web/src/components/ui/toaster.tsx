import { Toaster as Sonner } from "sonner";
import { usePreferencesStore } from "@/stores/preferences";
import { useResolvedTheme } from "@/hooks/use-theme";

/** Toast host. Bottom-right on desktop, top on mobile so it clears thumbs. */
export function Toaster() {
  const resolved = useResolvedTheme();
  const reducedMotion = usePreferencesStore((state) => state.reducedMotion);

  return (
    <Sonner
      theme={resolved}
      position="bottom-right"
      duration={reducedMotion ? 6000 : 4000}
      closeButton
      richColors={false}
      toastOptions={{
        classNames: {
          toast:
            "group !bg-popover !text-popover-foreground !border-border !rounded-lg !shadow-lg !text-sm !font-sans",
          title: "!font-semibold",
          description: "!text-muted-foreground !text-xs",
          actionButton: "!bg-primary !text-primary-foreground !rounded-md !text-xs",
          cancelButton: "!bg-secondary !text-secondary-foreground !rounded-md !text-xs",
          success: "[&_[data-icon]]:!text-success",
          error: "[&_[data-icon]]:!text-destructive",
          warning: "[&_[data-icon]]:!text-warning",
          info: "[&_[data-icon]]:!text-info",
        },
      }}
    />
  );
}
