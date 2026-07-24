import * as React from "react";
import { AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogBody,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

interface ConfirmDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  description: React.ReactNode;
  confirmLabel?: string;
  cancelLabel?: string;
  destructive?: boolean;
  loading?: boolean;
  onConfirm: () => void;
  confirmationText?: string;
}

/** Blocking confirmation for irreversible actions (deletes, purges). */
export function ConfirmDialog({
  open,
  onOpenChange,
  title,
  description,
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  destructive = false,
  loading = false,
  onConfirm,
  confirmationText,
}: ConfirmDialogProps) {
  const [typed, setTyped] = React.useState("");
  React.useEffect(() => {
    if (!open) setTyped("");
  }, [open]);
  const confirmed = !confirmationText || typed === confirmationText;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            {destructive && <AlertTriangle className="text-destructive size-4 shrink-0" aria-hidden />}
            {title}
          </DialogTitle>
        </DialogHeader>
        <DialogBody>
          <DialogDescription className="text-sm leading-relaxed">{description}</DialogDescription>
          {confirmationText && (
            <div className="mt-4 space-y-2">
              <label htmlFor="destructive-confirmation" className="text-sm font-medium">
                Type <span className="font-mono">{confirmationText}</span> to continue
              </label>
              <Input
                id="destructive-confirmation"
                value={typed}
                onChange={(event) => setTyped(event.target.value)}
                autoComplete="off"
              />
            </div>
          )}
        </DialogBody>
        <DialogFooter>
          <Button variant="ghost" onClick={() => onOpenChange(false)} disabled={loading}>
            {cancelLabel}
          </Button>
          <Button
            variant={destructive ? "destructive" : "default"}
            onClick={onConfirm}
            loading={loading}
            disabled={loading || !confirmed}
          >
            {confirmLabel}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
