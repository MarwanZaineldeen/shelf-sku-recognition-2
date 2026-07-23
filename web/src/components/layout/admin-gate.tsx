import * as React from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { KeyRound, ShieldCheck } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogBody,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { EmptyState } from "@/components/common/states";
import { useAdminStore } from "@/stores/admin";

interface AdminGateDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Where to go once unlocked. Defaults to staying put. */
  redirectTo?: string;
}

/** Passcode prompt for the developer-only continual learning workbench. */
export function AdminGateDialog({ open, onOpenChange, redirectTo }: AdminGateDialogProps) {
  const unlock = useAdminStore((state) => state.unlock);
  const navigate = useNavigate();
  const [passcode, setPasscode] = React.useState("");
  const [error, setError] = React.useState<string | null>(null);
  const inputId = React.useId();

  React.useEffect(() => {
    if (open) {
      setPasscode("");
      setError(null);
    }
  }, [open]);

  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    if (unlock(passcode)) {
      toast.success("Developer access granted");
      onOpenChange(false);
      if (redirectTo) navigate(redirectTo);
      return;
    }
    setError("Incorrect passcode. The default developer passcode is 0000.");
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-sm">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <ShieldCheck className="text-admin size-4" aria-hidden />
            Developer access
          </DialogTitle>
          <DialogDescription>
            Gallery curation and confusion mining can rewrite the production vector store.
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={submit}>
          <DialogBody className="space-y-2">
            <Label htmlFor={inputId}>
              <KeyRound className="size-3" aria-hidden />
              Developer passcode
            </Label>
            <Input
              id={inputId}
              type="password"
              autoFocus
              autoComplete="off"
              inputMode="numeric"
              value={passcode}
              onChange={(event) => {
                setPasscode(event.target.value);
                setError(null);
              }}
              aria-invalid={Boolean(error)}
              aria-describedby={error ? `${inputId}-error` : undefined}
              className="text-center font-mono text-lg tracking-[0.4em]"
              placeholder="••••"
            />
            {error && (
              <p id={`${inputId}-error`} role="alert" className="text-destructive text-xs font-medium">
                {error}
              </p>
            )}
          </DialogBody>
          <DialogFooter>
            <Button type="button" variant="ghost" onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
            <Button type="submit" variant="admin">
              Unlock workbench
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

/** Route guard: renders children only when the developer gate is unlocked. */
export function RequireAdmin({ children }: { children: React.ReactNode }) {
  const unlocked = useAdminStore((state) => state.unlocked);
  const location = useLocation();
  const [promptOpen, setPromptOpen] = React.useState(!unlocked);

  React.useEffect(() => {
    if (!unlocked) setPromptOpen(true);
  }, [unlocked]);

  if (unlocked) return <>{children}</>;

  return (
    <div className="p-6">
      <div className="border-admin/30 bg-admin-subtle/40 rounded-xl border">
        <EmptyState
          icon={ShieldCheck}
          title="Developer access required"
          description="The Continual Learning workbench executes gallery curation against the production vector store. Unlock it with the developer passcode to continue."
          action={
            <Button variant="admin" onClick={() => setPromptOpen(true)}>
              <KeyRound aria-hidden />
              Enter passcode
            </Button>
          }
        />
      </div>
      <AdminGateDialog
        open={promptOpen}
        onOpenChange={setPromptOpen}
        redirectTo={location.pathname}
      />
    </div>
  );
}
