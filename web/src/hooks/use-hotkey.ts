import * as React from "react";

interface HotkeyOptions {
  /** Require Ctrl (Windows/Linux) or ⌘ (macOS). */
  meta?: boolean;
  shift?: boolean;
  /** Fire even while a text field has focus. Off by default. */
  allowInInput?: boolean;
  enabled?: boolean;
}

function isEditableTarget(target: EventTarget | null) {
  if (!(target instanceof HTMLElement)) return false;
  return (
    target.isContentEditable ||
    ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)
  );
}

/**
 * Bind a document-level keyboard shortcut.
 *
 * `key` is matched case-insensitively against `event.key`, so pass "k", "/",
 * "Escape", etc.
 */
export function useHotkey(key: string, handler: () => void, options: HotkeyOptions = {}) {
  const { meta = false, shift = false, allowInInput = false, enabled = true } = options;
  const handlerRef = React.useRef(handler);
  handlerRef.current = handler;

  React.useEffect(() => {
    if (!enabled) return;

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key.toLowerCase() !== key.toLowerCase()) return;
      if (meta !== (event.metaKey || event.ctrlKey)) return;
      if (shift !== event.shiftKey) return;
      if (!allowInInput && isEditableTarget(event.target)) return;

      event.preventDefault();
      handlerRef.current();
    };

    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [key, meta, shift, allowInInput, enabled]);
}
