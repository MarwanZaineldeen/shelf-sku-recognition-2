import * as React from "react";

/** Subscribe to a CSS media query. Returns false during SSR/first paint. */
export function useMediaQuery(query: string): boolean {
  const subscribe = React.useCallback(
    (onChange: () => void) => {
      const media = window.matchMedia(query);
      media.addEventListener("change", onChange);
      return () => media.removeEventListener("change", onChange);
    },
    [query],
  );

  return React.useSyncExternalStore(
    subscribe,
    () => window.matchMedia(query).matches,
    () => false,
  );
}

/** Tailwind's `lg` breakpoint — the point where the sidebar becomes permanent. */
export function useIsDesktop() {
  return useMediaQuery("(min-width: 1024px)");
}
