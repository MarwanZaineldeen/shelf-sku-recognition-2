import { QueryCache, QueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { ApiError } from "./client";

/**
 * A single client for the app. Background refetching is deliberately
 * conservative: this dashboard talks to a local CPU inference service where an
 * accidental refetch storm is expensive.
 */
export const queryClient = new QueryClient({
  queryCache: new QueryCache({
    onError: (error, query) => {
      // Only shout about background failures once data is already on screen —
      // first-load errors are rendered inline by the route's error state.
      if (query.state.data === undefined) return;
      const message = error instanceof ApiError ? error.message : "Background refresh failed";
      toast.error(message);
    },
  }),
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      gcTime: 5 * 60_000,
      refetchOnWindowFocus: false,
      retry: (failureCount, error) => {
        // Client errors will not fix themselves; only retry transient failures.
        if (error instanceof ApiError && error.status < 500) return false;
        return failureCount < 2;
      },
    },
    mutations: { retry: false },
  },
});
