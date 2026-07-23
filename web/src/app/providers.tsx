import * as React from "react";
import { QueryClientProvider } from "@tanstack/react-query";
import { MotionConfig } from "framer-motion";
import { queryClient } from "@/lib/api/query-client";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Toaster } from "@/components/ui/toaster";
import { useApplyTheme } from "@/hooks/use-theme";

/** Every cross-cutting provider the app needs, in one place. */
export function Providers({ children }: { children: React.ReactNode }) {
  useApplyTheme();

  return (
    <QueryClientProvider client={queryClient}>
      {/* `reducedMotion="user"` makes Framer honour the OS setting globally. */}
      <MotionConfig reducedMotion="user">
        <TooltipProvider>
          {children}
          <Toaster />
        </TooltipProvider>
      </MotionConfig>
    </QueryClientProvider>
  );
}
