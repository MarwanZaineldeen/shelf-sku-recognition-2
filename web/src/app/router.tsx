import { lazy } from "react";
import { createBrowserRouter, Navigate } from "react-router-dom";
import { AppShell } from "@/components/layout/app-shell";
import { RequireAdmin } from "@/components/layout/admin-gate";
import { NotFoundPage } from "@/routes/not-found";

/**
 * Route-level code splitting: each page is its own chunk, so the initial
 * payload is just the shell plus the audit workspace.
 */
const AuditPage = lazy(() => import("@/routes/audit/audit-page"));
const ReviewPage = lazy(() => import("@/routes/review/review-page"));
const CatalogPage = lazy(() => import("@/routes/catalog/catalog-page"));
const OnboardingPage = lazy(() => import("@/routes/onboarding/onboarding-page"));
const PerformancePage = lazy(() => import("@/routes/performance/performance-page"));
const LearningPage = lazy(() => import("@/routes/learning/learning-page"));

export const router = createBrowserRouter([
  {
    path: "/",
    element: <AppShell />,
    errorElement: <NotFoundPage />,
    children: [
      { index: true, element: <Navigate to="/audit" replace /> },
      { path: "audit", element: <AuditPage /> },
      { path: "review", element: <ReviewPage /> },
      { path: "catalog", element: <CatalogPage /> },
      { path: "onboarding", element: <OnboardingPage /> },
      { path: "performance", element: <PerformancePage /> },
      {
        path: "learning",
        element: (
          <RequireAdmin>
            <LearningPage />
          </RequireAdmin>
        ),
      },
      { path: "*", element: <NotFoundPage /> },
    ],
  },
]);
