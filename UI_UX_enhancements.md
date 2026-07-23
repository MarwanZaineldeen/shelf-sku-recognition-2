# UI/UX Enhancements — Frontend Redesign Report

**Date:** 2026-07-23
**Scope:** Complete redesign and rebuild of the web frontend for the Retail AI Shelf SKU Recognition platform.

---

## Summary

The frontend was fully rebuilt from a single-page vanilla HTML/CSS/JS dashboard
(`server/static/index.html`, `app.js`, `style.css` — ~3,100 lines total) into a
modern, production-grade React application located in [`web/`](web/). All
backend routes, request/response contracts, and business logic in
`server/app.py` were preserved unchanged except for the minimum required to
serve the new single-page app.

The old vanilla frontend still exists at `server/static/` and is served as an
automatic fallback if the new build is ever absent.

---

## Technology Stack

| Layer | Choice |
|---|---|
| Framework | React 19 + Vite 6 + TypeScript (strict mode) |
| Styling | Tailwind CSS v4 + hand-written shadcn/ui-style component library |
| Routing | React Router 7, route-level code splitting via `React.lazy` |
| Server state | TanStack Query (caching, retries, optimistic cache updates) |
| Client state | Zustand (preferences, admin gate, audit view state) |
| Forms | React Hook Form + Zod validation |
| Animation | Framer Motion (respects `prefers-reduced-motion`) |
| Charts | Recharts |
| Icons | Lucide |

---

## Design System

Generated using the `ui-ux-pro-max` design-intelligence skill, tuned for a
dark-first enterprise ML-ops dashboard:

- **Palette:** "Analytics Dashboard" blue (`#2563eb` / `#4f8bf7`) over a deep-slate
  surface, with a reserved status vocabulary — emerald (automated), amber
  (needs review), rose (unknown/rejected class), violet (developer-only areas).
- **Typography:** Plus Jakarta Sans Variable for UI text, JetBrains Mono
  Variable for identifiers, scores, and bounding-box coordinates. Both are
  self-hosted (no external font CDN calls).
- **Tokens:** a full semantic token system (background/foreground/card/border/
  ring/status colors) driving both a light and a dark theme — each theme is
  independently authored, not a simple inversion.
- **Elevation & radius:** a single `--radius` variable drives the whole corner
  scale; a five-step, theme-matched shadow scale replaces ad-hoc box-shadows.
- **Charts:** validated with the `dataviz` skill's palette validator — a
  single-hue ordinal ramp (not categorical) for the ordered pipeline stages,
  and a 100%-stacked bar instead of a donut so close values stay comparable.

---

## Pages Rebuilt

| Route | What changed |
|---|---|
| `/audit` (Shelf Audit) | Drag-and-drop upload, live metrics tiles, an accessible bounding-box overlay (keyboard-navigable, zoomable, status-filterable), and a docked facing inspector with score gauges, candidate table, and approve/correct actions. |
| `/review` (Review Queue) | One row per facing actually routed to HITL review, with a searchable SKU picker pre-filled with the model's top guess and a single confirm action. |
| `/catalog` (SKU Catalogue) | Searchable/brand-filterable grid with multi-select and bulk delete; search/filter state lives in the URL so views are shareable. |
| `/onboarding` (Add New SKU) | Validated form (React Hook Form + Zod) with a drag-and-drop reference-crop dropzone and a live diagnostics panel (embedding count, catalogue card, shelf validation benchmark). |
| `/performance` (Performance) | Per-stage latency bar chart + share-of-time chart, plus an accessible data-table equivalent and an architecture flow diagram. |
| `/learning` (Continual Learning) | Developer-passcode-gated workbench for review persistence stats and gallery vector curation. |

Shared shell: collapsible sidebar (desktop) / drawer (mobile), a ⌘K command
palette, a live service-status indicator, and a light/dark theme toggle.

---

## Key UX Decisions

1. **Bounding-box overlay is DOM-based, not canvas-based.** Boxes are real
   focusable `<button>` elements positioned over the shelf image, making them
   keyboard-navigable and screen-reader readable, and keeping labels sharp at
   any zoom level. Labels reveal on hover/selection by default, with an
   explicit "show all" toggle — a dense shelf can have 150+ facings, and
   always-on labels made the image unreadable.
2. **Review Queue reflects only genuinely queued facings.** Testing against
   a live sample audit showed the annotation list also contains open-set
   ("Class Unknown") rejections that were never sent for human review. These
   are shown as red boxes on the canvas instead of inflating the queue count.
3. **Every async state has a purpose-built UI:** loading skeletons sized to
   the real content (no layout shift), empty states with a clear next action,
   and error states with a retry button — no blank screens or raw error text.

---

## Accessibility

- Semantic landmarks and a skip-to-content link.
- One consistent, visible focus treatment across every interactive element.
- Status is never color-alone — every badge pairs a color with an icon and a word.
- Dialogs, drawers, menus, and the command palette use Radix UI primitives for
  correct focus trapping and ARIA semantics.
- Forms wire `aria-describedby` / `aria-invalid` automatically via a shared
  form component layer.
- `prefers-reduced-motion` is honored both in CSS and in Framer Motion config.

---

## Performance

- Route-based code splitting — Recharts (~305 KB) only loads on the
  Performance page, not on initial load.
- Manual vendor chunking (`react`, `router`, `query`, `motion`, `charts`,
  `vendor`) so a single dependency bump doesn't invalidate the whole cache.
- Content-hashed build assets served with `immutable` caching; the HTML shell
  is always `no-store` so deploys are picked up immediately.
- `React.memo` applied to components that render per-row/per-card in long lists.

---

## Backend Changes

Minimal and additive, in [`server/app.py`](server/app.py):

- `GET /` now serves the built React app from `server/static/app/`, falling
  back to the legacy `server/static/index.html` if no build is present.
- A catch-all route serves the SPA shell for client-side paths (`/catalog`,
  `/review`, etc.) so deep links and page refreshes work correctly.
- Hashed asset files under `/static/app/assets` are served with long-term
  immutable caching instead of `no-cache`.
- Unknown paths under a recognized API prefix (`/v1`, `/api`, etc.) still
  correctly return a JSON 404, not the SPA shell.

**No existing API route, request/response schema, or multipart field name was
changed.**

---

## Verification Performed

- `tsc --noEmit` — clean, no type errors.
- Full production build (`npm run build`) — succeeds, output written to
  `server/static/app/`.
- End-to-end browser testing (headless Chrome) against the live FastAPI
  service: sample shelf audit (164 facings), bounding-box selection and
  inspector, all status filters, review queue confirm flow, catalogue search
  and delete, onboarding form, performance charts, admin gate unlock, light/dark
  theme switch, command palette, and mobile viewport (390px) — **zero console
  errors, zero failed network requests, no horizontal overflow.**
- Existing Python test suite (`python -m unittest discover -s tests`) —
  268/271 passing; the 3 failures are pre-existing and unrelated (missing
  `data/Nesquik` test fixture directory, not present in this checkout).

---

## How to Run

```bash
# Backend (serves the already-built frontend)
python -m uvicorn server.app:app --host 127.0.0.1 --port 8000
# Open http://127.0.0.1:8000/

# Frontend development (hot reload)
cd web
npm install
npm run dev
# Open http://localhost:5173/

# Rebuild frontend for production
cd web
npm run build   # outputs to ../server/static/app
```

Full architecture, folder structure, and design-system documentation:
see [`web/README.md`](web/README.md).
