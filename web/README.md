# Retail AI — Frontend

React 19 + Vite + TypeScript single-page app for the shelf SKU recognition and
HITL audit suite. It talks to the FastAPI service in [`../server`](../server)
and is served by that same service in production.

---

## Quick start

```bash
# 1. Start the API (from the repo root, in another terminal)
python -m uvicorn server.app:app --host 127.0.0.1 --port 8000

# 2. Start the frontend dev server
cd web
npm install
npm run dev          # http://localhost:5173, proxies /v1 /api /healthz to :8000
```

### Production build

```bash
cd web
npm run build        # typecheck + bundle -> ../server/static/app
```

Then run the API alone — it serves the built app at `/`:

```bash
python -m uvicorn server.app:app --host 127.0.0.1 --port 8000
# open http://127.0.0.1:8000/
```

`server/static/app/` is the build output. Commit it if you want the Python
service to run with no Node toolchain present; otherwise add it to
`.gitignore` and build as part of your deploy.

| Script | Purpose |
| --- | --- |
| `npm run dev` | Vite dev server with HMR and an API proxy |
| `npm run build` | `tsc -b --noEmit` then a production bundle |
| `npm run typecheck` | Types only |
| `npm run preview` | Serve the built bundle locally |

Set `VITE_API_ORIGIN` to point the dev proxy somewhere other than
`http://127.0.0.1:8000`. Set `VITE_API_BASE` to make the built app call a
different origin (default: same origin).

---

## Architecture

```
src/
├─ app/                  Composition root
│  ├─ providers.tsx      Query client, tooltips, motion config, toasts, theme
│  └─ router.tsx         Route table; every page is a lazy chunk
│
├─ routes/               One folder per page — the only place that composes screens
│  ├─ audit/             Shelf Audit workspace (default route)
│  ├─ review/            Human-in-the-loop queue
│  ├─ catalog/           SKU catalogue explorer
│  ├─ onboarding/        Few-shot SKU registration (+ its Zod schema)
│  ├─ performance/       Latency benchmark (+ its Recharts components)
│  └─ learning/          Continual learning workbench (developer-gated)
│
├─ components/
│  ├─ ui/                Design-system primitives (shadcn/ui conventions)
│  ├─ common/            App-level building blocks: PageHeader, StatCard,
│  │                     EmptyState, ErrorState, skeletons, ErrorBoundary
│  ├─ layout/            Shell: sidebar, topbar, command palette, admin gate
│  ├─ audit/             Shelf overlay, inspector, candidate table, SKU picker
│  └─ onboarding/        Reference-crop dropzone
│
├─ lib/
│  ├─ api/
│  │  ├─ client.ts       fetch wrapper + typed ApiError
│  │  ├─ endpoints.ts    One function per backend route — the only URL builder
│  │  ├─ queries.ts      TanStack Query hooks + query keys
│  │  └─ query-client.ts Cache policy
│  ├─ audit.ts           Domain model: AuditResponse -> Facing[] + report export
│  ├─ format.ts          Every number/label formatter
│  └─ utils.ts           `cn`
│
├─ stores/               Zustand: preferences (persisted), admin gate
│                        (session), audit view state (ephemeral)
├─ hooks/                use-theme, use-media-query, use-hotkey, use-facing-review
├─ config/               Navigation table, pipeline benchmark constants
├─ types/api.ts          Wire types mirroring server/schemas.py
└─ styles/globals.css    Design tokens, both themes, base layer
```

### Rules the structure enforces

- **Components never build URLs or call `fetch`.** They use hooks from
  `lib/api/queries.ts`, which call `endpoints.ts`, which calls `client.ts`.
- **`types/api.ts` mirrors the backend exactly** and is never reshaped. Anything
  the UI needs in a different shape is derived in `lib/audit.ts`.
- **Formatting lives in `lib/format.ts`.** No inline `toFixed` in JSX.
- **Colour, spacing, radius and type come from tokens.** Components use
  semantic classes (`bg-card`, `text-muted-foreground`), never raw palette
  values. The one deliberate exception is `shelf-canvas.tsx`, which paints
  bounding boxes in fixed status colours so the overlay reads identically in
  both themes and matches the exported report.

---

## Design system

Generated with the `ui-ux-pro-max` skill and tuned for a dark-first ML-ops
dashboard.

| Token group | Choice |
| --- | --- |
| Palette | "Analytics Dashboard" blue (`#2563eb` / `#4f8bf7`) over a deep-slate surface |
| Status | emerald = automated, amber = needs review, rose = unknown class, violet = developer-only |
| Headings & body | Plus Jakarta Sans Variable (self-hosted via `@fontsource-variable`) |
| Numerals, ids, scores | JetBrains Mono Variable |
| Radius | One `--radius` knob drives the whole scale |
| Elevation | Five layered shadows, colour-matched per theme |
| Z-index | Named scale (sticky 10, drawer 20, overlay 30, modal 40, toast 50) |

Both themes are authored, not derived: `.dark` has its own steps rather than an
inverted copy. `index.html` resolves the stored preference before first paint so
the shell never flashes the wrong theme.

### Charts

`config/pipeline.ts` holds a **single-hue ordinal ramp** rather than a
categorical palette, because pipeline stages are ordered and the encoded measure
is magnitude. Both ramps were checked with the `dataviz` validator: monotone
lightness, ≥0.06 ΔL between steps, and the light end clears its surface. Share
of total time is a 100% stacked bar, not a donut, so close values stay
comparable — and every chart has a table equivalent on the same page.

---

## Accessibility

- Semantic landmarks, a skip link, and one focus treatment applied globally.
- Bounding boxes are focusable `<button>` elements positioned over the image
  rather than shapes painted into a `<canvas>` — the overlay is keyboard
  navigable and screen-reader readable, and labels stay crisp at any zoom.
- Status is never colour-alone: every badge pairs a hue with an icon and a word.
- Dialogs, drawers, menus and the command palette are Radix primitives, so focus
  trapping, `aria-*` wiring and escape handling are correct by construction.
- Forms are `react-hook-form` + Zod through `components/ui/form.tsx`, which
  derives the `id`/`aria-describedby`/`aria-invalid` relationships automatically.
- `prefers-reduced-motion` is honoured globally in CSS and via Framer's
  `MotionConfig reducedMotion="user"`.

## Performance

- Route-level `React.lazy` — the initial payload is the shell plus the audit
  workspace; Recharts (~305 kB) only downloads on the Performance route.
- Manual vendor chunks (`react`, `router`, `query`, `motion`, `charts`,
  `vendor`) so a dependency bump does not invalidate everything.
- Content-hashed assets served with `immutable` caching; the entry HTML is
  always `no-store`.
- `React.memo` on the components that render per-row/per-card in long lists.
- TanStack Query caches catalogue and health data and dedupes in-flight
  requests; background refetching is deliberately conservative because the
  backend runs CPU inference.

---

## Backend contract

Unchanged from the previous UI. Every route, verb and multipart field name is
the same.

| Route | Used by |
| --- | --- |
| `POST /v1/audit/shelf` · `GET /v1/audit/sample` | Shelf Audit |
| `POST /v1/hitl/review` | Review Queue, Facing Inspector |
| `GET /api/catalog` · `POST /v1/catalog/delete` | SKU Catalogue, SKU picker |
| `GET /v1/next-class-id` · `POST /v1/onboard/sku` | Add New SKU |
| `GET /v1/exemplars/{class_id}` | Catalogue cards, candidate table |
| `GET /v1/active-learning/status` · `POST /v1/active-learning/curate` | Continual Learning |
| `GET /healthz` | Topbar service indicator |

The only backend change this redesign required is in
[`server/app.py`](../server/app.py): `/` now serves the built SPA (falling back
to the legacy `server/static/index.html` if no build is present), a catch-all
route returns the SPA shell for client-side paths like `/catalog`, and hashed
assets under `/static/app/assets` are cached instead of `no-store`. Unknown
paths under an API prefix still return a JSON 404.

---

## Notes

- The developer gate on Continual Learning is a **workflow guard, not
  authentication** — same client-side passcode the previous UI used. It keeps
  destructive curation out of a merchandiser's path. Anything that must be
  secured belongs behind a server-side check.
- An audit result lives in the TanStack Query cache for the session. A full page
  reload clears it, by design: the payload embeds the shelf image as a base64
  data URL and is not worth persisting.
- The review queue lists only facings the pipeline routed to `hitl_queue`.
  Open-set rejections returned in the annotation list render as red "unknown"
  boxes on the canvas and are triaged from the inspector.
