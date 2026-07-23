/** Display formatters. Keep every number/label transformation in this file. */

const integerFormatter = new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 });

export function formatInteger(value: number | null | undefined): string {
  return integerFormatter.format(value ?? 0);
}

/** `0.9165` → `91.7%` */
export function formatPercent(value: number | null | undefined, digits = 1): string {
  return `${((value ?? 0) * 100).toFixed(digits)}%`;
}

/** Fixed-precision score for tables, e.g. `0.9165`. */
export function formatScore(value: number | null | undefined, digits = 4): string {
  return (value ?? 0).toFixed(digits);
}

/** Milliseconds with a sensible unit: `842 ms`, `7.96 s`. */
export function formatDuration(ms: number | null | undefined): string {
  const value = ms ?? 0;
  if (value < 1000) return `${value.toFixed(value < 10 ? 1 : 0)} ms`;
  return `${(value / 1000).toFixed(2)} s`;
}

export function formatBBox(bbox: { x1: number; y1: number; x2: number; y2: number }): string {
  return `${Math.round(bbox.x1)}, ${Math.round(bbox.y1)} → ${Math.round(bbox.x2)}, ${Math.round(bbox.y2)}`;
}

/** Turn an arbitrary filename into something safe for a download attribute. */
export function slugifyFilename(value: string): string {
  return value.replace(/[^a-zA-Z0-9]+/g, "_").replace(/^_+|_+$/g, "") || "export";
}

export function pluralize(count: number, singular: string, plural = `${singular}s`): string {
  return `${formatInteger(count)} ${count === 1 ? singular : plural}`;
}

/** Trigger a client-side file download without leaking the object URL. */
export function downloadTextFile(filename: string, contents: string, mime = "text/plain") {
  const blob = new Blob([contents], { type: `${mime};charset=utf-8` });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}
