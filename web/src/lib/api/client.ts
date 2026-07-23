/**
 * Thin fetch wrapper around the FastAPI service.
 *
 * Responsibilities: absolute-URL resolution, JSON/multipart encoding, and
 * turning any non-2xx response into a typed `ApiError` carrying the backend's
 * `detail` message so the UI can surface something meaningful.
 */

/** Backend origin. Empty string = same origin (production) or Vite proxy (dev). */
const API_BASE = import.meta.env.VITE_API_BASE ?? "";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly detail?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export function resolveUrl(path: string): string {
  if (/^https?:\/\//.test(path)) return path;
  return `${API_BASE}${path}`;
}

/** Pull the most useful message out of a FastAPI error body. */
function extractDetail(body: unknown, status: number): string {
  if (typeof body === "string" && body.trim()) return body;
  if (body && typeof body === "object") {
    const detail = (body as { detail?: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      const first = detail[0] as { msg?: string; loc?: unknown[] } | undefined;
      if (first?.msg) {
        const field = Array.isArray(first.loc) ? first.loc.at(-1) : undefined;
        return field ? `${String(field)}: ${first.msg}` : first.msg;
      }
    }
  }
  return `Request failed with status ${status}`;
}

async function parseBody(response: Response): Promise<unknown> {
  const text = await response.text();
  if (!text) return null;
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return text;
  }
}

interface RequestOptions extends Omit<RequestInit, "body"> {
  body?: BodyInit | Record<string, unknown> | null;
}

export async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { body, headers, ...rest } = options;

  const isPlainObject =
    body !== null &&
    typeof body === "object" &&
    !(body instanceof FormData) &&
    !(body instanceof Blob) &&
    !(body instanceof ArrayBuffer);

  const response = await fetch(resolveUrl(path), {
    ...rest,
    headers: {
      ...(isPlainObject ? { "Content-Type": "application/json" } : {}),
      ...headers,
    },
    body: isPlainObject ? JSON.stringify(body) : (body as BodyInit | null | undefined),
  });

  const payload = await parseBody(response);

  if (!response.ok) {
    throw new ApiError(extractDetail(payload, response.status), response.status, payload);
  }

  return payload as T;
}

/** Build a `FormData` body, skipping nullish values and stringifying scalars. */
export function toFormData(
  fields: Record<string, string | number | boolean | Blob | null | undefined>,
): FormData {
  const form = new FormData();
  for (const [key, value] of Object.entries(fields)) {
    if (value === null || value === undefined) continue;
    form.append(key, value instanceof Blob ? value : String(value));
  }
  return form;
}
