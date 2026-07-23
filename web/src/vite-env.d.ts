/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Origin of the FastAPI service. Empty = same origin. */
  readonly VITE_API_BASE?: string;
  readonly VITE_API_ORIGIN?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
