/**
 * Reference latency benchmark for the four inference stages.
 *
 * These are the published CPU numbers for the shipped model configuration —
 * the same figures the previous dashboard reported. They are the baseline a
 * live audit is compared against; they are not fetched, because the service
 * does not expose per-stage timings.
 */
export interface PipelineStage {
  id: string;
  order: number;
  name: string;
  model: string;
  /** Wall-clock milliseconds for a full shelf (~163 facings). */
  totalMs: number;
  /** Amortised milliseconds per detected facing. */
  perFacingMs: number;
  description: string;
}

export const BENCHMARK_FACINGS = 163;

export const PIPELINE_STAGES: PipelineStage[] = [
  {
    id: "detect",
    order: 1,
    name: "Facing localisation",
    model: "YOLOv8l · SKU110K",
    totalMs: 521,
    perFacingMs: 3.2,
    description:
      "Class-agnostic bounding-box detection across the whole shelf. One forward pass, independent of catalogue size.",
  },
  {
    id: "embed",
    order: 2,
    name: "Visual embedding",
    model: "DINOv3 ViT-B/16",
    totalMs: 7956.1,
    perFacingMs: 48.8,
    description:
      "768-D L2-normalised feature extraction per crop. The dominant cost — it scales linearly with the number of facings.",
  },
  {
    id: "retrieve",
    order: 3,
    name: "Vector retrieval",
    model: "NumPy cosine index",
    totalMs: 182,
    perFacingMs: 1.1,
    description:
      "Top-7 class-unique nearest-neighbour search over the SQLite gallery, with per-class deduplication.",
  },
  {
    id: "rerank",
    order: 4,
    name: "VLM rerank & fusion",
    model: "Qwen2-VL",
    totalMs: 350,
    perFacingMs: 2.1,
    description:
      "Zero-shot packaging-text verification on ambiguous slates, fused 80% visual / 20% textual.",
  },
];

export const PIPELINE_TOTAL_MS = PIPELINE_STAGES.reduce((sum, stage) => sum + stage.totalMs, 0);

/**
 * Single-hue ordinal ramps, light→dark, one step per stage.
 *
 * Stages are ordered, so a value ramp is the correct encoding (a categorical
 * palette would imply the stages are unrelated identities). Both ramps pass the
 * ordinal checks — monotone lightness, ≥0.06 ΔL between steps, and the light
 * end clears its surface.
 */
export const STAGE_RAMP = {
  light: ["#60a5fa", "#3b82f6", "#2563eb", "#1e40af"],
  dark: ["#93c5fd", "#60a5fa", "#3b82f6", "#2563eb"],
} as const;
