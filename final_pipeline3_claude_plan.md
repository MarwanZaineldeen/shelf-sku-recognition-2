# Final Pipeline 3 Plan — Active Continual Learning & Gallery Curation

> **Status**: Agreed implementation plan. Supersedes `docs/pipeline_3_active_continual_learning_integration_spec.md` where the two conflict.
> **Target**: `shelf-sku-recognition-2`, branch `main`
> **Backbone**: DINOv3 ViT-B/16, 768-D, `ml.embeddings.dinov3.DINOv3Extractor`
> **Gallery**: `data/processed/crops/gt_clean/retail_sku_registry_dinov3.db` (31,656 × 768-D)

---

## 0. What changed from the original spec, and why

The original spec's four core ideas (curation, hard negatives, promotion gate, SupCon) are kept in full. The changes below fix correctness bugs, close a production-safety gap, and resolve contradictions with the code the spec integrates against.

| # | Change | Reason |
| :--- | :--- | :--- |
| 1 | Fix `kcenter_greedy_select`: prime distances over **all** seeds, mask selected indices, seed from the **medoid** instead of `np.random.randint` | Multi-seed incremental curation silently ignored all but the last seed; duplicate indices possible; non-reproducible in a repo that pins `random_seed: 42` |
| 2 | Promotion gate uses a **paired bootstrap on Δ** (challenger − champion), not challenger-CI vs. champion point estimate | Original test treats champion as noiseless; paired resampling cancels shared query-difficulty variance, so it is both more correct and more sensitive |
| 3 | Gate promotes on **automation rate at ≥95% precision**, not raw Top-1; requires a **Platt refit** on a validation split | Promoting a projected embedding space invalidates `a=15.0, b=-11.0` and the 0.75/0.92 cosine bands with no error raised — a silent production failure |
| 4 | Every gallery row and review row carries `source_image`; gate's test set **excludes any source image present in the gallery** | Feeding HITL crops into the gallery leaks near-duplicates of test queries and auto-promotes worse models |
| 5 | Reviews store the **768-D embedding BLOB** at review time | Curation and SupCon both need vectors; re-extracting requires the crop file to survive and DINOv3 to be loaded. It was already computed during the audit — free |
| 6 | Schema expresses **open-set rejection** (`NOT_IN_CATALOG`) | `true_class_id INTEGER NOT NULL` cannot represent "competitor product" — the most valuable signal in an open-set system |
| 7 | Curation **soft-deletes**; adds `active` / `pruned_in_version` to `sku_crops` | Hard `DELETE` is unrecoverable and `rollback_version` cannot restore pruned rows |
| 8 | Reviews live in a **separate `reviews.db`** | Keeps write-heavy review churn out of the read-mostly gallery; independent rollback |
| 9 | Pipeline 3 **does not change gating thresholds**; reads them from config | Spec says 0.94/0.78, orchestrator does 0.92/0.75, README says 0.85. Not this pipeline's job to arbitrate |
| 10 | `N=500` cap is **configurable and validated against the real per-class histogram** before being committed | 31,656 / 67 ≈ 472 average — a 500 cap prunes almost nothing unless heavily skewed. The "~22,000" figure is an unvalidated guess |
| 11 | SupCon head ships **opt-in, default OFF** | Top-1 93.65% / Top-5 98.70% means the failure mode is *ranking within* the top 5 — already the VLM reranker's job. A head trained on a few hundred reviews risks overfitting for ~2pp, and is the one change forcing full recalibration |

### Agreed decisions

| Decision | Resolution |
| :--- | :--- |
| `hitl_store.py` | **Verified dead**: zero references repo-wide; `insert_crop` defined nowhere, so `approve_task`/`correct_task` raise `AttributeError`. Both stores coexist through Phase 1 for side-by-side testing; `hitl_store.py` is deleted in Phase 5 once `store.py` demonstrably covers it. |
| Review DB location | Separate `data/processed/active_learning/reviews.db` |
| Scope | All 7 modules from the spec tree |
| SupCon head | Opt-in, default OFF, promoted only by the gate |

Retained unchanged from the spec: **D1** two-table review schema, **D2** batched 50-review update sessions, **D3** volume-based fine-tune trigger at N≥500 plus `--force`, **D4** SupCon loss.

---

## 1. Module layout

```
ml/active_learning/
├── __init__.py          # [NEW] package marker — currently absent
├── store.py             # review logging: reviews + review_candidates (reviews.db)
├── curation.py          # k-center greedy max-min, near-dup rejection, per-class cap
├── memory.py            # non-parametric index updater (SQLite + NumpyCosineIndex)
├── hard_negatives.py    # confusion-pair miner from review_candidates
├── finetune.py          # SupCon ProjectionHead trainer (opt-in)
├── gate.py              # paired-bootstrap champion/challenger promotion gate
├── loop.py              # master orchestration: session → curate → gate → promote
└── hitl_store.py        # [DELETE in Phase 5 after equivalence testing]
```

---

## 2. Schemas

### 2.1 `reviews.db` (new, separate)

```sql
CREATE TABLE IF NOT EXISTS reviews (
    review_id              TEXT PRIMARY KEY,
    crop_path              TEXT,
    source_image           TEXT NOT NULL,   -- leakage control (change #4)
    decision               TEXT NOT NULL,   -- APPROVED | CORRECTED | NOT_IN_CATALOG (#6)
    true_class_id          INTEGER,         -- NULL when NOT_IN_CATALOG (#6)
    top1_predicted_class_id INTEGER NOT NULL,
    top1_similarity        REAL NOT NULL,
    calibrated_probability REAL,
    is_correction          INTEGER NOT NULL,
    embedding              BLOB,            -- 768-D float32, captured at audit time (#5)
    embedding_dim          INTEGER,
    model_version          TEXT NOT NULL,   -- e.g. "dinov3_vitb16_raw768"
    reviewer_id            TEXT NOT NULL,
    consumed_in_batch      TEXT,            -- NULL = unconsumed; enables the N>=500 "new" trigger
    created_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS review_candidates (
    candidate_id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id    TEXT NOT NULL,
    rank         INTEGER NOT NULL,
    class_id     INTEGER NOT NULL,
    similarity   REAL NOT NULL,
    FOREIGN KEY(review_id) REFERENCES reviews(review_id)
);

CREATE INDEX IF NOT EXISTS idx_reviews_unconsumed ON reviews(consumed_in_batch);
CREATE INDEX IF NOT EXISTS idx_candidates_review  ON review_candidates(review_id);
```

### 2.2 Gallery DB migration (additive, backward-compatible)

```sql
ALTER TABLE sku_crops ADD COLUMN active            INTEGER DEFAULT 1;
ALTER TABLE sku_crops ADD COLUMN pruned_in_version INTEGER;
ALTER TABLE sku_crops ADD COLUMN origin            TEXT DEFAULT 'seed';  -- seed | continual
```

`fetch_all_references` gains `AND s.active = 1`. Existing rows default to `active=1`, so current behaviour is unchanged. Curation sets `active=0, pruned_in_version=<v>` — never `DELETE` (change #7).

---

## 3. Module contracts

### 3.1 `store.py`
`ReviewStore(IPlugin)` — `initialize/health_check/shutdown`, matching every other plugin in `ml/base.py`.

- `log_review(...) -> str` — writes one review + its Top-5 candidate rows in a single transaction
- `fetch_unconsumed(limit=None) -> list[ReviewRecord]`
- `mark_consumed(review_ids, batch_id)`
- `count_unconsumed() -> int` — drives the D3 trigger
- `fetch_embeddings_by_class(class_id) -> (np.ndarray, list[meta])`

### 3.2 `curation.py`

```python
def kcenter_greedy_select(
    vectors: np.ndarray,          # (N, D), L2-normalized (asserted)
    k: int,
    initial_indices: list[int] | None = None,
    seed_strategy: str = "medoid",  # deterministic (#1)
) -> list[int]:
```

Corrected algorithm:
1. Assert unit norms (gallery BLOBs are only re-normalized inside `NumpyCosineIndex.add`).
2. Seed: `initial_indices` if given, else the medoid — the vector maximizing mean cosine to the class set.
3. **Prime `min_distances` across every seed**, not just the last (fixes the multi-seed bug).
4. Each step: fold in the newest selection, **mask already-selected indices to `-inf`**, then `argmax`.

Plus:
- `reject_near_duplicates(vectors, threshold=0.98) -> keep_mask`
- `curate_class(vectors, meta, cap=500, near_dup=0.98) -> (keep_idx, prune_idx)`
- `class_size_histogram(db_path) -> dict[int, int]` — **run first**; the cap is only committed after inspecting the real distribution (change #10)

### 3.3 `memory.py`
Applies a curation decision transactionally: bump `gallery_version`, soft-delete pruned rows, insert promoted review crops with `origin='continual'` and their `source_image`, then rebuild the in-memory `NumpyCosineIndex`. Every write is reversible via `rollback_version`.

### 3.4 `hard_negatives.py`
Mines confusion pairs `(true_class, competing_class, mean_similarity, frequency)` from `review_candidates` where a non-true class outranked or nearly outranked the truth. Output feeds the batch sampler in `finetune.py` and produces a standalone confusion report — valuable on its own even with SupCon disabled.

### 3.5 `finetune.py` (opt-in, default OFF)
`SupConProjectionHead(768 → 512 → 128)`, L2-normalized output, as specced. Two changes:

- **LayerNorm instead of BatchNorm1d.** BN running stats are unstable when trained on a few hundred reviews, and train/eval behaviour diverges — exactly the silent-drift failure mode this pipeline must avoid.
- **Hard-negative-aware batch sampling**: build batches around mined confusion pairs so each batch carries multiple positives per class *and* its top confusers.

Trains on cached 768-D embeddings from `reviews.db` — no image I/O, no DINOv3 forward pass. Emits a versioned checkpoint. **Never auto-promoted.**

### 3.6 `gate.py`
The safety-critical module.

```python
def paired_bootstrap_delta(
    champion_neighbors, challenger_neighbors, query_labels,
    metric_fn, n_boot=1000, seed=42,
) -> dict:  # {delta_mean, ci_lower, ci_upper}
```

Resamples query indices **jointly** across both systems (change #2). Sibling to the existing `bootstrap_metrics` in `ml/evaluation/metrics.py:138`.

Promotion requires **all** of:

1. **Superiority** — `Δ Top-1` CI₉₅ lower bound **> +0.005**: a minimum effect size, not merely > 0, because the gate fires repeatedly against one test set (guards multiple-testing)
2. **Non-inferiority** — `Δ Top-5` CI₉₅ lower bound **> −0.005**
3. **Non-inferiority** — `Δ automation-rate-at-95%-precision` CI₉₅ lower bound **> −0.005**, computed *after* refitting Platt on a validation split via `calibrate_similarity_threshold` (`ml/evaluation/metrics.py:188`) — the business metric, and the only condition that catches a silently broken calibration (change #3)
4. **Leakage assertion**: no test-set `source_image` appears in the challenger gallery — hard failure, not a warning (change #4)

> **Revision during Phase 3 implementation.** Criteria 2 and 3 originally read "CI₉₅ lower bound ≥ 0". That is a *superiority* test wearing a non-inferiority label: it demands proof of improvement. With Top-5 already near its 98.7% ceiling, sampling noise exceeds any plausible gain, so a challenger that is genuinely flat or slightly better is rejected roughly half the time. Both now use a −0.5pp non-inferiority margin, asking the correct question: *is the regression provably smaller than what we are willing to lose?*
>
> **The margin is only meaningful relative to evaluation-set size.** At ~6,100 test queries the Top-5 CI half-width is ≈0.35pp, comfortably inside a 0.5pp margin. At a few hundred queries it is ≈2pp, and the criterion cannot pass at all. The gate therefore flags a failing criterion as `underpowered` when its interval *straddles* the −margin boundary — evidence is missing, rather than harm being proven. A regression whose whole interval lies below −margin is conclusively real and is never flagged, however wide.

A passing gate emits refitted Platt coefficients and gating bands **alongside** the checkpoint. They are promoted together or not at all.

### 3.7 `loop.py`
CLI orchestration: `--session` (consume ≤50 unconsumed reviews → curate → apply → gate → report), `--finetune` (D3 trigger at N≥500, `--force` to override), `--promote <challenger_id>`, `--rollback <version>`, `--report`. Dry-run by default; writes require `--apply`.

---

## 4. Testing

No `data/` directory, no gallery DB, and no model weights exist on this machine. Everything therefore ships with **synthetic fixtures and must run model-free**. `store`, `curation`, `memory`, `hard_negatives`, and `gate` are pure numpy/sqlite; `finetune` needs only torch (2.7.1+cpu, available).

`tests/active_learning/`:

- `test_curation.py` — **multi-seed regression test** (the bug in change #1: assert every seed contributes), no duplicate indices, determinism across runs, near-dup rejection at 0.98, cap enforcement, `k >= n` passthrough
- `test_store.py` — round-trip, embedding BLOB fidelity, `NOT_IN_CATALOG` with NULL `true_class_id`, unconsumed accounting, candidate FK integrity
- `test_gate.py` — identical systems produce Δ CI straddling 0 (no promotion); a synthetically better challenger promotes; **leakage assertion fires**; a Top-1 gain paired with an automation-rate collapse is **rejected**
- `test_memory.py` — soft-delete then `rollback_version` restores the pre-curation vector count
- `test_hard_negatives.py` — planted confusion pairs are recovered in frequency order
- `test_finetune.py` — head output is (B, 128) and unit-norm; SupCon loss decreases on separable synthetic data; LayerNorm gives identical train/eval output for the same input

---

## 5. Phases

| Phase | Deliverable | Gate to proceed |
| :--- | :--- | :--- |
| **1** | `__init__.py`, `store.py`, tests. `hitl_store.py` untouched. | Store tests green; `store.py` shown to cover every `hitl_store.py` use case |
| **2** | `curation.py` + `memory.py` + gallery migration, tests | Multi-seed and rollback tests green; per-class histogram reviewed and cap ratified |
| **3** | `gate.py` with paired bootstrap + Platt refit + leakage assertion | Gate tests green, including the automation-rate-collapse rejection |
| **4** | `hard_negatives.py`, `finetune.py` (default OFF), `loop.py` | Confusion report produces sensible pairs |
| **5** | Delete `hitl_store.py`; wire `/v1/hitl/review` ([`server/app.py:381`](server/app.py#L381)) to `ReviewStore` | Phase 1 equivalence confirmed |

Phases 1–3 are the load-bearing half and are worth shipping even if 4 is deferred.

---

## 6. Out of scope

- Changing gating thresholds (change #9) — needs its own decision, separate from this pipeline
- The hardcoded `d:/Marwan/...` paths across 10 files, which currently prevent the server from starting on this machine. Independent bug, tracked separately; Pipeline 3 modules will use repo-relative paths from the outset and will not inherit it.
- Retraining the YOLO detector from review data — a later loop
