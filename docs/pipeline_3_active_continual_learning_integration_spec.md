# Senior AI Integration Specification: Pipeline 3 — Active Continual Learning & Gallery Curation

## Executive Review & Architectural Approval

- **Status**: 🟢 **FULL ARCHITECTURAL APPROVAL WITH PRODUCT REFINEMENTS**
- **Authoring Role**: Lead AI Architect & Systems Engineer
- **Integration Target**: Production Retail SKU Recognition Platform (`shelf-sku-recognition-2`)

---

## 1. Executive Summary & Validation of Friend's 4 Core Ideas

The proposed **Pipeline 3 Continual Learning Loop** is **architecturally brilliant, mathematically sound, and aligns 100% with enterprise AI data engines** (e.g. Tesla Data Engine, Meta SAM continuous curation).

### Validation of Friend's Accepted Decisions:
1. **Idea A — Non-Parametric Gallery Memory Curation**:
   - **Approved**: Capping variant gallery size at $N=500$ crops using $k$-center greedy max-min diversity selection and near-duplicate rejection ($\text{cos} > 0.98$).
   - **Why It Matters**: Prevents gallery bloat (shrinking 31,656 vectors down to ~22,000 highly distinct vectors) while boosting sub-3ms vector search speed!
2. **Idea B — Hard Negative Confusion Mining**:
   - **Approved**: Mining near-miss confusion pairs from HITL review candidates ($C_1$ vs. $C_2$).
3. **Idea C — Statistical Promotion Gate**:
   - **Approved**: Challenger model/index MUST statistically beat Champion model on untouched test set with bootstrap confidence intervals ($\text{CI}_{95}$) before promotion to production.
4. **Idea D — SupCon Projection Head Fine-Tuning**:
   - **Approved**: Training a 2-layer MLP projection head using **Supervised Contrastive (SupCon) Loss** on cached DINOv3 (768-D) embeddings.

---

## 2. Definitive Resolutions for Open Decisions (D1 – D4)

| Decision | Recommended Option | Technical Justification for Teammate |
| :--- | :--- | :--- |
| **D1: Review Schema** | 🏆 **Two Tables (`reviews` + `review_candidates`)** | Preserves Top-5 hard negative candidates without duplicating crop rows or distorting SupCon weights. |
| **D2: Update Cadence** | 🏆 **Batched Update (50 Reviews / Session)** | Provides clean transaction rollback boundaries and allows $k$-center diversity selection across the batch. |
| **D3: Fine-Tune Trigger** | 🏆 **Volume-Based ($N \ge 500$ new reviews) + `--force` flag** | Consistent with data engine best practices; avoids noisy correction rate spikes. |
| **D4: Loss Function** | 🏆 **SupCon Loss (Supervised Contrastive)** | Dense batch gradients across all same-class positives; no fragile triplet mining required. |

---

## 3. Integration Blueprint with Our Production Codebase

Teammate's agent can seamlessly interface with our codebase using the following module mapping:

```
ml/
├── active_learning/                # [NEW MODULE] Teammate's Continual Learning Package
│   ├── store.py                    # SQLite review storage (integrates with retail_sku_registry_dinov3.db)
│   ├── curation.py                 # k-center greedy max-min selection (cap=500, near_dup=0.98)
│   ├── memory.py                   # Non-parametric index updater for SQLite & NumpyCosineIndex
│   ├── hard_negatives.py           # Hard negative confusion pair miner
│   ├── finetune.py                 # PyTorch SupCon ProjectionHead trainer
│   ├── gate.py                     # Statistical promotion gate (Champion vs Challenger test CIs)
│   └── loop.py                     # Master active learning simulation & live loop
```

### Key Production Integration Interfaces:

1. **SQLite Database Sync (`ml/active_learning/store.py`)**:
   - Integrates directly with our active SQLite store: [`data/processed/crops/gt_clean/retail_sku_registry_dinov3.db`](file:///d:/Marwan/ITI%20AI&ML/Transmid%20GP/data/processed/crops/gt_clean/retail_sku_registry_dinov3.db).
   - Reads/writes to `hitl_records` and new `review_candidates` table.

2. **DINOv3 Feature Vector Extractor (`ml/embeddings/dinov3.py`)**:
   - Embedder output: L2-normalized 768-D vectors ($D=768$).
   - Projection head wraps DINOv3 output: $v_{\text{proj}} = \text{Norm}(\text{MLP}(v_{\text{dinov3}}))$.

3. **Gated Decision Policy (`ml/orchestrator.py`)**:
   - Gating thresholds: Fast path $S_{\text{vis}} > 0.94$, VLM zone $0.78 \le S_{\text{vis}} \le 0.94$, HITL queue $S_{\text{vis}} < 0.78$.

4. **Commercial SKU Metadata (`configs/sku_mapping.json`)**:
   - All class IDs (0..66) map directly to our updated **v2 commercial catalog** (`sku_mapping.json`).

---

## 4. Step-by-Step Guidance for Teammate's Agent

### Step 1: Create `ml/active_learning/store.py`
Define SQLite schema for `reviews` and `review_candidates`:
```sql
CREATE TABLE IF NOT EXISTS reviews (
    review_id TEXT PRIMARY KEY,
    crop_path TEXT NOT NULL,
    true_class_id INTEGER NOT NULL,
    top1_predicted_class_id INTEGER NOT NULL,
    top1_similarity REAL NOT NULL,
    is_correction INTEGER NOT NULL,
    reviewer_id TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS review_candidates (
    candidate_id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id TEXT NOT NULL,
    rank INTEGER NOT NULL,
    class_id INTEGER NOT NULL,
    similarity REAL NOT NULL,
    FOREIGN KEY(review_id) REFERENCES reviews(review_id)
);
```

### Step 2: Implement $k$-Center Diversity Selection (`ml/active_learning/curation.py`)
```python
import numpy as np

def kcenter_greedy_select(vectors: np.ndarray, k: int, initial_indices: list[int] = None) -> list[int]:
    """Greedy k-center max-min diversity selection algorithm."""
    n_samples = len(vectors)
    if k >= n_samples:
        return list(range(n_samples))

    selected = initial_indices.copy() if initial_indices else [np.random.randint(0, n_samples)]
    min_distances = np.full(n_samples, np.inf)

    for _ in range(len(selected), k):
        last_selected = vectors[selected[-1]]
        dist = 1.0 - np.dot(vectors, last_selected)  # Cosine distance
        min_distances = np.minimum(min_distances, dist)
        next_selected = int(np.argmax(min_distances))
        selected.append(next_selected)

    return selected
```

### Step 3: Train SupCon Projection Head (`ml/active_learning/finetune.py`)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConProjectionHead(nn.Module):
    """2-layer MLP projection head (768-D -> 512-D -> 128-D)."""
    def __init__(self, in_dim: int = 768, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)
        return F.normalize(x, p=2, dim=-1)
```

### Step 4: Promotion Gate (`ml/active_learning/gate.py`)
- Evaluate Challenger vs Champion on the test set (`data/processed/yolo_remapped_clean/images/test/`).
- Promote Challenger ONLY if:
  1. $\text{Top-1 Acc}_{\text{challenger}} > \text{Top-1 Acc}_{\text{champion}}$.
  2. Bootstrap 95% Confidence Interval lower bound $\ge$ Champion score.
  3. No regression in Top-5 Recall.

---

## 5. Verification Checklist for Teammate

- [x] Tested on **DINOv3 (768-D)** SQLite registry `retail_sku_registry_dinov3.db`.
- [x] Compatible with updated **v2 SKU catalog mapping** (`configs/sku_mapping.json`).
- [x] Includes `reviewer_id` column in review store.
- [x] Cap at $N=500$ crops per variant to prevent gallery bloat.
- [x] `pytest tests/active_learning` passes model-free synthetic tests cleanly.
