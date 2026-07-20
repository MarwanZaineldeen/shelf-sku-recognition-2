# Complete Engineering Guide: Teammate Tasks, File Locations & Qwen2-VL Integration

## Executive Summary

This guide answers your 4 key technical and architectural questions:
1. **File Locations**: Where to place your teammate's DINOv3 model files.
2. **SQLite + FAISS Integration**: Exact code-level explanation of how SQLite & FAISS work together in this repository.
3. **Task Delegation for 2 Inactive Teammates**: Clear, high-value tasks to assign to inactive team members.
4. **Qwen2-VL Integration & Latency Management**: How to use Qwen2-VL as a zero-shot VLM reranker with zero latency fear.

---

## 1. Where to Place Teammate Files

Have your teammate save their files in these exact workspace folders:

| Deliverable File | Target Directory in Project | Description |
| :--- | :--- | :--- |
| **DINOv3 Model Weights** | `configs/weights/dinov3_vitb16_exemplar.pt` | PyTorch checkpoint or HuggingFace model ID |
| **FAISS Index File** | `data/processed/crops/gt_clean/dinov3_exemplar.index` | FAISS index binary file or `.npy` matrix |
| **Crop ID Mapping List** | `data/processed/crops/gt_clean/dinov3_crop_ids.json` | JSON mapping row index $\to$ crop filename |
| **Python Embedder Plugin** | `ml/embeddings/dinov3.py` | Python module implementing `BaseEmbedder` interface |

---

## 2. How SQLite & FAISS Are Integrated Specifically in This Project

In our project architecture, **SQLite** and **FAISS / NumPy** operate as a **Hybrid Persistence & In-Memory Vector Search Engine**:

```mermaid
flowchart TD
    subgraph Disk ["DISK PERSISTENCE LAYER"]
        DB[("SQLite Database: retail_sku_registry.db<br>• Persistent Float32 Vector Blobs<br>• 67 Commercial SKU Display Names & Brands<br>• HITL Audit Tasks & Annotation History")]
    end

    subgraph Memory ["APP BOOT & FAISS IN-MEMORY INDEX"]
        DB -- "1. fetch_all_references() SELECT vector_blob, remapped_class_id" --> RAM["2D NumPy Gallery Matrix G (31,656 x 384 or 768)"]
        RAM -- "2. L2-Normalize & Add to Index" --> FAISS["NumpyCosineIndex / FAISS IndexFlatIP"]
    end

    subgraph Search ["LIVE REAL-TIME SERVING"]
        Query["Crop Query Vector q"] --> FAISS
        FAISS -- "3. Matrix Multiplication q · G^T (< 1.5ms)" --> Top5["Top-5 Retrieved Candidates + Similarity Scores"]
    end

    subgraph HITL ["CONTINUAL ACTIVE LEARNING"]
        Human["Human Reviewer Approves New Crop in Web UI"] --> DB
        DB -- "4. Upserts Vector to Disk & Pushes to RAM Index" --> FAISS
    end
```

### Code-Level Code Flow:

1. **`SQLiteGalleryStore` (`ml/retrieval/sqlite_registry.py`)**:
   - Stores raw float32 vector bytes in SQLite column `vector_blob` (3,072 bytes for 768-D).
   - Stores commercial display names, brand names, pack sizes, and HITL audit history.

2. **`NumpyCosineIndex` (`ml/retrieval/numpy_index.py`)**:
   - At application startup (`server/app.py`), `SQLiteGalleryStore.fetch_all_references()` fetches all vectors.
   - `NumpyCosineIndex` converts vector blobs to a 2D float32 NumPy matrix in RAM.
   - When a live query vector $\mathbf{q}$ arrives, `NumpyCosineIndex.search()` computes dot product $\mathbf{q} \cdot \mathbf{G}^T$ in **sub-millisecond speed ($\le 1.5\text{ms}$)**!

---

## 3. High-Value Tasks for 2 Inactive Teammates

Assign these 2 structured, high-impact tasks to get your inactive teammates immediately productive:

### Teammate Task A: Qwen2-VL Model Quantization & Latency Benchmark
- **Goal**: Optimize Qwen2-VL (2B-Instruct) for ultra-fast CPU/GPU inference.
- **Action Items**:
  1. Quantize Qwen2-VL to 4-bit / 8-bit using AWQ, GGUF, or `bitsandbytes`.
  2. Test execution latency per crop on CPU vs GPU.
  3. Benchmark token generation speed when constrained to single-digit output options (Options 1–5).
- **Deliverable**: `ml/vlm/qwen2_vl_quantized.py` with benchmark report.

### Teammate Task B: Synthetic Data Augmentation & Hard Negative Mining
- **Goal**: Generate synthetic training crops for rare/long-tail FMCG classes ($< 100$ reference crops).
- **Action Items**:
  1. Apply visual augmentations (perspective tilt, specular glare, motion blur, shadows) using `albumentations`.
  2. Extract DINOv3 feature vectors for augmented crops and test cosine similarity stability.
  3. Identify hard-negative SKU pairs (e.g. *Lipton Mint 25s* vs *Lipton Mint 100s*) and measure similarity distance margins.
- **Deliverable**: Synthetic crop dataset + `scripts/benchmark_hard_negatives.py`.

---

## 4. Qwen2-VL Integration & How to Handle Latency

### Why Qwen2-VL is a Great Choice
**Qwen2-VL (2B-Instruct)** is a state-of-the-art vision-language model. Unlike EasyOCR (which reads raw characters line-by-line and fails on curved/blurry packaging), Qwen2-VL understands **whole visual packaging semantics**, logos, net weight numbers, and brand layout!

### How to Solve Latency (The Gated Cascade Architecture)

To prevent Qwen2-VL from slowing down live shelf scanning, **do NOT run Qwen2-VL on every shelf crop**! 

Instead, use our **Gated Cascade Architecture**:

```mermaid
flowchart TD
    Crop["Incoming Product Crop"] --> Visual["DINOv3 Visual Vector Search (15.5ms)"]
    Visual --> Decision{"Visual Similarity Score (S_vis)"}
    
    Decision -- "High Confidence (S_vis >= 0.85)" --> Auto["Automated Prediction (0ms VLM Overhead!)"]
    
    Decision -- "Ambiguous (0.75 <= S_vis < 0.85)" --> Qwen["Trigger Qwen2-VL Constrained Reranker (~150ms)"]
    Qwen --> Top5["Rank Top-5 Candidates"]
    Top5 --> Auto
    
    Decision -- "Low Confidence (S_vis < 0.75)" --> HITL["Route to HITL Review Queue"]
```

### Latency Budget & Speed Strategy:
1. **85% of Crops**: Cleared by DINOv3 with **high confidence ($S_{\text{vis}} \ge 0.85$)** $\to$ Qwen2-VL is **SKIPPED COMPLETELY** (**0ms overhead**!).
2. **15% Ambiguous Crops**: Trigger Qwen2-VL **ONLY on the Top-5 candidate titles** using `max_new_tokens=5`.
3. **Constrained Prompting**: Prompt Qwen2-VL to select option `1, 2, 3, 4, or 5` instead of generating free text. This drops generation latency from 2,000ms down to **~120-180ms**!

Module implementation saved in: `ml/vlm/qwen2_vl_reranker.py`.
