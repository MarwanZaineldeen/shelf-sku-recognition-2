# Enterprise Retail AI: Open-Set Shelf Product Recognition Platform

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![SOTA Backbone: DINOv3](https://img.shields.io/badge/Backbone-DINOv3--ViT--B16-brightgreen.svg)](https://github.com/facebookresearch/dinov3)
[![Pipeline 1 & 2: Complete](https://img.shields.io/badge/Pipeline-1_%26_2_Complete-success.svg)]()
[![Pipeline 3: Active Learning](https://img.shields.io/badge/Pipeline-3_Active_Learning-orange.svg)]()

An enterprise-grade, modular computer vision platform for automatic supermarket shelf product localization, fine-grained visual SKU recognition, zero-shot VLM/OCR reranking, and dynamic product onboarding.

---

## 💡 System Vision & Open-Set Production Architecture

In traditional computer vision, training a closed-set detector requires re-training and re-deploying models every time a retailer adds a new product SKU. **This is not scalable for production.**

This platform adopts a **Zero-Retraining Open-Set Architecture**:

1. **Class-Agnostic Shelf Product Detection (Stage 1)**:
   Localizes all product packaging facings on a store shelf using class-agnostic object detection (fine-tuned on **SKU110K**). Fully decoupled from product identity.
2. **SOTA Visual Feature Search (Stage 2)**:
   Extracts dense $L_2$-normalized visual embeddings using **DINOv3 ViT-B/16 Exemplar** ($D=768$, achieving **93.65% Top-1 / 99.20% Top-5 Gallery Recall**) and queries a reference SQLite Vector Registry with 31,656 vectors using Cosine Nearest-Neighbor Search. *(Includes **DINOv2-small** ($D=384$) as a lightweight fallback for edge devices)*.
3. **Class-Unique Candidate Deduplication (Stage 2.5)**:
   Queries Top-50 nearest neighbors and returns **5 Class-Unique Diverse Candidates** (one per product class), expanding the VLM disambiguation space across different product categories.
4. **Qwen2-VL Zero-Shot Late Fusion Reranking (Stage 3)**:
   Selectively executes **Qwen2-VL Zero-Shot Text Matcher** on crops in the uncertainty zone ($0.75 \le S_{\text{visual}} < 0.92$) to verify brand, flavor, variant, and pack-size from packaging. Adds $+0.12$ fused similarity boost to the VLM-verified candidate.
5. **Platt Calibration & Gated Decision Engine (Stage 4)**:
   Maps raw similarity scores into calibrated probabilities ($P \in [0, 1]$). Predictions meeting target precision constraints ($P \ge 80\%$) are auto-approved; uncertain items are safely routed to the **Human-in-the-Loop (HITL) Queue**.
6. **Zero-Downtime Dynamic SKU Onboarding**:
   Adding a new product SKU requires **zero model re-training**. Simply upload reference crop photos via `/v1/onboard/sku` — the system dynamically updates active search indexes in real time.

---

## 🏆 SOTA 5-Model Vision Embedding Benchmark

We evaluated **5 vision embedding architectures** on our 67-class commercial FMCG retail dataset:

| Rank | Model Architecture | Model Type | Vector Dim | Top-1 Acc | Top-3 Acc | Top-5 Acc ⭐ | MRR | Latency | Decision |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **#1** | 🏆 **DINOv3 (ViT-B/16)** | Self-Supervised Vision | **768** | **93.65%** | **97.75%** | **99.20%** | **0.9574** | **2.93 ms** | **SELECTED — SOTA Production Backbone** |
| **#2** | ⚡ **DINOv2 (ViT-S/14)** | Self-Supervised Vision | **384** | **92.00%** | **97.25%** | **98.60%** | **0.9468** | **0.95 ms** | **Lightweight Fallback for Edge Devices** |
| **#3** | 🥈 RADIO CLIP (RADIOv2.5-L) | Multimodal Vision-Lang | 3072 | 83.44% | 91.57% | 94.41% | 0.8767 | 52.14 ms | *Unused — 4.7× slower, heavy RAM* |
| **#4** | 🥉 SigLIP (SO400M) | Multimodal Vision-Lang | 1152 | 77.18% | 87.82% | 91.55% | 0.8272 | 29.75 ms | *Unused — High embedding latency* |
| **#5** | ❌ CLIP (ViT-B/32) | Generic Zero-Shot | 512 | 74.16% | 83.10% | 90.53% | 0.8057 | 3.20 ms | *Unused — Drops accuracy on FMCG packaging* |

**Why DINOv3 wins**: Achieves the highest Top-1 accuracy (93.65%) and near-perfect Top-5 gallery recall (99.20%), ensuring downstream VLM/calibrators receive the correct candidate in 99/100 queries.

---

## 🏗️ Production Architecture

```
                  Raw Shelf Image (JPEG/PNG)
                              │
                              ▼
                 ┌─────────────────────────┐
                 │  Stage 1: Class-Agnostic│
                 │  Product Localizer      │ (YOLOv8l fine-tuned on SKU110K)
                 └────────────┬────────────┘
                              │ Bounding Boxes (BBoxDTO)
                              ▼
                 ┌─────────────────────────┐
                 │  Stage 2: DINOv3 ViT-B  │
                 │  Feature Extractor      │ (Normalized Vector D=768)
                 └────────────┬────────────┘
                              │ EmbeddingDTO
                              ▼
                 ┌─────────────────────────────────┐
                 │  SQLite Vector Registry         │
                 │  31,656 reference embeddings    │ (Top-50 → 5 Class-Unique Candidates)
                 └────────────┬────────────────────┘
                              │ Top-5 Class-Unique Candidates (S_visual)
                              ▼
                 ┌─────────────────────────┐
                 │  Selective Gating       │
                 └──────┬───────────┬──────┘
                        │           │
   S_visual >= 0.92     │           │  0.75 <= S_visual < 0.92
   (Fast Path <3ms)     │           │  (Ambiguous Zone — VLM Activated)
                        │           ▼
                        │     ┌─────────────────────────┐
                        │     │ Stage 3: Qwen2-VL       │ (Zero-shot text/variant verification)
                        │     │ Zero-Shot VLM Reranker  │ (+0.12 boost on verified candidate)
                        │     └────────────┬────────────┘
                        │                  │ Verified Candidates (S_fused)
                        └──────────┬───────┘
                                   │ Fused Similarity Scores
                                   ▼
                      ┌─────────────────────────┐
                      │ Stage 4: Platt          │
                      │ Calibration Engine      │ (Calibrated Probability P)
                      └────────────┬────────────┘
                                   │
                                   ▼
                      ┌─────────────────────────┐
                      │ Gated Decision Policy   │
                      └──────┬────────────┬─────┘
                             │            │
         P >= 0.80           │            │  P < 0.80 / Non-Catalog
     (Auto-Approved SKU)     │            │  (Routed to HITL Queue)
                             ▼            ▼
                     Auto Annotation    HITL Queue
                                            │
                                            ▼
                             ┌─────────────────────────┐
                             │  Pipeline 3: Active     │ ← Teammate Integration
                             │  Continual Learning     │   sku_crops + hitl_records
                             └─────────────────────────┘
```

---

## 🚀 Quickstart & Server Launch

### 1. Launch FastAPI Interactive Web Dashboard & REST API
```bash
python -m uvicorn server.app:app --host 127.0.0.1 --port 8000 --reload
```
Open `http://127.0.0.1:8000` in your browser — the HITL Audit Dashboard loads automatically with a sample shelf scan.

### 2. Run Live Benchmark Evaluation
```bash
python scratch/compare_dinov2_vs_dinov3_live.py
```

### 3. Run End-to-End Pipeline Test
```bash
python scratch/test_e2e_process.py
```

---

## 📁 Repository Structure

```
.
├── configs/
│   ├── weights/dinov3_vitb16/          # DINOv3 fine-tuned exemplar checkpoint (768-D)
│   │   ├── config.json                 #   Architecture config
│   │   ├── model.safetensors           #   Fine-tuned weights (~327 MB, gitignored)
│   │   └── preprocessor_config.json   #   Processor config
│   ├── sku_mapping_v2.json             # Authoritative 67-class SKU catalog (training_class_id keys)
│   └── retrieval_config.json           # Search thresholds & pipeline parameters
│
├── data/
│   ├── processed/crops/gt_clean/
│   │   └── retail_sku_registry_dinov3.db  # 31,656 DINOv3 vectors (~125 MB, gitignored)
│   ├── processed/Sku Preview/          # Manually curated high-res reference images per class
│   └── processed/yolo_remapped_clean/images/test/  # 127 held-out test shelf images (gitignored)
│
├── runs/detect/yolo8l_sku110k/
│   └── yolov8l-sku110k.pt              # YOLOv8l fine-tuned on SKU110K (~84 MB, gitignored)
│
├── ml/                                 # Core ML pipeline modules
│   ├── detection/                      # YOLOv8 class-agnostic localizer
│   ├── embeddings/                     # DINOv3 & DINOv2 feature extractors
│   ├── retrieval/                      # SQLite vector store & cosine search
│   ├── vlm/                            # Qwen2-VL zero-shot reranker
│   ├── orchestrator.py                 # End-to-end pipeline (detect → embed → rank → decide)
│   ├── ocr/                            # EasyOCR & character n-gram fusion
│   └── calibrators/                    # Platt sigmoidal probability calibrators
│
├── server/
│   ├── app.py                          # FastAPI REST API & HITL review endpoints
│   ├── schemas.py                      # Pydantic response models
│   └── static/                         # Audit Dashboard (HTML/CSS/JS)
│
├── docs/
│   ├── pipeline_3_active_continual_learning_integration_spec.md  # ← Teammate: read this first
│   └── reports/
│
└── scratch/                            # Dev & validation scripts
    ├── test_e2e_process.py             # Full pipeline smoke test
    ├── verify_vlm_offline.py           # VLM weight & reranking verification
    └── compare_dinov2_vs_dinov3_live.py
```

---

## 🔗 Pipeline 3: Active Continual Learning (Teammate Integration)

> **Status**: Foundation implemented — teammate integration in progress.

The HITL review system is fully wired to the Active Learning feedback loop:

| Step | Action |
| :--- | :--- |
| **1. Correction Submitted** | Merchandiser corrects prediction via `POST /v1/hitl/review` |
| **2. Embedding Persisted** | 768-D DINOv3 embedding + `true_class_id` stored in `sku_crops` table |
| **3. Hard Negative Mining** | Pipeline 3 reads `hitl_records` to find ($C_{\text{true}}$, $C_{\text{predicted}}$) pairs |
| **4. Diversity Sampling** | $k$-center greedy selection with $N \le 500$ crop budget cap |
| **5. SupCon Fine-Tuning** | Supervised Contrastive head trained on curated hard pairs |

### 🗂️ Key Files for Teammate:

| Path | Purpose |
| :--- | :--- |
| `data/processed/crops/gt_clean/retail_sku_registry_dinov3.db` | Source gallery (31,656 vectors) + HITL correction records |
| `configs/weights/dinov3_vitb16/` | Fine-tuned DINOv3 exemplar checkpoint used to build the DB |
| `configs/sku_mapping_v2.json` | Authoritative SKU catalog keyed by `training_class_id` |
| `runs/detect/yolo8l_sku110k/yolov8l-sku110k.pt` | Shelf product detector weights |
| `data/processed/yolo_remapped_clean/images/test/` | 127 held-out test shelf images |
| `docs/pipeline_3_active_continual_learning_integration_spec.md` | Full integration spec & DB schema |

> ⚠️ All large binary assets (`*.db`, `*.pt`, `model.safetensors`, test images) are **gitignored**. Transfer them separately via the direct file paths above.

### 🗄️ Database Schema for Pipeline 3:

```sql
-- Reference gallery (31,656 crop vectors)
SELECT id, crop_path, remapped_class_id, embedding_blob, gallery_version
FROM sku_crops;

-- HITL correction records (hard negatives for SupCon fine-tuning)
SELECT review_id, crop_path, true_class_id, top1_predicted_class_id,
       reviewer_id, is_correction, reviewed_at
FROM hitl_records;
```

Full technical integration spec → [`docs/pipeline_3_active_continual_learning_integration_spec.md`](docs/pipeline_3_active_continual_learning_integration_spec.md)
