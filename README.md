# Enterprise Retail AI: Open-Set Supermarket Shelf Audit & SKU Recognition Platform

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![SOTA Backbone: DINOv3](https://img.shields.io/badge/Backbone-DINOv3--ViT--B16-brightgreen.svg)](https://github.com/facebookresearch/dinov3)
[![Pipeline 1: Audit Engine](https://img.shields.io/badge/Pipeline--1-Shelf_Audit_SOTA-success.svg)]()
[![Pipeline 2: Onboarding](https://img.shields.io/badge/Pipeline--2-Dynamic_Onboarding-blue.svg)]()
[![Pipeline 3: Active Learning](https://img.shields.io/badge/Pipeline--3-Active_Continual_Learning-orange.svg)]()

An enterprise-grade, open-set computer vision platform for automatic supermarket shelf product localization, fine-grained visual SKU recognition, zero-shot VLM/OCR variant verification, zero-downtime new SKU onboarding, and continuous active learning.

---

## 💡 System Vision & Open-Set Architecture

Traditional closed-set image classification models require expensive re-training and service downtime whenever a retailer adds a new product SKU. **This platform adopts an open-set, zero-retraining production architecture across 3 decoupled pipelines:**

```
                  ┌─────────────────────────────────────────────────────────┐
                  │                 RAW SUPERMARKET SCAN                    │
                  └────────────────────────────┬────────────────────────────┘
                                               │
                                               ▼
                  ┌─────────────────────────────────────────────────────────┐
                  │   STAGE 1: YOLOv8 Class-Agnostic Product Localizer      │
                  │   (Extracts all product packaging facings, imgsz=640)   │
                  └────────────────────────────┬────────────────────────────┘
                                               │ Bounding Box Crops
                                               ▼
                  ┌─────────────────────────────────────────────────────────┐
                  │   STAGE 2: DINOv3 ViT-B/16 SOTA Visual Feature Search   │
                  │   (768-D dense L2-normalized vector, <3ms query time)   │
                  └────────────────────────────┬────────────────────────────┘
                                               │ Cosine Search (NumpyCosineIndex)
                                               ▼
                  ┌─────────────────────────────────────────────────────────┐
                  │   STAGE 3: Qwen2-VL Zero-Shot Variant Reranking          │
                  │   (Score fusion: 80% Visual + 20% Text Verification)    │
                  └────────────────────────────┬────────────────────────────┘
                                               │ Fused Similarity Score S
                                               ▼
                  ┌─────────────────────────────────────────────────────────┐
                  │   STAGE 4: 4-Region Dynamic Gating Policy               │
                  └───────┬─────────────────┬─────────────────┬─────────────┘
                          │                 │                 │
     S >= 0.84 & ΔS >= 0.10│ 0.78 <= S < 0.84  │ 0.62 <= S < 0.78│ S < 0.62
       (Region A: Fast)   │ (Region B: Auto)│ (Region C: HITL)│ (Region D: Noise)
                          │                 │                 │
                          ▼                 ▼                 ▼
                  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
                  │ Auto-Approve │  │ Auto-Approve │  │ HITL Review  │
                  │  (Instant)   │  │ (VLM Tagged) │  │ Audit Queue  │
                  └──────────────┘  └──────────────┘  └──────┬───────┘
                                                             │ Auditor Feedback
                                                             ▼
                                                    ┌─────────────────┐
                                                    │ Active Learning │
                                                    │ Vector Upsert   │
                                                    └─────────────────┘
```

---

## 🏆 SOTA 5-Model Vision Embedding Benchmark

We evaluated **5 vision embedding architectures** on our 67-class commercial FMCG retail dataset:

| Rank | Model Architecture | Model Type | Vector Dim | Top-1 Acc | Top-3 Acc | Top-5 Acc ⭐ | MRR | Latency | Decision |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **#1** | 🏆 **DINOv3 (ViT-B/16)** | Self-Supervised Vision | **768** | **93.65%** | **97.75%** | **99.20%** | **0.9574** | **2.93 ms** | **SELECTED — SOTA Production Backbone** |
| **#2** | ⚡ **DINOv2 (ViT-S/14)** | Self-Supervised Vision | **384** | **92.00%** | **97.25%** | **98.60%** | **0.9468** | **0.95 ms** | **Lightweight Fallback for Edge Devices** |
| **#3** | 🥈 RADIO CLIP (RADIOv2.5-L) | Multimodal Vision-Lang | 3072 | 83.44% | 91.57% | 94.41% | 0.8767 | 52.14 ms | *Unused — Heavy memory overhead* |
| **#4** | 🥉 SigLIP (SO400M) | Multimodal Vision-Lang | 1152 | 77.18% | 87.82% | 91.55% | 0.8272 | 29.75 ms | *Unused — High embedding latency* |
| **#5** | ❌ CLIP (ViT-B/32) | Generic Zero-Shot | 512 | 74.16% | 83.10% | 90.53% | 0.8057 | 3.20 ms | *Unused — Drops accuracy on FMCG packaging* |

---

## ⚙️ The 3 Platform Pipelines

### 🔍 **Pipeline 1: Supermarket Shelf Audit & Visual Search**
- **YOLOv8 Localizer**: Bounding-box detection fine-tuned on SKU110K.
- **DINOv3 Vector Search**: Queries 31,656 L2-normalized 768-D reference vectors in SQLite matrix memory via `NumpyCosineIndex`.
- **4-Region Decision Policy**:
  - **Region A ($S \ge 0.84, \Delta S \ge 0.10$)**: High confidence, dominant margin gap $\implies$ Instant Auto-Approval ($<3\text{ ms}$).
  - **Region B ($0.78 \le S < 0.84$)**: Mid-high confidence $\implies$ Auto-Approved with inline `Qwen2-VL Verified` badge.
  - **Region C ($0.62 \le S < 0.78$)**: Low-mid confidence $\implies$ Pre-ranked by Qwen2-VL and routed to human auditor HITL queue.
  - **Region D ($S < 0.62$)**: Non-catalog noise $\implies$ Strictly assigned `Class Unknown (-1)` and omitted from HITL review queue.

### 🆕 **Pipeline 2: Zero-Downtime New SKU Onboarding**
- **Dynamic Catalog Expansion**: Field auditors upload reference photos of new SKUs via `/v1/onboard/sku`.
- **Automatic Crop & Augmentation**: Isolates product bounding boxes using YOLOv8 and generates lighting/tilt variants.
- **Real-Time Memory Upsert**: Vector embeddings are instantly added to `retail_sku_registry_dinov3.db` without restarting the server or retraining any weights.

### 🔁 **Pipeline 3: Active Continual Learning & HITL Feedback**
- **Auditor Curation**: Human corrections made in the HITL Review Queue are logged in `data/processed/hitl_active_learning.db`.
- **Continual Vector Injection**: Verified crop embeddings are automatically upserted into the SQLite vector database, expanding gallery representation for real-world store conditions.

---

## 💻 Web Dashboard & Senior UI/UX Features

The web frontend (`server/static/index.html`) is built with modern vanilla CSS glassmorphism, Google Fonts (`Inter`, `Outfit`), and FontAwesome icons:

- **Shelf Audit Inspection Tab**: Interactive shelf canvas with real-time color-coded bounding boxes (Emerald = Automated Match, Crimson = HITL Queue Match), expandable Crop SKU Inspector drawer, and uncompressed similarity metrics.
- **HITL Review Queue Tab**: 6-column aligned review queue (`Crop Preview`, `Crop ID & Details`, `Top Candidate Prediction`, `Visual Similarity`, `Reject Reason`, `Actions`), featuring a 67-class dropdown and `Save & Upsert` action.
- **Commercial Catalog Explorer**: Interactive 67-SKU catalog grid with real-time client-side search filtering.
- **Latency & Architecture Tab**: Real-time breakdown of per-facing stage latency ($130\text{ ms}$ CPU total).

---

## 🚀 Quickstart & Server Run

### 1. Requirements & Setup
```bash
git clone https://github.com/MarwanZaineldeen/shelf-sku-recognition-2.git
cd shelf-sku-recognition-2
python -m pip install -r requirements.txt
```

### 2. Launch Local Server
```bash
python -m uvicorn server.app:app --host 127.0.0.1 --port 8000
```
Open your browser to: **`http://127.0.0.1:8000/`**

---

## 🧪 Automated Testing

Run system unit and integration test suites:
```bash
python -m unittest discover -s tests
```

---

## 📄 License
Distributed under the **MIT License**. See `LICENSE` for details.
