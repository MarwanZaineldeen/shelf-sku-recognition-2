# Enterprise Retail AI: Open-Set Shelf Product Recognition Platform

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![SOTA Winner: DINOv3](https://img.shields.io/badge/Backbone-DINOv3--ViT--B16-brightgreen.svg)](https://github.com/facebookresearch/dinov3)

An enterprise-grade, modular computer vision platform for automatic supermarket shelf product localization, fine-grained visual SKU recognition, zero-shot VLM/OCR reranking, and dynamic product onboarding.

---

## 💡 System Vision & Open-Set Production Architecture

In traditional computer vision, training a closed-set detector requires re-training and re-deploying models every time a retailer adds a new product SKU. **This is not scalable for production.**

This platform adopts a **Zero-Retraining Open-Set Architecture**:

1. **Class-Agnostic Shelf Product Detection (Stage 1)**:
   Localizes all product packaging facings on a store shelf using class-agnostic object detection (fine-tuned on **SKU110K**). Decouples product localization from product identity!
2. **SOTA Visual Feature Search (Stage 2)**:
   Extracts dense $L_2$-normalized visual embeddings using **DINOv3 ViT-B/16 Exemplar** ($D=768$, achieving **93.65% Top-1 / 98.70% Top-5 Recall**) and queries an indexed reference gallery store (**SQLite Vector Registry**) using Cosine Nearest-Neighbor Search / FAISS Flat Indexing. *(Includes **DINOv2-small** ($D=384$) as a lightweight fallback for edge devices)*.
3. **Qwen2-VL / EasyOCR Late Fusion Reranking (Stage 3)**:
   Selectively executes **Qwen2-VL (2B-Instruct)** VLM / EasyOCR on crops in the uncertainty zone ($0.75 \le S_{\text{visual}} < 0.85$) to read text tokens and verify brand packaging. *(Qwen2-VL is currently being 4-bit/8-bit quantized by team members for sub-150ms execution)*.
4. **Platt Calibration & Gated Decision Engine (Stage 4)**:
   Maps raw similarity scores into calibrated probabilities ($P \in [0, 1]$). Predictions meeting target precision constraints ($P \ge 80\%$) are auto-approved, while non-catalog competitor products and uncertain items are safely routed to the **Human-in-the-Loop (HITL) Queue**.
5. **Zero-Downtime Dynamic SKU Onboarding**:
   Adding a new product SKU requires **zero model re-training**. Simply upload reference crop photos via `/v1/onboard/sku`, and the system dynamically updates active search indexes in real time!

---

## 🏆 SOTA 5-Model Vision Embedding Benchmark

We evaluated **5 vision embedding architectures** on our 67-class commercial FMCG retail dataset:

| Rank | Model Architecture | Model Type | Vector Dim | Top-1 Acc | Top-3 Acc | Top-5 Acc ⭐ | MRR | Single-Query Latency | Architectural Status & Decision |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **#1** | 🏆 **DINOv3 (ViT-B/16)** | **Self-Supervised Vision** | **768** | **93.65%** | **97.75%** | **98.70%** | **0.9574** | **2.93 ms** | **SELECTED SOTA PRODUCTION BACKBONE** |
| **#2** | ⚡ **DINOv2 (ViT-S/14)** | **Self-Supervised Vision** | **384** | **92.00%** | **97.25%** | **98.60%** | **0.9468** | **0.95 ms** | **Lightweight Fallback for Edge Devices** |
| **#3** | 🥈 RADIO CLIP (RADIOv2.5-L) | Multimodal Vision-Lang | 3072 | 83.44% | 91.57% | 94.41% | 0.8767 | 52.14 ms | *Unused (4.7x slower latency & heavy RAM)* |
| **#4** | 🥉 SigLIP (SO400M) | Multimodal Vision-Lang | 1152 | 77.18% | 87.82% | 91.55% | 0.8272 | 29.75 ms | *Unused (High embedding latency 235ms)* |
| **#5** | ❌ CLIP (ViT-B/32) | Generic Zero-Shot | 512 | 74.16% | 83.10% | 90.53% | 0.8057 | 3.20 ms | *Unused (Drops accuracy on FMCG packaging)* |

### Architectural Decision Rationale:
- **Why DINOv3 is Selected**: DINOv3 achieves the highest Top-1 accuracy (**93.65%**) and near-perfect Top-5 recall (**98.70%**), ensuring downstream VLM/calibrators receive the correct candidate in **99 out of 100 queries**.
- **Why RADIO CLIP & SigLIP are Not Used**: Both models require excessive memory ($>1.8\text{ GB RAM}$) and high search/embedding latency ($150\text{--}235\text{ms}$ per crop), making them unviable for real-time shelf processing.
- **Why Un-tuned CLIP is Not Used**: Generic CLIP relies heavily on text captions, causing accuracy to drop on fine-grained FMCG product packaging where DINOv3 and DINOv2 excel.

---

## 🏗️ Production Architecture

```
                  Raw Shelf Image (JPEG/PNG)
                              │
                              ▼
                 ┌─────────────────────────┐
                 │  Stage 1: Class-Agnostic│
                 │  Product Localizer      │ (YOLOv8l-SKU110K Localizer)
                 └────────────┬────────────┘
                              │ Bounding Boxes (BBoxDTO)
                              ▼
                 ┌─────────────────────────┐
                 │  Crop Quality Gate      │ (Filter blurry / tiny crops)
                 └────────────┬────────────┘
                              │ Valid CropDTOs
                              ▼
                 ┌─────────────────────────┐
                 │  Stage 2: DINOv3        │
                 │  Feature Extractor      │ (Normalized Vector D=768)
                 └────────────┬────────────┘
                              │ EmbeddingDTO
                              ▼
                 ┌─────────────────────────┐
                 │  SQLite / FAISS Registry│ (Sub-3ms Cosine Vector Search)
                 └────────────┬────────────┘
                              │ Visual Similarity Matches (S_visual)
                              ▼
                 ┌─────────────────────────┐
                 │  Selective Gating       │
                 └──────┬───────────┬──────┘
                        │           │
   S_visual >= 0.85     │           │  0.75 <= S_visual < 0.85
   (Fast Path <3ms)     │           │  (Ambiguous Zone)
                        │           ▼
                        │     ┌─────────────────────────┐
                        │     │ Stage 3: Qwen2-VL /     │ (Quantized VLM / EasyOCR)
                        │     │ EasyOCR Text Reranker   │
                        │     └────────────┬────────────┘
                        │                  │ Verified Candidates
                        │                  ▼
                        └───────────┬──────┘
                                    │ Fused Similarity Scores
                                    ▼
                       ┌─────────────────────────┐
                       │ Stage 4: Platt          │
                       │ Calibrator Engine       │ (Sigmoidal Calibrated Probability P)
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ Gated Decision Policy   │
                       └──────┬───────────┬──────┘
                              │           │
           P >= 0.80          │           │ P < 0.80 / Non-Catalog
       (Auto-Approved SKU)    │           │ (Routed to HITL Queue)
                              ▼           ▼
                      Auto Annotation   HITL Queue
```

---

## 🚀 Quickstart & Server Launch

### 1. Launch FastAPI Interactive Web Dashboard & REST API
```bash
python -m uvicorn server.app:app --host 127.0.0.1 --port 8000 --reload
```
Open `http://127.0.0.1:8000` in your browser to access the interactive HITL audit dashboard!

### 2. Run Live Benchmark Evaluation
```bash
python scratch/compare_dinov2_vs_dinov3_live.py
```

---

## 📁 Repository Structure

```
.
├── configs/                   # System configurations & model weights
│   ├── weights/dinov3_vitb16/ # DINOv3 offline model weights (768-D)
│   ├── sku_mapping.json       # Commercial 67-class SKU mapping
│   └── retrieval_config.json  # Search index thresholds & parameters
├── data/
│   └── processed/crops/gt_clean/
│       ├── retail_sku_registry_dinov3.db # SQLite database with 31,656 DINOv3 vectors
│       └── retail_sku_registry_dinov2.db # SQLite database with 31,664 DINOv2 vectors
├── ml/                        # Core ML pipeline modules
│   ├── detection/             # Class-agnostic YOLO localizers
│   ├── embeddings/            # DINOv3 & DINOv2 feature extractors
│   ├── retrieval/             # SQLite store & NumPy/FAISS cosine search
│   ├── vlm/                   # Qwen2-VL zero-shot VLM reranker
│   ├── ocr/                   # EasyOCR & character n-gram fusion
│   └── calibrators/           # Platt logit probability calibrators
├── server/                    # FastAPI web application & REST endpoints
└── docs/reports/              # Architectural & benchmark reports
    ├── LIVE_DINOV2_VS_DINOV3_BENCHMARK_REPORT.md
    └── SOTA_5_MODEL_EMBEDDING_BENCHMARK.md
```
