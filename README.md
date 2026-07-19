# Enterprise Retail AI: Open-Set Shelf Product Recognition Platform

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade, modular computer vision platform for automatic supermarket shelf product localization, fine-grained visual SKU recognition, OCR text late fusion, and dynamic product onboarding.

---

## 💡 System Vision & Open-Set Production Architecture

In traditional computer vision, training a closed-set detector requires re-training and re-deploying models every time a retailer adds a new product SKU. **This is not scalable for production.**

This platform adopts a **Zero-Retraining Open-Set Architecture**:

1. **Class-Agnostic Shelf Product Detection (Stage 1)**:
   Localizes all product packaging facings on a store shelf using class-agnostic object detection (fine-tuned on **SKU110K**). Decouples product localization from product identity!
2. **Visual Feature Search (Stage 2)**:
   Extracts dense $L_2$-normalized visual embeddings using **DINOv2** ($D=384$) and queries an indexed reference gallery store (**SQLite Vector Registry**) using Cosine Nearest-Neighbor Search / 2-Layer Brand Centroid Search.
3. **Gated EasyOCR Late Fusion (Stage 3)**:
   Selectively executes EasyOCR on crops in the uncertainty zone ($0.85 \le S_{\text{visual}} \le 0.96$) to read text tokens and apply lexicon-based similarity boosts.
4. **Platt Calibration & Gated Decision Engine (Stage 4)**:
   Maps raw similarity scores into calibrated probabilities ($P \in [0, 1]$). Predictions meeting target precision constraints ($P \ge 0.95$) are auto-approved, while uncertain items are safely routed to the **Human-in-the-Loop (HITL) Queue**.
5. **Zero-Downtime Dynamic SKU Onboarding**:
   Adding a new product SKU requires **zero model re-training**. Simply upload reference crop photos via `/v1/onboard/sku`, and the system dynamically updates active search indexes in real time!

---

## 🏗️ System Architecture

```
                  Raw Shelf Image (JPEG/PNG)
                              │
                              ▼
                 ┌─────────────────────────┐
                 │  Stage 1: Class-Agnostic│
                 │  Product Detector       │ (SKU110K Bounding Box Localizer)
                 └────────────┬────────────┘
                              │ Bounding Boxes (BBoxDTO)
                              ▼
                 ┌─────────────────────────┐
                 │  Crop Quality Check Gate│ (Filter blurry / tiny crops)
                 └────────────┬────────────┘
                              │ Valid CropDTOs
                              ▼
                 ┌─────────────────────────┐
                 │  Stage 2: DINOv2        │
                 │  Feature Extractor      │ (Normalized Vector D=384)
                 └────────────┬────────────┘
                              │ EmbeddingDTO
                              ▼
                 ┌─────────────────────────┐
                 │  Vector Search Engine   │ (SQLite Gallery / 2-Layer Hierarchical Index)
                 └────────────┬────────────┘
                              │ Visual Similarity Matches (S_visual)
                              ▼
                 ┌─────────────────────────┐
                 │  Selective CPU Gating   │
                 └──────┬───────────┬──────┘
                        │           │
   S_visual > 0.96      │           │  0.85 <= S_visual <= 0.96
   (Fast Path <15ms)    │           │  (Uncertainty Zone)
                        │           ▼
                        │     ┌─────────────────────────┐
                        │     │ Stage 3: EasyOCR Engine │ (Timeout Budget: 300ms)
                        │     └────────────┬────────────┘
                        │                  │ Extracted Text (OCRResultDTO)
                        │                  ▼
                        │     ┌─────────────────────────┐
                        │     │ Lexicon Late Fusion     │ (Class Keyword Boost)
                        │     └────────────┬────────────┘
                        │                  │
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
           P >= 0.95          │           │ P < 0.95 / Rejected
       (Auto-Approved SKU)    │           │ (Routed to HITL Queue)
                              ▼           ▼
                      Auto Annotation   HITL Queue
```

---

## 📁 Repository Structure

```
.
├── configs/                   # System configurations (YAML settings, lexicons, mappings)
│   ├── retrieval_config.yaml  # Model hyperparameters & evaluation thresholds
│   ├── class_lexicons.json    # Target class OCR keyword dictionaries
│   └── class_id_mapping.json  # Continuous class ID mapping tables
├── ml/                        # Core Clean Architecture Machine Learning Domain
│   ├── base.py                # Abstract plugin interfaces (IPlugin) & typed Pydantic DTOs
│   ├── orchestrator.py        # Master shelf audit pipeline orchestrator
│   ├── detection/             # Object detector plugins (YOLOv8 / SKU110K)
│   ├── embeddings/            # Feature extractor plugins (DINOv2 / CLIP)
│   ├── retrieval/             # Vector stores (SQLite Gallery Store, NumPy Index, Hierarchical Index)
│   ├── ocr/                   # Text recognition plugins (Timeout-protected EasyOCR)
│   ├── calibrators/           # Probability mapping engines (Platt Scaling)
│   ├── fusion/                # Late fusion strategies (Lexicon keyword score boosting)
│   ├── decision/              # Gated decision policies (Precision-constrained routing)
│   └── data_quality/          # Bbox quality gates & leakage reconciliation modules
├── server/                    # FastAPI Web Application Service
│   ├── app.py                 # REST API routes (/healthz, /v1/audit/shelf, /v1/onboard/sku)
│   └── schemas.py             # Pydantic HTTP request/response schemas
├── scripts/                   # CLI Execution Tools & Utilities
│   ├── process_shelf.py       # Command line end-to-end shelf audit runner
│   └── migrate_pickle_to_sqlite.py # SQLite database migration script
├── tests/                     # Comprehensive Unit & Integration Test Suites
│   ├── test_sqlite_registry.py
│   ├── test_late_fusion.py
│   ├── test_hierarchical_index.py
│   └── test_api_audit_e2e.py
├── requirements.txt           # Project dependencies
└── README.md                  # System documentation
```

---

## 📊 Evaluation & Benchmark Results

### 1. Feature Embedding Backbone Comparison (Held-Out Test Set: 6,105 Query Crops)

| Metric | DINOv2-small ($D=384$) | CLIP-ViT-B/32 ($D=512$) | Advantage |
| :--- | :---: | :---: | :---: |
| **Recall@1 (Top-1 Accuracy)** | **80.59%** $\pm$ 0.55% | 74.16% $\pm$ 0.63% | 🏆 **+6.43% DINOv2** |
| **Recall@5 (Top-5 Accuracy)** | **93.75%** $\pm$ 0.34% | 90.53% $\pm$ 0.42% | 🏆 **+3.22% DINOv2** |
| **Macro Recall@1 (Class-Balanced)** | **72.86%** | 65.63% | 🏆 **+7.23% DINOv2** |
| **Macro Recall@5 (Class-Balanced)** | **89.45%** | 84.67% | 🏆 **+4.78% DINOv2** |
| **Mean Reciprocal Rank (MRR)** | **85.87%** $\pm$ 0.42% | 80.57% $\pm$ 0.50% | 🏆 **+5.30% DINOv2** |
| **NDCG@5 (Normalized Gain)** | **228.00%** $\pm$ 1.31% | 205.03% $\pm$ 1.47% | 🏆 **+22.97% DINOv2** |

---

## 🚀 Quick Start

### 1. Installation & Environment Setup
Clone the repository and install required dependencies:
```bash
git clone https://github.com/MarwanZaineldeen/shelf-sku-recognition-2.git
cd shelf-sku-recognition-2

pip install -r requirements.txt
```

### 2. Running the Unit & Integration Test Suite
Verify that all system components, SQLite databases, and API routes are operating cleanly:
```bash
$env:PYTHONPATH="."
python -m unittest discover -s tests -p "test_*.py"
```

### 3. Launching the FastAPI REST Microservice
Start the production Uvicorn web server:
```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```
Access interactive API documentation at `http://localhost:8000/docs`.

### 4. End-to-End Command Line Shelf Audit
Process any shelf image and output latency statistics and predictions:
```bash
python scripts/process_shelf.py --image path/to/shelf_image.jpg --device cpu
```

---

## 🛠️ API Endpoints

- `GET /healthz`: System health diagnostics, loaded model status, and active database schema version.
- `POST /v1/audit/shelf`: Upload a raw shelf image to run full detection, feature extraction, retrieval search, OCR fusion, and confidence routing. Returns JSON auto-annotations and HITL queue records.
- `POST /v1/onboard/sku`: Onboard reference crops for a new SKU category into the SQLite vector database and active memory index without server downtime.

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for details.
