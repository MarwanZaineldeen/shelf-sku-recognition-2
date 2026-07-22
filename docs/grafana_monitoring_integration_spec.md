# Technical Specification: Grafana Telemetry & Monitoring Suite

### Target Role: MLOps & Infrastructure Monitoring Lead
### System: Enterprise Retail AI Platform (`shelf-sku-recognition-2`)
### Authoring Team: AI Architecture Team

---

## 1. Executive Summary & Telemetry Architecture

Our **Enterprise Retail AI Platform** executes an end-to-end, multi-stage computer vision pipeline:
$$\text{Shelf Image} \xrightarrow{\text{YOLOv8l}} \text{BBoxes} \xrightarrow{\text{DINOv3 (768-D)}} \text{Vector Search} \xrightarrow{\text{Qwen2-VL}} \text{VLM Rerank} \xrightarrow{\text{Platt}} \text{Gated Decision}$$

To provide complete visibility for executive stakeholders, MLOps engineers, and annotators, the monitoring stack uses **Prometheus** for metric scraping and **Grafana** for real-time visual dashboards.

```
  ┌─────────────────────────┐        ┌─────────────────────────┐        ┌─────────────────────────┐
  │  FastAPI Backend Server │ ─────► │   Prometheus Scraper    │ ─────► │    Grafana Dashboard    │
  │  (Port 8000 /metrics)   │        │   (Port 9090)           │        │    (Port 3000)          │
  └─────────────────────────┘        └─────────────────────────┘        └─────────────────────────┘
```

---

## 2. Complete API Endpoint Inventory Reference

The system exposes **9 FastAPI REST endpoints** that require metrics instrumentation:

| # | HTTP Method | Endpoint Path | Summary | Input Payload | Output Response Schema |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **1** | `GET` | `/` | Serves Web UI Dashboard | None | `FileResponse` (`index.html`) |
| **2** | `GET` | `/healthz` | System Health & DB Version | None | `HealthResponse` (`status`, `loaded_models`, `db_version`) |
| **3** | `GET` | `/api/catalog` | Commercial Catalog 67 SKUs | None | `JSON` (`classes` dictionary keyed by `training_class_id`) |
| **4** | `GET` | `/v1/skus` | SKU List for Dropdowns | None | `JSON` (`classes`: array of `class_id`, `display_name`, `brand`) |
| **5** | `GET` | `/v1/exemplars/{class_id}` | Product Crop Thumbnail | `class_id` (`int` path) | `Response` (`image/jpeg` or fallback SVG) |
| **6** | `POST` | `/v1/audit/shelf` | **Core Pipeline**: Process Shelf | `file` (`UploadFile` JPEG/PNG) | `AuditResponse` (Annotations, HITL Queue, `processing_time_ms`) |
| **7** | `GET` | `/v1/audit/sample` | Sample Shelf Audit Demo | None | `AuditResponse` (Default test shelf analysis) |
| **8** | `POST` | `/v1/hitl/review` | **Pipeline 3**: Save Review | Form: `hitl_id`, `crop_id`, `assigned_class_id`, `reviewer_id` | `JSON` (`status`, `hitl_id`, `assigned_class_id`) |
| **9** | `POST` | `/v1/onboard/sku` | **Pipeline 2**: Few-Shot Onboard | Form: `brand`, `product_name`, `class_id`, Files: `reference_images` | `OnboardResponse` (`status`, `version`, `crops_added`, `message`) |

---

## 3. FastAPI Prometheus Instrumentation Guide

To instrument the application, install `prometheus-fastapi-instrumentator` and add telemetry hooks in `server/app.py`.

```bash
pip install prometheus-fastapi-instrumentator prometheus-client
```

### In `server/app.py`:

```python
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge

# 1. Initialize Prometheus Exporter
instrumentator = Instrumentator().instrument(app)

# 2. Custom Business & ML Metrics Definition
stage_latency_histogram = Histogram(
    "retail_ai_stage_latency_seconds",
    "Execution time per pipeline stage in seconds",
    ["stage"]  # yolo_detector, dinov3_embedder, vector_search, qwen2_vl, platt_calibrator
)

auto_annotation_ratio_gauge = Gauge(
    "retail_ai_auto_annotation_ratio",
    "Ratio of auto-approved facings vs total detected facings"
)

facings_detected_counter = Counter(
    "retail_ai_facings_detected_total",
    "Total product facings detected across shelf scans"
)

vlm_triggers_counter = Counter(
    "retail_ai_vlm_triggers_total",
    "Total times Qwen2-VL reranker was activated for ambiguous crops"
)

hitl_reviews_counter = Counter(
    "retail_ai_hitl_reviews_total",
    "Total HITL reviews submitted by merchandisers",
    ["type"]  # confirmation vs correction
)

sku_registry_count_gauge = Gauge(
    "retail_ai_sqlite_vector_count",
    "Total 768-D DINOv3 vectors indexed in SQLite registry"
)

# 3. Expose /metrics endpoint on startup
@app.on_event("startup")
def expose_metrics():
    instrumentator.expose(app, endpoint="/metrics")
```

---

## 4. High-Impact Grafana Dashboard Specifications

Implement these **4 specialized Grafana dashboards** for complete platform observability:

---

### 📊 Dashboard 1: Executive KPI & Retail Operations Overview
> **Target Audience**: Executive CTOs, Retail Operations Directors, Store Managers.

```
┌──────────────────────────────────────┬──────────────────────────────────────┐
│  Auto-Annotation Rate (Target >=80%) │  Live Shelf Scan Throughput          │
│       [ Gauge: 84.2% (EMERALD) ]     │      [ Time-Series: Scans / Min ]    │
├──────────────────────────────────────┼──────────────────────────────────────┤
│ Total Product Facings Analyzed       │  HITL Review Queue Backlog           │
│       [ Stat: 142,850 Facings ]      │      [ Bar Gauge: 127 Pending ]      │
├──────────────────────────────────────┴──────────────────────────────────────┤
│  Catalog SKU Coverage & Onboarded Products Growth                           │
│      [ Time Series: Active SKUs in SQLite Vector Registry (67 -> 72) ]      │
└─────────────────────────────────────────────────────────────────────────────┘
```

* **Panel 1 (Gauge)**: `Auto-Annotation Efficiency Rate`
  * *PromQL*: `(sum(rate(retail_ai_facings_detected_total{status="auto"}[5m])) / sum(rate(retail_ai_facings_detected_total[5m]))) * 100`
  * *Thresholds*: Red ($<70\%$), Amber ($70\text{--}80\%$), Emerald ($\ge 80\%$).
* **Panel 2 (Stat)**: `Total Product Facings Analyzed`
  * *PromQL*: `sum(retail_ai_facings_detected_total)`
* **Panel 3 (Bar Gauge)**: `HITL Queue Backlog` by store unit.
* **Panel 4 (Time Series)**: `Catalog SKU Growth` (showing Pipeline 2 onboarding additions over time).

---

### ⚡ Dashboard 2: ML Pipeline Latency & Stage Bottleneck Inspector
> **Target Audience**: Senior AI Engineers, MLOps, Infrastructure Engineers.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ End-to-End Per-Facing Latency Breakdown (ms)                                │
│ [ Stacked Bar: YOLO (12ms) | DINOv3 (2.9ms) | VLM (180ms) | Platt (0.1ms) ]  │
├──────────────────────────────────────┬──────────────────────────────────────┤
│ Stage 1: YOLOv8l Detection Latency   │ Stage 2: DINOv3 Vector Search        │
│    [ Stat: p50=11.4ms | p99=18.2ms ] │    [ Stat: 2.93 ms (Sub-3ms SLA) ]   │
├──────────────────────────────────────┼──────────────────────────────────────┤
│ Stage 3: Qwen2-VL Activation Rate    │ CPU & PyTorch RAM Memory Footprint   │
│    [ Gauge: 14.5% of Crops Triggered]│    [ Gauge: 4.2 GB / 16 GB RAM ]     │
└──────────────────────────────────────┴──────────────────────────────────────┘
```

* **Panel 1 (Stacked Bar)**: Latency breakdown by module (`stage="yolo"`, `stage="dinov3"`, `stage="vlm"`, `stage="platt"`).
* **Panel 2 (Stat)**: YOLOv8l detection latency percentiles ($p_{50}, p_{95}, p_{99}$).
* **Panel 3 (Stat)**: DINOv3 768-D cosine search query latency (demonstrating sub-3ms query speed).
* **Panel 4 (Gauge)**: Qwen2-VL activation rate (verifying gating isolates VLM execution to $0.75 \le S_{\text{vis}} < 0.92$).

---

### 🔄 Dashboard 3: Pipeline 2 & 3 Active Learning & HITL Tracker
> **Target Audience**: Data Engine Leads, Annotation Supervisors, Pipeline 3 Engineers.

```
┌──────────────────────────────────────┬──────────────────────────────────────┐
│ HITL Corrections vs Confirmations    │ Top Hard-Negative Confusion Pairs    │
│  [ Donut: 78% Confirm | 22% Correct] │  [ Heatmap / Bar: Class 6 vs Class 9 ]│
├──────────────────────────────────────┼──────────────────────────────────────┤
│ Pipeline 2 Onboarding Velocity       │ SupCon Fine-Tuning Trigger Progress  │
│  [ Stat: 15 Crops Onboarded / Min ]  │  [ Progress Bar: 184 / 500 Reviews ] │
├──────────────────────────────────────┴──────────────────────────────────────┤
│ Merchandiser Reviewer Activity Leaderboard                                  │
│  [ Table: Reviewer ID | Reviews Submitted | Mean Correction Time ]          │
└─────────────────────────────────────────────────────────────────────────────┘
```

* **Panel 1 (Donut Chart)**: HITL confirmations vs corrections ratio.
* **Panel 2 (Heatmap)**: Hard-negative confusion pairs ($C_1$ vs $C_2$), identifying pairs for Pipeline 3 contrastive head fine-tuning.
* **Panel 3 (Progress Bar)**: Progress towards $N=500$ review threshold for SupCon retraining.
* **Panel 4 (Table)**: Merchandiser review activity log (`reviewer_id`, total reviews).

---

### 🎯 Dashboard 4: Model Calibration & Vector Registry Health
> **Target Audience**: ML Research Engineers, Quality Assurance Auditors.

```
┌──────────────────────────────────────┬──────────────────────────────────────┐
│ Calibrated Probability Distribution  │ Visual Similarity (S_vis) Histogram  │
│  [ Histogram: P in [0.0, 1.0] ]      │  [ Histogram: Cosine Sim Peaks ]     │
├──────────────────────────────────────┴──────────────────────────────────────┤
│ Gated Decision Outcome Split                                                │
│  [ Pie Chart: Auto-Approved (84.2%) | Low Conf HITL (12.8%) | Non-Catalog (3%)]│
├─────────────────────────────────────────────────────────────────────────────┤
│ SQLite Database Vector Growth Curve                                         │
│  [ Time Series: 31,656 -> 31,810 vectors (version 4) ]                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

* **Panel 1 (Histogram)**: Platt calibrated probability distribution ($P \in [0, 1]$).
* **Panel 2 (Pie Chart)**: Decision gating split (Auto-Approved vs HITL Low Confidence vs Non-Catalog).
* **Panel 3 (Time Series)**: Total 768-D vectors in `retail_sku_registry_dinov3.db`.

---

## 🎯 5. Step-by-Step Deployment Instructions

1. **Install Prometheus Exporter**: Add `prometheus-fastapi-instrumentator` to `server/app.py`.
2. **Configure Prometheus Scraper**:
   Create `prometheus.yml`:
   ```yaml
   global:
     scrape_interval: 5s

   scrape_configs:
     - job_name: 'retail_ai_backend'
       metrics_path: '/metrics'
       static_configs:
         - targets: ['localhost:8000']
   ```
3. **Connect Grafana Data Source**: Point Grafana (`http://localhost:3000`) to Prometheus (`http://localhost:9090`).
4. **Set Up Alert Rules**:
   - `Auto-Annotation Rate < 75%` (Triggers alert for potential model drift).
   - `FastAPI 500 Error Spikes > 2%` (Triggers server error alert).
   - `Per-Facing Latency > 500ms` (Triggers latency bottleneck alert).
