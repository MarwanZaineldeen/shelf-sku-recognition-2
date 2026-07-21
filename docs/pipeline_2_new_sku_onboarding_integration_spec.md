# Technical Specification: Pipeline 2 (New SKU Onboarding Flow)

This specification outlines the architecture, API endpoints, design variants, and integration strategies for **Pipeline 2: Dynamic Few-Shot SKU Onboarding**. 

---

## 1. Executive Summary & Core Objective

The central challenge in production retail shelf audit systems is **dynamic class expansion**. Traditional closed-set object detection models (e.g. YOLO trained directly on 67 SKU classes) require full model retraining, evaluation, and redeployment whenever a retailer introduces a new SKU. **This is completely unviable.**

This platform adopts an **Open-Set Retrieval Architecture** that decouples the pipeline:
1. **Stage 1 (Localization)**: Class-agnostic product bounding box detection.
2. **Stage 2 (Classification)**: Nearest-neighbor embedding match against a dynamically updated vector gallery registry (`sku_crops` in SQLite).

**Pipeline 2** allows product managers to onboard a new SKU category using only **10 to 50 exemplar photos**, updating the search gallery in real time with **zero model retraining, zero system downtime, and sub-second availability**.

```
  [10-50 SKU Photos] ──► [Onboard Ingestion] ──► [DINOv3 Embedding Extraction]
                                                          │
  [Dynamic Search Index] ◄── [Vector Append] ◄── [SQLite Registry Insert]
```

---

## 2. API Endpoint Contract

The primary ingestion route is exposed via FastAPI as a multipart upload:

### `POST /v1/onboard/sku`

* **Request Content-Type**: `multipart/form-data`
* **Request Fields**:
  * `class_id` (`int`, Required): The new stable integer class target identifier.
  * `old_class_id` (`int`, Required): Legacy class code or external catalog reference ID.
  * `family_id` (`str`, Required): Family category grouping (e.g., `tea`, `soda`).
  * `source_image` (`str`, Required): Metadata indicating source context image or batch reference.
  * `reference_images` (`List[UploadFile]`, Required): Binary image payloads of the new SKU.

* **Response Schema (`OnboardResponse`)**:
  ```json
  {
    "status": "success",
    "class_id": 99,
    "gallery_version": 4,
    "crops_added": 15,
    "message": "Successfully onboarded 15 new crop references for class 99."
  }
  ```

---

## 3. Step-by-Step Architecture & Design Variants

To provide flexibility during implementation, each pipeline step supports multiple technical approaches (Variants). The teammate can select the most appropriate option based on compute budgets and deployment environments.

---

### Step 1: Input Ingestion & Metadata Registration

Receives reference photos of the new product packaging and registers metadata in the catalog database.

```
                  ┌───────────────────────────────┐
                  │   Reference Image Payload     │
                  └──────────────┬────────────────┘
                                 ▼
         ┌─────────────────────────────────────────────────┐
         │          Select Ingestion Variant               │
         └──────┬───────────────────┬───────────────────┬──┘
                │                   │                   │
                ▼                   ▼                   ▼
         [Variant A:         [Variant B:         [Variant C:
          Single-Item]        Batch Upload]       Directory Scan]
```

* **Variant A: Single-Item Inboarding**
  * *Description*: Accepts one image per API request. Excellent for simple, single-item additions from handheld scanner units.
  * *Implementation*: Loop calls to a lightweight endpoint. High network overhead for larger batches.
* **Variant B: Multi-Image Batch Upload (Primary Option)**
  * *Description*: Accepts a list of files (`reference_images: List[UploadFile]`) in a single HTTP request. Processes the batch within a single SQLite transaction to ensure database consistency.
  * *Implementation*: FastAPI parses multipart files asynchronously, allowing concurrent feature extraction.
* **Variant C: Folder-Based Auto-Import**
  * *Description*: A background service scans a structured local folder (e.g., `data/onboarding/class_{class_id}/`) and registers files automatically.
  * *Implementation*: Useful for offline bulk setup during tenant migration. Runs as a background task.

---

### Step 2: Reference Crop Generation

Isolates the actual product facing boundaries from the uploaded reference photos.

```
                      ┌───────────────────────┐
                      │ Ingested Image Files  │
                      └───────────┬───────────┘
                                  ▼
         ┌─────────────────────────────────────────────────┐
         │            Select Cropping Variant              │
         └──────┬───────────────────┬───────────────────┬──┘
                │                   │                   │
                ▼                   ▼                   ▼
         [Variant A:         [Variant B:         [Variant C:
          Pre-Cropped]        YOLO Auto-Detect]   Interactive Box]
```

* **Variant A: Pre-Cropped Images (User's Curated Base)**
  * *Description*: Assumes the uploaded images are already cropped tightly to the product boundary (such as the files manually curated in `data/processed/Sku Preview`).
  * *Implementation*: Coordinates default to the full image boundaries (`x1=0.0, y1=0.0, x2=w, y2=h`). This bypasses localization algorithms entirely.
* **Variant B: Class-Agnostic YOLO Auto-Detection**
  * *Description*: Runs the incoming raw images through our fine-tuned YOLO detector (`runs/detect/yolo8l_sku110k/yolov8l-sku110k.pt`) to locate the product bounding box automatically.
  * *Implementation*: Selects the bounding box with the highest confidence score, crops the region, and discards background borders. Saves time for end-users.
* **Variant C: Interactive Manual Box Inputs**
  * *Description*: The API receives coordinate arrays along with each file.
  * *Implementation*: Allows merchandisers to draw bounding boxes on the web UI before submitting the request, guaranteeing exact boundaries.

> [!NOTE]
> As requested, **the crop image quality gate is neglected** at this stage to avoid rejecting reference photos that may contain shadows or slight angles but are still valuable for few-shot matching.

---

## Step 3: Embedding Extraction & Preprocessing

Generates visual feature vectors that represent the new SKU.

```
                    ┌─────────────────────────┐
                    │  Cropped Image Bounds   │
                    └────────────┬────────────┘
                                 ▼
         ┌─────────────────────────────────────────────────┐
         │          Select Preprocessing Variant           │
         └──────────────┬───────────────────┬──────────────┘
                        │                   │
                        ▼                   ▼
                 [Variant A:         [Variant B:
                  Aspect-Ratio        Direct Resize]
                  Preserving]
```

* **Variant A: Aspect-Ratio Preserving Gray-Canvas Padding (Recommended)**
  * *Description*: Keeps original proportions by resizing the longest side to 224 pixels and padding the remaining borders with gray canvas values ($128, 128, 128$).
  * *Implementation*: Matches the exact input distribution expected by our DINOv3 model, preventing visual distortion.
* **Variant B: Direct Bilinear Resize**
  * *Description*: Stretches the image directly to $224 \times 224$ pixels.
  * *Implementation*: Faster to execute but distorts aspect ratios, which can slightly degrade visual matching accuracy.

#### Embedding Extraction Logic
We extract the $L_2$-normalized 768-dimensional visual embedding vector using **DINOv3 ViT-B/16**:
```python
# Extract normalized embedding vector using our production extractor
embedding_dto = embedder_plugin.extract_dto(crop_dto)
vector = embedding_dto.vector  # 768-D Float list
```

---

## Step 4: Vector Registry & Dynamic Memory Index Updates

Saves the new embeddings to disk and updates the active search indexes.

```
                     ┌───────────────────────┐
                     │ DINOv3 Feature Vector │
                     └───────────┬───────────┘
                                 ▼
             ┌──────────────────────────────────────┐
             │         Database SQLite Save         │
             │   (Insert record to `sku_crops`)     │
             └───────────────────┬──────────────────┘
                                 ▼
         ┌─────────────────────────────────────────────────┐
         │             Select Indexing Variant             │
         └──────────────┬───────────────────┬──────────────┘
                        │                   │
                        ▼                   ▼
                 [Variant A:         [Variant B:
                  Dynamic Append]     Full Index Rebuild]
```

* **Variant A: Dynamic Memory Index Append (Instant Availability)**
  * *Description*: Directly appends the new embedding vectors to the active search index in-memory buffer without rebooting FastAPI.
  * *Implementation*: Call `retriever_plugin.add(...)` with the new embedding array. Updates take place in less than 1ms.
* **Variant B: Full Index Rebuild**
  * *Description*: Writes the data to SQLite, increments the registry version, and schedules an index rebuild task.
  * *Implementation*: Re-reads the `sku_crops` table and reconstructs the FAISS/NumPy index. Guarantees index optimization but consumes more CPU resources.

---

## Step 5: Onboarding Validation & Quality Verification

Verifies that the new SKU can be successfully recognized by the system.

```
                     ┌───────────────────────┐
                     │ Updated Search Index  │
                     └───────────┬───────────┘
                                 ▼
         ┌─────────────────────────────────────────────────┐
         │            Select Validation Variant            │
         └──────────────┬───────────────────┬──────────────┘
                        │                   │
                        ▼                   ▼
                 [Variant A:         [Variant B:
                  Self-Retrieval]     Shelf Scan Test]
```

* **Variant A: Self-Retrieval Validation (Fast Checker)**
  * *Description*: Queries the newly uploaded reference crops against the updated registry.
  * *Implementation*: Asserts that the Top-1 cosine similarity match for each reference crop points to its own newly onboarded `class_id`. Should achieve 100% accuracy.
* **Variant B: Mock Shelf Image Validation**
  * *Description*: Runs a mock audit on selected test shelf images containing the new SKU.
  * *Implementation*: Verifies that the system correctly localizes and classifies the new SKU within a realistic layout.

---

## 4. Database Schema Reference

The teammate's ingestion logic should store the reference crops directly in the `sku_crops` table inside [`retail_sku_registry_dinov3.db`](file:///d:/Marwan/ITI%20AI&ML/Transmid%20GP/data/processed/crops/gt_clean/retail_sku_registry_dinov3.db):

```sql
CREATE TABLE sku_crops (
    id VARCHAR(255) PRIMARY KEY,       -- Format: 'onboard_crop_{filename}'
    crop_path TEXT NOT NULL,           -- Local filename references
    remapped_class_id INTEGER NOT NULL,-- training_class_id
    old_class_id INTEGER NOT NULL,     -- Original catalog class ID
    family_id TEXT,                    -- Family category (e.g. 'tea')
    source_image_name TEXT,            -- Bounding box parent image
    bbox_x1 REAL NOT NULL,             -- Bounding box coordinates
    bbox_y1 REAL NOT NULL,
    bbox_x2 REAL NOT NULL,
    bbox_y2 REAL NOT NULL,
    embedding_blob BLOB NOT NULL,      -- DINOv3 768-D float32 array
    gallery_version INTEGER NOT NULL   -- Database update version
);
```

---

## 5. Teammate Package Structure Guidelines

To ensure clean pull requests that merge easily into GitHub, the teammate should implement their onboarding logic under a dedicated package:

```
ml/
├── onboarding/                     # [NEW PACKAGE]
│   ├── __init__.py
│   ├── ingestion.py                # Handles file reads & multi-part uploads
│   ├── preprocessing.py            # Aspect-ratio resizer
│   ├── registry_updater.py         # SQLite DB writer and dynamic index appender
│   └── validation.py               # Self-retrieval validator
```

All modifications to the API layer should be confined to [`server/app.py`](file:///d:/Marwan/ITI%20AI&ML/Transmid%20GP/server/app.py) under the `@app.post("/v1/onboard/sku")` endpoint.
