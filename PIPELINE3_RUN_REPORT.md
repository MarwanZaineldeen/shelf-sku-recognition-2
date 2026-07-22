# Pipeline 3 — Active Continual Learning: Execution Report

**Scope**: end-to-end execution of the Pipeline 3 continual learning loop against the production gallery.
**Verdict**: **all Pipeline 3 stages functioned correctly.** Four defects were found and fixed to reach this point; five open issues are documented, of which one belongs to Pipeline 3 and four belong to the base retrieval stack or the data.

---

## 1. Environment

| Component | Version / value |
| :--- | :--- |
| Python | 3.12.10 (Windows, CPU only) |
| torch | 2.7.1+cpu |
| transformers | 5.5.3 |
| ultralytics | 8.4.103 |
| scikit-learn / numpy / pydantic | 1.6.1 / 2.2.6 / 2.12.5 |

### Data verified present

| Asset | Size | Verified contents |
| :--- | ---: | :--- |
| `retail_sku_registry_dinov3.db` | 131 MB | 31,656 rows × 768-D, 67 classes, all unit-norm |
| `configs/weights/dinov3_vitb16/` | 343 MB | DINOv3 ViT-B/16, 85.7M params, 0 missing keys |
| `yolov8l-sku110k.pt` | 88 MB | loads, class-agnostic detector |
| `images/test/` | — | 127 shelf images |

---

## 2. Test suite

| Suite | Result |
| :--- | :--- |
| `tests/active_learning` | **241 passed** (~12 s) |
| Whole repo, excl. weight-dependent e2e | **256 passed** |

All Pipeline 3 tests are model-free: no database, no weights, no network.

---

## 3. Shelf audit — the input to the loop

Sample image `Transmed Others 246.jpg`, CPU:

| Metric | Value |
| :--- | ---: |
| Detections (YOLOv8l) | 164 |
| Auto-annotated | **24 (14.6%)** |
| Routed to HITL | 140 |
| Gallery index load | 3.9 s |
| Embedding throughput | 246 ms/crop |
| Total audit | ~36–40 s |

HITL routing reasons: `LOW_VISUAL_CONFIDENCE` 120, `LOW_CONFIDENCE` 20.

Sample auto-annotations (calibrated probability):

```
crop_6    class 41   p=0.961   Lipton Green Tea Mint Value Pack - 100 Tea Bags
crop_8    class  5   p=0.958   Lipton Green Tea Lemon Value Pack - 100 Tea Bags
crop_20   class 15   p=0.908   Lipton Anise Herbal Infusion - 20 Tea Bags
crop_23   class 58   p=0.906   Lipton Forest Fruits Black Tea
crop_22   class 16   p=0.854   Lipton Chamomile Herbal Infusion - 20 Tea Bags
```

Commercial metadata resolved correctly for all 67 classes.

---

## 4. Pipeline 3 results

### 4.1 Review ingest — all three decision paths

Submitted through the live `POST /v1/hitl/review` endpoint:

| Crop | Predicted | Reviewer chose | Decision recorded | Embedding captured |
| :--- | ---: | ---: | :--- | :--- |
| `crop_1` | 12 | 7 | `CORRECTED` | ✅ 768-D |
| `crop_2` | 39 | 39 | `APPROVED` | ✅ 768-D |
| `crop_3` | 63 | −1 (not in catalog) | `NOT_IN_CATALOG` | ✅ 768-D |

Every review captured its audit-time DINOv3 vector — no second backbone pass, no recomputation. `NOT_IN_CATALOG` correctly stored a NULL `true_class_id`.

Before this work the endpoint was a `print` statement that returned success and persisted nothing.

### 4.2 Hard negative mining

From 2 verified reviews (the open-set rejection is correctly excluded — it has no ground-truth class):

```
true  confused   freq  outranked  mean_sim  mean_margin
   7        12      2          2    0.8184      -0.8184
   7        64      2          2    0.7986      -0.7986
   7         3      1          1    0.7973      -0.7973
  39         8      1          0    0.6360      +0.0090
```

The negative margins on class 7 show its true class never entered the Top-5 at all — the strongest possible confusion signal. Class 39 vs 8 is a near-miss: 39 won, but by only 0.009.

### 4.3 Gallery curation — the headline result

Session dry run at `--cap 500` on the full gallery:

| Metric | Value |
| :--- | ---: |
| Gallery before | 31,658 (31,656 seed + 2 promoted reviews) |
| Gallery after | **22,007** |
| Pruned | **9,651 (30.5%)** |
| Skipped from promotion | 1 (`NO_VERIFIED_CLASS`) |
| Runtime | ~4 min |

**The original specification estimated "~22,000". The measured result is 22,007.**

### 4.4 Cap ratification

This was the one open decision blocking production use. Measured per-class distribution:

| min | p25 | median | mean | p75 | max |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 196 | 406 | 472.5 | 718 | 1,457 |

**29 of 67 classes exceed 500.** Cap sweep (cap stage only; near-duplicate rejection removes a further ~800):

| cap | classes over | kept | pruned |
| ---: | ---: | ---: | ---: |
| **500** | 29 | 22,794 | 28.0% |
| 400 | 35 | 19,712 | 37.7% |
| 300 | 44 | 15,866 | 49.9% |
| 200 | 49 | 11,281 | 64.4% |

**Decision: `DEFAULT_CLASS_CAP = 500` is ratified — no code change needed.**

> **Correction to earlier analysis.** During planning I argued that a 500 cap would prune almost nothing, reasoning that 31,656 / 67 ≈ 472 average sits below the cap. That reasoning was wrong: the distribution is heavily skewed (median 406, max 1,457), so the mean was misleading. The specification's estimate was correct and my objection was not.

### 4.5 Rollback — verified on production data

The dry run writes to the real database and then restores it. Verified afterwards:

| Check | Result |
| :--- | ---: |
| Seed rows active | 31,656 / 31,656 |
| Rows left pruned | **0** |
| Visible to retrieval | **31,656** (original state) |
| Continual rows | 2 — present but hidden via deactivated version |

Nothing is hard-deleted: pruning deactivates rows, and promotion is hidden by deactivating its gallery version. Both are restorable.

### 4.6 Promotion gate

Exercised on synthetic champion/challenger evaluations (6,000 paired test queries):

**Case 1 — genuinely better challenger → PROMOTE**
```
PASS  top1_gain:      0.8417 -> 0.9287  (delta +0.0870, CI95 [+0.0745, +0.0993])
PASS  top5:           0.9833 -> 0.9933  (delta +0.0100)
PASS  automation@95%: 0.8817 -> 0.9705  (delta +0.0888)
```

**Case 2 — better on every accuracy metric, but rejected**
```
PASS  top1_gain:      0.8417 -> 0.9345  (delta +0.0928)
PASS  top5:           0.9833 -> 0.9938  (delta +0.0105)
FAIL  automation@95%: 0.8817 -> 0.0000  (delta -0.8817)
```

This is the case the gate exists for. The challenger is **+9.3pp Top-1 and +1.1pp Top-5** — better on everything an accuracy-based gate would measure — yet its similarity scores no longer separate correct from incorrect, so reaching 95% precision forces a threshold that automates nothing. A gate promoting on Top-1 would have shipped it.

**Case 3 — leakage → hard failure**
```
LeakageError: 1 shelf image(s) appear in both the challenger gallery and the
test set: shelf_02.jpg. The challenger would be graded on its own reference
crops; re-split before gating.
```

---

## 5. Defects found and fixed

| # | Defect | Impact |
| :--- | :--- | :--- |
| 1 | `ml/embeddings/dinov3.py` — `config.json` declares `model_type: "dinov2"` while architecture and tensors are DINOv3. `AutoModel` dispatches on `model_type`, built a DINOv2 graph, matched **zero** parameters, and randomly initialised the entire backbone. | **Silent corruption.** transformers 5.x raises; older versions loaded it quietly and produced well-formed, meaningless embeddings. Now loads `DINOv3ViTModel` explicitly: 0 missing / 0 unexpected keys. |
| 2 | `ml/embeddings/dinov3.py` — weights path hardcoded to another machine | Extractor unusable. Now repo-relative (`DINOV3_WEIGHTS_DIR` overrides). |
| 3 | `server/app.py` — `workspace_root` hardcoded to another machine | Server could not start. Now repo-relative (`RETAIL_AI_ROOT` overrides). One line fixed all ten dependent paths. |
| 4 | `requirements.txt` omitted torch, transformers, fastapi, uvicorn, pydantic, pyyaml, pillow, safetensors, **python-multipart** | Server failed at import with an opaque multipart error. Now complete. |

Fixed earlier during implementation, each with a regression test: the k-center multi-seed defect, a destructive `rollback_version`, the `fetch_all_references` return-type mismatch that crashed the hierarchical index, and a dry run whose preview disagreed with the applied run.

---

## 6. Open issues

### 6.1 End-to-end automation is 14.6%, not the README's 93.65% — *base retrieval, not Pipeline 3*

These measure different things. 93.65% is a clean-crop retrieval benchmark (gallery crop vs gallery crop). 14.6% is the share of **real detector boxes** clearing the auto-annotate gate; 120 of 164 crops scored below 0.75 similarity.

Most likely a domain gap: every gallery row has `source_image_name = "exemplar_gallery"` — clean isolated reference crops — while queries are angled, occluded, partially cropped shelf detections. This is the number that will be asked about in review, and it is worth investigating before presenting the system.

### 6.2 Qwen2-VL never loads — *base pipeline*

The AWQ checkpoint requires `gptqmodel`. Without it the reranker sets `is_ready=False` and the orchestrator skips Tier 2 silently. That band (0.75 ≤ S_vis < 0.92) is exactly where 20 of the sample image's crops landed, so `pip install gptqmodel` may recover a meaningful share of the 14.6%.

### 6.3 The gate's leakage check is inert for the seed gallery — *data provenance*

All 31,656 seed rows carry `source_image_name = "exemplar_gallery"`, a single distinct value. Criterion 4 has nothing to compare against for them — their provenance was lost when the gallery was built. It works correctly for crops promoted from reviews, which carry real shelf image names. If the gallery is ever rebuilt, preserve the true source image per crop.

### 6.4 Very small classes are fragile — *Pipeline 3*

Class 50 has 2 vectors, class 45 has 4, class 46 has 11. Near-duplicate rejection can reduce these further, and curation has no per-class floor. Adding a minimum that exempts small classes from pruning is worth doing before relying on `--apply` in production.

### 6.5 Reviewed crop images are not stored — *Pipeline 3, by design*

The review store keeps the 768-D embedding, not the crop image, and `crop_path` is currently NULL. This is sufficient for curation, promotion, and SupCon training, which only touch vectors — but it is a regression against the deleted `hitl_store.py`, which had a `crop_bytes` column. You cannot visually re-audit a decision, re-embed after a backbone change, or show a reviewer what they labelled.

Fix is small: add `crop_bytes BLOB` to the schema and carry `pred.crop_bytes` through `ReviewContextCache` (the orchestrator already populates it). Cost ~5–15 KB per review.

---

## 7. Summary

| Stage | Status |
| :--- | :--- |
| Review ingest (3 decision paths, embeddings captured) | ✅ |
| Hard negative confusion mining | ✅ |
| Gallery curation (31,658 → 22,007) | ✅ |
| Versioned rollback on production data | ✅ |
| Promotion gate (promote / reject / leakage) | ✅ |
| SupCon challenger training (opt-in, unpromoted) | ✅ |
| Cap ratification | ✅ 500 confirmed |

Pipeline 3 works. The one substantive number that underperforms expectation — 14.6% automation — originates in the base retrieval stack that Pipeline 3 sits on top of and is designed to improve over time.
