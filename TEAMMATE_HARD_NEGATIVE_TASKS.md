# 📋 Team Action Plan: Hard Negative SKU Separation & Optimization Tasks

This document outlines the empirical findings from our **31,656 DINOv3 (768-D) visual embedding analysis** across 67 active SKU categories and defines actionable engineering tasks to improve class separation.

---

## 📊 How Inter-Class Cosine Similarity Scores Are Computed

The inter-class similarity score $S_{A, B}$ measures the raw visual feature overlap between two SKU classes:

1. **Centroid Extraction**: For every SKU class $c \in \{0 \dots 66\}$, we calculate its L2-normalized 768-D mean visual centroid vector $\mathbf{\mu}_c$ from all stored DINOv3 crop embeddings in `retail_sku_registry_dinov3.db`.
2. **Cross-Class Dot Product**:
   $$S_{A, B} = \mathbf{\mu}_A \cdot \mathbf{\mu}_B^T \quad \in [-1, 1]$$
3. **What a Score of 99.5%+ Means**:
   DINOv3 self-supervised visual features focus heavily on overall box geometry, background colors, and brand logos. Because `Lipton Yellow Label 400g` and `800g` share identical yellow packaging and logo placement, their visual embedding vectors are $\sim 99.5\%$ identical in 768-D space.

---

## 🏆 Top-20 Hard Negative SKU Pairs

| Rank | Inter-Class Similarity | Class A | Class B | Primary Cause of Confusion |
| :---: | :---: | :--- | :--- | :--- |
| **#01** | **99.63%** | [Class 25] Lipton Green Tea Mint - 50s | [Class 36] Lipton Green Tea Mint Saver Box 50s | Promo Banner ("Saver Box") |
| **#02** | **99.57%** | [Class 28] Lipton Yellow Label Tea - 400g | [Class 32] Lipton Yellow Label Tea - 800g | Net Weight Digit (`400g` vs `800g`) |
| **#03** | **99.51%** | [Class 0] Lipton Green Tea Lemon - 50s | [Class 37] Lipton Green Tea Lemon Saver Box 50s | Promo Banner ("Saver Box") |
| **#04** | **99.47%** | [Class 24] Lipton Green Tea Pure - 50s | [Class 35] Lipton Green Tea Pure - 50s | Duplicate Class Labeling |
| **#05** | **99.44%** | [Class 0] Lipton Green Tea Lemon - 50s | [Class 61] Lipton Green Tea Lemon - 100s | Pack Count (`50s` vs `100s`) |
| **#06** | **99.43%** | [Class 3] Lipton Yellow Special Offer 100s | [Class 44] Lipton Yellow Standard 100s | Promo Banner ("Special Offer") |
| **#07** | **99.37%** | [Class 4] Lipton Yellow Label Tea - 200s | [Class 26] Lipton Yellow Label Tea - 200s | Duplicate Class Labeling |
| **#08** | **99.36%** | [Class 32] Lipton Yellow Label Tea - 800g | [Class 38] Lipton Yellow Special Offer 800g | Promo Banner ("Special Offer") |
| **#09** | **99.32%** | [Class 38] Lipton Yellow Special Offer 800g | [Class 63] Lipton Yellow Super Saver 400g | Size + Promo Banner Combo |
| **#10** | **99.23%** | [Class 29] Lipton Green Tea Pure - 100s | [Class 41] Lipton Green Tea Mint Value Pack 100s | Flavor Accent (Pure vs. Mint) |
| **#11** | **99.12%** | [Class 28] Lipton Yellow Label Tea - 400g | [Class 38] Lipton Yellow Special Offer 800g | Size + Promo Banner |
| **#12** | **99.09%** | [Class 37] Lipton Green Tea Lemon Saver 50s | [Class 61] Lipton Green Tea Lemon - 100s | Pack Count (`50s` vs `100s`) |
| **#13** | **98.99%** | [Class 35] Lipton Green Tea Pure - 50s | [Class 36] Lipton Green Tea Mint Saver 50s | Flavor Accent (Pure vs. Mint) |
| **#14** | **98.98%** | [Class 28] Lipton Yellow Label Tea - 400g | [Class 63] Lipton Yellow Super Saver 400g | Promo Banner ("Super Saver") |
| **#15** | **98.93%** | [Class 10] Brooke Bond Red Label - 800g | [Class 31] Brooke Bond Red Label - 400g | Net Weight Digit (`800g` vs `400g`) |
| **#16** | **98.87%** | [Class 29] Lipton Green Tea Pure - 100s | [Class 35] Lipton Green Tea Pure - 50s | Pack Count (`100s` vs `50s`) |
| **#17** | **98.86%** | [Class 36] Lipton Green Tea Mint Saver 50s | [Class 41] Lipton Green Tea Mint Value Pack 100s | Pack Count (`50s` vs `100s`) |
| **#18** | **98.86%** | [Class 8] Brooke Bond Red Label - 100s | [Class 48] Brooke Bond Special Offer - 100s | Promo Banner ("Special Offer") |
| **#19** | **98.73%** | [Class 24] Lipton Green Tea Pure - 50s | [Class 25] Lipton Green Tea Mint - 50s | Flavor Accent (Pure vs. Mint) |
| **#20** | **98.73%** | [Class 25] Lipton Green Tea Mint - 50s | [Class 41] Lipton Green Tea Mint Value Pack 100s | Pack Count (`50s` vs `100s`) |

---

## 🛠️ Assigned Actionable Engineering Tasks

### Task 1: Duplicate Class Remapping & Merging
- **Assigned To**: Teammate A
- **Target File**: `configs/sku_mapping.json`
- **Objective**: Merge duplicate class entries (e.g. Class 24 & Class 35 for Lipton Green Tea Pure 50s; Class 4 & Class 26 for Lipton Yellow 200s) into single canonical class IDs.

### Task 2: Sub-Region Localized Patch Token Pooling
- **Assigned To**: Teammate B
- **Target Files**: `ml/embeddings/dinov3.py`, `ml/orchestrator.py`
- **Objective**: Extract DINOv3 $14 \times 14$ patch tokens specifically from the bottom-right corner (net weight digits) and top header (promo banners) for tight variant ties.

### Task 3: Regex Digit Overrides in Qwen2-VL Reranking
- **Assigned To**: Teammate C
- **Target Files**: `ml/vlm/qwen2_vl_reranker.py`
- **Objective**: Implement strict regex digit parsing (`r'(\d+)\s*(g|kg|bags)'`) on OCR/VLM text outputs to filter out incorrect pack size candidates when text explicitly specifies weight.

### Task 4: Exponential Repulsive Force in Pipeline 3 SupCon Loss
- **Assigned To**: Teammate D
- **Target Files**: `ml/active_learning/hard_negatives.py`, `ml/active_learning/finetune.py`
- **Objective**: Weight contrastive repulsive loss proportionally to cross-class similarity ($w_{ij} = e^{S_{ij}}$) to push confused class vectors apart during active learning fine-tuning.
