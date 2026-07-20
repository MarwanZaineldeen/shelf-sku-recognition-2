# DINOv3 V2 short results report

## Decision

Use **DINOv3 ViT-B/16 exemplar/all** as the V2 exact-SKU baseline. Flat retrieval is the fair backbone comparison and gives the highest accuracy. Adaptive two-tier retrieval is the project architecture and gives nearly the same accuracy while searching one brand-specific index per query.

Prototype/10 is much smaller and faster at FAISS search, but it loses too much exact-SKU accuracy to replace exemplar/all.

## Held-out test results

The test set contains 5,099 queries from 65 supported classes. Values below were produced only after validation choices were frozen.

| Gallery | Retrieval | Top-1 | Top-3 | Top-5 | Macro R@1 | MRR | Search p95 |
|---|---|---:|---:|---:|---:|---:|---:|
| **Exemplar/all** | **Flat** | **83.37%** | **96.51%** | **99.12%** | **77.12%** | **0.8999** | 15.466 ms |
| Exemplar/all | Adaptive two-tier | 83.19% | 96.31% | 98.92% | 76.89% | 0.8980 | 12.312 ms |
| Prototype/10 | Flat | 58.15% | 83.78% | 92.06% | 58.66% | 0.7243 | **0.201 ms** |
| Prototype/10 | Adaptive two-tier | 57.99% | 83.62% | 91.90% | 58.45% | 0.7227 | 0.204 ms |

The grouped 95% confidence interval for flat exemplar Top-1 is **80.31%–86.52%**. For flat Prototype/10 it is **54.61%–61.53%**.

## What the result means

- **Top-1 83.37%:** the first exemplar/all prediction is correct for about 83 of every 100 test crops.
- **Top-5 99.12%:** the correct SKU is somewhere in the five unique SKU candidates for about 99 of every 100 crops. This makes OCR or a fine-grained reranker a strong next stage.
- **Macro R@1 77.12%:** after giving every supported SKU equal weight, performance is lower than overall Top-1. Some classes are therefore much harder than the common classes.
- **Two-tier routing 99.80%:** the correct brand was selected for 5,089 of 5,099 test queries. Its 0.18-point Top-1 loss versus flat is small, but it did not improve accuracy.
- **Prototype/10:** 656 selected training crops were averaged into 67 SKU vectors. This reduces the searchable gallery from 31,656 to 67 vectors—about 472.5× fewer—but flat Top-1 falls by 25.22 percentage points.

## Confidence automation

Validation selected a cosine-similarity threshold of `0.977133` for a 95% precision target. On test, exemplar/all automatically accepted 81 queries and got 80 correct: **98.77% precision at only 1.59% coverage**. Most queries still require review or a second stage. Prototype/10 found no non-empty validation threshold that met 95% precision, so its automation coverage is 0%.

## Main errors and limitations

The most frequent flat exemplar confusions were `class_28 → class_32`, `class_3 → class_44`, `class_32 → class_28`, `class_31 → class_10`, and `class_48 → class_8`. These are mainly fine package-variant errors that a global DINO embedding may not separate reliably.

The dataset has no exact cross-split duplicate crops and no repeated recorded source IDs, but four suspected same-capture/session groups cross splits. Classes `class_45` and `class_50` have no validation/test queries. Unknown products, detector-generated crops, and online serving latency were not tested.

Prototype/10 was run after exemplar test results were already available, so it is a **post-hoc efficiency ablation**, not a second untouched confirmatory result.

## Recommended next step

Keep exemplar/all as the accuracy reference. Test a two-stage system that uses DINO Top-5 candidates followed by OCR or a hard-negative variant classifier, then evaluate it once on a new source/session-separated holdout containing real detector crops and unknown SKUs.
