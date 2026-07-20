# Embedding Baseline Comparison Report

> Note: results not yet available for: dinov3. Comparison below reflects only the models that have been run.

> Val and test are both reported below. Val exists for inspection/calibration; the winner is selected on test only (see `select_winner`), the untouched final benchmark.

## Accuracy

| Model | Split | Dim | Flat Top-1 | Flat Top-3 | Flat Top-5 | Flat Macro-P | Flat Macro-R | Clustered Top-1 | Clustered Top-3 | Clustered Top-5 | Brand Near-Miss Rate | Gallery Index |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| siglip | val | 1152 | 0.7623 | 0.8791 | 0.9210 | 0.6999 | 0.6751 | 0.7620 | 0.8786 | 0.9203 | 0.0829 | built |
| siglip | test | 1152 | 0.7718 | 0.8782 | 0.9155 | 0.7184 | 0.6992 | 0.7718 | 0.8782 | 0.9155 | 0.0540 | built |
| radio_clip | val | 3072 | 0.8275 | 0.9178 | 0.9459 | 0.7909 | 0.7640 | 0.8268 | 0.9164 | 0.9445 | 0.0409 | built |
| radio_clip | test | 3072 | 0.8344 | 0.9157 | 0.9441 | 0.7750 | 0.7555 | 0.8336 | 0.9150 | 0.9433 | 0.0287 | built |

## Ranking Quality & Macro Recall@k

Macro Recall@k averages per-class hit rate across variants (long-tail-fair), unlike the size-weighted Top-k accuracy above. MRR and NDCG@5 score *where* the true label ranked, not just whether it cleared a cutoff.

| Model | Split | Flat Macro-R@3 | Flat Macro-R@5 | Flat MRR | Flat NDCG@5 | Clustered Macro-R@3 | Clustered Macro-R@5 | Clustered MRR | Clustered NDCG@5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| siglip | val | 0.8088 | 0.8625 | 0.8239 | 0.8482 | 0.8085 | 0.8622 | 0.8234 | 0.8477 |
| siglip | test | 0.8093 | 0.8441 | 0.8272 | 0.8493 | 0.8093 | 0.8441 | 0.8272 | 0.8493 |
| radio_clip | val | 0.8733 | 0.9086 | 0.8746 | 0.8925 | 0.8717 | 0.9070 | 0.8736 | 0.8915 |
| radio_clip | test | 0.8684 | 0.9000 | 0.8767 | 0.8936 | 0.8677 | 0.8993 | 0.8759 | 0.8928 |

## 95% Bootstrap Confidence Intervals (1,000 resamples)

| Model | Split | Flat Top-1 CI | Flat Top-3 CI | Flat Top-5 CI | Clustered Top-1 CI | Clustered Top-3 CI | Clustered Top-5 CI |
|---|---|---|---|---|---|---|---|
| siglip | val | [0.7523, 0.7727] | [0.8709, 0.8878] | [0.9139, 0.9282] | [0.7520, 0.7724] | [0.8704, 0.8872] | [0.9132, 0.9275] |
| siglip | test | [0.7609, 0.7830] | [0.8698, 0.8870] | [0.9077, 0.9228] | [0.7609, 0.7830] | [0.8698, 0.8870] | [0.9077, 0.9228] |
| radio_clip | val | [0.8173, 0.8368] | [0.9104, 0.9248] | [0.9399, 0.9519] | [0.8164, 0.8363] | [0.9090, 0.9234] | [0.9382, 0.9503] |
| radio_clip | test | [0.8242, 0.8444] | [0.9079, 0.9232] | [0.9374, 0.9503] | [0.8235, 0.8436] | [0.9071, 0.9222] | [0.9368, 0.9494] |

## Calibration @ 95% Target Precision

Calibrated threshold = lowest top-1 confidence score, walking queries from most- to least-confident, whose cumulative precision still meets 95%. Automation coverage = fraction of queries at/above that threshold -- i.e. what fraction could skip human review at that precision bar. `n/a` means no threshold, however small, cleared 95%. Each split calibrates its own threshold independently (not fit on val and applied to test).

| Model | Split | Flat Threshold | Flat Coverage | Clustered Threshold | Clustered Coverage |
|---|---|---:|---:|---:|---:|
| siglip | val | 0.9913 | 0.0081 | 0.9913 | 0.0081 |
| siglip | test | 0.9942 | 0.0010 | 0.9942 | 0.0010 |
| radio_clip | val | 0.9861 | 0.0176 | 0.9861 | 0.0176 |
| radio_clip | test | 0.9834 | 0.0561 | 0.9834 | 0.0561 |

## Latency (ms)

Embed = model forward pass per crop. Flat/Clustered search = index lookup only, excludes embedding time -- the two are measured separately so they're never conflated.

> **Flat Search column corrected for single-query inference.** The pipeline's raw `mean_search_latency_ms` times one *batched* FAISS call over all val/test queries at once and divides by query count -- a fast, amortized number no single real query at inference time actually gets (matrix-matrix GEMM vs. matrix-vector GEMV). The values below instead estimate true single-query latency, taken from a diagnostic benchmark run at matching scale (same gallery size ~31.7k, k=5, and embedding dim per model) that timed flat search one query at a time. Clustered Search was already single-query in the original run (its evaluation loop always searches one query at a time), so it needed no correction.

| Model | Split | Embed ms/crop | Flat Search ms/query | Clustered Search ms/query |
|---|---|---:|---:|---:|
| siglip | val | 233.95 | 29.75 | 11.71 |
| siglip | test | 235.58 | 29.75 | 11.93 |
| radio_clip | val | 150.27 | 52.14 | 31.82 |
| radio_clip | test | 150.52 | 52.14 | 31.73 |

Clustered search is actually **faster** than flat per single query -- consistent with searching one brand partition (~half the 2-brand gallery) instead of the full gallery.

## Clustering Cost (flat top-1 minus clustered top-1)

| Model | Split | Recall Lost to Clustering |
|---|---|---:|
| siglip | val | 0.0004 |
| siglip | test | 0.0000 |
| radio_clip | val | 0.0007 |
| radio_clip | test | 0.0008 |

## Winner: `radio_clip`

Selected by highest flat-exact top-1 accuracy on the **test** split, ties broken by lowest per-crop test query embedding latency. This model is the recommended fine-tuning candidate.