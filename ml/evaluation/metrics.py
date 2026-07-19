from typing import List, Dict, Tuple, Any
import numpy as np


def compute_recall_at_k(neighbor_labels: np.ndarray, query_labels: np.ndarray, k: int) -> float:
    """Computes standard Recall@K (micro-averaged accuracy at K).

    Args:
        neighbor_labels: Integer array of shape (M, K) containing neighbor class labels.
        query_labels: Integer array of shape (M,) containing query class labels.
        k: The rank threshold.

    Returns:
        float: Recall@K rate.
    """
    if len(query_labels) == 0:
        return 0.0
    matches = 0
    for i in range(len(query_labels)):
        if query_labels[i] in neighbor_labels[i, :k]:
            matches += 1
    return float(matches / len(query_labels))


def compute_macro_recall_at_k(neighbor_labels: np.ndarray, query_labels: np.ndarray, k: int) -> float:
    """Computes class-balanced (macro-averaged) Recall@K.

    Args:
        neighbor_labels: Integer array of shape (M, K) containing neighbor labels.
        query_labels: Integer array of shape (M,) containing query labels.
        k: The rank threshold.

    Returns:
        float: Macro Recall@K rate.
    """
    unique_classes = np.unique(query_labels)
    if len(unique_classes) == 0:
        return 0.0
    
    class_recalls = []
    for c in unique_classes:
        class_mask = (query_labels == c)
        c_neighbors = neighbor_labels[class_mask, :k]
        c_queries = query_labels[class_mask]
        
        matches = 0
        for i in range(len(c_queries)):
            if c_queries[i] in c_neighbors[i]:
                matches += 1
        class_recalls.append(matches / len(c_queries))
        
    return float(np.mean(class_recalls))


def compute_mrr(neighbor_labels: np.ndarray, query_labels: np.ndarray) -> float:
    """Computes Mean Reciprocal Rank (MRR) for query retrievals.

    Args:
        neighbor_labels: Integer array of shape (M, K) containing neighbor labels.
        query_labels: Integer array of shape (M,) containing query labels.

    Returns:
        float: MRR score.
    """
    if len(query_labels) == 0:
        return 0.0
    
    reciprocal_ranks = []
    for i in range(len(query_labels)):
        q_label = query_labels[i]
        n_labels = neighbor_labels[i]
        
        found_rank = 0
        for rank_idx, val in enumerate(n_labels):
            if val == q_label:
                found_rank = rank_idx + 1
                break
        
        if found_rank > 0:
            reciprocal_ranks.append(1.0 / found_rank)
        else:
            reciprocal_ranks.append(0.0)
            
    return float(np.mean(reciprocal_ranks))


def compute_ndcg_at_k(neighbor_labels: np.ndarray, query_labels: np.ndarray, k: int) -> float:
    """Computes Normalized Discounted Cumulative Gain (NDCG) at K for single-label retrieval.

    Args:
        neighbor_labels: Integer array of shape (M, K) containing neighbor labels.
        query_labels: Integer array of shape (M,) containing query labels.
        k: The rank threshold.

    Returns:
        float: NDCG@K score.
    """
    if len(query_labels) == 0 or k <= 0:
        return 0.0
    
    # Calculate the Ideal DCG at K: assuming all top K retrieved are correct matches
    idcg = sum(1.0 / np.log2(r + 2) for r in range(k))
    if idcg == 0.0:
        return 0.0
    
    ndcgs = []
    for i in range(len(query_labels)):
        q_label = query_labels[i]
        n_labels = neighbor_labels[i, :k]
        
        dcg = 0.0
        for rank_idx, val in enumerate(n_labels):
            if val == q_label:
                dcg += 1.0 / np.log2(rank_idx + 2)
                
        ndcgs.append(dcg / idcg)
        
    return float(np.mean(ndcgs))


def compute_cmc_curve(neighbor_labels: np.ndarray, query_labels: np.ndarray, max_k: int = 10) -> List[float]:
    """Computes Cumulative Match Characteristic (CMC) curve from Rank-1 to Rank-max_k.

    Args:
        neighbor_labels: Integer array of shape (M, K) containing neighbor labels.
        query_labels: Integer array of shape (M,) containing query labels.
        max_k: Maximum rank limit to evaluate.

    Returns:
        List[float]: Recall rates from Rank-1 to Rank-max_k.
    """
    cmc = []
    for k in range(1, max_k + 1):
        cmc.append(compute_recall_at_k(neighbor_labels, query_labels, k))
    return cmc


def bootstrap_metrics(
    neighbor_labels: np.ndarray,
    query_labels: np.ndarray,
    top_k: int = 5,
    num_bootstraps: int = 1000,
    seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """Performs bootstrap resampling to estimate standard errors and 95% Confidence Intervals.

    Args:
        neighbor_labels: Integer array of shape (M, K) containing neighbor labels.
        query_labels: Integer array of shape (M,) containing query labels.
        top_k: Top-K evaluation parameter for NDCG and Recall.
        num_bootstraps: Number of bootstrap iterations.
        seed: Random seed for reproducibility.

    Returns:
        Dict[str, Dict[str, float]]: Standard deviations and CI limits for Recall@1, Recall@5, NDCG, and MRR.
    """
    rng = np.random.default_rng(seed)
    M = len(query_labels)
    if M == 0:
        return {}

    boot_recall1 = []
    boot_recall5 = []
    boot_ndcg = []
    boot_mrr = []

    for _ in range(num_bootstraps):
        boot_idx = rng.choice(M, size=M, replace=True)
        boot_queries = query_labels[boot_idx]
        boot_neighbors = neighbor_labels[boot_idx]

        boot_recall1.append(compute_recall_at_k(boot_neighbors, boot_queries, 1))
        boot_recall5.append(compute_recall_at_k(boot_neighbors, boot_queries, top_k))
        boot_ndcg.append(compute_ndcg_at_k(boot_neighbors, boot_queries, top_k))
        boot_mrr.append(compute_mrr(boot_neighbors, boot_queries))

    metrics_ci = {}
    for name, vals in [("recall@1", boot_recall1), ("recall@5", boot_recall5), ("ndcg", boot_ndcg), ("mrr", boot_mrr)]:
        metrics_ci[name] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "ci_lower": float(np.percentile(vals, 2.5)),
            "ci_upper": float(np.percentile(vals, 97.5))
        }
    return metrics_ci


def calibrate_similarity_threshold(
    top_scores: np.ndarray,
    top_correct: np.ndarray,
    target_precision: float = 0.95
) -> Tuple[float, float]:
    """Calibrates similarity thresholds to satisfy a target precision constraint.

    Args:
        top_scores: Cosine similarities of the top predictions, shape (M,).
        top_correct: Boolean array indicating if the top prediction is correct, shape (M,).
        target_precision: Target precision rate constraint (e.g. 0.95).

    Returns:
        Tuple[float, float]: Calibrated threshold and corresponding automation rate.
    """
    if len(top_scores) == 0:
        return 1.0, 0.0

    # Sweep thresholds at 0.005 granularity
    thresholds = np.linspace(0.0, 1.0, 201)
    valid_thresholds = []

    for tau in thresholds:
        mask = (top_scores >= tau)
        num_auto = np.sum(mask)
        
        if num_auto == 0:
            precision = 1.0
        else:
            precision = np.sum(top_correct[mask]) / num_auto

        auto_rate = num_auto / len(top_scores)
        
        if precision >= target_precision:
            valid_thresholds.append((tau, auto_rate))

    if valid_thresholds:
        # Select the lowest threshold that satisfies the constraint
        valid_thresholds.sort(key=lambda x: x[0])
        best_tau, best_auto_rate = valid_thresholds[0]
    else:
        # Fall back to highest threshold if constraint is unreachable
        best_tau = 1.0
        best_auto_rate = 0.0

    return float(best_tau), float(best_auto_rate)
