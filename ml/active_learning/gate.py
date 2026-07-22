"""Statistical promotion gate for Pipeline 3 (Champion vs Challenger).

Nothing reaches production without clearing this gate. A challenger — a
curated gallery, a promoted review batch, or a SupCon projection head —
must beat the incumbent by a margin that survives a paired bootstrap, and
must not have seen the test set.

Three design points that differ from a naive gate:

1. **Paired, not independent.** Both systems are measured on the same
   queries, so the bootstrap resamples query indices *jointly* and tests the
   distribution of the difference. Comparing a challenger's confidence
   interval against a champion's point estimate would treat the incumbent as
   noiseless and throw away the variance cancellation that pairing buys.

2. **Promotion turns on automation rate, not Top-1.** A projected embedding
   space has a different similarity distribution, which silently invalidates
   the Platt coefficients and the cosine gating bands. Top-1 accuracy cannot
   see that failure; automation-rate-at-target-precision can. Each system is
   therefore recalibrated on a validation split before being scored.

3. **Leakage is a hard failure.** Continual learning pushes reviewed shelf
   crops into the gallery. If those crops came from shelf images that also
   supply test queries, the challenger is being graded on its own training
   data and will promote itself on inflated numbers.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional, Iterable, Set
import numpy as np
from pydantic import BaseModel, Field

from ml.evaluation.metrics import calibrate_similarity_threshold


# Minimum Top-1 gain the challenger must clear. Not merely > 0: the gate
# fires repeatedly against one test set, so demanding a real effect size
# guards against eventually promoting noise.
DEFAULT_MIN_TOP1_GAIN = 0.005

# Largest tolerable regression for the two "must not get worse" criteria.
#
# These are non-inferiority tests, not superiority tests. Requiring a CI
# lower bound at or above zero would demand *proof of improvement*, and with
# Top-5 recall already near its ceiling (~98.7%) the sampling noise is wider
# than any plausible gain — a challenger that is genuinely flat or slightly
# better would be rejected about half the time. Allowing the bound down to
# -margin asks the right question: is the regression provably smaller than
# what we are willing to lose?
DEFAULT_MAX_TOP5_REGRESSION = 0.005
DEFAULT_MAX_AUTOMATION_REGRESSION = 0.005

DEFAULT_TARGET_PRECISION = 0.95
DEFAULT_N_BOOTSTRAP = 1000
DEFAULT_SEED = 42

# Probability assigned when a calibration split contains only one outcome,
# making a logistic fit degenerate. |z| = 20 saturates the sigmoid.
_DEGENERATE_LOGIT = 20.0


class LeakageError(RuntimeError):
    """Raised when the challenger gallery overlaps the evaluation set."""


class DeltaCI(BaseModel):
    """Paired bootstrap confidence interval for a challenger-minus-champion delta."""
    champion_mean: float = Field(..., description="Champion metric on the full test set.")
    challenger_mean: float = Field(..., description="Challenger metric on the full test set.")
    delta_mean: float = Field(..., description="Observed challenger - champion.")
    ci_lower: float = Field(..., description="2.5th percentile of the bootstrapped delta.")
    ci_upper: float = Field(..., description="97.5th percentile of the bootstrapped delta.")


class CriterionResult(BaseModel):
    """Outcome of one promotion criterion."""
    name: str = Field(..., description="Criterion identifier.")
    passed: bool = Field(..., description="Whether the criterion was satisfied.")
    observed: Optional[float] = Field(None, description="Observed statistic (usually a CI lower bound).")
    required: Optional[float] = Field(None, description="Threshold the statistic had to clear.")
    underpowered: bool = Field(
        False,
        description="True if sampling noise exceeds the margin, so the test cannot resolve it.",
    )
    detail: str = Field("", description="Human-readable explanation.")


class GateDecision(BaseModel):
    """Full promotion verdict."""
    promoted: bool = Field(..., description="True only if every criterion passed.")
    champion_name: str = Field(..., description="Incumbent system identifier.")
    challenger_name: str = Field(..., description="Candidate system identifier.")
    criteria: List[CriterionResult] = Field(default_factory=list)
    top1: Optional[DeltaCI] = Field(None, description="Paired delta for Top-1 accuracy.")
    top5: Optional[DeltaCI] = Field(None, description="Paired delta for Top-5 recall.")
    automation: Optional[DeltaCI] = Field(None, description="Paired delta for automation rate.")
    champion_platt: Optional[Tuple[float, float]] = Field(None, description="Champion (a, b) after refit.")
    challenger_platt: Optional[Tuple[float, float]] = Field(None, description="Challenger (a, b) after refit.")
    champion_threshold: Optional[float] = Field(None, description="Champion operating point on calibrated probability.")
    challenger_threshold: Optional[float] = Field(None, description="Challenger operating point.")
    champion_test_precision: Optional[float] = Field(None, description="Champion realized precision on test.")
    challenger_test_precision: Optional[float] = Field(None, description="Challenger realized precision on test.")
    n_test_queries: int = Field(0, description="Paired test queries evaluated.")

    def summary(self) -> str:
        """Renders a one-line-per-criterion verdict for logs and CLI output."""
        verdict = "PROMOTE" if self.promoted else "REJECT"
        lines = [
            f"[{verdict}] {self.challenger_name} vs {self.champion_name} "
            f"({self.n_test_queries} paired test queries)"
        ]
        for criterion in self.criteria:
            mark = "PASS" if criterion.passed else "FAIL"
            lines.append(f"  {mark}  {criterion.name}: {criterion.detail}")
        return "\n".join(lines)


@dataclass
class SystemEvaluation:
    """Retrieval output for one system across a validation and a test split.

    The validation split exists purely to fit the calibrator and choose an
    operating point. Doing either on the test split would let the system
    tune against the data it is graded on.

    Attributes:
        name: System identifier.
        val_neighbor_labels: (Mv, K) retrieved class labels, validation.
        val_query_labels: (Mv,) true labels, validation.
        val_top_scores: (Mv,) rank-1 similarity, validation.
        test_neighbor_labels: (Mt, K) retrieved class labels, test.
        test_query_labels: (Mt,) true labels, test.
        test_top_scores: (Mt,) rank-1 similarity, test.
    """
    name: str
    val_neighbor_labels: np.ndarray
    val_query_labels: np.ndarray
    val_top_scores: np.ndarray
    test_neighbor_labels: np.ndarray
    test_query_labels: np.ndarray
    test_top_scores: np.ndarray

    def __post_init__(self) -> None:
        for split in ("val", "test"):
            neighbors = getattr(self, f"{split}_neighbor_labels")
            labels = getattr(self, f"{split}_query_labels")
            scores = getattr(self, f"{split}_top_scores")

            if neighbors.ndim != 2:
                raise ValueError(
                    f"{self.name}: {split}_neighbor_labels must be 2-D (M, K), "
                    f"got shape {neighbors.shape}."
                )
            if not (neighbors.shape[0] == labels.shape[0] == scores.shape[0]):
                raise ValueError(
                    f"{self.name}: {split} split row counts disagree — "
                    f"neighbors {neighbors.shape[0]}, labels {labels.shape[0]}, "
                    f"scores {scores.shape[0]}."
                )


# ── Per-query indicators ─────────────────────────────────────────────
# Every metric the gate uses reduces to a per-query 0/1 indicator, which is
# what makes a single paired bootstrap sufficient for all of them.

def recall_indicators(
    neighbor_labels: np.ndarray,
    query_labels: np.ndarray,
    k: int,
) -> np.ndarray:
    """Returns a (M,) float array: 1.0 where the true label is in the top k."""
    if k <= 0:
        raise ValueError("k must be positive.")
    top_k = neighbor_labels[:, :k]
    return (top_k == query_labels[:, None]).any(axis=1).astype(np.float64)


def top1_correct(neighbor_labels: np.ndarray, query_labels: np.ndarray) -> np.ndarray:
    """Returns a (M,) boolean array of rank-1 correctness."""
    return neighbor_labels[:, 0] == query_labels


# ── Calibration ──────────────────────────────────────────────────────

def fit_platt(similarities: np.ndarray, correct: np.ndarray) -> Tuple[float, float]:
    """Fits Platt scaling coefficients mapping similarity to probability.

    Args:
        similarities: (M,) rank-1 cosine similarities.
        correct: (M,) boolean rank-1 correctness.

    Returns:
        Tuple[float, float]: (a, b) for z = a * similarity + b, matching the
        form PlattCalibrator applies in production.
    """
    from sklearn.linear_model import LogisticRegression

    sims = np.asarray(similarities, dtype=np.float64).reshape(-1, 1)
    labels = np.asarray(correct).astype(int).ravel()

    if sims.shape[0] == 0:
        raise ValueError("Cannot fit Platt scaling on an empty split.")

    # A split with one outcome has no decision boundary to fit. Return a
    # saturated constant rather than letting sklearn raise.
    unique = np.unique(labels)
    if unique.shape[0] < 2:
        return 0.0, (_DEGENERATE_LOGIT if unique[0] == 1 else -_DEGENERATE_LOGIT)

    model = LogisticRegression(solver="lbfgs")
    model.fit(sims, labels)
    return float(model.coef_[0][0]), float(model.intercept_[0])


def platt_probabilities(similarities: np.ndarray, a: float, b: float) -> np.ndarray:
    """Applies Platt scaling to an array of similarities.

    Vectorized equivalent of PlattCalibrator.calibrate, including its
    overflow clipping, so gate estimates match production behaviour.
    """
    z = a * np.asarray(similarities, dtype=np.float64) + b
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


def _operating_point(
    probabilities: np.ndarray,
    correct: np.ndarray,
    target_precision: float,
) -> float:
    """Lowest calibrated-probability threshold meeting the precision target."""
    threshold, _ = calibrate_similarity_threshold(
        probabilities, np.asarray(correct, dtype=bool), target_precision=target_precision
    )
    return float(threshold)


# ── Paired bootstrap ─────────────────────────────────────────────────

def paired_bootstrap_delta(
    champion_per_query: np.ndarray,
    challenger_per_query: np.ndarray,
    n_boot: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> DeltaCI:
    """Bootstraps the challenger-minus-champion difference on paired queries.

    Both arrays must be per-query values for the *same* queries in the *same*
    order. Each replicate resamples query indices once and applies them to
    both systems, so the shared query-difficulty variance cancels — a
    strictly tighter interval than bootstrapping the two systems separately.

    Args:
        champion_per_query: (M,) per-query metric values for the incumbent.
        challenger_per_query: (M,) per-query metric values for the candidate.
        n_boot: Bootstrap replicates.
        seed: RNG seed, for reproducibility.

    Returns:
        DeltaCI: Observed means, the observed delta, and its 95% interval.

    Raises:
        ValueError: If the arrays differ in length or are empty.
    """
    champion = np.asarray(champion_per_query, dtype=np.float64).ravel()
    challenger = np.asarray(challenger_per_query, dtype=np.float64).ravel()

    if champion.shape != challenger.shape:
        raise ValueError(
            f"Paired arrays must have equal length, got {champion.shape} and {challenger.shape}."
        )
    if champion.shape[0] == 0:
        raise ValueError("Cannot bootstrap an empty evaluation set.")

    n_queries = champion.shape[0]
    rng = np.random.default_rng(seed)

    # One index draw per replicate, applied to both systems — this is the
    # pairing.
    idx = rng.integers(0, n_queries, size=(n_boot, n_queries))
    deltas = challenger[idx].mean(axis=1) - champion[idx].mean(axis=1)

    return DeltaCI(
        champion_mean=float(champion.mean()),
        challenger_mean=float(challenger.mean()),
        delta_mean=float(challenger.mean() - champion.mean()),
        ci_lower=float(np.percentile(deltas, 2.5)),
        ci_upper=float(np.percentile(deltas, 97.5)),
    )


# ── Leakage control ──────────────────────────────────────────────────

def assert_gallery_test_disjoint(
    gallery_source_images: Iterable[str],
    test_source_images: Iterable[str],
) -> None:
    """Raises if any shelf image supplies both gallery references and test queries.

    Args:
        gallery_source_images: source_image_name values in the challenger gallery.
        test_source_images: source images backing the test queries.

    Raises:
        LeakageError: On any overlap, naming offending images.
    """
    gallery: Set[str] = {str(name) for name in gallery_source_images}
    test: Set[str] = {str(name) for name in test_source_images}

    overlap = sorted(gallery & test)
    if overlap:
        shown = ", ".join(overlap[:10])
        suffix = f" (+{len(overlap) - 10} more)" if len(overlap) > 10 else ""
        raise LeakageError(
            f"{len(overlap)} shelf image(s) appear in both the challenger gallery "
            f"and the test set: {shown}{suffix}. The challenger would be graded "
            f"on its own reference crops; re-split before gating."
        )


# ── The gate ─────────────────────────────────────────────────────────

def evaluate_promotion(
    champion: SystemEvaluation,
    challenger: SystemEvaluation,
    gallery_source_images: Optional[Iterable[str]] = None,
    test_source_images: Optional[Iterable[str]] = None,
    target_precision: float = DEFAULT_TARGET_PRECISION,
    min_top1_gain: float = DEFAULT_MIN_TOP1_GAIN,
    max_top5_regression: float = DEFAULT_MAX_TOP5_REGRESSION,
    max_automation_regression: float = DEFAULT_MAX_AUTOMATION_REGRESSION,
    n_boot: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> GateDecision:
    """Decides whether a challenger may replace the champion.

    Promotion requires all of:
      1. Top-1 delta CI lower bound above `min_top1_gain` (superiority).
      2. Top-5 delta CI lower bound above `-max_top5_regression`
         (non-inferiority).
      3. Automation-rate delta CI lower bound above
         `-max_automation_regression`, measured after recalibrating each
         system on its validation split (non-inferiority).
      4. No overlap between the challenger gallery and the test set.

    Args:
        champion: Incumbent evaluation across validation and test splits.
        challenger: Candidate evaluation over the *same* test queries.
        gallery_source_images: Shelf images backing the challenger gallery.
        test_source_images: Shelf images backing the test queries. Supply
            both to enable the leakage check.
        target_precision: Precision constraint defining the operating point.
        min_top1_gain: Minimum Top-1 improvement required.
        max_top5_regression: Largest tolerable Top-5 recall loss.
        max_automation_regression: Largest tolerable automation rate loss.
        n_boot: Bootstrap replicates.
        seed: RNG seed.

    Returns:
        GateDecision: Verdict plus every intermediate statistic.

    Raises:
        LeakageError: If the challenger gallery overlaps the test set.
        ValueError: If the two systems were not evaluated on identical,
            identically-ordered test queries.
    """
    # 1. Leakage first — a contaminated comparison is not worth computing.
    if gallery_source_images is not None and test_source_images is not None:
        assert_gallery_test_disjoint(gallery_source_images, test_source_images)

    # 2. Pairing is a precondition, not a nicety: the bootstrap below applies
    # one index draw to both systems, which is only meaningful if row i
    # refers to the same query in each.
    if champion.test_query_labels.shape != challenger.test_query_labels.shape:
        raise ValueError(
            f"Champion and challenger must share a test set: got "
            f"{champion.test_query_labels.shape[0]} vs "
            f"{challenger.test_query_labels.shape[0]} queries."
        )
    if not np.array_equal(champion.test_query_labels, challenger.test_query_labels):
        raise ValueError(
            "Champion and challenger test query labels differ. The paired "
            "bootstrap requires identical queries in identical order."
        )

    criteria: List[CriterionResult] = []

    # 3. Top-1 accuracy.
    top1 = paired_bootstrap_delta(
        recall_indicators(champion.test_neighbor_labels, champion.test_query_labels, 1),
        recall_indicators(challenger.test_neighbor_labels, challenger.test_query_labels, 1),
        n_boot=n_boot, seed=seed,
    )
    criteria.append(CriterionResult(
        name="top1_gain",
        passed=top1.ci_lower > min_top1_gain,
        observed=top1.ci_lower,
        required=min_top1_gain,
        detail=(
            f"Top-1 {top1.champion_mean:.4f} -> {top1.challenger_mean:.4f} "
            f"(delta {top1.delta_mean:+.4f}, CI95 [{top1.ci_lower:+.4f}, {top1.ci_upper:+.4f}]); "
            f"requires CI lower > {min_top1_gain:+.4f}"
        ),
    ))

    # 4. Top-5 recall must not regress — it is the reranker's input.
    top5 = paired_bootstrap_delta(
        recall_indicators(champion.test_neighbor_labels, champion.test_query_labels, 5),
        recall_indicators(challenger.test_neighbor_labels, challenger.test_query_labels, 5),
        n_boot=n_boot, seed=seed,
    )
    top5_underpowered = _underpowered(top5, max_top5_regression)
    criteria.append(CriterionResult(
        name="top5_no_regression",
        passed=top5.ci_lower > -max_top5_regression,
        observed=top5.ci_lower,
        required=-max_top5_regression,
        underpowered=top5_underpowered,
        detail=(
            f"Top-5 {top5.champion_mean:.4f} -> {top5.challenger_mean:.4f} "
            f"(delta {top5.delta_mean:+.4f}, CI95 [{top5.ci_lower:+.4f}, {top5.ci_upper:+.4f}]); "
            f"requires CI lower > {-max_top5_regression:+.4f}"
            + (
                f" [UNDERPOWERED: CI half-width exceeds the margin at "
                f"{champion.test_query_labels.shape[0]} queries — enlarge the "
                f"evaluation set or widen max_top5_regression]"
                if top5_underpowered else ""
            )
        ),
    ))

    # 5. Automation rate at target precision, each system on its own refit
    # calibrator. This is the business metric and the only criterion that
    # detects a broken calibration.
    champ_auto, champ_platt, champ_tau, champ_precision = _automation_indicators(
        champion, target_precision
    )
    chall_auto, chall_platt, chall_tau, chall_precision = _automation_indicators(
        challenger, target_precision
    )

    automation = paired_bootstrap_delta(
        champ_auto, chall_auto, n_boot=n_boot, seed=seed
    )
    automation_underpowered = _underpowered(automation, max_automation_regression)
    criteria.append(CriterionResult(
        name="automation_rate_no_regression",
        passed=automation.ci_lower > -max_automation_regression,
        observed=automation.ci_lower,
        required=-max_automation_regression,
        underpowered=automation_underpowered,
        detail=(
            f"Automation@{target_precision:.0%}p "
            f"{automation.champion_mean:.4f} -> {automation.challenger_mean:.4f} "
            f"(delta {automation.delta_mean:+.4f}, "
            f"CI95 [{automation.ci_lower:+.4f}, {automation.ci_upper:+.4f}]); "
            f"realized test precision {champ_precision:.4f} -> {chall_precision:.4f}; "
            f"requires CI lower > {-max_automation_regression:+.4f}"
            + (
                f" [UNDERPOWERED: CI half-width exceeds the margin at "
                f"{champion.test_query_labels.shape[0]} queries — enlarge the "
                f"evaluation set or widen max_automation_regression]"
                if automation_underpowered else ""
            )
        ),
    ))

    return GateDecision(
        promoted=all(c.passed for c in criteria),
        champion_name=champion.name,
        challenger_name=challenger.name,
        criteria=criteria,
        top1=top1,
        top5=top5,
        automation=automation,
        champion_platt=champ_platt,
        challenger_platt=chall_platt,
        champion_threshold=champ_tau,
        challenger_threshold=chall_tau,
        champion_test_precision=champ_precision,
        challenger_test_precision=chall_precision,
        n_test_queries=int(champion.test_query_labels.shape[0]),
    )


def _underpowered(delta: DeltaCI, margin: float) -> bool:
    """True when a failure is inconclusive rather than a proven regression.

    The interval straddles the decision boundary: its lower bound falls
    below -margin (so the criterion fails) while its upper bound sits above
    -margin (so acceptable performance is still consistent with the data).
    That is a shortage of evidence, not evidence of harm — the fix is a
    larger evaluation set, not a different challenger.

    A regression whose entire interval lies below -margin is conclusively
    real and is emphatically *not* underpowered, however wide the interval.
    """
    return delta.ci_lower <= -margin < delta.ci_upper


def _automation_indicators(
    system: SystemEvaluation,
    target_precision: float,
) -> Tuple[np.ndarray, Tuple[float, float], float, float]:
    """Recalibrates a system and returns its per-query automation decisions.

    Platt coefficients and the operating point are both fitted on the
    validation split, then applied unchanged to test. Choosing the threshold
    on test would be circular and would inflate the measured automation rate.

    Returns:
        Tuple of (per-query automated flags on test, (a, b), threshold,
        realized test precision among automated queries).
    """
    val_correct = top1_correct(system.val_neighbor_labels, system.val_query_labels)
    a, b = fit_platt(system.val_top_scores, val_correct)

    val_probs = platt_probabilities(system.val_top_scores, a, b)
    threshold = _operating_point(val_probs, val_correct, target_precision)

    test_correct = top1_correct(system.test_neighbor_labels, system.test_query_labels)
    test_probs = platt_probabilities(system.test_top_scores, a, b)
    automated = test_probs >= threshold

    n_automated = int(automated.sum())
    precision = float(test_correct[automated].mean()) if n_automated else 1.0

    return automated.astype(np.float64), (a, b), threshold, precision
