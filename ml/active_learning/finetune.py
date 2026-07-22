"""SupCon projection head training for Pipeline 3. OPT-IN, DEFAULT OFF.

Trains a small MLP on cached DINOv3 embeddings so same-SKU crops pull
together and confusable variants push apart. No image I/O and no backbone
forward pass: the vectors were already computed during the audit and stored
on the review rows.

**This head is never promoted automatically.** Serving a projected space
changes the similarity distribution, which invalidates the Platt
coefficients and the cosine gating bands calibrated against raw DINOv3.
``gate.evaluate_promotion`` refits and re-measures before anything ships,
and a promotion carries its recalibrated coefficients with it.

Worth stating plainly: with Top-1 at 93.65% and Top-5 at 98.70%, retrieval
almost always *contains* the right answer, so the failure mode is ranking
within the Top-5 — which the Qwen2-VL reranker already addresses. A head
trained on a few hundred reviews is a real overfitting risk for a couple of
points. Prefer curation until the review corpus is large.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, Field

from ml.active_learning.hard_negatives import ConfusionReport


DEFAULT_IN_DIM = 768
DEFAULT_HIDDEN_DIM = 512
DEFAULT_OUT_DIM = 128
DEFAULT_TEMPERATURE = 0.07

# SupCon needs at least two samples of a class in a batch to have any
# positive pair at all.
MIN_SAMPLES_PER_CLASS = 2


class TrainingReport(BaseModel):
    """Outcome of a projection head training run."""
    epochs: int = Field(..., description="Epochs completed.")
    n_samples: int = Field(..., description="Embeddings trained on.")
    n_classes: int = Field(..., description="Distinct classes represented.")
    initial_loss: float = Field(..., description="Mean loss of the first epoch.")
    final_loss: float = Field(..., description="Mean loss of the last epoch.")
    loss_history: List[float] = Field(default_factory=list)
    checkpoint_path: Optional[str] = Field(None, description="Where the head was saved.")


class SupConProjectionHead(nn.Module):
    """2-layer MLP projecting backbone embeddings onto a unit hypersphere.

    Uses LayerNorm rather than the specification's BatchNorm1d. BatchNorm
    keeps running statistics that are unstable when estimated from a few
    hundred reviews, and it behaves differently in train and eval mode —
    precisely the silent train/serve drift this pipeline exists to prevent.
    LayerNorm normalizes per sample, so a vector projects identically
    regardless of batch composition or mode.
    """

    def __init__(
        self,
        in_dim: int = DEFAULT_IN_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        out_dim: int = DEFAULT_OUT_DIM,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns L2-normalized projections, matching the gallery convention."""
        x = self.relu(self.norm(self.fc1(x)))
        return F.normalize(self.fc2(x), p=2, dim=-1)

    @torch.inference_mode()
    def project_array(self, vectors: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """Projects a (N, in_dim) array to (N, out_dim), for re-indexing a gallery."""
        self.eval()
        outputs: List[np.ndarray] = []
        for start in range(0, len(vectors), batch_size):
            chunk = torch.from_numpy(
                np.asarray(vectors[start:start + batch_size], dtype=np.float32)
            )
            outputs.append(self(chunk).cpu().numpy().astype(np.float32))
        if not outputs:
            return np.empty((0, self.out_dim), dtype=np.float32)
        return np.concatenate(outputs, axis=0)


def supcon_loss(
    projections: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = DEFAULT_TEMPERATURE,
) -> torch.Tensor:
    """Supervised Contrastive loss (Khosla et al., 2020).

    For each anchor, every same-class sample in the batch is a positive and
    every other sample a negative — dense gradients without the fragile
    triplet mining that a margin loss would need.

    Args:
        projections: (B, D) L2-normalized projections.
        labels: (B,) integer class labels.
        temperature: Softmax temperature; lower sharpens the contrast.

    Returns:
        torch.Tensor: Scalar loss. Anchors with no positive in the batch are
        excluded, and a batch with no positive pairs at all returns 0.

    Raises:
        ValueError: If temperature is not positive, or shapes disagree.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if projections.dim() != 2:
        raise ValueError(f"projections must be 2-D (B, D), got {tuple(projections.shape)}.")
    if projections.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Batch mismatch: {projections.shape[0]} projections, {labels.shape[0]} labels."
        )

    batch_size = projections.shape[0]
    device = projections.device

    logits = projections @ projections.T / temperature
    # Subtract the row max before exponentiating — standard log-sum-exp
    # stabilization; detached so it contributes no gradient.
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
    positive_mask = (labels[:, None] == labels[None, :]) & ~self_mask

    exp_logits = torch.exp(logits).masked_fill(self_mask, 0.0)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    n_positives = positive_mask.sum(dim=1)
    valid = n_positives > 0
    if not valid.any():
        return projections.sum() * 0.0

    mean_log_prob = (
        (positive_mask * log_prob).sum(dim=1)[valid] / n_positives[valid]
    )
    return -mean_log_prob.mean()


def build_confusion_batches(
    labels: np.ndarray,
    batch_classes: int = 8,
    samples_per_class: int = 4,
    confusion: Optional[ConfusionReport] = None,
    seed: int = 42,
) -> List[np.ndarray]:
    """Builds SupCon batches, grouping confusable classes together.

    Every batch holds `samples_per_class` samples from each of up to
    `batch_classes` classes, so positives always exist. When a confusion
    report is supplied, batches are seeded from a class and its known
    confusers, putting the hardest negatives in the same softmax denominator
    as the positives — where the loss can actually act on them.

    Args:
        labels: (N,) class label per training sample.
        batch_classes: Classes per batch.
        samples_per_class: Samples drawn per class, with replacement when a
            class is short.
        confusion: Mined confusion structure. Falls back to random class
            groups when omitted.
        seed: RNG seed.

    Returns:
        List[np.ndarray]: Index arrays, one per batch.
    """
    if batch_classes < 2:
        raise ValueError("batch_classes must be at least 2 for contrastive learning.")
    if samples_per_class < MIN_SAMPLES_PER_CLASS:
        raise ValueError(
            f"samples_per_class must be at least {MIN_SAMPLES_PER_CLASS}; "
            f"SupCon needs a positive pair."
        )

    rng = np.random.default_rng(seed)
    by_class: Dict[int, np.ndarray] = {
        int(c): np.flatnonzero(labels == c) for c in np.unique(labels)
    }
    # A class with a single sample can still serve as a hard negative, but it
    # can never contribute a positive pair, so it cannot seed a batch.
    eligible = sorted(c for c, idx in by_class.items() if len(idx) >= MIN_SAMPLES_PER_CLASS)
    if not eligible:
        return []

    groups: List[Tuple[int, ...]] = []
    if confusion is not None:
        eligible_set = set(eligible)
        for group in confusion.confusion_groups():
            members = [c for c in group if c in eligible_set]
            if len(members) >= 2:
                groups.append(tuple(members))

    batches: List[np.ndarray] = []
    # One batch per confusion group, then random groups to cover classes the
    # confusion report never mentioned.
    covered: set = set()
    for group in groups:
        chosen = list(group[:batch_classes])
        covered.update(chosen)
        batches.append(_draw_batch(chosen, by_class, samples_per_class, rng))

    uncovered = [c for c in eligible if c not in covered]
    rng.shuffle(uncovered)
    for start in range(0, len(uncovered), batch_classes):
        chosen = uncovered[start:start + batch_classes]
        if len(chosen) < 2:
            # A trailing singleton has no negatives; pad from elsewhere.
            filler = [c for c in eligible if c not in chosen][:1]
            chosen = chosen + filler
        if len(chosen) >= 2:
            batches.append(_draw_batch(chosen, by_class, samples_per_class, rng))

    return batches


def _draw_batch(
    class_ids: Sequence[int],
    by_class: Dict[int, np.ndarray],
    samples_per_class: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Samples `samples_per_class` indices from each class, with replacement if short."""
    picked: List[int] = []
    for class_id in class_ids:
        pool = by_class[class_id]
        replace = len(pool) < samples_per_class
        picked.extend(rng.choice(pool, size=samples_per_class, replace=replace).tolist())
    return np.asarray(picked, dtype=int)


def train_projection_head(
    embeddings: np.ndarray,
    labels: np.ndarray,
    confusion: Optional[ConfusionReport] = None,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    temperature: float = DEFAULT_TEMPERATURE,
    batch_classes: int = 8,
    samples_per_class: int = 4,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    out_dim: int = DEFAULT_OUT_DIM,
    seed: int = 42,
    device: str = "cpu",
) -> Tuple[SupConProjectionHead, TrainingReport]:
    """Trains a SupCon projection head on cached embeddings.

    Args:
        embeddings: (N, D) backbone vectors.
        labels: (N,) verified class labels.
        confusion: Mined confusions steering batch construction.
        epochs: Passes over the batch schedule.
        learning_rate: Adam learning rate.
        temperature: SupCon temperature.
        batch_classes: Classes per batch.
        samples_per_class: Samples per class per batch.
        hidden_dim / out_dim: Head geometry.
        seed: Seed for batch sampling and weight init.
        device: Torch device.

    Returns:
        Tuple[SupConProjectionHead, TrainingReport]: Trained head and history.

    Raises:
        ValueError: If inputs disagree in length or no class has a positive pair.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(labels)

    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D (N, D), got {embeddings.shape}.")
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Length mismatch: {embeddings.shape[0]} embeddings, {labels.shape[0]} labels."
        )

    batches = build_confusion_batches(
        labels, batch_classes=batch_classes, samples_per_class=samples_per_class,
        confusion=confusion, seed=seed,
    )
    if not batches:
        raise ValueError(
            "No class has enough samples to form a positive pair. SupCon needs at "
            f"least {MIN_SAMPLES_PER_CLASS} reviews of the same class."
        )

    torch.manual_seed(seed)
    head = SupConProjectionHead(
        in_dim=embeddings.shape[1], hidden_dim=hidden_dim, out_dim=out_dim
    ).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate)

    tensor_x = torch.from_numpy(embeddings).to(device)
    tensor_y = torch.from_numpy(labels.astype(np.int64)).to(device)

    head.train()
    history: List[float] = []
    for _ in range(epochs):
        epoch_losses: List[float] = []
        for batch_idx in batches:
            idx = torch.from_numpy(batch_idx).to(device)
            optimizer.zero_grad()
            loss = supcon_loss(head(tensor_x[idx]), tensor_y[idx], temperature=temperature)
            if loss.requires_grad:
                loss.backward()
                optimizer.step()
            epoch_losses.append(float(loss.item()))
        history.append(float(np.mean(epoch_losses)))

    report = TrainingReport(
        epochs=epochs,
        n_samples=int(embeddings.shape[0]),
        n_classes=int(len(np.unique(labels))),
        initial_loss=history[0],
        final_loss=history[-1],
        loss_history=history,
    )
    return head, report


def save_checkpoint(
    head: SupConProjectionHead,
    path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Saves a head plus the metadata needed to rebuild and audit it.

    Geometry is stored alongside the weights so a checkpoint can be loaded
    without the caller having to remember how it was configured.
    """
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "state_dict": head.state_dict(),
        "in_dim": head.in_dim,
        "hidden_dim": head.hidden_dim,
        "out_dim": head.out_dim,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": metadata or {},
    }
    torch.save(payload, checkpoint_path)
    return str(checkpoint_path)


def load_checkpoint(
    path: str,
    device: str = "cpu",
) -> Tuple[SupConProjectionHead, Dict[str, Any]]:
    """Loads a head saved by save_checkpoint.

    Returns:
        Tuple[SupConProjectionHead, Dict[str, Any]]: Head in eval mode, and
        the stored payload minus the weights.
    """
    payload = torch.load(path, map_location=device, weights_only=False)

    head = SupConProjectionHead(
        in_dim=payload["in_dim"],
        hidden_dim=payload["hidden_dim"],
        out_dim=payload["out_dim"],
    )
    head.load_state_dict(payload["state_dict"])
    head.to(device).eval()

    info = {k: v for k, v in payload.items() if k != "state_dict"}
    return head, info
