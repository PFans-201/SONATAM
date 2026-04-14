"""
evaluate.py
===========
Evaluation utilities for GraphSAGE link prediction on the SONATAM
knowledge graph.

Provides:

* ``evaluate_link_prediction()`` -- full evaluation on a test split
* ``compute_mrr()``             -- Mean Reciprocal Rank
* ``compute_hits_at_k()``       -- Hits@K
* ``plot_training_curves()``    -- loss / AUC over epochs
* ``plot_embedding_tsne()``     -- t-SNE of node embeddings
* ``plot_score_distribution()`` -- histogram of predicted edge scores
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

__all__ = [
    "evaluate_link_prediction",
    "compute_mrr",
    "compute_hits_at_k",
    "plot_training_curves",
    "plot_embedding_tsne",
    "plot_score_distribution",
]

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
#  Core evaluation
# --------------------------------------------------------------------------

@torch.no_grad()
def evaluate_link_prediction(
    model,
    data,
    edge_type: Tuple[str, str, str] = ("MusicalPiece", "hasGenre", "Genre"),
    ks: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Evaluate a trained link prediction model on a data split.

    Parameters
    ----------
    model : GraphSAGELinkPredModel
        A trained model.
    data : HeteroData
        A data split (e.g. the test split from ``create_link_split``).
    edge_type : tuple
        The target edge type with ``edge_label`` and ``edge_label_index``.
    ks : list[int], optional
        Values of K for Hits@K (default ``[1, 3, 5, 10]``).

    Returns
    -------
    dict
        Metrics: ``auc_roc``, ``avg_precision``, ``mrr``, ``hits@k``, ...
    """
    if ks is None:
        ks = [1, 3, 5, 10]

    model.eval()
    et = edge_type

    # Forward pass
    pred = model(
        data.x_dict,
        data.edge_index_dict,
        data[et].edge_label_index,
        src_type=et[0],
        dst_type=et[2],
    )
    probs = torch.sigmoid(pred).cpu().numpy()
    labels = data[et].edge_label.cpu().numpy()

    metrics: Dict[str, float] = {}

    # -- AUC-ROC + Average Precision --
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        metrics["auc_roc"] = float(roc_auc_score(labels, probs))
        metrics["avg_precision"] = float(average_precision_score(labels, probs))
    except Exception as exc:
        log.warning("sklearn metrics failed: %s", exc)
        metrics["auc_roc"] = 0.0
        metrics["avg_precision"] = 0.0

    # -- MRR --
    metrics["mrr"] = compute_mrr(probs, labels)

    # -- Hits@K --
    for k in ks:
        metrics[f"hits@{k}"] = compute_hits_at_k(probs, labels, k)

    return metrics


# --------------------------------------------------------------------------
#  Ranking metrics
# --------------------------------------------------------------------------

def compute_mrr(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).

    For each positive edge, rank it among all scored edges and compute
    1 / rank.  MRR is the mean of these reciprocal ranks.

    Parameters
    ----------
    scores : np.ndarray  (N,)
        Predicted probabilities / scores.
    labels : np.ndarray  (N,)
        Binary labels (1 = positive, 0 = negative).

    Returns
    -------
    float
        MRR in [0, 1].
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    pos_mask = labels == 1

    if pos_mask.sum() == 0:
        return 0.0

    # Rank all scores in descending order
    sorted_indices = np.argsort(-scores)
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(1, len(scores) + 1)

    pos_ranks = ranks[pos_mask]
    reciprocal_ranks = 1.0 / pos_ranks.astype(float)

    return float(reciprocal_ranks.mean())


def compute_hits_at_k(
    scores: np.ndarray,
    labels: np.ndarray,
    k: int = 10,
) -> float:
    """
    Compute Hits@K -- fraction of positive edges ranked in the top-K.

    Parameters
    ----------
    scores : np.ndarray  (N,)
    labels : np.ndarray  (N,)
    k : int

    Returns
    -------
    float
        Hits@K in [0, 1].
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    pos_mask = labels == 1
    n_pos = pos_mask.sum()

    if n_pos == 0:
        return 0.0

    # Top-K indices by descending score
    top_k_indices = np.argsort(-scores)[:k]
    hits = labels[top_k_indices].sum()

    return float(hits / n_pos)


# --------------------------------------------------------------------------
#  Visualisation
# --------------------------------------------------------------------------

def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
) -> None:
    """
    Plot training loss and validation AUC over epochs.

    Parameters
    ----------
    history : dict
        Must contain ``train_loss`` and ``val_auc`` lists.
    save_path : str, optional
        If given, save the figure instead of showing it.
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax1.plot(epochs, history["train_loss"], "b-", label="Train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # AUC
    if "val_auc" in history:
        ax2.plot(epochs, history["val_auc"], "r-", label="Val AUC")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("AUC-ROC")
        ax2.set_title("Validation AUC")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved training curves -> {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_embedding_tsne(
    z_dict: Dict[str, torch.Tensor],
    node_type: str = "MusicalPiece",
    labels: Optional[np.ndarray] = None,
    n_samples: int = 2000,
    save_path: Optional[str] = None,
) -> None:
    """
    Visualise node embeddings with t-SNE.

    Parameters
    ----------
    z_dict : dict[str, Tensor]
        Node embeddings per type (output of ``model.encode()``).
    node_type : str
        Which node type to visualise.
    labels : np.ndarray, optional
        Colour-coding labels (e.g. genre IDs).
    n_samples : int
        Max samples to plot (for speed).
    save_path : str, optional
        Save figure path.
    """
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    z = z_dict[node_type].cpu().numpy()
    if len(z) > n_samples:
        idx = np.random.choice(len(z), n_samples, replace=False)
        z = z[idx]
        if labels is not None:
            labels = np.asarray(labels)[idx]

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    coords = tsne.fit_transform(z)

    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=labels, cmap="tab20", s=5, alpha=0.6,
    )
    if labels is not None:
        plt.colorbar(scatter, ax=ax, fraction=0.03)
    ax.set_title(f"t-SNE -- {node_type} embeddings")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved t-SNE plot -> {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_score_distribution(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot histograms of predicted scores for positive and negative edges.

    Parameters
    ----------
    pos_scores, neg_scores : np.ndarray
        Predicted probabilities for true and false edges.
    save_path : str, optional
        Save figure path.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(neg_scores, bins=50, alpha=0.6, label="Negative", color="steelblue", density=True)
    ax.hist(pos_scores, bins=50, alpha=0.6, label="Positive", color="coral", density=True)
    ax.set_xlabel("Predicted score")
    ax.set_ylabel("Density")
    ax.set_title("Edge Score Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved score distribution -> {save_path}")
    else:
        plt.show()
    plt.close(fig)
