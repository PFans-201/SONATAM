"""
evaluate.py
===========
Evaluation utilities for trained classifiers.

Functions
---------
classification_report_df(model, loader, idx2label, device)
    Run inference and return a detailed per-class DataFrame.

confusion_matrix_plot(model, loader, idx2label, device)
    Plot a seaborn confusion-matrix heatmap.

tsne_plot(model, loader, idx2label, device, layer_name)
    Extract penultimate embeddings and plot a 2-D t-SNE scatter.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

__all__ = ["classification_report_df", "confusion_matrix_plot", "tsne_plot"]


def _run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple:
    """Return (y_true, y_pred) numpy arrays."""
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_true.extend(y.numpy())
            all_pred.extend(preds)
    return np.array(all_true), np.array(all_pred)


def classification_report_df(
    model: nn.Module,
    loader: DataLoader,
    idx2label: Dict[int, str],
    device: str = "cpu",
) -> "pd.DataFrame":  # type: ignore[name-defined]
    """
    Run inference and return a per-class metrics DataFrame.

    Returns
    -------
    pd.DataFrame with columns: class, precision, recall, f1, support.
    """
    import pandas as pd
    from sklearn.metrics import classification_report

    y_true, y_pred = _run_inference(model, loader, device)
    target_names   = [idx2label[i] for i in sorted(idx2label)]
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    return pd.DataFrame(report).T


def confusion_matrix_plot(
    model: nn.Module,
    loader: DataLoader,
    idx2label: Dict[int, str],
    device: str = "cpu",
    figsize: tuple = (12, 10),
    save_path: Optional[str] = None,
) -> None:
    """Plot a normalised confusion-matrix heatmap (seaborn)."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    y_true, y_pred = _run_inference(model, loader, device)
    labels = [idx2label[i] for i in sorted(idx2label)]
    cm     = confusion_matrix(y_true, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt=".2f",
        xticklabels=labels, yticklabels=labels,
        cmap="Blues", ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalised Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def tsne_plot(
    model: nn.Module,
    loader: DataLoader,
    idx2label: Dict[int, str],
    device: str = "cpu",
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None,
    perplexity: int = 30,
) -> None:
    """
    Extract the penultimate layer's embeddings, reduce to 2-D with t-SNE,
    and plot a colour-coded scatter plot.

    Works for GenreClassifier (accesses ``model.net[:-1]``) and
    ChordTransformer (uses the Transformer output before the head).
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    model.eval()
    embeddings_list: List[np.ndarray] = []
    labels_list:     List[int] = []

    # Register a hook on the last hidden layer
    hook_out: List[torch.Tensor] = []

    def _hook(module, inp, out):
        hook_out.append(out.detach().cpu())

    # Determine hook target
    try:
        # GenreClassifier exposes model.net
        handle = model.net[-2].register_forward_hook(_hook)   # type: ignore[attr-defined]
    except AttributeError:
        # ChordTransformer — hook on transformer output
        handle = model.transformer.register_forward_hook(_hook)  # type: ignore[attr-defined]

    with torch.no_grad():
        for x, y in loader:
            hook_out.clear()
            model(x.to(device))
            if hook_out:
                emb = hook_out[0]
                if emb.dim() == 3:          # (B, L, d_model) — mean-pool
                    emb = emb.mean(dim=1)
                embeddings_list.append(emb.numpy())
                labels_list.extend(y.numpy())

    handle.remove()

    if not embeddings_list:
        print("No embeddings collected — check model architecture.")
        return

    X = np.vstack(embeddings_list)
    y = np.array(labels_list)

    print(f"Running t-SNE on {X.shape[0]} samples, {X.shape[1]} dims …")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X2   = tsne.fit_transform(X)

    unique_labels = sorted(set(y.tolist()))
    cmap = plt.cm.get_cmap("tab20", len(unique_labels))

    fig, ax = plt.subplots(figsize=figsize)
    for i, label_idx in enumerate(unique_labels):
        mask = y == label_idx
        ax.scatter(X2[mask, 0], X2[mask, 1], s=10, alpha=0.6,
                   color=cmap(i), label=idx2label.get(label_idx, str(label_idx)))
    ax.legend(markerscale=2, fontsize=7, loc="best", ncol=2)
    ax.set_title("t-SNE of Penultimate Embeddings")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
