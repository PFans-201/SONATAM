"""
train.py
========
Training loop for GraphSAGE link prediction on the SONATAM heterogeneous
knowledge graph.

Main components
---------------
TrainerConfig   -- dataclass holding all hyper-parameters
LinkPredTrainer -- orchestrates the training / validation loop

Usage
-----
::

    from sonata.models.train import LinkPredTrainer, TrainerConfig

    cfg = TrainerConfig.from_config()        # reads config.yaml
    trainer = LinkPredTrainer(cfg)
    history = trainer.fit(data)              # HeteroData (PyG)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from sonata.config.settings import CFG, resolve_path
from sonata.models.graph_models import GraphSAGELinkPredModel

__all__ = ["LinkPredTrainer", "TrainerConfig"]

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
#  Configuration dataclass
# --------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    """All hyper-parameters for link-prediction training."""

    # GraphSAGE architecture
    hidden_channels: int = 128
    num_sage_layers: int = 2
    num_pred_layers: int = 2
    pred_hidden: int = 64
    dropout: float = 0.3
    aggregator: str = "mean"

    # Training loop
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    clip_grad_norm: float = 1.0
    patience: int = 15
    save_every: int = 10

    # Data splits
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    neg_sampling_ratio: float = 1.0

    # Target edge type for link prediction
    target_edge_type: Tuple[str, str, str] = ("MusicalPiece", "hasGenre", "Genre")

    # System
    device: str = "auto"
    seed: int = 42
    checkpoint_dir: str = "checkpoints"

    @classmethod
    def from_config(cls, cfg: Optional[Dict] = None) -> "TrainerConfig":
        """
        Load training configuration from ``config.yaml``.

        Falls back to defaults for any missing keys.
        """
        c = cfg or CFG
        gl = c.get("graph_learning", {})
        gs = gl.get("graphsage", {})
        lp = gl.get("link_predictor", {})
        tr = gl.get("training", {})

        return cls(
            hidden_channels=gs.get("hidden_channels", 128),
            num_sage_layers=gs.get("num_layers", 2),
            num_pred_layers=lp.get("num_layers", 2),
            pred_hidden=lp.get("hidden_channels", 64),
            dropout=gs.get("dropout", 0.3),
            aggregator=gs.get("aggregator", "mean"),
            epochs=tr.get("epochs", 100),
            lr=tr.get("lr", 1e-3),
            weight_decay=tr.get("weight_decay", 1e-4),
            batch_size=tr.get("batch_size", 512),
            clip_grad_norm=tr.get("clip_grad_norm", 1.0),
            patience=tr.get("patience", 15),
            save_every=tr.get("save_every", 10),
            val_ratio=tr.get("val_frac", 0.15),
            test_ratio=tr.get("test_frac", 0.15),
            neg_sampling_ratio=tr.get("negative_sampling_ratio", 1.0),
            device=tr.get("device", "auto"),
            seed=tr.get("seed", 42),
            checkpoint_dir=gl.get("checkpoint_dir", "checkpoints"),
        )

    @property
    def resolved_device(self) -> torch.device:
        """Return the resolved torch device."""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


# --------------------------------------------------------------------------
#  Trainer
# --------------------------------------------------------------------------

class LinkPredTrainer:
    """
    Train a :class:`GraphSAGELinkPredModel` for link prediction.

    Parameters
    ----------
    config : TrainerConfig
        Hyper-parameter configuration.
    """

    def __init__(self, config: Optional[TrainerConfig] = None) -> None:
        self.cfg = config or TrainerConfig.from_config()
        self.model: Optional[GraphSAGELinkPredModel] = None
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_auc": [],
        }

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def fit(self, data, verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the model on a HeteroData object.

        Parameters
        ----------
        data : torch_geometric.data.HeteroData
            The full heterogeneous graph.  A link split is created
            internally based on ``self.cfg.target_edge_type``.
        verbose : bool
            Print epoch-level logs.

        Returns
        -------
        dict
            Training history with keys ``train_loss``, ``val_auc``.
        """
        from sonata.kg.converter import HeteroGraphConverter

        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        device = self.cfg.resolved_device
        et = self.cfg.target_edge_type

        # -- Link split --
        converter = HeteroGraphConverter()
        splits = converter.create_link_split(
            data,
            edge_type=et,
            val_ratio=self.cfg.val_ratio,
            test_ratio=self.cfg.test_ratio,
            neg_sampling_ratio=self.cfg.neg_sampling_ratio,
            seed=self.cfg.seed,
        )
        train_data = splits["train"].to(device)
        val_data = splits["val"].to(device)

        # -- Build model --
        in_channels_dict = {}
        for nt in data.node_types:
            if hasattr(data[nt], "x") and data[nt].x is not None:
                in_channels_dict[nt] = data[nt].x.size(-1)

        self.model = GraphSAGELinkPredModel(
            metadata=data.metadata(),
            in_channels_dict=in_channels_dict,
            hidden_channels=self.cfg.hidden_channels,
            num_sage_layers=self.cfg.num_sage_layers,
            num_pred_layers=self.cfg.num_pred_layers,
            pred_hidden=self.cfg.pred_hidden,
            dropout=self.cfg.dropout,
            aggregator=self.cfg.aggregator,
        ).to(device)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        # -- Training loop --
        best_val_auc = 0.0
        patience_counter = 0
        ckpt_dir = resolve_path(self.cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()
            loss = self._train_epoch(train_data, optimizer, et)
            val_auc = self._eval_epoch(val_data, et)
            elapsed = time.time() - t0

            self.history["train_loss"].append(loss)
            self.history["val_auc"].append(val_auc)

            if verbose:
                print(
                    f"  Epoch {epoch:3d}/{self.cfg.epochs} | "
                    f"loss={loss:.4f} | val_AUC={val_auc:.4f} | "
                    f"{elapsed:.1f}s"
                )

            # -- Checkpointing --
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                torch.save(
                    self.model.state_dict(),
                    ckpt_dir / "best_model.pt",
                )
            else:
                patience_counter += 1

            if epoch % self.cfg.save_every == 0:
                torch.save(
                    self.model.state_dict(),
                    ckpt_dir / f"model_epoch{epoch:03d}.pt",
                )

            # -- Early stopping --
            if patience_counter >= self.cfg.patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch} (patience={self.cfg.patience})")
                break

        if verbose:
            print(f"  Training complete -- best val AUC = {best_val_auc:.4f}")

        return self.history

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _train_epoch(self, data, optimizer, et) -> float:
        """Run one training epoch; return average loss."""
        self.model.train()
        optimizer.zero_grad()

        # Forward
        pred = self.model(
            data.x_dict,
            data.edge_index_dict,
            data[et].edge_label_index,
            src_type=et[0],
            dst_type=et[2],
        )
        labels = data[et].edge_label.float()
        loss = F.binary_cross_entropy_with_logits(pred, labels)

        # Backward
        loss.backward()
        if self.cfg.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.clip_grad_norm,
            )
        optimizer.step()

        return loss.item()

    @torch.no_grad()
    def _eval_epoch(self, data, et) -> float:
        """Evaluate on validation data; return AUC-ROC."""
        self.model.eval()

        pred = self.model(
            data.x_dict,
            data.edge_index_dict,
            data[et].edge_label_index,
            src_type=et[0],
            dst_type=et[2],
        )
        labels = data[et].edge_label

        # AUC-ROC
        try:
            from sklearn.metrics import roc_auc_score
            probs = torch.sigmoid(pred).cpu().numpy()
            auc = roc_auc_score(labels.cpu().numpy(), probs)
        except Exception:
            auc = 0.0

        return float(auc)


# --------------------------------------------------------------------------
#  CLI entry point
# --------------------------------------------------------------------------

def main() -> None:
    """Train GraphSAGE from the command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Train GraphSAGE link predictor")
    parser.add_argument("--data", type=str, required=True, help="Path to HeteroData .pt file")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = TrainerConfig.from_config()
    if args.epochs:
        cfg.epochs = args.epochs
    if args.lr:
        cfg.lr = args.lr
    if args.device:
        cfg.device = args.device

    data = torch.load(args.data, weights_only=False)
    trainer = LinkPredTrainer(cfg)
    trainer.fit(data)


if __name__ == "__main__":
    main()
