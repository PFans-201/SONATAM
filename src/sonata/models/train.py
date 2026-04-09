"""
train.py
========
Generic training loop with checkpointing for both GenreClassifier and ChordTransformer.

Usage
-----
>>> from sonata.models.train import Trainer, TrainerConfig
>>> trainer = Trainer(model, train_loader, val_loader, config)
>>> trainer.fit()
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

__all__ = ["Trainer", "TrainerConfig"]


class TrainerConfig:
    """
    Hyper-parameter bundle passed to :class:`Trainer`.

    Attributes
    ----------
    epochs : int
    lr : float
    weight_decay : float
    clip_grad_norm : float | None
        Gradient clipping value.  None = disabled.
    checkpoint_dir : str
        Directory for ``.pt`` checkpoint files.
    save_every : int
        Save a checkpoint every N epochs.
    device : str
        ``'cuda'``, ``'cpu'``, or ``'auto'``.
    """

    def __init__(
        self,
        epochs: int = 30,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        clip_grad_norm: Optional[float] = 1.0,
        checkpoint_dir: str = "checkpoints",
        save_every: int = 5,
        device: str = "auto",
    ) -> None:
        self.epochs         = epochs
        self.lr             = lr
        self.weight_decay   = weight_decay
        self.clip_grad_norm = clip_grad_norm
        self.checkpoint_dir = checkpoint_dir
        self.save_every     = save_every
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device


class Trainer:
    """
    Training loop with:
    * AdamW optimiser + CosineAnnealingLR scheduler
    * Optional gradient clipping
    * Best-val-loss checkpointing
    * Per-epoch train / val loss + accuracy logging

    Parameters
    ----------
    model : nn.Module
    train_loader : DataLoader
    val_loader : DataLoader
    config : TrainerConfig
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Optional[TrainerConfig] = None,
    ) -> None:
        self.model    = model
        self.train_dl = train_loader
        self.val_dl   = val_loader
        self.cfg      = config or TrainerConfig()

        self.model.to(self.cfg.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(
            model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.cfg.epochs)

        Path(self.cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.history: Dict[str, list] = {
            "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []
        }
        self._best_val_loss = float("inf")

    # ─────────────────────────────────────────────────────────────────────
    #  Public
    # ─────────────────────────────────────────────────────────────────────

    def fit(self) -> Dict[str, list]:
        """
        Run the full training loop.

        Returns
        -------
        history dict with lists: train_loss, val_loss, train_acc, val_acc.
        """
        print(f"Training on {self.cfg.device}  |  {self.cfg.epochs} epochs\n")
        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = self._run_epoch(self.train_dl, train=True)
            va_loss, va_acc = self._run_epoch(self.val_dl,   train=False)
            self.scheduler.step()
            elapsed = time.time() - t0

            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(va_loss)
            self.history["train_acc"].append(tr_acc)
            self.history["val_acc"].append(va_acc)

            print(
                f"  Epoch {epoch:3d}/{self.cfg.epochs}  "
                f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.3f}  "
                f"val_loss={va_loss:.4f}  val_acc={va_acc:.3f}  "
                f"({elapsed:.1f}s)"
            )

            # Save best checkpoint
            if va_loss < self._best_val_loss:
                self._best_val_loss = va_loss
                self._save_checkpoint(epoch, tag="best")

            # Periodic checkpoint
            if epoch % self.cfg.save_every == 0:
                self._save_checkpoint(epoch, tag=f"epoch{epoch:03d}")

        print("\n  ✓ Training complete.")
        return self.history

    # ─────────────────────────────────────────────────────────────────────
    #  Private
    # ─────────────────────────────────────────────────────────────────────

    def _run_epoch(self, loader: DataLoader, train: bool):
        self.model.train(train)
        total_loss, total_correct, total_n = 0.0, 0, 0

        with torch.set_grad_enabled(train):
            for x, y in loader:
                x = x.to(self.cfg.device)
                y = y.to(self.cfg.device)

                logits = self.model(x)
                loss   = self.criterion(logits, y)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.cfg.clip_grad_norm:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad_norm)
                    self.optimizer.step()

                total_loss    += loss.item() * y.size(0)
                total_correct += (logits.argmax(1) == y).sum().item()
                total_n       += y.size(0)

        avg_loss = total_loss / max(total_n, 1)
        accuracy = total_correct / max(total_n, 1)
        return avg_loss, accuracy

    def _save_checkpoint(self, epoch: int, tag: str = "") -> None:
        path = os.path.join(
            self.cfg.checkpoint_dir,
            f"checkpoint_{tag}.pt" if tag else f"checkpoint_epoch{epoch:03d}.pt"
        )
        torch.save(
            {
                "epoch":       epoch,
                "model_state": self.model.state_dict(),
                "optim_state": self.optimizer.state_dict(),
                "val_loss":    self._best_val_loss,
            },
            path,
        )
