#!/usr/bin/env python3
"""
train_model.py
==============
CLI entry point — train a GenreClassifier or ChordTransformer.

Usage
-----
    python scripts/train_model.py
    python scripts/train_model.py --model transformer --epochs 50
    python scripts/train_model.py --model mlp --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
from torch.utils.data import DataLoader, Subset

from sonata.config.settings import CFG
from sonata.models.architectures.classifier import GenreClassifier
from sonata.models.architectures.transformer import ChordTransformer
from sonata.models.dataset import HarmonicDataset
from sonata.models.train import Trainer, TrainerConfig


def parse_args() -> argparse.Namespace:
    dl_cfg  = CFG.get("deep_learning", {})
    tr_cfg  = dl_cfg.get("training", {})
    ds_cfg  = CFG.get("dataset", {})

    p = argparse.ArgumentParser(description="Train SONATA deep learning model.")
    p.add_argument(
        "--input",
        type=str,
        default=ds_cfg.get("parquet_file", "data/processed/curated_dataset.parquet"),
    )
    p.add_argument("--model",   type=str,   default=dl_cfg.get("model", "mlp"),
                   choices=["mlp", "transformer"])
    p.add_argument("--epochs",  type=int,   default=tr_cfg.get("epochs", 30))
    p.add_argument("--lr",      type=float, default=tr_cfg.get("lr", 3e-4))
    p.add_argument("--batch",   type=int,   default=tr_cfg.get("batch_size", 64))
    p.add_argument("--device",  type=str,   default=tr_cfg.get("device", "auto"))
    p.add_argument("--checkpoint-dir", type=str,
                   default=dl_cfg.get("checkpoint_dir", "checkpoints"))
    p.add_argument("--label-col", type=str, default=dl_cfg.get("label_col", "primary_genre"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dl_cfg = CFG.get("deep_learning", {})

    print(f"  Loading dataset from: {args.input}")
    df = pd.read_parquet(args.input)
    print(f"  Loaded {len(df):,} rows.\n")

    mode = "sequence" if args.model == "transformer" else "classification"
    dataset = HarmonicDataset(df, label_col=args.label_col, mode=mode)

    tr_cfg_d = CFG.get("deep_learning", {}).get("training", {})
    idx_train, idx_val, _ = dataset.split(
        val_frac  = tr_cfg_d.get("val_frac", 0.15),
        test_frac = tr_cfg_d.get("test_frac", 0.15),
        seed      = tr_cfg_d.get("seed", 42),
    )

    train_loader = DataLoader(Subset(dataset, idx_train), batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(Subset(dataset, idx_val),   batch_size=args.batch, shuffle=False)

    if args.model == "mlp":
        mlp_cfg = dl_cfg.get("mlp", {})
        model = GenreClassifier(
            input_dim   = dataset.input_dim,
            num_classes = dataset.num_classes,
            hidden_dims = mlp_cfg.get("hidden_dims", [256, 128, 64]),
            dropout     = mlp_cfg.get("dropout", 0.3),
        )
    else:
        tf_cfg = dl_cfg.get("transformer", {})
        model = ChordTransformer(
            vocab_size      = dataset.vocab_size,
            num_classes     = dataset.num_classes,
            mode            = "classify",
            d_model         = tf_cfg.get("d_model", 128),
            nhead           = tf_cfg.get("nhead", 4),
            num_layers      = tf_cfg.get("num_layers", 3),
            dim_feedforward = tf_cfg.get("dim_feedforward", 256),
            max_seq_len     = tf_cfg.get("max_seq_len", 128),
            dropout         = tf_cfg.get("dropout", 0.1),
        )

    config = TrainerConfig(
        epochs         = args.epochs,
        lr             = args.lr,
        checkpoint_dir = args.checkpoint_dir,
        device         = args.device,
        weight_decay   = tr_cfg_d.get("weight_decay", 1e-4),
        clip_grad_norm = tr_cfg_d.get("clip_grad_norm", 1.0),
        save_every     = tr_cfg_d.get("save_every", 5),
    )

    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.fit()


if __name__ == "__main__":
    main()
