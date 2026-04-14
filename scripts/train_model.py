#!/usr/bin/env python3#!/usr/bin/env python3

""""""

train_model.pytrain_model.py

============================

CLI entry point — train the GraphSAGE link-prediction model.CLI entry point — train a GenreClassifier or ChordTransformer.



UsageUsage

----------

    python scripts/train_model.py --data data/processed/hetero_data.pt    python scripts/train_model.py

    python scripts/train_model.py --data data/processed/hetero_data.pt --epochs 50    python scripts/train_model.py --model transformer --epochs 50

    python scripts/train_model.py --data data/processed/hetero_data.pt --device cuda    python scripts/train_model.py --model mlp --device cuda

""""""



from __future__ import annotationsfrom __future__ import annotations



import argparseimport argparse

from pathlib import Path

import torchfrom typing import Optional



from sonata.config.settings import CFGimport pandas as pd

from sonata.models import LinkPredTrainer, TrainerConfig, evaluate_link_prediction, plot_training_curvesfrom torch.utils.data import DataLoader, Subset



from sonata.config.settings import CFG

def parse_args() -> argparse.Namespace:from sonata.models.architectures.classifier import GenreClassifier

    gl_cfg = CFG.get("graph_learning", {})from sonata.models.architectures.transformer import ChordTransformer

    tr_cfg = gl_cfg.get("training", {})from sonata.models.dataset import HarmonicDataset

from sonata.models.train import Trainer, TrainerConfig

    p = argparse.ArgumentParser(description="Train SONATAM GraphSAGE link-prediction model.")

    p.add_argument(

        "-d", "--data", type=str, required=True,def parse_args() -> argparse.Namespace:

        help="Path to HeteroData .pt file (from build_kg.py).",    dl_cfg  = CFG.get("deep_learning", {})

    )    tr_cfg  = dl_cfg.get("training", {})

    p.add_argument("--epochs",  type=int,   default=tr_cfg.get("epochs", 100))    ds_cfg  = CFG.get("dataset", {})

    p.add_argument("--lr",      type=float, default=tr_cfg.get("lr", 1e-3))

    p.add_argument("--device",  type=str,   default=tr_cfg.get("device", "auto"))    p = argparse.ArgumentParser(description="Train SONATA deep learning model.")

    p.add_argument("--patience", type=int,  default=tr_cfg.get("patience", 15))    p.add_argument(

    p.add_argument("--checkpoint-dir", type=str, default=gl_cfg.get("checkpoint_dir", "checkpoints"))        "--input",

    p.add_argument("--plot",    action="store_true", help="Save training curve plots.")        type=str,

    return p.parse_args()        default=ds_cfg.get("parquet_file", "data/processed/curated_dataset.parquet"),

    )

    p.add_argument("--model",   type=str,   default=dl_cfg.get("model", "mlp"),

def main() -> None:                   choices=["mlp", "transformer"])

    args = parse_args()    p.add_argument("--epochs",  type=int,   default=tr_cfg.get("epochs", 30))

    p.add_argument("--lr",      type=float, default=tr_cfg.get("lr", 3e-4))

    print(f"  Loading HeteroData: {args.data}")    p.add_argument("--batch",   type=int,   default=tr_cfg.get("batch_size", 64))

    data = torch.load(args.data, weights_only=False)    p.add_argument("--device",  type=str,   default=tr_cfg.get("device", "auto"))

    print(f"  Node types: {data.node_types}")    p.add_argument("--checkpoint-dir", type=str,

    print(f"  Edge types: {data.edge_types}\n")                   default=dl_cfg.get("checkpoint_dir", "checkpoints"))

    p.add_argument("--label-col", type=str, default=dl_cfg.get("label_col", "primary_genre"))

    # Configure    return p.parse_args()

    config = TrainerConfig.from_config()

    config.epochs = args.epochs

    config.lr = args.lrdef main() -> None:

    config.patience = args.patience    args = parse_args()

    config.checkpoint_dir = args.checkpoint_dir    dl_cfg = CFG.get("deep_learning", {})

    if args.device != "auto":

        config.device = args.device    print(f"  Loading dataset from: {args.input}")

    df = pd.read_parquet(args.input)

    # Train    print(f"  Loaded {len(df):,} rows.\n")

    trainer = LinkPredTrainer(config)

    model, history = trainer.fit(data, verbose=True)    mode = "sequence" if args.model == "transformer" else "classification"

    dataset = HarmonicDataset(df, label_col=args.label_col, mode=mode)

    # Plot

    if args.plot:    tr_cfg_d = CFG.get("deep_learning", {}).get("training", {})

        plot_training_curves(history, save_path="data/processed/training_curves.png")    idx_train, idx_val, _ = dataset.split(

        val_frac  = tr_cfg_d.get("val_frac", 0.15),

    # Final evaluation        test_frac = tr_cfg_d.get("test_frac", 0.15),

    from sonata.kg.converter import HeteroGraphConverter        seed      = tr_cfg_d.get("seed", 42),

    )

    converter = HeteroGraphConverter()

    splits = converter.create_link_split(    train_loader = DataLoader(Subset(dataset, idx_train), batch_size=args.batch, shuffle=True)

        data,    val_loader   = DataLoader(Subset(dataset, idx_val),   batch_size=args.batch, shuffle=False)

        edge_type=config.target_edge_type,

        val_ratio=config.val_ratio,    if args.model == "mlp":

        test_ratio=config.test_ratio,        mlp_cfg = dl_cfg.get("mlp", {})

        seed=config.seed,        model = GenreClassifier(

    )            input_dim   = dataset.input_dim,

    test_metrics = evaluate_link_prediction(            num_classes = dataset.num_classes,

        model, splits["test"], config.target_edge_type, device=config.device,            hidden_dims = mlp_cfg.get("hidden_dims", [256, 128, 64]),

    )            dropout     = mlp_cfg.get("dropout", 0.3),

        )

    print("\n  ═══ Test Metrics ═══")    else:

    for k, v in test_metrics.items():        tf_cfg = dl_cfg.get("transformer", {})

        print(f"    {k:>12s}: {v:.4f}")        model = ChordTransformer(

            vocab_size      = dataset.vocab_size,

            num_classes     = dataset.num_classes,

if __name__ == "__main__":            mode            = "classify",

    main()            d_model         = tf_cfg.get("d_model", 128),

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
