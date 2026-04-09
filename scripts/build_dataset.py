#!/usr/bin/env python3
"""
build_dataset.py
================
CLI entry point — run the full dataset pipeline.

Usage
-----
    python scripts/build_dataset.py
    python scripts/build_dataset.py --max-tracks 500 --min-score 0.7
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sonata.config.settings import CFG
from sonata.data.analyzer import MIDIHarmonicAnalyzer
from sonata.data.linker import LakhMSDLinker


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the SONATA curated dataset.")
    p.add_argument("--max-tracks",  type=int,   default=CFG.get("dataset", {}).get("max_tracks"))
    p.add_argument("--min-score",   type=float, default=CFG.get("dataset", {}).get("min_match_score", 0.70))
    p.add_argument("--pick-midi",   type=str,   default=CFG.get("dataset", {}).get("pick_midi", "best"))
    p.add_argument("--no-harmonic", action="store_true", help="Skip harmonic analysis (faster)")
    p.add_argument("--output-dir",  type=str,   default=CFG.get("dataset", {}).get("output_dir", "data/processed"))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_cfg = CFG.get("data", {})
    midi_root = data_cfg.get("midi_root", "")
    h5_root   = data_cfg.get("h5_root", "")

    # Load match scores if present
    match_scores_path = data_cfg.get("match_scores_path", "data/raw/match_scores.json")
    match_scores = {}
    if Path(match_scores_path).exists():
        with open(match_scores_path) as f:
            match_scores = json.load(f)
        print(f"  Loaded match scores for {len(match_scores):,} tracks.")

    analyzer = None if args.no_harmonic else MIDIHarmonicAnalyzer(
        key_window  = CFG.get("analyzer", {}).get("key_window", 6),
        key_confidence = CFG.get("analyzer", {}).get("key_confidence", 0.80),
    )

    linker = LakhMSDLinker(
        midi_root    = midi_root,
        h5_root      = h5_root,
        analyzer     = analyzer,
        match_scores = match_scores,
    )

    df = linker.build_dataset(
        max_tracks            = args.max_tracks,
        min_score             = args.min_score,
        with_harmonic_features= not args.no_harmonic,
        pick_midi             = args.pick_midi,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = out_dir / "curated_dataset.parquet"
    csv_path     = out_dir / "curated_dataset.csv"

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    print(f"\n  ✓ Saved → {parquet_path}")
    print(f"  ✓ Saved → {csv_path}")


if __name__ == "__main__":
    main()
