#!/usr/bin/env python3
"""
build_kg.py
===========
CLI entry point — build the RDF knowledge graph from the curated dataset.

Usage
-----
    python scripts/build_kg.py
    python scripts/build_kg.py --input data/processed/curated_dataset.parquet
    python scripts/build_kg.py --format ntriples
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sonata.config.settings import CFG
from sonata.kg.builder import KGBuilder


def parse_args() -> argparse.Namespace:
    kg_cfg  = CFG.get("knowledge_graph", {})
    ds_cfg  = CFG.get("dataset", {})
    p = argparse.ArgumentParser(description="Build the SONATA RDF knowledge graph.")
    p.add_argument(
        "--input",
        type=str,
        default=ds_cfg.get("parquet_file", "data/processed/curated_dataset.parquet"),
        help="Path to the curated dataset parquet file.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=kg_cfg.get("output_dir", "data/processed"),
    )
    p.add_argument(
        "--format",
        type=str,
        choices=["turtle", "ntriples", "json-ld", "n3", "xml"],
        default="turtle",
        help="RDF serialisation format.",
    )
    p.add_argument(
        "--include-progressions",
        action="store_true",
        default=kg_cfg.get("include_progressions", False),
        help="Include chord-level progression nodes (larger graph).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"  Loading dataset from: {args.input}")
    df = pd.read_parquet(args.input)
    print(f"  Loaded {len(df):,} rows.\n")

    builder = KGBuilder()
    g = builder.from_dataframe(df, include_progressions=args.include_progressions)

    ext_map = {
        "turtle":    ".ttl",
        "ntriples":  ".nt",
        "json-ld":   ".jsonld",
        "n3":        ".n3",
        "xml":       ".xml",
    }
    ext = ext_map.get(args.format, ".ttl")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"harmonic_kg{ext}"

    builder.save(g, str(out_path), fmt=args.format)


if __name__ == "__main__":
    main()
