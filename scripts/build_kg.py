#!/usr/bin/env python3#!/usr/bin/env python3

""""""

build_kg.pybuild_kg.py

======================

CLI entry point — build the RDF knowledge graph and HeteroData from theCLI entry point — build the RDF knowledge graph from the curated dataset.

curated feature dataset.

Usage

Usage-----

-----    python scripts/build_kg.py

    python scripts/build_kg.py    python scripts/build_kg.py --input data/processed/curated_dataset.parquet

    python scripts/build_kg.py --input data/processed/curated_dataset.parquet    python scripts/build_kg.py --format ntriples

    python scripts/build_kg.py --format ntriples --no-heterodata"""

"""

from __future__ import annotations

from __future__ import annotations

import argparse

import argparsefrom pathlib import Path

from pathlib import Path

import pandas as pd

import pandas as pd

import torchfrom sonata.config.settings import CFG

from sonata.kg.builder import KGBuilder

from sonata.config.settings import CFG

from sonata.kg import KGBuilder, HeteroGraphConverter

def parse_args() -> argparse.Namespace:

    kg_cfg  = CFG.get("knowledge_graph", {})

def parse_args() -> argparse.Namespace:    ds_cfg  = CFG.get("dataset", {})

    kg_cfg = CFG.get("knowledge_graph", {})    p = argparse.ArgumentParser(description="Build the SONATA RDF knowledge graph.")

    ds_cfg = CFG.get("dataset", {})    p.add_argument(

    p = argparse.ArgumentParser(description="Build SONATAM RDF knowledge graph + HeteroData.")        "--input",

    p.add_argument(        type=str,

        "--input", type=str,        default=ds_cfg.get("parquet_file", "data/processed/curated_dataset.parquet"),

        default=ds_cfg.get("parquet_file", "data/processed/curated_dataset.parquet"),        help="Path to the curated dataset parquet file.",

        help="Path to the curated dataset parquet file.",    )

    )    p.add_argument(

    p.add_argument("--output-dir", type=str, default=kg_cfg.get("output_dir", "data/processed"))        "--output-dir",

    p.add_argument(        type=str,

        "--format", type=str, choices=["turtle", "ntriples", "json-ld", "n3", "xml"],        default=kg_cfg.get("output_dir", "data/processed"),

        default="turtle", help="RDF serialisation format.",    )

    )    p.add_argument(

    p.add_argument(        "--format",

        "--include-progressions", action="store_true",        type=str,

        default=kg_cfg.get("include_progressions", False),        choices=["turtle", "ntriples", "json-ld", "n3", "xml"],

        help="Include chord-level progression nodes.",        default="turtle",

    )        help="RDF serialisation format.",

    p.add_argument(    )

        "--no-heterodata", action="store_true",    p.add_argument(

        help="Skip PyTorch Geometric HeteroData conversion.",        "--include-progressions",

    )        action="store_true",

    return p.parse_args()        default=kg_cfg.get("include_progressions", False),

        help="Include chord-level progression nodes (larger graph).",

    )

def main() -> None:    return p.parse_args()

    args = parse_args()



    print(f"  Loading dataset: {args.input}")def main() -> None:

    df = pd.read_parquet(args.input)    args = parse_args()

    print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns.\n")

    print(f"  Loading dataset from: {args.input}")

    # ── RDF graph ─────────────────────────────────────────────────────    df = pd.read_parquet(args.input)

    builder = KGBuilder()    print(f"  Loaded {len(df):,} rows.\n")

    g = builder.from_dataframe(df, include_progressions=args.include_progressions)

    builder = KGBuilder()

    ext_map = {"turtle": ".ttl", "ntriples": ".nt", "json-ld": ".jsonld", "n3": ".n3", "xml": ".xml"}    g = builder.from_dataframe(df, include_progressions=args.include_progressions)

    ext = ext_map.get(args.format, ".ttl")

    ext_map = {

    out_dir = Path(args.output_dir)        "turtle":    ".ttl",

    out_dir.mkdir(parents=True, exist_ok=True)        "ntriples":  ".nt",

    rdf_path = out_dir / f"harmonic_kg{ext}"        "json-ld":   ".jsonld",

        "n3":        ".n3",

    builder.save(g, str(rdf_path), fmt=args.format)        "xml":       ".xml",

    print(f"  ✓ RDF graph → {rdf_path}  ({len(g):,} triples)")    }

    ext = ext_map.get(args.format, ".ttl")

    # ── HeteroData ────────────────────────────────────────────────────

    if not args.no_heterodata:    out_dir = Path(args.output_dir)

        converter = HeteroGraphConverter(normalize_features=True)    out_dir.mkdir(parents=True, exist_ok=True)

        hetero_data = converter.convert(df)    out_path = out_dir / f"harmonic_kg{ext}"



        hetero_path = out_dir / "hetero_data.pt"    builder.save(g, str(out_path), fmt=args.format)

        torch.save(hetero_data, hetero_path)

        print(f"  ✓ HeteroData → {hetero_path}")

        print(f"    Node types: {hetero_data.node_types}")if __name__ == "__main__":

        print(f"    Edge types: {hetero_data.edge_types}")    main()



if __name__ == "__main__":
    main()
