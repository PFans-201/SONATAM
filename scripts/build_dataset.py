#!/usr/bin/env python3#!/usr/bin/env python3

""""""

build_dataset.pybuild_dataset.py

================================

CLI entry point — run the dual-branch feature extraction pipeline.CLI entry point — run the full dataset pipeline.



UsageUsage

----------

    python scripts/build_dataset.py    python scripts/build_dataset.py

    python scripts/build_dataset.py --max-tracks 500 --min-score 0.7    python scripts/build_dataset.py --max-tracks 500 --min-score 0.7

""""""



from __future__ import annotationsfrom __future__ import annotations



import argparseimport argparse

import jsonimport json

from pathlib import Pathfrom pathlib import Path



from sonata.config.settings import CFGfrom sonata.config.settings import CFG

from sonata.data import LakhMSDLinkerfrom sonata.data.analyzer import MIDIHarmonicAnalyzer

from sonata.data.linker import LakhMSDLinker



def parse_args() -> argparse.Namespace:

    ds_cfg = CFG.get("dataset", {})def parse_args() -> argparse.Namespace:

    p = argparse.ArgumentParser(description="Build SONATAM curated dataset with dual-branch features.")    p = argparse.ArgumentParser(description="Build the SONATA curated dataset.")

    p.add_argument("--max-tracks",    type=int,   default=ds_cfg.get("max_tracks"))    p.add_argument("--max-tracks",  type=int,   default=CFG.get("dataset", {}).get("max_tracks"))

    p.add_argument("--min-score",     type=float, default=ds_cfg.get("min_match_score", 0.70))    p.add_argument("--min-score",   type=float, default=CFG.get("dataset", {}).get("min_match_score", 0.70))

    p.add_argument("--pick-midi",     type=str,   default=ds_cfg.get("pick_midi", "best"))    p.add_argument("--pick-midi",   type=str,   default=CFG.get("dataset", {}).get("pick_midi", "best"))

    p.add_argument("--no-jsymbolic",  action="store_true", help="Skip jSymbolic2 feature extraction")    p.add_argument("--no-harmonic", action="store_true", help="Skip harmonic analysis (faster)")

    p.add_argument("--no-semantic",   action="store_true", help="Skip semantic feature extraction")    p.add_argument("--output-dir",  type=str,   default=CFG.get("dataset", {}).get("output_dir", "data/processed"))

    p.add_argument("--output-dir",    type=str,   default=ds_cfg.get("output_dir", "data/processed"))    return p.parse_args()

    return p.parse_args()



def main() -> None:

def main() -> None:    args = parse_args()

    args = parse_args()

    data_cfg = CFG.get("data", {})    data_cfg = CFG.get("data", {})

    midi_root = data_cfg.get("midi_root", "")

    # Load match scores    h5_root   = data_cfg.get("h5_root", "")

    scores_path = data_cfg.get("match_scores_path", "data/raw/match_scores.json")

    match_scores = {}    # Load match scores if present

    if Path(scores_path).exists():    match_scores_path = data_cfg.get("match_scores_path", "data/raw/match_scores.json")

        with open(scores_path) as f:    match_scores = {}

            match_scores = json.load(f)    if Path(match_scores_path).exists():

        print(f"  Loaded match scores for {len(match_scores):,} tracks.")        with open(match_scores_path) as f:

            match_scores = json.load(f)

    # Optional feature extractors        print(f"  Loaded match scores for {len(match_scores):,} tracks.")

    jsymbolic = None

    semantic = None    analyzer = None if args.no_harmonic else MIDIHarmonicAnalyzer(

        key_window  = CFG.get("analyzer", {}).get("key_window", 6),

    if not args.no_jsymbolic:        key_confidence = CFG.get("analyzer", {}).get("key_confidence", 0.80),

        try:    )

            from sonata.data import JSymbolicExtractor

            feat_cfg = CFG.get("features", {}).get("jsymbolic", {})    linker = LakhMSDLinker(

            jsymbolic = JSymbolicExtractor(        midi_root    = midi_root,

                jar_path=feat_cfg.get("jar_path", "lib/jSymbolic2.jar"),        h5_root      = h5_root,

                timeout=feat_cfg.get("timeout_sec", 120),        analyzer     = analyzer,

            )        match_scores = match_scores,

            print("  jSymbolic2 extractor ready.")    )

        except Exception as e:

            print(f"  ⚠ jSymbolic2 not available: {e}")    df = linker.build_dataset(

        max_tracks            = args.max_tracks,

    if not args.no_semantic:        min_score             = args.min_score,

        try:        with_harmonic_features= not args.no_harmonic,

            from sonata.data import SemanticAnalyzer        pick_midi             = args.pick_midi,

            semantic = SemanticAnalyzer()    )

            print("  SemanticAnalyzer ready.")

        except Exception as e:    out_dir = Path(args.output_dir)

            print(f"  ⚠ SemanticAnalyzer not available: {e}")    out_dir.mkdir(parents=True, exist_ok=True)



    linker = LakhMSDLinker(    parquet_path = out_dir / "curated_dataset.parquet"

        midi_root            = data_cfg.get("midi_root", ""),    csv_path     = out_dir / "curated_dataset.csv"

        h5_root              = data_cfg.get("h5_root", ""),

        match_scores         = match_scores,    df.to_parquet(parquet_path, index=False)

        jsymbolic_extractor  = jsymbolic,    df.to_csv(csv_path, index=False)

        semantic_analyzer    = semantic,    print(f"\n  ✓ Saved → {parquet_path}")

    )    print(f"  ✓ Saved → {csv_path}")



    df = linker.build_dataset(

        max_tracks     = args.max_tracks,if __name__ == "__main__":

        min_score      = args.min_score,    main()

        pick_midi      = args.pick_midi,
        with_jsymbolic = not args.no_jsymbolic,
        with_semantic  = not args.no_semantic,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = out_dir / "curated_dataset.parquet"
    csv_path     = out_dir / "curated_dataset.csv"

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    print(f"\n  ✓ Saved → {parquet_path}  ({len(df):,} rows × {len(df.columns)} cols)")
    print(f"  ✓ Saved → {csv_path}")


if __name__ == "__main__":
    main()
