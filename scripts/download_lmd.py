#!/usr/bin/env python3
"""
download_lmd.py
===============
Download and unpack the Lakh MIDI Dataset (LMD) + MSD HDF5 files
into ``data/raw/`` automatically.

Files fetched
-------------
  lmd_matched         ~1.5 GB tar.gz  → data/raw/lmd_matched/
  lmd_aligned         ~1.6 GB tar.gz  → data/raw/lmd_aligned/
  lmd_matched_h5      ~2.5 GB tar.gz  → data/raw/lmd_matched_h5/
  match_scores.json   ~9  MB          → data/raw/match_scores.json
  match_scores_aligned.json           → data/raw/match_scores_aligned.json

Usage
-----
    # Full download (all five assets)
    python scripts/download_lmd.py

    # Skip large archives already downloaded
    python scripts/download_lmd.py --skip-existing

    # Download only specific assets
    python scripts/download_lmd.py --only match_scores match_scores_aligned

    # Place files somewhere other than data/raw/
    python scripts/download_lmd.py --dest /mnt/data/lmd

References
----------
    https://colinraffel.com/projects/lmd/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
import time
from pathlib import Path
from typing import Dict, Optional
from urllib.request import Request, urlopen

# ── optional tqdm progress bar ───────────────────────────────────────────────
try:
    from tqdm import tqdm as _tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

ROOT = Path(__file__).parents[1]   # project root

def _rp(path: Path) -> str:
    """Return path relative to project root, or str(path) if outside."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)

# ── asset registry ────────────────────────────────────────────────────────────
ASSETS: Dict[str, Dict] = {
    "lmd_matched": {
        "url":      "http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz",
        "filename": "lmd_matched.tar.gz",
        "unpack":   True,
        "unpack_to": "lmd_matched",
        "description": "LMD-matched  (~45 k MIDI files matched to MSD)",
    },
    "lmd_aligned": {
        "url":      "http://hog.ee.columbia.edu/craffel/lmd/lmd_aligned.tar.gz",
        "filename": "lmd_aligned.tar.gz",
        "unpack":   True,
        "unpack_to": "lmd_aligned",
        "description": "LMD-aligned  (LMD-matched, beat-aligned to MSD previews)",
    },
    "lmd_matched_h5": {
        "url":      "http://hog.ee.columbia.edu/craffel/lmd/lmd_matched_h5.tar.gz",
        "filename": "lmd_matched_h5.tar.gz",
        "unpack":   True,
        "unpack_to": "lmd_matched_h5",
        "description": "MSD HDF5 files for every LMD-matched entry",
    },
    "match_scores": {
        "url":      "http://hog.ee.columbia.edu/craffel/lmd/match_scores.json",
        "filename": "match_scores.json",
        "unpack":   False,
        "description": "DTW match scores for LMD-matched",
    },
    "match_scores_aligned": {
        "url":      "https://colinraffel.com/projects/lmd/match_scores.json",
        "filename": "match_scores_aligned.json",
        "unpack":   False,
        "description": "DTW match scores for LMD-aligned",
    },
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _human(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def _download(url: str, dest: Path, skip_existing: bool) -> Path:
    """Stream *url* to *dest* with a progress bar."""
    if dest.exists() and skip_existing:
        print(f"    ↩  Already exists — skipping  ({_human(dest.stat().st_size)})")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)

    req  = Request(url, headers={"User-Agent": "Mozilla/5.0 SONATAM-downloader"})
    resp = urlopen(req, timeout=30)

    total = int(resp.headers.get("Content-Length", 0))
    chunk = 1 << 20  # 1 MB chunks

    print(f"    ↓  {url}")
    print(f"       → {_rp(dest)}  ({_human(total) if total else 'size unknown'})")

    downloaded = 0
    t0 = time.time()

    if HAS_TQDM and total:
        bar = _tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024, ncols=80)
    else:
        bar = None

    with open(dest, "wb") as fh:
        while True:
            block = resp.read(chunk)
            if not block:
                break
            fh.write(block)
            downloaded += len(block)
            if bar:
                bar.update(len(block))
            elif total:
                pct  = downloaded / total * 100
                speed = downloaded / max(time.time() - t0, 1e-3)
                print(f"\r       {pct:5.1f}%  {_human(downloaded)}/{_human(total)}"
                      f"  @ {_human(int(speed))}/s", end="", flush=True)

    if bar:
        bar.close()
    else:
        print()

    elapsed = time.time() - t0
    print(f"       ✓  {_human(downloaded)} downloaded in {elapsed:.1f}s")
    return dest


def _unpack(archive: Path, target_dir: Path, skip_existing: bool) -> None:
    """Extract *archive* (tar.gz) into *target_dir*."""
    if target_dir.exists() and skip_existing:
        n = sum(1 for _ in target_dir.rglob("*") if _.is_file())
        print(f"    ↩  Already unpacked  ({n:,} files in {_rp(target_dir)})")
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"    📦 Unpacking {archive.name} → {_rp(target_dir)} …", flush=True)

    with tarfile.open(archive, "r:gz") as tf:
        members  = tf.getmembers()
        n_total  = len(members)

        if HAS_TQDM:
            bar = _tqdm(total=n_total, unit="files", ncols=80)
        else:
            bar = None

        for i, member in enumerate(members, 1):
            # Strip the single top-level directory the archive may contain
            parts = Path(member.name).parts
            if len(parts) > 1:
                member.name = str(Path(*parts[1:]))
            tf.extract(member, path=target_dir)
            if bar:
                bar.update(1)
            elif i % 5000 == 0:
                print(f"\r       {i:,}/{n_total:,} files …", end="", flush=True)

        if bar:
            bar.close()
        else:
            print()

    n_extracted = sum(1 for _ in target_dir.rglob("*") if _.is_file())
    print(f"    ✓  {n_extracted:,} files extracted to {_rp(target_dir)}")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download & unpack the Lakh MIDI Dataset into data/raw/."
    )
    p.add_argument(
        "--dest",
        default=str(ROOT / "data" / "raw"),
        help="Destination directory  (default: data/raw/)",
    )
    p.add_argument(
        "--only",
        nargs="+",
        choices=list(ASSETS),
        metavar="ASSET",
        help=f"Download only these assets. Choices: {', '.join(ASSETS)}",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip download/unpack if the target already exists (default: on).",
    )
    p.add_argument(
        "--no-skip",
        dest="skip_existing",
        action="store_false",
        help="Re-download and re-unpack even if files already exist.",
    )
    p.add_argument(
        "--no-unpack",
        action="store_true",
        help="Download tar.gz archives but do NOT unpack them.",
    )
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    dest   = Path(args.dest).resolve()
    assets = {k: v for k, v in ASSETS.items() if not args.only or k in args.only}

    dest.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print("  SONATAM — Lakh MIDI Dataset downloader")
    print(f"  Destination : {_rp(dest)}")
    print(f"  Assets      : {', '.join(assets)}")
    print(f"{'='*65}\n")

    for key, asset in assets.items():
        print(f"\n── {key}  ──  {asset['description']}")
        archive = dest / asset["filename"]

        try:
            _download(asset["url"], archive, skip_existing=args.skip_existing)
        except Exception as exc:
            print(f"    ✗  Download failed: {exc}", file=sys.stderr)
            continue

        if asset["unpack"] and not args.no_unpack:
            target = dest / asset["unpack_to"]
            try:
                _unpack(archive, target, skip_existing=args.skip_existing)
            except Exception as exc:
                print(f"    ✗  Unpack failed: {exc}", file=sys.stderr)

    # ── Write a manifest so the notebook can find everything ─────────────────
    manifest = {
        key: {
            "archive": str(dest / asset["filename"]),
            "unpacked": str(dest / asset["unpack_to"]) if asset.get("unpack") else None,
            "url": asset["url"],
        }
        for key, asset in ASSETS.items()
    }
    manifest_path = dest / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  ✓  All done.  Manifest written to {_rp(manifest_path)}")
    print(f"{'='*65}\n")
    print("  Next step:")
    print("    python scripts/build_dataset.py")


if __name__ == "__main__":
    main()
