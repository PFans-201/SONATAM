"""
jsymbolic_wrapper.py
====================
Batch wrapper for jSymbolic2 — extracts 200+ statistical features from MIDI
files by shelling out to the jSymbolic2 JAR.

jSymbolic2 is a Java application; this wrapper requires a JRE (≥11) on PATH
and the JAR to be downloaded into ``lib/jSymbolic2.jar`` (or configured in
``config/config.yaml`` at ``features.jsymbolic.jar_path``).

Main class
----------
JSymbolicExtractor
    extract(midi_path)       → dict[str, float]
    extract_batch(paths)     → pd.DataFrame   (rows = files, cols = features)
"""

from __future__ import annotations

import csv
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from sonata.config.settings import CFG

__all__ = ["JSymbolicExtractor"]

# Default feature set when no config is supplied (all features)
_DEFAULT_TIMEOUT = 120  # seconds per file


class JSymbolicExtractor:
    """
    Extract statistical audio features from MIDI files using jSymbolic2.

    jSymbolic2 produces a CSV with one row per file and one column per
    feature (e.g. *Number of Pitches*, *Range*, *Pitch Variety*, …).

    Parameters
    ----------
    jar_path : str or Path, optional
        Path to ``jSymbolic2.jar``.  Falls back to
        ``CFG["features"]["jsymbolic"]["jar_path"]``.
    config_path : str or Path, optional
        jSymbolic XML configuration file (selects which features to compute).
        ``None`` → compute all available features.
    timeout : int
        Per-file timeout in seconds.

    Raises
    ------
    FileNotFoundError
        If the JAR does not exist at the specified path.
    RuntimeError
        If Java is not on PATH.
    """

    def __init__(
        self,
        jar_path: str | Path | None = None,
        config_path: str | Path | None = None,
        timeout: int | None = None,
    ) -> None:
        cfg_js = CFG.get("features", {}).get("jsymbolic", {})

        self.jar_path = Path(jar_path or cfg_js.get("jar_path", "lib/jSymbolic2.jar"))
        if not self.jar_path.exists():
            raise FileNotFoundError(
                f"jSymbolic2 JAR not found: {self.jar_path}\n"
                "Download from https://sourceforge.net/projects/jmir/files/jSymbolic/"
            )

        self.config_path = Path(config_path) if config_path else (
            Path(cfg_js["config_path"]) if cfg_js.get("config_path") else None
        )
        self.timeout = timeout or cfg_js.get("timeout_sec", _DEFAULT_TIMEOUT)

        # Verify Java is available
        try:
            subprocess.run(
                ["java", "-version"],
                capture_output=True, check=True, timeout=10,
            )
        except (FileNotFoundError, subprocess.SubprocessError) as exc:
            raise RuntimeError(
                "Java (JRE ≥ 11) is required for jSymbolic2 but was not found on PATH."
            ) from exc

    # ─────────────────────────────────────────────────────────────────────
    #  Single-file extraction
    # ─────────────────────────────────────────────────────────────────────

    def extract(self, midi_path: str | Path) -> Dict[str, float]:
        """
        Extract jSymbolic2 features from a single MIDI file.

        Returns
        -------
        dict[str, float]
            Feature name → numeric value.
        """
        df = self.extract_batch([midi_path])
        if df.empty:
            return {}
        return df.iloc[0].to_dict()

    # ─────────────────────────────────────────────────────────────────────
    #  Batch extraction
    # ─────────────────────────────────────────────────────────────────────

    def extract_batch(
        self,
        midi_paths: Sequence[str | Path],
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Extract features from multiple MIDI files in one jSymbolic2 run.

        Parameters
        ----------
        midi_paths : sequence of str or Path
            MIDI files to process.
        verbose : bool
            Print progress.

        Returns
        -------
        pd.DataFrame
            One row per MIDI file, columns = feature names.
            Index = file stem (without extension).
        """
        if not midi_paths:
            return pd.DataFrame()

        midi_paths = [Path(p) for p in midi_paths]
        for p in midi_paths:
            if not p.exists():
                raise FileNotFoundError(f"MIDI file not found: {p}")

        with tempfile.TemporaryDirectory(prefix="jsymbolic_") as tmp:
            tmp_dir = Path(tmp)
            input_list = tmp_dir / "input_files.txt"
            output_csv = tmp_dir / "features.csv"
            output_def = tmp_dir / "feature_definitions.csv"

            # Write input file list
            input_list.write_text("\n".join(str(p) for p in midi_paths))

            # Build jSymbolic2 command
            cmd: List[str] = [
                "java", "-Xmx2g",
                "-jar", str(self.jar_path),
                "-csv", str(input_list),
                str(output_csv),
                str(output_def),
            ]
            if self.config_path:
                cmd.extend(["-configrun", str(self.config_path)])

            if verbose:
                print(f"  ⏳ Running jSymbolic2 on {len(midi_paths)} file(s) …")

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout * len(midi_paths),
                )
            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    f"jSymbolic2 timed out after {self.timeout * len(midi_paths)}s"
                )

            if result.returncode != 0:
                raise RuntimeError(
                    f"jSymbolic2 failed (exit {result.returncode}):\n{result.stderr[:500]}"
                )

            # Parse output CSV
            df = self._parse_output(output_csv, output_def, midi_paths)

            if verbose:
                print(f"  ✓ Extracted {len(df.columns)} features × {len(df)} files")

            return df

    # ─────────────────────────────────────────────────────────────────────
    #  Output parsing
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_output(
        csv_path: Path,
        def_path: Path,
        midi_paths: Sequence[Path],
    ) -> pd.DataFrame:
        """Parse jSymbolic2 CSV output into a DataFrame."""
        if not csv_path.exists():
            raise FileNotFoundError(f"jSymbolic2 output not found: {csv_path}")

        # Read feature definitions for column names
        col_names: List[str] = []
        if def_path.exists():
            with open(def_path, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    if row:
                        col_names.append(row[0].strip())

        # Read feature values
        df = pd.read_csv(csv_path, header=None)

        # First column is the file path — drop it and use as index
        file_col = df.iloc[:, 0]
        df = df.iloc[:, 1:]

        if col_names and len(col_names) == len(df.columns):
            df.columns = col_names
        else:
            df.columns = [f"jsym_{i}" for i in range(len(df.columns))]

        # Use file stems as index
        df.index = [Path(p).stem for p in file_col]
        df.index.name = "file"

        # Convert to numeric
        df = df.apply(pd.to_numeric, errors="coerce")

        # Prefix all column names for easy identification
        df.columns = [f"jsym_{c}" if not c.startswith("jsym_") else c for c in df.columns]

        return df


# ── CLI entry point ──────────────────────────────────────────────────────
def main() -> None:
    """Run jSymbolic2 extraction from the command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Extract jSymbolic2 features from MIDI files")
    parser.add_argument("midi_files", nargs="+", help="MIDI files to process")
    parser.add_argument("-o", "--output", default="jsymbolic_features.parquet")
    parser.add_argument("--jar", default=None, help="Path to jSymbolic2.jar")
    args = parser.parse_args()

    extractor = JSymbolicExtractor(jar_path=args.jar)
    df = extractor.extract_batch(args.midi_files)
    df.to_parquet(args.output)
    print(f"  ✓ Saved → {args.output}")
