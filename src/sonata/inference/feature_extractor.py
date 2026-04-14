"""
feature_extractor.py
====================
Inference-time feature extraction pipeline.

Takes a **raw audio file** (or pre-existing MIDI) and produces the
dual-branch feature vector that can be fed into the trained GraphSAGE
model for link prediction / recommendation.

Pipeline::

    MP3 / WAV / FLAC
        │
        ▼
    AudioTranscriber (MT3 / basic-pitch)
        │
        ▼
    MIDI file
        │
    ┌───┴───┐
    │       │
    ▼       ▼
  jSymbolic2   SemanticAnalyzer
  (statistical) (semantic)
    │       │
    └───┬───┘
        │
        ▼
    Merged feature vector  (pd.Series / dict)

Main class
----------
FeatureExtractor
    extract(audio_or_midi_path) → dict[str, float]
    extract_batch(paths)        → pd.DataFrame
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from sonata.config.settings import CFG

__all__ = ["FeatureExtractor"]


class FeatureExtractor:
    """
    End-to-end feature extraction pipeline for inference.

    Accepts either:
    * An **audio file** (MP3, WAV, FLAC) → transcribed to MIDI first
    * A **MIDI file** → directly analysed

    Then runs the dual-branch extraction (jSymbolic2 + SemanticAnalyzer)
    and returns a merged feature vector compatible with the trained
    GraphSAGE model.

    Parameters
    ----------
    transcriber : AudioTranscriber, optional
        Pre-configured audio transcriber.  If ``None``, one is created
        from config when needed.
    jsymbolic : JSymbolicExtractor, optional
        Pre-configured jSymbolic2 wrapper.  ``None`` → skip.
    semantic : SemanticAnalyzer, optional
        Pre-configured semantic analyser.  ``None`` → skip.
    feature_cols : list[str], optional
        Expected feature column names (from training).  If provided,
        the output is aligned to these columns (with 0 for missing).
    normalize_params : dict, optional
        ``{"mean": array, "std": array}`` from the training set.
        If provided, z-score normalisation is applied.
    """

    _AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma"}
    _MIDI_EXTENSIONS = {".mid", ".midi", ".MID", ".MIDI"}

    def __init__(
        self,
        transcriber: Optional[Any] = None,
        jsymbolic: Optional[Any] = None,
        semantic: Optional[Any] = None,
        feature_cols: Optional[List[str]] = None,
        normalize_params: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self.transcriber = transcriber
        self.jsymbolic = jsymbolic
        self.semantic = semantic
        self.feature_cols = feature_cols
        self.normalize_params = normalize_params

    def _ensure_transcriber(self):
        """Lazy-init transcriber if needed."""
        if self.transcriber is None:
            from sonata.inference.audio_transcriber import AudioTranscriber
            self.transcriber = AudioTranscriber()

    def _is_audio(self, path: Path) -> bool:
        return path.suffix.lower() in self._AUDIO_EXTENSIONS

    def _is_midi(self, path: Path) -> bool:
        return path.suffix.lower() in self._MIDI_EXTENSIONS

    # ─────────────────────────────────────────────────────────────────────
    #  Single-file extraction
    # ─────────────────────────────────────────────────────────────────────

    def extract(
        self,
        path: str | Path,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Extract the full feature vector from a single file.

        Parameters
        ----------
        path : str or Path
            Audio file (MP3/WAV/FLAC) or MIDI file.
        verbose : bool
            Print progress.

        Returns
        -------
        dict[str, float]
            Feature name → value, aligned to ``self.feature_cols`` if set.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # ── Step 1: Transcribe if audio ───────────────────────────────
        if self._is_audio(path):
            self._ensure_transcriber()
            midi_path = self.transcriber.transcribe(path, verbose=verbose)
        elif self._is_midi(path):
            midi_path = path
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        features: Dict[str, float] = {}

        # ── Step 2: jSymbolic2 (statistical) ──────────────────────────
        if self.jsymbolic is not None:
            try:
                jsym_feats = self.jsymbolic.extract(midi_path)
                features.update(jsym_feats)
            except Exception as exc:
                if verbose:
                    print(f"  ⚠ jSymbolic2 failed: {exc}")

        # ── Step 3: Semantic analysis ─────────────────────────────────
        if self.semantic is not None:
            try:
                report = self.semantic.analyze(midi_path, verbose=False)
                sem_feats = self.semantic.extract_features(report)
                # Only keep numeric values
                for k, v in sem_feats.items():
                    try:
                        features[k] = float(v)
                    except (ValueError, TypeError):
                        pass
            except Exception as exc:
                if verbose:
                    print(f"  ⚠ Semantic analysis failed: {exc}")

        # ── Step 4: Align to training columns ─────────────────────────
        if self.feature_cols:
            aligned = {col: features.get(col, 0.0) for col in self.feature_cols}
            features = aligned

        # ── Step 5: Normalise ─────────────────────────────────────────
        if self.normalize_params:
            mean = self.normalize_params["mean"]
            std = self.normalize_params["std"]
            if self.feature_cols:
                values = np.array([features[c] for c in self.feature_cols], dtype=np.float32)
                values = (values - mean) / np.where(std == 0, 1.0, std)
                features = dict(zip(self.feature_cols, values.tolist()))

        return features

    # ─────────────────────────────────────────────────────────────────────
    #  Batch extraction
    # ─────────────────────────────────────────────────────────────────────

    def extract_batch(
        self,
        paths: Sequence[str | Path],
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Extract features from multiple files.

        Returns
        -------
        pd.DataFrame
            One row per file, columns = feature names.
        """
        rows = []
        n = len(paths)
        for i, p in enumerate(paths):
            try:
                feats = self.extract(p, verbose=False)
                feats["_file"] = Path(p).stem
                rows.append(feats)
            except Exception as exc:
                if verbose:
                    print(f"  ⚠ [{i+1}/{n}] {Path(p).name}: {exc}")

            if verbose and n >= 5 and (i + 1) % max(1, n // 5) == 0:
                print(f"  Progress: {i+1}/{n} files …")

        df = pd.DataFrame(rows)
        if "_file" in df.columns:
            df = df.set_index("_file")
        if verbose:
            print(f"  ✓ Extracted features for {len(df)} files")
        return df


# ── CLI entry point ──────────────────────────────────────────────────────
def main() -> None:
    """Run inference-time feature extraction from the command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Extract features from audio/MIDI files")
    parser.add_argument("files", nargs="+", help="Audio or MIDI files")
    parser.add_argument("-o", "--output", default="inferred_features.parquet")
    parser.add_argument("--no-jsymbolic", action="store_true")
    parser.add_argument("--no-semantic", action="store_true")
    args = parser.parse_args()

    jsym = None
    if not args.no_jsymbolic:
        try:
            from sonata.data.jsymbolic_wrapper import JSymbolicExtractor
            jsym = JSymbolicExtractor()
        except Exception as exc:
            print(f"⚠ jSymbolic2 not available: {exc}")

    sem = None
    if not args.no_semantic:
        from sonata.data.semantic_analyzer import SemanticAnalyzer
        sem = SemanticAnalyzer()

    extractor = FeatureExtractor(jsymbolic=jsym, semantic=sem)
    df = extractor.extract_batch(args.files)
    df.to_parquet(args.output)
    print(f"  ✓ Saved → {args.output}")
