"""
linker.py
=========
LakhMSDLinker — link Lakh MIDI Dataset files with MSD HDF5 metadata,
with optional DTW match-score filtering.

Class
-----
LakhMSDLinker
    discover_tracks(max_tracks, min_score) → list of track dicts
    build_dataset(...)                     → pd.DataFrame
"""

from __future__ import annotations

import os
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .harmonic_analyzer import MIDIHarmonicAnalyzer
from .msd_reader import read_msd_metadata

__all__ = ["LakhMSDLinker"]


class LakhMSDLinker:
    """
    Link Lakh MIDI Dataset files with Million Song Dataset HDF5 metadata,
    with optional DTW match-score quality filtering.

    Parameters
    ----------
    midi_root : str
        Path to ``lmd_matched/`` (contains ``TR.../`` subdirectories with ``.mid`` files).
    h5_root : str
        Path to ``lmd_matched_h5/`` (contains ``TRXXX.h5`` files with the same sub-path).
    analyzer : MIDIHarmonicAnalyzer, optional
        Harmonic feature extractor. Pass ``None`` for metadata-only (much faster).
    match_scores : dict, optional
        Loaded ``match_scores.json`` — ``{track_id: {md5_hash: score}}``.
        Enables score-based filtering and ``pick_midi='best'`` mode.
        When omitted, every MIDI gets an implicit score of 0.0.

    Notes
    -----
    The MIDI ↔ HDF5 path mapping is::

        lmd_matched/<A>/<B>/<C>/<TRXXX>/  ↔  lmd_matched_h5/<A>/<B>/<C>/<TRXXX>.h5

    The MD5 hash is the bare filename (without extension) of each ``.mid`` inside
    the track folder, matching keys in ``match_scores.json``.
    """

    def __init__(
        self,
        midi_root: str,
        h5_root: str,
        analyzer: Optional[MIDIHarmonicAnalyzer] = None,
        match_scores: Optional[Dict] = None,
    ) -> None:
        self.midi_root    = Path(midi_root)
        self.h5_root      = Path(h5_root)
        self.analyzer     = analyzer
        self.match_scores = match_scores or {}

    # ─────────────────────────────────────────────────────────────────────
    #  Discovery
    # ─────────────────────────────────────────────────────────────────────

    def discover_tracks(
        self,
        max_tracks: Optional[int] = None,
        min_score: float = 0.0,
    ) -> List[Dict]:
        """
        Walk ``lmd_matched/`` and return a list of matched track dicts.

        Each dict contains::

            {
              "track_id"  : str,
              "h5_path"   : str,
              "midi_paths": [(path_str, score_float), ...],  # sorted best-first
            }

        Parameters
        ----------
        max_tracks : int, optional
            Stop after collecting this many tracks (for prototyping).
        min_score : float
            Discard any MIDI file whose match score is strictly below this value.
            Tracks where *no* MIDI survives the threshold are also discarded.
            Has no effect when ``match_scores`` was not provided.
        """
        pairs: List[Dict] = []

        for track_dir in sorted(self.midi_root.rglob("TR*")):
            if not track_dir.is_dir():
                continue

            track_id = track_dir.name

            # Derive expected HDF5 path
            rel     = track_dir.relative_to(self.midi_root)
            h5_path = self.h5_root / rel.parent / f"{track_id}.h5"
            if not h5_path.exists():
                continue

            midi_files = sorted(track_dir.glob("*.mid")) + sorted(track_dir.glob("*.midi"))
            if not midi_files:
                continue

            # Attach match scores; filter by min_score
            scores_for_track = self.match_scores.get(track_id, {})
            scored_midis: List[tuple] = []
            for midi_path in midi_files:
                md5   = os.path.splitext(midi_path.name)[0]
                score = scores_for_track.get(md5, 0.0)
                if score >= min_score:
                    scored_midis.append((str(midi_path), score))

            if not scored_midis:
                continue  # all MIDIs below threshold

            # Sort best-score first so index 0 is always the best MIDI
            scored_midis.sort(key=lambda x: -x[1])

            pairs.append(
                {
                    "track_id":   track_id,
                    "h5_path":    str(h5_path),
                    "midi_paths": scored_midis,
                }
            )

            if max_tracks and len(pairs) >= max_tracks:
                break

        return pairs

    # ─────────────────────────────────────────────────────────────────────
    #  Dataset builder
    # ─────────────────────────────────────────────────────────────────────

    def build_dataset(
        self,
        tracks: Optional[List[Dict]] = None,
        max_tracks: Optional[int] = None,
        min_score: float = 0.0,
        with_harmonic_features: bool = True,
        pick_midi: str = "best",
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Build a unified DataFrame where each row = one MIDI linked to its MSD metadata.

        Parameters
        ----------
        tracks : list, optional
            Output of :meth:`discover_tracks`. If ``None``, discovery is run first.
        max_tracks : int, optional
            Passed to :meth:`discover_tracks` when ``tracks`` is ``None``.
        min_score : float
            Minimum DTW match score — passed to :meth:`discover_tracks`.
        with_harmonic_features : bool
            If ``True`` (and ``self.analyzer`` is set) run harmonic analysis on
            each MIDI and merge the scalar feature vector.
        pick_midi : ``'best'`` | ``'first'`` | ``'all'``
            * ``'best'``  — one row per track, highest-scoring MIDI (recommended).
            * ``'first'`` — one row per track, first MIDI alphabetically.
            * ``'all'``   — one row per surviving MIDI (multiple rows per track).
        verbose : bool
            Print progress messages.

        Returns
        -------
        pd.DataFrame
            Columns: track / file identifiers, ``match_score``, MSD metadata
            scalars, and (optionally) harmonic feature scalars.
        """
        if tracks is None:
            if verbose:
                print(f"Discovering tracks  (min_score ≥ {min_score}) …")
            tracks = self.discover_tracks(max_tracks=max_tracks, min_score=min_score)
            if verbose:
                total_midis = sum(len(t["midi_paths"]) for t in tracks)
                print(f"  Found {len(tracks):,} tracks  |  {total_midis:,} MIDI files after filtering\n")

        rows: List[Dict] = []
        n = len(tracks)

        for idx, trk in enumerate(tracks):
            track_id     = trk["track_id"]
            h5_path      = trk["h5_path"]
            scored_midis = trk["midi_paths"]   # list of (path_str, score_float)

            # ── MSD metadata ──────────────────────────────────────────
            try:
                meta = read_msd_metadata(h5_path)
            except Exception as exc:
                if verbose:
                    print(f"  ⚠ [{idx+1}/{n}] {track_id}: HDF5 read failed — {exc}")
                continue

            # ── Select MIDI subset ────────────────────────────────────
            if pick_midi == "best":
                midi_subset = scored_midis[:1]                                  # already sorted
            elif pick_midi == "first":
                midi_subset = [min(scored_midis, key=lambda x: x[0])]          # alphabetical
            else:   # 'all'
                midi_subset = scored_midis

            for midi_path, match_score in midi_subset:
                row: Dict = OrderedDict()
                row["track_id"]    = track_id
                row["midi_file"]   = os.path.basename(midi_path)
                row["midi_path"]   = midi_path
                row["h5_path"]     = h5_path
                row["match_score"] = match_score

                # Merge MSD metadata (skip raw list columns)
                for k, v in meta.items():
                    if k in ("artist_terms", "artist_terms_freq"):
                        continue
                    row[k] = v

                # ── Harmonic features (optional) ──────────────────────
                if with_harmonic_features and self.analyzer is not None:
                    try:
                        report   = self.analyzer.analyze(midi_path, verbose=False)
                        features = self.analyzer.extract_features(report)
                        for k, v in features.items():
                            if k.startswith("_") or k in ("top_bigrams", "top_chords"):
                                continue
                            row[k] = str(v) if k == "interval_class_vector" else v
                    except Exception as exc:
                        if verbose:
                            print(
                                f"  ⚠ [{idx+1}/{n}] {track_id}/"
                                f"{os.path.basename(midi_path)}: {exc}"
                            )

                rows.append(row)

            if verbose and n >= 10 and (idx + 1) % max(1, n // 10) == 0:
                print(f"  Progress: {idx+1}/{n} tracks processed …")

        df = pd.DataFrame(rows)
        if verbose:
            print(f"\n  ✓ Dataset built: {len(df):,} rows × {len(df.columns)} columns")
        return df
