"""
linker.py
=========
Match Lakh MIDI <-> MSD HDF5 and run dual-branch feature extraction.

The **LakhMSDLinker** class is the main entry point for building the
curated dataset.  It:

1. Discovers matched ``(track_id, midi_path, h5_path)`` triples.
2. Reads MSD metadata via :func:`sonata.data.msd_reader.read_msd_metadata`.
3. Runs jSymbolic2 (statistical branch -> ``jsym_*`` columns).
4. Runs the semantic analyser (semantic branch -> ``sem_*`` columns).
5. Returns a single ``pd.DataFrame`` ready for KG ingestion.

Main class
----------
LakhMSDLinker
    discover_tracks()      -> list[dict]
    build_dataset(tracks)  -> pd.DataFrame
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from sonata.config.settings import CFG
from sonata.data.msd_reader import read_msd_metadata

__all__ = ["LakhMSDLinker"]

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
#  Helpers
# --------------------------------------------------------------------------

def _track_id_to_h5_path(track_id: str, h5_root: Path) -> Path:
    """
    Convert a track ID like ``TRAAAAV128F421A322`` into the matching
    HDF5 file path:  ``<h5_root>/A/B/C/TRAAAAV128F421A322.h5``.
    """
    return h5_root / track_id[2] / track_id[3] / track_id[4] / f"{track_id}.h5"


def _find_midi_for_track(track_id: str, midi_root: Path) -> List[Path]:
    """Return every MIDI file under ``midi_root/<A>/<B>/<C>/<track_id>/``."""
    subdir = midi_root / track_id[2] / track_id[3] / track_id[4] / track_id
    if not subdir.is_dir():
        return []
    return sorted(subdir.glob("*.mid")) + sorted(subdir.glob("*.midi"))


def _pick_best_midi(
    midi_files: Sequence[Path],
    match_scores: Optional[Dict[str, Any]] = None,
    track_id: str = "",
    strategy: str = "best",
) -> Optional[Path]:
    """
    Pick one MIDI from a list, using the match-score file if available.

    Parameters
    ----------
    strategy : str
        ``"best"``  -> highest DTW match score.
        ``"first"`` -> lexicographically first file.
        ``"all"``   -> not handled here; caller should iterate.
    """
    if not midi_files:
        return None
    if strategy == "first" or match_scores is None:
        return midi_files[0]

    # "best" strategy - use DTW match scores
    scores: Dict[str, float] = {}
    if track_id in (match_scores or {}):
        raw = match_scores[track_id]
        if isinstance(raw, dict):
            scores = {k: float(v) for k, v in raw.items()}
        elif isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    scores.update({k: float(v) for k, v in item.items()})

    best_path: Optional[Path] = None
    best_score: float = -1.0
    for mp in midi_files:
        key = mp.name
        sc = scores.get(key, scores.get(str(mp), 0.0))
        if sc > best_score:
            best_score = sc
            best_path = mp

    return best_path or midi_files[0]


# --------------------------------------------------------------------------
#  Main class
# --------------------------------------------------------------------------

class LakhMSDLinker:
    """
    Build the SONATAM feature dataset from Lakh MIDI <-> MSD.

    Parameters
    ----------
    midi_root : str or Path, optional
        Root of the ``lmd_matched`` directory tree.
    h5_root : str or Path, optional
        Root of the ``lmd_matched_h5`` directory tree.
    match_scores_path : str or Path, optional
        Path to the DTW ``match_scores.json`` file.
    min_match_score : float
        Quality gate - skip tracks below this DTW score.
    pick_midi : str
        Strategy for selecting among multiple MIDIs per track.
    max_tracks : int or None
        Cap the number of tracks (for prototyping).
    jsymbolic_extractor : object or None
        An instance of :class:`JSymbolicExtractor` (optional).
    semantic_analyzer : object or None
        An instance of :class:`SemanticAnalyzer` (optional).
    """

    def __init__(
        self,
        midi_root=None,
        h5_root=None,
        match_scores_path=None,
        min_match_score: float = 0.70,
        pick_midi: str = "best",
        max_tracks=None,
        jsymbolic_extractor=None,
        semantic_analyzer=None,
    ) -> None:
        cfg_data = CFG.get("data", {})
        cfg_ds = CFG.get("dataset", {})

        self.midi_root = Path(midi_root or cfg_data.get("midi_root", "data/raw/lmd_matched"))
        self.h5_root = Path(h5_root or cfg_data.get("h5_root", "data/raw/lmd_matched_h5"))
        self.min_match_score = min_match_score or cfg_ds.get("min_match_score", 0.70)
        self.pick_midi = pick_midi or cfg_ds.get("pick_midi", "best")
        self.max_tracks = max_tracks or cfg_ds.get("max_tracks")

        # Optional match scores
        self.match_scores: Optional[Dict] = None
        ms_path = match_scores_path or cfg_data.get("match_scores_path")
        if ms_path and Path(ms_path).exists():
            log.info("Loading match scores from %s ...", ms_path)
            with open(ms_path, "r") as fh:
                self.match_scores = json.load(fh)

        # Optional feature extractors
        self.jsymbolic = jsymbolic_extractor
        self.semantic = semantic_analyzer

    # ------------------------------------------------------------------
    #  Track discovery
    # ------------------------------------------------------------------

    def discover_tracks(self) -> List[Dict[str, Any]]:
        """
        Walk the HDF5 tree and find every ``(track_id, h5_path, midi_path)``
        triple that passes the quality gate.

        Returns
        -------
        list[dict]
            Each dict has keys ``track_id``, ``h5_path``, ``midi_path``.
        """
        tracks: List[Dict[str, Any]] = []

        if not self.h5_root.is_dir():
            log.warning("h5_root does not exist: %s", self.h5_root)
            return tracks

        for h5_file in sorted(self.h5_root.rglob("*.h5")):
            track_id = h5_file.stem
            if not track_id.startswith("TR"):
                continue

            # Quality gate via match scores
            if self.match_scores and track_id in self.match_scores:
                raw_score = self.match_scores[track_id]
                if isinstance(raw_score, (int, float)):
                    if raw_score < self.min_match_score:
                        continue
                elif isinstance(raw_score, dict):
                    best = max(raw_score.values(), default=0)
                    if best < self.min_match_score:
                        continue

            # Find matching MIDI
            midi_files = _find_midi_for_track(track_id, self.midi_root)
            midi_path = _pick_best_midi(
                midi_files, self.match_scores, track_id, self.pick_midi,
            )
            if midi_path is None:
                continue

            tracks.append({
                "track_id": track_id,
                "h5_path": str(h5_file),
                "midi_path": str(midi_path),
            })

            if self.max_tracks and len(tracks) >= self.max_tracks:
                break

        log.info("Discovered %d tracks (max_tracks=%s)", len(tracks), self.max_tracks)
        return tracks

    # ------------------------------------------------------------------
    #  Dataset construction
    # ------------------------------------------------------------------

    def build_dataset(
        self,
        tracks=None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Build the full feature DataFrame.

        1. Read MSD metadata for each track.
        2. (Optional) Run jSymbolic2 -> ``jsym_*`` columns.
        3. (Optional) Run semantic analyser -> ``sem_*`` columns.

        Parameters
        ----------
        tracks : list[dict], optional
            Output of :meth:`discover_tracks`.
            If ``None``, calls ``discover_tracks()`` automatically.
        verbose : bool
            Print progress.

        Returns
        -------
        pd.DataFrame
        """
        if tracks is None:
            tracks = self.discover_tracks()
        if not tracks:
            warnings.warn("No tracks found - returning empty DataFrame.")
            return pd.DataFrame()

        rows: List[Dict[str, Any]] = []
        total = len(tracks)

        for idx, trk in enumerate(tracks, 1):
            track_id = trk["track_id"]
            h5_path = trk["h5_path"]
            midi_path = trk["midi_path"]

            if verbose and idx % 500 == 0:
                print(f"  [{idx}/{total}] {track_id}")

            # -- MSD metadata --
            try:
                row = read_msd_metadata(h5_path)
            except Exception as exc:
                log.warning("Failed to read %s: %s", h5_path, exc)
                continue

            row["midi_path"] = midi_path

            # -- jSymbolic2 (statistical branch) --
            if self.jsymbolic is not None:
                try:
                    jsym_feats = self.jsymbolic.extract(midi_path)
                    for k, v in jsym_feats.items():
                        col = k if k.startswith("jsym_") else f"jsym_{k}"
                        row[col] = v
                except Exception as exc:
                    log.debug("jSymbolic failed for %s: %s", track_id, exc)

            # -- Semantic analyser (semantic branch) --
            if self.semantic is not None:
                try:
                    report = self.semantic.analyze(midi_path, verbose=False)
                    flat = self.semantic.extract_features(report)
                    for k, v in flat.items():
                        col = k if k.startswith("sem_") else f"sem_{k}"
                        row[col] = v
                except Exception as exc:
                    log.debug("Semantic analysis failed for %s: %s", track_id, exc)

            rows.append(row)

        df = pd.DataFrame(rows)
        if verbose:
            print(f"  Done: {len(df)} rows x {len(df.columns)} columns")
        return df
