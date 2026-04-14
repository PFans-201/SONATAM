"""
converter.py
============
Convert the feature DataFrame (+ optional user interactions) into a
**PyTorch Geometric HeteroData** object for GraphSAGE training.

This is the bridge between the *knowledge-graph* layer and the *GNN*
layer.  The converter:

1. Assigns integer IDs to every entity (piece, artist, genre, key, era, user).
2. Builds edge-index tensors for each relation type.
3. Attaches numeric feature tensors to MusicalPiece nodes (jSymbolic2 + semantic).
4. Optionally creates train / val / test edge splits for link prediction.

Main class
----------
HeteroGraphConverter
    convert(df, user_df)           → torch_geometric.data.HeteroData
    create_link_split(data, ...)   → dict[str, HeteroData]
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

__all__ = ["HeteroGraphConverter"]


class HeteroGraphConverter:
    """
    Convert a SONATAM feature DataFrame into PyTorch Geometric HeteroData.

    Parameters
    ----------
    feature_cols : list[str], optional
        Columns to use as MusicalPiece node features.  If ``None``,
        all ``jsym_*`` and numeric ``sem_*`` columns are used automatically.
    normalize_features : bool
        If True, z-score normalise the feature matrix.
    """

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        normalize_features: bool = True,
    ) -> None:
        self.feature_cols = feature_cols
        self.normalize = normalize_features

        # ID maps (populated during convert)
        self.piece_ids: Dict[str, int] = {}
        self.artist_ids: Dict[str, int] = {}
        self.genre_ids: Dict[str, int] = {}
        self.key_ids: Dict[str, int] = {}
        self.era_ids: Dict[str, int] = {}
        self.user_ids: Dict[str, int] = {}

    # ─────────────────────────────────────────────────────────────────────
    #  Main conversion
    # ─────────────────────────────────────────────────────────────────────

    def convert(
        self,
        df: pd.DataFrame,
        user_df: Optional[pd.DataFrame] = None,
        verbose: bool = True,
    ):
        """
        Build a HeteroData graph from the feature DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Output of ``LakhMSDLinker.build_dataset()`` — contains MSD
            metadata + jSymbolic2 (``jsym_*``) + semantic (``sem_*``) columns.
        user_df : pd.DataFrame, optional
            User interaction data (``user_id, track_id, play_count, rating``).
        verbose : bool
            Print summary.

        Returns
        -------
        torch_geometric.data.HeteroData
        """
        from torch_geometric.data import HeteroData

        data = HeteroData()

        # ── Determine feature columns ────────────────────────────────
        if self.feature_cols:
            feat_cols = [c for c in self.feature_cols if c in df.columns]
        else:
            feat_cols = [
                c for c in df.columns
                if (c.startswith("jsym_") or c.startswith("sem_") or c.startswith("musif_"))
                and df[c].dtype in ("float64", "float32", "int64", "int32")
            ]

        # ── MusicalPiece nodes + features ─────────────────────────────
        self.piece_ids = {
            str(tid): idx for idx, tid in enumerate(df["track_id"].unique())
        }
        n_pieces = len(self.piece_ids)

        # Build feature matrix
        if feat_cols:
            # Map rows to piece IDs (handle duplicates by taking first)
            piece_to_row = {}
            for i, row in df.iterrows():
                tid = str(row["track_id"])
                if tid not in piece_to_row:
                    piece_to_row[tid] = i

            feat_matrix = np.zeros((n_pieces, len(feat_cols)), dtype=np.float32)
            for tid, pid in self.piece_ids.items():
                if tid in piece_to_row:
                    row_idx = piece_to_row[tid]
                    for j, col in enumerate(feat_cols):
                        val = df.at[row_idx, col]
                        if pd.notna(val):
                            try:
                                feat_matrix[pid, j] = float(val)
                            except (ValueError, TypeError):
                                pass

            # Normalise
            if self.normalize:
                mean = feat_matrix.mean(axis=0)
                std = feat_matrix.std(axis=0)
                std[std == 0] = 1.0
                feat_matrix = (feat_matrix - mean) / std

            data["MusicalPiece"].x = torch.tensor(feat_matrix, dtype=torch.float32)
        else:
            # No features — use one-hot or identity
            data["MusicalPiece"].x = torch.eye(n_pieces, dtype=torch.float32)

        data["MusicalPiece"].num_nodes = n_pieces

        # ── Entity nodes ──────────────────────────────────────────────
        # Artist
        artists = df["artist_id"].dropna().unique().astype(str)
        self.artist_ids = {a: i for i, a in enumerate(artists)}
        data["Artist"].num_nodes = len(self.artist_ids)

        # Genre
        genres = set()
        for col in ("top3_genres", "primary_genre"):
            if col in df.columns:
                for val in df[col].dropna():
                    for gl in str(val).split(";"):
                        gl = gl.strip()
                        if gl:
                            genres.add(gl)
        self.genre_ids = {g: i for i, g in enumerate(sorted(genres))}
        data["Genre"].num_nodes = len(self.genre_ids)

        # MusicalKey
        keys = set()
        for col in ("sem_global_key", "global_key", "msd_key_name"):
            if col in df.columns:
                for val in df[col].dropna():
                    keys.add(str(val))
        self.key_ids = {k: i for i, k in enumerate(sorted(keys))}
        data["MusicalKey"].num_nodes = len(self.key_ids)

        # Era
        eras = set()
        if "year" in df.columns:
            for yr in df["year"].dropna():
                yr = int(yr)
                if yr > 0:
                    decade = (yr // 10) * 10
                    eras.add(f"{decade}s")
        self.era_ids = {e: i for i, e in enumerate(sorted(eras))}
        data["Era"].num_nodes = len(self.era_ids)

        # ── Edge indices ──────────────────────────────────────────────
        # piece → artist
        src, dst = [], []
        for _, row in df.iterrows():
            tid = str(row.get("track_id", ""))
            aid = str(row.get("artist_id", ""))
            if tid in self.piece_ids and aid in self.artist_ids:
                src.append(self.piece_ids[tid])
                dst.append(self.artist_ids[aid])
        if src:
            data["MusicalPiece", "hasArtist", "Artist"].edge_index = torch.tensor(
                [src, dst], dtype=torch.long
            )

        # piece → genre
        src, dst = [], []
        for _, row in df.iterrows():
            tid = str(row.get("track_id", ""))
            if tid not in self.piece_ids:
                continue
            genre_val = row.get("top3_genres") or row.get("primary_genre")
            if pd.notna(genre_val):
                for gl in str(genre_val).split(";"):
                    gl = gl.strip()
                    if gl in self.genre_ids:
                        src.append(self.piece_ids[tid])
                        dst.append(self.genre_ids[gl])
        if src:
            data["MusicalPiece", "hasGenre", "Genre"].edge_index = torch.tensor(
                [src, dst], dtype=torch.long
            )

        # piece → key
        src, dst = [], []
        for _, row in df.iterrows():
            tid = str(row.get("track_id", ""))
            if tid not in self.piece_ids:
                continue
            for col in ("sem_global_key", "global_key", "msd_key_name"):
                if col in row and pd.notna(row[col]):
                    k = str(row[col])
                    if k in self.key_ids:
                        src.append(self.piece_ids[tid])
                        dst.append(self.key_ids[k])
                    break
        if src:
            data["MusicalPiece", "hasGlobalKey", "MusicalKey"].edge_index = torch.tensor(
                [src, dst], dtype=torch.long
            )

        # piece → era
        src, dst = [], []
        if "year" in df.columns:
            for _, row in df.iterrows():
                tid = str(row.get("track_id", ""))
                if tid not in self.piece_ids:
                    continue
                if pd.notna(row.get("year")) and int(row["year"]) > 0:
                    era_label = f"{(int(row['year']) // 10) * 10}s"
                    if era_label in self.era_ids:
                        src.append(self.piece_ids[tid])
                        dst.append(self.era_ids[era_label])
        if src:
            data["MusicalPiece", "hasEra", "Era"].edge_index = torch.tensor(
                [src, dst], dtype=torch.long
            )

        # ── User interaction edges ────────────────────────────────────
        if user_df is not None and not user_df.empty:
            self._add_user_edges(data, user_df)

        if verbose:
            print("  ✓ HeteroData built:")
            for nt in data.node_types:
                print(f"      {nt}: {data[nt].num_nodes} nodes")
            for et in data.edge_types:
                ei = data[et].edge_index
                print(f"      {et}: {ei.size(1)} edges")

        return data

    # ─────────────────────────────────────────────────────────────────────
    #  User edges
    # ─────────────────────────────────────────────────────────────────────

    def _add_user_edges(self, data, user_df: pd.DataFrame) -> None:
        """Add User → MusicalPiece edges to the HeteroData."""
        # Build user IDs
        users = user_df["user_id"].unique().astype(str)
        self.user_ids = {u: i for i, u in enumerate(users)}
        data["User"].num_nodes = len(self.user_ids)

        src, dst, weights = [], [], []
        for _, row in user_df.iterrows():
            uid = str(row["user_id"])
            tid = str(row["track_id"])
            if uid in self.user_ids and tid in self.piece_ids:
                src.append(self.user_ids[uid])
                dst.append(self.piece_ids[tid])
                w = float(row.get("play_count", 1))
                weights.append(w)

        if src:
            data["User", "listenedTo", "MusicalPiece"].edge_index = torch.tensor(
                [src, dst], dtype=torch.long
            )
            data["User", "listenedTo", "MusicalPiece"].edge_attr = torch.tensor(
                weights, dtype=torch.float32
            ).unsqueeze(-1)

    # ─────────────────────────────────────────────────────────────────────
    #  Link prediction splits
    # ─────────────────────────────────────────────────────────────────────

    def create_link_split(
        self,
        data,
        edge_type: Tuple[str, str, str] = ("MusicalPiece", "hasGenre", "Genre"),
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        neg_sampling_ratio: float = 1.0,
        seed: int = 42,
    ):
        """
        Create train / val / test edge splits for link prediction.

        Parameters
        ----------
        data : HeteroData
            The full heterogeneous graph.
        edge_type : tuple
            The edge type to split for link prediction.
        val_ratio, test_ratio : float
            Fraction of edges to hold out.
        neg_sampling_ratio : float
            Ratio of negative to positive edges.
        seed : int
            Random seed.

        Returns
        -------
        dict with keys ``'train'``, ``'val'``, ``'test'`` containing
        HeteroData objects with ``edge_label`` and ``edge_label_index``.
        """
        from torch_geometric.transforms import RandomLinkSplit

        transform = RandomLinkSplit(
            num_val=val_ratio,
            num_test=test_ratio,
            neg_sampling_ratio=neg_sampling_ratio,
            edge_types=[edge_type],
            rev_edge_types=[(edge_type[2], f"rev_{edge_type[1]}", edge_type[0])],
            is_undirected=False,
            add_negative_train_samples=True,
        )

        torch.manual_seed(seed)
        train_data, val_data, test_data = transform(data)

        return {"train": train_data, "val": val_data, "test": test_data}

    # ─────────────────────────────────────────────────────────────────────
    #  Serialisation helpers
    # ─────────────────────────────────────────────────────────────────────

    def save_heterodata(self, data, path: str) -> None:
        """Save HeteroData to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, str(p))
        print(f"  ✓ Saved HeteroData → {p}")

    @staticmethod
    def load_heterodata(path: str):
        """Load HeteroData from disk."""
        data = torch.load(path, weights_only=False)
        print(f"  ✓ Loaded HeteroData ← {path}")
        return data
