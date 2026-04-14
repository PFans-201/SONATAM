"""
graph_querier.py
================
Query a trained GraphSAGE model to make predictions for **new** musical
pieces — the core of the inductive inference pipeline.

Given a new piece's feature vector, this module:

1. Creates a temporary node in the HeteroData graph.
2. Runs the GraphSAGE encoder to get the node embedding.
3. Scores candidate edges (genres, artists, eras) via the link predictor.
4. Returns ranked predictions.

This is what makes GraphSAGE **inductive**: new nodes are embedded using
the learned aggregation function, without retraining.

Main class
----------
GraphQuerier
    predict_links(features, ...)   → dict of ranked predictions
    recommend(features, ...)       → list of similar pieces
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np

__all__ = ["GraphQuerier"]


class GraphQuerier:
    """
    Query a trained GraphSAGE model for link prediction and recommendation.

    Parameters
    ----------
    model : GraphSAGELinkPredModel
        Trained model.
    data : HeteroData
        The full heterogeneous graph (used for message passing context).
    id_maps : dict
        Mapping dicts from the HeteroGraphConverter:
        ``{"piece_ids": {...}, "genre_ids": {...}, "artist_ids": {...}, ...}``
    device : str
        Device to run inference on.
    """

    def __init__(
        self,
        model,
        data,
        id_maps: Dict[str, Dict[str, int]],
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.data = data.to(device)
        self.device = device
        self.id_maps = id_maps

        # Reverse ID maps for label lookup
        self.reverse_maps: Dict[str, Dict[int, str]] = {}
        for key, mapping in id_maps.items():
            self.reverse_maps[key] = {v: k for k, v in mapping.items()}

    # ─────────────────────────────────────────────────────────────────────
    #  Link prediction for a new piece
    # ─────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_links(
        self,
        features: Dict[str, float],
        target_types: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Predict links for a new musical piece.

        Parameters
        ----------
        features : dict[str, float]
            Feature vector for the new piece (from FeatureExtractor).
        target_types : list[str], optional
            Node types to predict links to.  Default: Genre, Era, MusicalKey.
        top_k : int
            Number of top predictions per target type.

        Returns
        -------
        dict mapping target type → list of ``{"label": str, "score": float}``
        """
        if target_types is None:
            target_types = ["Genre", "Era", "MusicalKey"]

        # ── Add new node to the graph temporarily ─────────────────────
        data = self.data.clone()
        piece_x = data["MusicalPiece"].x
        n_pieces = piece_x.size(0)
        new_piece_idx = n_pieces

        # Build feature tensor for the new piece
        feat_dim = piece_x.size(1)
        new_feat = torch.zeros(1, feat_dim, device=self.device)

        # Map feature dict to tensor positions
        # This assumes feature columns are in the same order as training
        feat_keys = sorted(features.keys())
        for i, key in enumerate(feat_keys):
            if i < feat_dim:
                new_feat[0, i] = features[key]

        # Append to MusicalPiece features
        data["MusicalPiece"].x = torch.cat([piece_x, new_feat], dim=0)
        data["MusicalPiece"].num_nodes = n_pieces + 1

        # ── Encode all nodes ──────────────────────────────────────────
        z_dict = self.model.encode(data.x_dict, data.edge_index_dict)
        z_new = z_dict["MusicalPiece"][new_piece_idx].unsqueeze(0)

        # ── Score candidate edges per target type ─────────────────────
        results: Dict[str, List[Dict[str, Any]]] = {}

        for target_type in target_types:
            if target_type not in z_dict:
                continue

            z_targets = z_dict[target_type]
            n_targets = z_targets.size(0)

            # Score all candidate edges: new_piece → each target
            z_src = z_new.expand(n_targets, -1)
            scores = self.model.predictor(z_src, z_targets)
            probs = torch.sigmoid(scores)

            # Get top-K
            top_vals, top_idx = torch.topk(probs, min(top_k, n_targets))

            # Map IDs back to labels
            reverse_key = f"{target_type.lower()}_ids"
            reverse_map = self.reverse_maps.get(reverse_key, {})

            predictions = []
            for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
                label = reverse_map.get(idx, f"{target_type}_{idx}")
                predictions.append({"label": label, "score": round(val, 4)})

            results[target_type] = predictions

        return results

    # ─────────────────────────────────────────────────────────────────────
    #  Recommendation: find similar pieces
    # ─────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def recommend(
        self,
        features: Dict[str, float],
        top_k: int = 10,
        method: str = "cosine",
    ) -> List[Dict[str, Any]]:
        """
        Recommend similar musical pieces based on embedding similarity.

        Parameters
        ----------
        features : dict[str, float]
            Feature vector for the query piece.
        top_k : int
            Number of recommendations.
        method : ``'cosine'`` | ``'dot'``
            Similarity method.

        Returns
        -------
        list of ``{"track_id": str, "score": float}``
        """
        data = self.data.clone()
        piece_x = data["MusicalPiece"].x
        n_pieces = piece_x.size(0)

        # Build feature tensor for the query
        feat_dim = piece_x.size(1)
        new_feat = torch.zeros(1, feat_dim, device=self.device)
        feat_keys = sorted(features.keys())
        for i, key in enumerate(feat_keys):
            if i < feat_dim:
                new_feat[0, i] = features[key]

        data["MusicalPiece"].x = torch.cat([piece_x, new_feat], dim=0)
        data["MusicalPiece"].num_nodes = n_pieces + 1

        z_dict = self.model.encode(data.x_dict, data.edge_index_dict)
        z_pieces = z_dict["MusicalPiece"]

        z_query = z_pieces[-1].unsqueeze(0)
        z_existing = z_pieces[:-1]

        if method == "cosine":
            z_query_norm = z_query / z_query.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            z_existing_norm = z_existing / z_existing.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            scores = (z_query_norm @ z_existing_norm.T).squeeze(0)
        else:
            scores = (z_query @ z_existing.T).squeeze(0)

        top_vals, top_idx = torch.topk(scores, min(top_k, n_pieces))

        reverse_pieces = self.reverse_maps.get("piece_ids", {})
        recommendations = []
        for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
            track_id = reverse_pieces.get(idx, f"piece_{idx}")
            recommendations.append({"track_id": track_id, "score": round(val, 4)})

        return recommendations

    # ─────────────────────────────────────────────────────────────────────
    #  Full inference pipeline
    # ─────────────────────────────────────────────────────────────────────

    def infer(
        self,
        features: Dict[str, float],
        predict_types: Optional[List[str]] = None,
        recommend_k: int = 10,
        predict_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Run full inference: link prediction + recommendation.

        Parameters
        ----------
        features : dict[str, float]
            Feature vector.
        predict_types : list[str], optional
            Node types for link prediction.
        recommend_k : int
            Number of similar pieces to recommend.
        predict_k : int
            Number of top predictions per type.

        Returns
        -------
        dict with keys ``"predictions"`` and ``"recommendations"``
        """
        return {
            "predictions": self.predict_links(features, predict_types, predict_k),
            "recommendations": self.recommend(features, recommend_k),
        }
