"""
graph_models.py
===============
GraphSAGE-based heterogeneous GNN for **link prediction** and
**hybrid recommendation** on the SONATAM musical knowledge graph.

Architecture
------------
::

    MusicalPiece features (jsym + sem)
            │
            ▼
    ┌───────────────────────────┐
    │  HeteroGraphSAGE          │   2-layer heterogeneous GraphSAGE
    │  (per-relation message    │   with per-type linear projections
    │   passing + aggregation)  │
    └───────────────────────────┘
            │
            ▼
    Node embeddings  z_u, z_v
            │
            ▼
    ┌───────────────────────────┐
    │  LinkPredictor (MLP)      │   σ(MLP(z_u ∥ z_v))  →  edge prob
    └───────────────────────────┘

Key design decisions
--------------------
* **Inductive**: GraphSAGE learns an *aggregation function*, not fixed
  embeddings → new nodes (e.g. a new MP3) can be scored without retraining.
* **Heterogeneous**: separate message-passing weights per edge type
  (``hasGenre``, ``hasArtist``, ``listenedTo``, …).
* **Link prediction**: trained to predict missing edges
  (genre, composer, era) for graph completion.
* **Hybrid recommendation**: user–piece edges enable collaborative
  filtering alongside content-based (feature) signals.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["HeteroGraphSAGE", "LinkPredictor", "GraphSAGELinkPredModel"]


class HeteroGraphSAGE(nn.Module):
    """
    Heterogeneous GraphSAGE encoder.

    For each node type that has input features (``x``), applies a shared
    linear projection.  For node types without features, learns a
    trainable embedding table.

    Then runs ``num_layers`` rounds of heterogeneous message passing using
    ``SAGEConv`` per edge type.

    Parameters
    ----------
    metadata : tuple
        ``(node_types, edge_types)`` from ``HeteroData.metadata()``.
    in_channels_dict : dict[str, int]
        Input feature dimensionality per node type.
        Types missing from this dict get a learnable embedding of size
        ``hidden_channels``.
    hidden_channels : int
        Hidden & output dimensionality for all node types.
    num_layers : int
        Number of GraphSAGE message-passing layers.
    dropout : float
        Dropout rate between layers.
    aggregator : str
        SAGEConv aggregator: ``"mean"`` | ``"max"`` | ``"lstm"``.
    """

    def __init__(
        self,
        metadata: Tuple,
        in_channels_dict: Dict[str, int],
        hidden_channels: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        aggregator: str = "mean",
    ) -> None:
        super().__init__()
        from torch_geometric.nn import HeteroConv, SAGEConv

        node_types, edge_types = metadata
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout

        # ── Input projections / embeddings ────────────────────────────
        self.input_projs = nn.ModuleDict()
        self.input_embeds = nn.ModuleDict()

        for nt in node_types:
            if nt in in_channels_dict and in_channels_dict[nt] > 0:
                self.input_projs[nt] = nn.Linear(in_channels_dict[nt], hidden_channels)
            else:
                # Learnable embedding for featureless node types
                # num_nodes will be set at forward time via lazy init
                self.input_embeds[nt] = None  # placeholder

        # ── Heterogeneous SAGEConv layers ────────────────────────────
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for et in edge_types:
                conv_dict[et] = SAGEConv(
                    (-1, -1), hidden_channels, aggr=aggregator,
                )
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        self.norms = nn.ModuleList([
            nn.ModuleDict({nt: nn.LayerNorm(hidden_channels) for nt in node_types})
            for _ in range(num_layers)
        ])

    def _init_embeddings(self, x_dict: Dict[str, torch.Tensor]) -> None:
        """Lazy-initialise embedding tables for featureless node types."""
        for nt in self.node_types:
            if nt in self.input_embeds and self.input_embeds[nt] is None:
                num_nodes = x_dict[nt].size(0) if nt in x_dict else 0
                if num_nodes > 0:
                    device = next(self.parameters()).device
                    self.input_embeds[nt] = nn.Embedding(
                        num_nodes, self.hidden_channels
                    ).to(device)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: project inputs → message passing → node embeddings.

        Parameters
        ----------
        x_dict : dict[str, Tensor]
            Node features per type.
        edge_index_dict : dict[tuple, Tensor]
            Edge indices per edge type.

        Returns
        -------
        dict[str, Tensor]
            Node embeddings per type, each of shape ``(N, hidden_channels)``.
        """
        self._init_embeddings(x_dict)

        # ── Input projection ─────────────────────────────────────────
        h_dict: Dict[str, torch.Tensor] = {}
        for nt in self.node_types:
            if nt in self.input_projs and nt in x_dict:
                h_dict[nt] = self.input_projs[nt](x_dict[nt])
            elif nt in self.input_embeds and self.input_embeds[nt] is not None:
                n = x_dict[nt].size(0) if nt in x_dict else self.input_embeds[nt].num_embeddings
                idx = torch.arange(n, device=self.input_embeds[nt].weight.device)
                h_dict[nt] = self.input_embeds[nt](idx)
            elif nt in x_dict:
                # Fallback: identity (features already correct dim)
                h_dict[nt] = x_dict[nt]

        # ── Message passing layers ───────────────────────────────────
        for i, conv in enumerate(self.convs):
            h_dict_new = conv(h_dict, edge_index_dict)
            for nt in h_dict_new:
                h = self.norms[i][nt](h_dict_new[nt])
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
                # Residual connection
                if nt in h_dict and h_dict[nt].shape == h.shape:
                    h = h + h_dict[nt]
                h_dict[nt] = h

        return h_dict


class LinkPredictor(nn.Module):
    """
    MLP-based link predictor: predicts the probability of an edge
    between two node embeddings.

    Score = σ(MLP(z_src ∥ z_dst))

    Parameters
    ----------
    in_channels : int
        Dimensionality of each node embedding (hidden_channels from encoder).
    hidden_channels : int
        MLP hidden size.
    num_layers : int
        Number of MLP layers (including final output layer).
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        in_channels: int = 128,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []

        # Input: concatenation of src + dst embeddings
        current = in_channels * 2

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current, hidden_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current = hidden_channels

        layers.append(nn.Linear(current, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        z_src: torch.Tensor,
        z_dst: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict edge probabilities.

        Parameters
        ----------
        z_src : Tensor  (E, d)
            Source node embeddings.
        z_dst : Tensor  (E, d)
            Destination node embeddings.

        Returns
        -------
        Tensor  (E,)
            Edge probabilities (after sigmoid).
        """
        h = torch.cat([z_src, z_dst], dim=-1)
        return self.mlp(h).squeeze(-1)


class GraphSAGELinkPredModel(nn.Module):
    """
    End-to-end model combining the HeteroGraphSAGE encoder with
    the LinkPredictor decoder.

    This is the main model used for training and inference.

    Parameters
    ----------
    metadata : tuple
        ``(node_types, edge_types)`` from ``HeteroData.metadata()``.
    in_channels_dict : dict[str, int]
        Input feature dimensionality per node type.
    hidden_channels : int
        Hidden dimensionality for GraphSAGE layers.
    num_sage_layers : int
        Number of GraphSAGE message-passing layers.
    num_pred_layers : int
        Number of MLP layers in the link predictor.
    pred_hidden : int
        Hidden size of the link predictor MLP.
    dropout : float
        Dropout rate.
    aggregator : str
        SAGEConv aggregator.
    """

    def __init__(
        self,
        metadata: Tuple,
        in_channels_dict: Dict[str, int],
        hidden_channels: int = 128,
        num_sage_layers: int = 2,
        num_pred_layers: int = 2,
        pred_hidden: int = 64,
        dropout: float = 0.3,
        aggregator: str = "mean",
    ) -> None:
        super().__init__()
        self.encoder = HeteroGraphSAGE(
            metadata=metadata,
            in_channels_dict=in_channels_dict,
            hidden_channels=hidden_channels,
            num_layers=num_sage_layers,
            dropout=dropout,
            aggregator=aggregator,
        )
        self.predictor = LinkPredictor(
            in_channels=hidden_channels,
            hidden_channels=pred_hidden,
            num_layers=num_pred_layers,
            dropout=dropout,
        )

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple, torch.Tensor],
        edge_label_index: torch.Tensor,
        src_type: str,
        dst_type: str,
    ) -> torch.Tensor:
        """
        Full forward pass: encode → predict link scores.

        Parameters
        ----------
        x_dict, edge_index_dict : as in HeteroGraphSAGE.forward
        edge_label_index : Tensor (2, E)
            Edges to score (src indices in row 0, dst in row 1).
        src_type, dst_type : str
            Node types for source and destination.

        Returns
        -------
        Tensor (E,)
            Predicted edge scores (logits, pre-sigmoid).
        """
        z_dict = self.encoder(x_dict, edge_index_dict)
        z_src = z_dict[src_type][edge_label_index[0]]
        z_dst = z_dict[dst_type][edge_label_index[1]]
        return self.predictor(z_src, z_dst)

    def encode(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Get node embeddings without predicting edges."""
        return self.encoder(x_dict, edge_index_dict)
