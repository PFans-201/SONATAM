"""
transformer.py
==============
Transformer-based model for chord-sequence modelling.

Two tasks are supported:
  1. **Genre classification** — encode the full chord sequence → class logit
  2. **Next-chord prediction** — autoregressive language model over chord tokens

Architecture
------------
  Embedding(vocab_size, d_model)
  + positional encoding (learned)
  → N × TransformerEncoderLayer(d_model, nhead, dim_feedforward)
  → [CLS] pool → Linear(d_model, num_classes)    (classification head)
  OR
  → Linear(d_model, vocab_size)                  (LM head)

Usage
-----
>>> from sonata.models.architectures.transformer import ChordTransformer
>>> model = ChordTransformer(vocab_size=50, num_classes=10, mode='classify')
>>> logits = model(token_ids)   # (B, L) → (B, num_classes)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ["ChordTransformer"]


class _PositionalEncoding(nn.Module):
    """Learned positional embeddings (simpler than sinusoidal for short sequences)."""

    def __init__(self, d_model: int, max_len: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout   = nn.Dropout(p=dropout)
        self.pos_embed = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        B, L, _ = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)  # (B, L)
        return self.dropout(x + self.pos_embed(positions))


class ChordTransformer(nn.Module):
    """
    Transformer encoder over chord token sequences.

    Parameters
    ----------
    vocab_size : int
        Size of the chord vocabulary (``HarmonicDataset.vocab_size``).
    num_classes : int
        Number of output classes.
    mode : ``'classify'`` | ``'lm'``
        * ``'classify'`` — mean-pool → linear → class logits
        * ``'lm'``       — per-token linear → next-token logits
    d_model : int
        Transformer embedding dimension.
    nhead : int
        Number of attention heads (must divide d_model).
    num_layers : int
        Number of Transformer encoder layers.
    dim_feedforward : int
        Size of the inner FFN in each encoder layer.
    max_seq_len : int
        Maximum sequence length (must match the Dataset).
    dropout : float
        Dropout applied in embedding + encoder.
    pad_token_id : int
        Index of the padding token (used to mask attention).
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        mode: str = "classify",
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        assert mode in ("classify", "lm"), f"Unknown mode: {mode!r}"
        self.mode         = mode
        self.d_model      = d_model
        self.pad_token_id = pad_token_id

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_enc   = _PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = dim_feedforward,
            dropout         = dropout,
            batch_first     = True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if mode == "classify":
            self.head = nn.Linear(d_model, num_classes)
        else:   # 'lm'
            self.head = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.embedding.weight, std=0.02)
        if self.embedding.padding_idx is not None:
            nn.init.zeros_(self.embedding.weight[self.embedding.padding_idx])
        nn.init.xavier_uniform_(self.head.weight)

    def forward(
        self,
        token_ids: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        token_ids : LongTensor (B, L)
        src_key_padding_mask : BoolTensor (B, L), True where tokens are padding.
            Built automatically from pad_token_id if not supplied.

        Returns
        -------
        classify mode : FloatTensor (B, num_classes)
        lm mode       : FloatTensor (B, L, vocab_size)
        """
        if src_key_padding_mask is None:
            src_key_padding_mask = (token_ids == self.pad_token_id)  # (B, L)

        x = self.embedding(token_ids) * math.sqrt(self.d_model)   # (B, L, d_model)
        x = self.pos_enc(x)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        if self.mode == "classify":
            # Mean-pool over non-padding positions
            mask_f = (~src_key_padding_mask).unsqueeze(-1).float()  # (B, L, 1)
            pooled = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
            return self.head(pooled)                                 # (B, num_classes)
        else:
            return self.head(x)                                      # (B, L, vocab_size)

    def predict(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices (classify mode) or next-token ids (lm mode)."""
        self.eval()
        with torch.no_grad():
            out = self.forward(token_ids)
            return out.argmax(dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
