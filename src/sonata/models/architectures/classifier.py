"""
classifier.py
=============
Multi-layer perceptron (MLP) for genre classification from the harmonic
feature vector produced by ``MIDIHarmonicAnalyzer.extract_features()``.

Architecture
------------
Input  (N_FEATURES)
  └─ Linear → BatchNorm → ReLU → Dropout
     └─ Linear → BatchNorm → ReLU → Dropout
        └─ Linear → BatchNorm → ReLU → Dropout
           └─ Linear (N_CLASSES)

Usage
-----
>>> from sonata.models.architectures.classifier import GenreClassifier
>>> model = GenreClassifier(input_dim=19, num_classes=10)
>>> logits = model(features_tensor)   # shape: (B, N_CLASSES)
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

__all__ = ["GenreClassifier"]


class GenreClassifier(nn.Module):
    """
    Feed-forward MLP for genre classification from a harmonic feature vector.

    Parameters
    ----------
    input_dim : int
        Number of input features (matches ``HarmonicDataset.input_dim``).
    num_classes : int
        Number of output classes.
    hidden_dims : list of int
        Sizes of hidden layers.  Default: [256, 128, 64].
    dropout : float
        Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: List[int] = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers: List[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, input_dim)

        Returns
        -------
        torch.Tensor, shape (B, num_classes)  — raw logits
        """
        return self.net(x)

    # ── Convenience helpers ────────────────────────────────────────────

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices (no gradient)."""
        self.eval()
        with torch.no_grad():
            return self.forward(x).argmax(dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
