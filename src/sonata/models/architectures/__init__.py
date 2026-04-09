"""sonata.models.architectures — neural network architecture definitions."""

from sonata.models.architectures.classifier import GenreClassifier
from sonata.models.architectures.transformer import ChordTransformer

__all__ = ["GenreClassifier", "ChordTransformer"]
