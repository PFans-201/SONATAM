"""models — Neural network architectures for the Harmonic-KG project."""

from .genre_classifier import GenreClassifier
from .sequence_model   import ChordTransformer

__all__ = ["GenreClassifier", "ChordTransformer"]
