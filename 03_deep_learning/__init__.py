"""
03_deep_learning — Deep learning modeling for harmonic analysis.

Modules
-------
dataset         : HarmonicDataset — PyTorch Dataset wrapper
models/genre_classifier    : MLP genre classifier from feature vector
models/sequence_model      : Transformer chord-sequence model
train           : training loop with checkpointing
evaluate        : evaluation utilities (accuracy, confusion matrix, t-SNE)
"""

from .dataset import HarmonicDataset, build_vocab

__all__ = ["HarmonicDataset", "build_vocab"]
