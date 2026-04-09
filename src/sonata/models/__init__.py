"""sonata.models — deep learning architectures, dataset, training, and evaluation."""

from sonata.models.architectures.classifier import GenreClassifier
from sonata.models.architectures.transformer import ChordTransformer
from sonata.models.dataset import HarmonicDataset, build_vocab
from sonata.models.train import Trainer, TrainerConfig
from sonata.models.evaluate import classification_report_df, confusion_matrix_plot, tsne_plot

__all__ = [
    "GenreClassifier",
    "ChordTransformer",
    "HarmonicDataset",
    "build_vocab",
    "Trainer",
    "TrainerConfig",
    "classification_report_df",
    "confusion_matrix_plot",
    "tsne_plot",
]
