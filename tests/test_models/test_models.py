"""
test_models.py
==============
Unit tests for sonata.models (GenreClassifier, ChordTransformer, HarmonicDataset).
"""

import pytest


def test_genre_classifier_import():
    from sonata.models.architectures.classifier import GenreClassifier
    assert GenreClassifier is not None


def test_genre_classifier_instantiation():
    import torch
    from sonata.models.architectures.classifier import GenreClassifier

    model = GenreClassifier(input_dim=19, num_classes=5)
    assert model.count_parameters() > 0

    x = torch.randn(4, 19)
    logits = model(x)
    assert logits.shape == (4, 5)


def test_chord_transformer_import():
    from sonata.models.architectures.transformer import ChordTransformer
    assert ChordTransformer is not None


def test_chord_transformer_instantiation():
    import torch
    from sonata.models.architectures.transformer import ChordTransformer

    model = ChordTransformer(vocab_size=30, num_classes=5, mode="classify")
    assert model.count_parameters() > 0

    token_ids = torch.randint(0, 30, (4, 16))
    logits = model(token_ids)
    assert logits.shape == (4, 5)
