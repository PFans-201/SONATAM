""""""

test_models.pytest_models.py

============================

Unit tests for sonata.models (GraphSAGE, LinkPredictor, trainer config).Unit tests for sonata.models (GenreClassifier, ChordTransformer, HarmonicDataset).

""""""



import pytestimport pytest





def test_graph_models_import():def test_genre_classifier_import():

    from sonata.models.graph_models import (    from sonata.models.architectures.classifier import GenreClassifier

        HeteroGraphSAGE,    assert GenreClassifier is not None

        LinkPredictor,

        GraphSAGELinkPredModel,

    )def test_genre_classifier_instantiation():

    assert HeteroGraphSAGE is not None    import torch

    assert LinkPredictor is not None    from sonata.models.architectures.classifier import GenreClassifier

    assert GraphSAGELinkPredModel is not None

    model = GenreClassifier(input_dim=19, num_classes=5)

    assert model.count_parameters() > 0

def test_link_predictor_instantiation():

    import torch    x = torch.randn(4, 19)

    from sonata.models.graph_models import LinkPredictor    logits = model(x)

    assert logits.shape == (4, 5)

    pred = LinkPredictor(in_channels=64, hidden_channels=32, num_layers=2)

    z_src = torch.randn(10, 64)

    z_dst = torch.randn(10, 64)def test_chord_transformer_import():

    scores = pred(z_src, z_dst)    from sonata.models.architectures.transformer import ChordTransformer

    assert scores.shape == (10,)    assert ChordTransformer is not None





def test_trainer_config_defaults():def test_chord_transformer_instantiation():

    from sonata.models.train import TrainerConfig    import torch

    from sonata.models.architectures.transformer import ChordTransformer

    config = TrainerConfig()

    assert config.epochs == 100    model = ChordTransformer(vocab_size=30, num_classes=5, mode="classify")

    assert config.hidden_channels == 128    assert model.count_parameters() > 0

    assert config.lr == 1e-3

    assert config.device in ("auto", "cpu", "cuda")    token_ids = torch.randint(0, 30, (4, 16))

    logits = model(token_ids)

    assert logits.shape == (4, 5)

def test_trainer_config_from_config():
    from sonata.models.train import TrainerConfig

    config = TrainerConfig.from_config()
    assert isinstance(config.epochs, int)
    assert isinstance(config.lr, float)
    assert config.device in ("cpu", "cuda")


def test_evaluate_imports():
    from sonata.models.evaluate import (
        evaluate_link_prediction,
        compute_mrr,
        compute_hits_at_k,
        plot_training_curves,
    )
    assert evaluate_link_prediction is not None
    assert compute_mrr is not None
    assert compute_hits_at_k is not None


def test_compute_mrr():
    import torch
    from sonata.models.evaluate import compute_mrr

    scores = torch.tensor([0.9, 0.1, 0.5, 0.3])
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    mrr = compute_mrr(scores, labels)
    assert 0.0 <= mrr <= 1.0


def test_compute_hits_at_k():
    import torch
    from sonata.models.evaluate import compute_hits_at_k

    scores = torch.tensor([0.9, 0.8, 0.7, 0.1])
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    hits = compute_hits_at_k(scores, labels, k=3)
    assert 0.0 <= hits <= 1.0
