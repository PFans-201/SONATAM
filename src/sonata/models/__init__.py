"""sonata.models — GraphSAGE link prediction & hybrid recommendation."""

from sonata.models.graph_models import (
    GraphSAGELinkPredModel,
    HeteroGraphSAGE,
    LinkPredictor,
)
from sonata.models.train import LinkPredTrainer, TrainerConfig
from sonata.models.evaluate import (
    evaluate_link_prediction,
    compute_mrr,
    compute_hits_at_k,
    plot_training_curves,
    plot_embedding_tsne,
    plot_score_distribution,
)

__all__ = [
    # Models
    "GraphSAGELinkPredModel",
    "HeteroGraphSAGE",
    "LinkPredictor",
    # Training
    "LinkPredTrainer",
    "TrainerConfig",
    # Evaluation
    "evaluate_link_prediction",
    "compute_mrr",
    "compute_hits_at_k",
    "plot_training_curves",
    "plot_embedding_tsne",
    "plot_score_distribution",
]
