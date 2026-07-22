"""Testes unitários do HybridMlpRecommender (forward pass)."""

from __future__ import annotations

import torch

from recommender.config.model_config import (
    EmbeddingConfig,
    HybridModelConfig,
    MlpConfig,
)
from recommender.models.hybrid_mlp import HybridMlpRecommender


def _build_model(tabular_feature_dim: int = 5) -> HybridMlpRecommender:
    """Monta um modelo pequeno, só para exercitar o forward pass."""
    config = HybridModelConfig(
        embedding=EmbeddingConfig(num_users=10, num_products=20),
        mlp=MlpConfig(tabular_feature_dim=tabular_feature_dim),
    )
    return HybridMlpRecommender(config)


def test_forward_returns_one_logit_per_example() -> None:
    """A saída deve ter exatamente um logit por exemplo do batch."""
    model = _build_model()
    batch_size = 4
    user_ids = torch.randint(0, 10, (batch_size,))
    product_ids = torch.randint(0, 20, (batch_size,))
    features = torch.rand(batch_size, 5)

    logits = model(user_ids, product_ids, features)

    assert logits.shape == (batch_size,)


def test_forward_output_is_finite_and_not_constant() -> None:
    """Os logits devem ser números válidos (sem NaN/Inf) e não todos iguais."""
    model = _build_model()
    user_ids = torch.randint(0, 10, (8,))
    product_ids = torch.randint(0, 20, (8,))
    features = torch.rand(8, 5)

    logits = model(user_ids, product_ids, features)

    assert torch.isfinite(logits).all()
    assert logits.unique().numel() > 1


def test_forward_respects_tabular_feature_dim() -> None:
    """O modelo deve aceitar o número de features tabulares configurado."""
    model = _build_model(tabular_feature_dim=3)
    user_ids = torch.randint(0, 10, (2,))
    product_ids = torch.randint(0, 20, (2,))
    features = torch.rand(2, 3)

    logits = model(user_ids, product_ids, features)

    assert logits.shape == (2,)
