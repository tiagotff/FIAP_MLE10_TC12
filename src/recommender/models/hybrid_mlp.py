"""Modelo híbrido de recomendação: embeddings + MLP.

Satisfaz o requisito "MLP ou embedding-based" do desafio ao ser as duas
coisas: embeddings de usuário/produto alimentam uma MLP no topo, junto
com features tabulares.
"""

from __future__ import annotations

import torch
from torch import nn

from recommender.config.model_config import HybridModelConfig
from recommender.models.base import RecommenderModel


class HybridMlpRecommender(RecommenderModel):
    """Gera embeddings de usuário/produto e pontua via uma MLP."""

    def __init__(self, config: HybridModelConfig) -> None:
        super().__init__()
        self._config = config
        emb = config.embedding

        self.user_embedding = nn.Embedding(emb.num_users, emb.user_embedding_dim)
        self.product_embedding = nn.Embedding(
            emb.num_products, emb.product_embedding_dim
        )
        input_dim = (
            emb.user_embedding_dim
            + emb.product_embedding_dim
            + config.mlp.tabular_feature_dim
        )
        self.mlp = self._build_mlp(input_dim, config.mlp)

    @staticmethod
    def _build_mlp(input_dim: int, mlp_config) -> nn.Sequential:  # noqa: ANN001
        """Monta a MLP como uma sequência de blocos Linear/BN/ReLU/Dropout."""
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in mlp_config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if mlp_config.use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(mlp_config.dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)

    def forward(
        self,
        user_ids: torch.Tensor,
        product_ids: torch.Tensor,
        tabular_features: torch.Tensor,
    ) -> torch.Tensor:
        """Ver RecommenderModel.forward."""
        user_vec = self.user_embedding(user_ids)
        product_vec = self.product_embedding(product_ids)
        x = torch.cat([user_vec, product_vec, tabular_features], dim=1)
        logits = self.mlp(x).squeeze(-1)
        return logits
