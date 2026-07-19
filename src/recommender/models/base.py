"""Classe base abstrata para modelos de recomendação.

Definir essa interface mantém o loop de treino e o Factory desacoplados
de qualquer arquitetura específica (Dependency Inversion / Open-Closed:
novos tipos de modelo são adicionados implementando esse contrato, sem
modificar código existente).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class RecommenderModel(nn.Module, ABC):
    """Contrato que todo modelo de recomendação deve cumprir."""

    @abstractmethod
    def forward(
        self,
        user_ids: torch.Tensor,
        product_ids: torch.Tensor,
        tabular_features: torch.Tensor,
    ) -> torch.Tensor:
        """Calcula os logits de probabilidade de recompra (reorder).

        Args:
            user_ids: Tensor de inteiros longos, shape (batch,).
            product_ids: Tensor de inteiros longos, shape (batch,).
            tabular_features: Tensor float, shape (batch, n_features).

        Returns:
            Tensor de logits, shape (batch,). Aplique sigmoid para obter
            probabilidades.
        """
        raise NotImplementedError
