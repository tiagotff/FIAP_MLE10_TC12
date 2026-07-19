"""Abstract base class for recommendation models.

Defining this interface keeps the training loop and the Factory
decoupled from any specific architecture (Dependency Inversion /
Open-Closed: new model types are added by implementing this contract,
not by modifying existing code).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class RecommenderModel(nn.Module, ABC):
    """Contract every recommendation model must satisfy."""

    @abstractmethod
    def forward(
        self,
        user_ids: torch.Tensor,
        product_ids: torch.Tensor,
        tabular_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reorder-probability logits.

        Args:
            user_ids: Long tensor of shape (batch,).
            product_ids: Long tensor of shape (batch,).
            tabular_features: Float tensor of shape (batch, n_features).

        Returns:
            Logit tensor of shape (batch,). Apply sigmoid for probabilities.
        """
        raise NotImplementedError
