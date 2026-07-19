"""Factory Pattern for model instantiation.

Centralizes model creation so the training pipeline depends only on
`ModelFactory.create`, not on concrete model classes. Adding a new
architecture means registering it here, without touching callers
(Open-Closed Principle).
"""

from __future__ import annotations

from collections.abc import Callable

from recommender.config.model_config import HybridModelConfig
from recommender.models.base import RecommenderModel
from recommender.models.hybrid_mlp import HybridMlpRecommender

_ModelBuilder = Callable[[HybridModelConfig], RecommenderModel]


class ModelFactory:
    """Builds `RecommenderModel` instances from a config's `model_type`."""

    _registry: dict[str, _ModelBuilder] = {
        "hybrid_mlp": HybridMlpRecommender,
    }

    @classmethod
    def create(cls, config: HybridModelConfig) -> RecommenderModel:
        """Instantiate the model registered under `config.model_type`.

        Args:
            config: Model configuration; `config.model_type` selects the
                builder.

        Returns:
            An initialized `RecommenderModel`.

        Raises:
            ValueError: If `config.model_type` is not registered.
        """
        builder = cls._registry.get(config.model_type)
        if builder is None:
            known = ", ".join(sorted(cls._registry))
            raise ValueError(
                f"Unknown model_type '{config.model_type}'. Known types: {known}"
            )
        return builder(config)

    @classmethod
    def register(cls, model_type: str, builder: _ModelBuilder) -> None:
        """Register a new model builder under `model_type`."""
        cls._registry[model_type] = builder
