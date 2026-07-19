"""Configuration schemas for model architecture and training.

Etapa 2 will wire these to environment variables via Pydantic Settings.
For now, these are plain, typed dataclasses used by the Factory and
training pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EmbeddingConfig:
    """Dimensions for the user/product embedding tables.

    Attributes:
        num_users: Cardinality of the user vocabulary.
        num_products: Cardinality of the product vocabulary.
        user_embedding_dim: Size of each user embedding vector.
        product_embedding_dim: Size of each product embedding vector.
    """

    num_users: int
    num_products: int
    user_embedding_dim: int = 32
    product_embedding_dim: int = 32


@dataclass(frozen=True)
class MlpConfig:
    """Hyperparameters for the MLP head of the hybrid model.

    Attributes:
        tabular_feature_dim: Number of tabular (non-embedding) input features.
        hidden_dims: Sizes of the hidden layers, in order.
        dropout: Dropout probability applied after each hidden layer.
        use_batchnorm: Whether to apply BatchNorm1d after each hidden layer.
    """

    tabular_feature_dim: int
    hidden_dims: tuple[int, ...] = (128, 64, 32)
    dropout: float = 0.3
    use_batchnorm: bool = True


@dataclass(frozen=True)
class HybridModelConfig:
    """Full configuration for the hybrid embedding + MLP model.

    Attributes:
        embedding: Embedding table configuration.
        mlp: MLP head configuration.
        model_type: Identifier consumed by the ModelFactory.
    """

    embedding: EmbeddingConfig
    mlp: MlpConfig
    model_type: str = "hybrid_mlp"


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters for the training loop.

    Attributes:
        learning_rate: Optimizer learning rate.
        batch_size: Mini-batch size.
        max_epochs: Upper bound on training epochs.
        early_stopping_patience: Epochs without val AUC improvement before stop.
        random_seed: Seed applied to torch/numpy/random for reproducibility.
    """

    learning_rate: float = 1e-3
    batch_size: int = 512
    max_epochs: int = 30
    early_stopping_patience: int = 5
    random_seed: int = 42
    metrics: list[str] = field(
        default_factory=lambda: ["auc_roc", "recall", "precision", "f1"]
    )
