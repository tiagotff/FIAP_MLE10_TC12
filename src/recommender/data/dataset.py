"""PyTorch Dataset wrapping preprocessed Instacart features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class InstacartReorderDataset(Dataset):
    """Yields (user_id, product_id, tabular_features, label) tuples.

    Args:
        user_ids: Encoded user id per row.
        product_ids: Encoded product id per row.
        tabular_features: Feature matrix, shape (n_rows, n_features).
        labels: Binary reorder label per row (1 = reordered).
    """

    def __init__(
        self,
        user_ids: np.ndarray,
        product_ids: np.ndarray,
        tabular_features: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self._user_ids = torch.as_tensor(user_ids, dtype=torch.long)
        self._product_ids = torch.as_tensor(product_ids, dtype=torch.long)
        self._tabular_features = torch.as_tensor(tabular_features, dtype=torch.float32)
        self._labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return self._labels.shape[0]

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self._user_ids[index],
            self._product_ids[index],
            self._tabular_features[index],
            self._labels[index],
        )

    @classmethod
    def from_dataframe(
        cls, df: pd.DataFrame, feature_columns: list[str], label_column: str
    ) -> InstacartReorderDataset:
        """Build a dataset directly from a preprocessed dataframe."""
        return cls(
            user_ids=df["user_id"].to_numpy(),
            product_ids=df["product_id"].to_numpy(),
            tabular_features=df[feature_columns].to_numpy(dtype=np.float32),
            labels=df[label_column].to_numpy(),
        )
