"""Dataset PyTorch que encapsula as features pré-processadas do Instacart."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class InstacartReorderDataset(Dataset):
    """Retorna tuplas (user_id, product_id, tabular_features, label).

    Args:
        user_ids: Id de usuário codificado, por linha.
        product_ids: Id de produto codificado, por linha.
        tabular_features: Matriz de features, shape (n_linhas, n_features).
        labels: Rótulo binário de recompra por linha (1 = houve reorder).
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
        """Retorna o número de exemplos no dataset."""
        return self._labels.shape[0]

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retorna o exemplo de índice `index` como tupla de tensores."""
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
        """Constrói o dataset diretamente a partir de um dataframe pré-processado."""
        return cls(
            user_ids=df["user_id"].to_numpy(),
            product_ids=df["product_id"].to_numpy(),
            tabular_features=df[feature_columns].to_numpy(dtype=np.float32),
            labels=df[label_column].to_numpy(),
        )
