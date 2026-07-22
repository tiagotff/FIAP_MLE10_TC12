"""Testes unitários do InstacartReorderDataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from recommender.data.dataset import InstacartReorderDataset


def _sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [0, 1, 2],
            "product_id": [5, 6, 7],
            "feat_a": [0.1, 0.2, 0.3],
            "feat_b": [1.0, 2.0, 3.0],
            "reordered": [1, 0, 1],
        }
    )


def test_len_matches_number_of_rows() -> None:
    """O tamanho do dataset deve bater com o número de linhas de entrada."""
    df = _sample_dataframe()
    dataset = InstacartReorderDataset.from_dataframe(
        df, ["feat_a", "feat_b"], "reordered"
    )
    assert len(dataset) == 3


def test_getitem_returns_expected_tensor_types_and_values() -> None:
    """Cada item deve retornar (user_id, product_id, features, label) como tensores."""
    df = _sample_dataframe()
    dataset = InstacartReorderDataset.from_dataframe(
        df, ["feat_a", "feat_b"], "reordered"
    )

    user_id, product_id, features, label = dataset[0]

    assert user_id.item() == 0
    assert product_id.item() == 5
    assert torch.allclose(features, torch.tensor([0.1, 1.0], dtype=torch.float32))
    assert label.item() == 1


def test_from_dataframe_handles_multiple_feature_columns() -> None:
    """from_dataframe deve montar a matriz de features na ordem das colunas pedidas."""
    df = _sample_dataframe()
    dataset = InstacartReorderDataset.from_dataframe(
        df, ["feat_b", "feat_a"], "reordered"
    )
    _, _, features, _ = dataset[1]
    assert torch.allclose(features, torch.tensor([2.0, 0.2], dtype=torch.float32))


def test_dataset_accepts_numpy_arrays_directly() -> None:
    """O construtor direto (sem from_dataframe) deve aceitar arrays numpy."""
    dataset = InstacartReorderDataset(
        user_ids=np.array([0, 1]),
        product_ids=np.array([1, 2]),
        tabular_features=np.array([[0.5, 0.5], [0.1, 0.9]]),
        labels=np.array([1, 0]),
    )
    assert len(dataset) == 2
