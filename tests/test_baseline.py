"""Testes unitários para o baseline Scikit-Learn (Regressão Logística)."""

from __future__ import annotations

import pandas as pd

from recommender.pipeline.baseline import _compute_metrics, _train_logistic_regression
from recommender.pipeline.feature_eng import FEATURE_COLUMNS, LABEL_COLUMN


def _synthetic_dataset(n: int = 200) -> pd.DataFrame:
    """Gera um dataset pequeno e sintético, só para exercitar o treino/avaliação."""
    import numpy as np

    rng = np.random.default_rng(42)
    data = {col: rng.random(n) for col in FEATURE_COLUMNS}
    data[LABEL_COLUMN] = rng.integers(0, 2, n)
    return pd.DataFrame(data)


def test_train_logistic_regression_fits_without_error() -> None:
    """O treino deve rodar sem erro e produzir um modelo com predict_proba."""
    train_df = _synthetic_dataset()
    model = _train_logistic_regression(train_df, random_seed=42)
    assert hasattr(model, "predict_proba")


def test_compute_metrics_returns_all_four_expected_keys() -> None:
    """As métricas calculadas devem ser exatamente as 4 exigidas pelo desafio."""
    train_df = _synthetic_dataset()
    val_df = _synthetic_dataset(n=50)
    model = _train_logistic_regression(train_df, random_seed=42)

    metrics = _compute_metrics(model, val_df)

    assert set(metrics.keys()) == {"auc_roc", "recall", "precision", "f1"}
    assert all(0.0 <= value <= 1.0 for value in metrics.values())
