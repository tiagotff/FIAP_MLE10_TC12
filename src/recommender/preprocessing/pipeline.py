"""Classe de contexto que orquestra uma lista de preprocessing strategies."""

from __future__ import annotations

import pandas as pd

from recommender.preprocessing.base import PreprocessingStrategy


class FeaturePipeline:
    """Executa uma sequência de `PreprocessingStrategy` e concatena os resultados."""

    def __init__(self, strategies: list[PreprocessingStrategy]) -> None:
        self._strategies = strategies

    def fit(self, raw_orders: pd.DataFrame) -> FeaturePipeline:
        """Ajusta (fit) cada estratégia sobre os pedidos brutos."""
        for strategy in self._strategies:
            strategy.fit(raw_orders)
        return self

    def transform(self, raw_orders: pd.DataFrame) -> pd.DataFrame:
        """Transforma os pedidos brutos na matriz completa de features tabulares."""
        feature_frames = [s.transform(raw_orders) for s in self._strategies]
        return pd.concat(feature_frames, axis=1)

    def fit_transform(self, raw_orders: pd.DataFrame) -> pd.DataFrame:
        """Atalho conveniente para `fit(df).transform(df)`."""
        return self.fit(raw_orders).transform(raw_orders)
