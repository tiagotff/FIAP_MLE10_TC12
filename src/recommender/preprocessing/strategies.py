"""Estratégias concretas de pré-processamento para dados do Instacart."""

from __future__ import annotations

import pandas as pd

from recommender.preprocessing.base import PreprocessingStrategy


class RecencyFrequencyStrategy(PreprocessingStrategy):
    """Calcula features de recência e frequência de compra por par usuário-produto."""

    def __init__(self) -> None:
        self._global_mean_frequency: float | None = None

    def fit(self, raw_orders: pd.DataFrame) -> RecencyFrequencyStrategy:
        """Armazena a frequência média global de compra para referência futura."""
        counts = raw_orders.groupby(["user_id", "product_id"]).size()
        self._global_mean_frequency = float(counts.mean())
        return self

    def transform(self, raw_orders: pd.DataFrame) -> pd.DataFrame:
        """Calcula purchase_count e days_since_last_order por usuário-produto."""
        grouped = raw_orders.groupby(["user_id", "product_id"])
        features = grouped.agg(
            purchase_count=("order_id", "count"),
            days_since_last_order=("days_since_prior_order", "min"),
        )
        return features.reset_index(drop=True)


class TemporalPatternStrategy(PreprocessingStrategy):
    """Calcula features de horário do pedido e dia da semana."""

    def fit(self, raw_orders: pd.DataFrame) -> TemporalPatternStrategy:
        """Não há estatísticas a aprender; retorna self por consistência."""
        return self

    def transform(self, raw_orders: pd.DataFrame) -> pd.DataFrame:
        """Extrai as colunas de hora do pedido e dia da semana."""
        return raw_orders[["order_hour_of_day", "order_dow"]].reset_index(drop=True)


class BasketSizeStrategy(PreprocessingStrategy):
    """Calcula o tamanho médio do carrinho (pedido) do usuário."""

    def fit(self, raw_orders: pd.DataFrame) -> BasketSizeStrategy:
        """Não há estatísticas a aprender; retorna self por consistência."""
        return self

    def transform(self, raw_orders: pd.DataFrame) -> pd.DataFrame:
        """Calcula o número de produtos em cada pedido, por linha."""
        basket_sizes = raw_orders.groupby("order_id")["product_id"].transform("count")
        return pd.DataFrame({"basket_size": basket_sizes}).reset_index(drop=True)
