"""Strategy Pattern para pré-processamento de features.

Cada estratégia concreta encapsula uma forma de transformar dados brutos
de pedidos do Instacart em features prontas para o modelo. Trocar de
estratégia (ex: para um estudo de ablação ou um novo conjunto de
features) não exige alterar o pipeline que as consome (Single
Responsibility / Open-Closed).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class PreprocessingStrategy(ABC):
    """Contrato para uma única estratégia de engenharia de features."""

    @abstractmethod
    def fit(self, raw_orders: pd.DataFrame) -> PreprocessingStrategy:
        """Aprende as estatísticas necessárias no momento do transform.

        Args:
            raw_orders: Dataframe bruto de orders/order_products.

        Returns:
            self, para permitir o uso fluente
            `strategy.fit(df).transform(df)`.
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, raw_orders: pd.DataFrame) -> pd.DataFrame:
        """Produz as colunas de features pelas quais essa estratégia é responsável.

        Args:
            raw_orders: Dataframe bruto de orders/order_products.

        Returns:
            Um dataframe com o mesmo índice de linhas de `raw_orders` e
            uma coluna por feature calculada por essa estratégia.
        """
        raise NotImplementedError
