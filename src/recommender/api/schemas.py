"""Schemas de request/response da API de inferência (Pydantic)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Payload de entrada para uma predição de recompra."""

    user_id: int = Field(..., description="ID original do usuário (Instacart).")
    product_id: int = Field(..., description="ID original do produto (Instacart).")
    purchase_count: float = Field(
        ..., description="Nº de vezes que o usuário comprou esse produto."
    )
    days_since_last_order: float = Field(
        ..., description="Dias desde o último pedido do usuário."
    )
    order_hour_of_day: float = Field(..., description="Hora do pedido (0-23).")
    order_dow: float = Field(..., description="Dia da semana do pedido (0-6).")
    basket_size: float = Field(..., description="Nº de itens no carrinho.")


class PredictResponse(BaseModel):
    """Resultado de uma predição de recompra."""

    reorder_probability: float = Field(
        ..., description="Probabilidade de recompra, entre 0 e 1."
    )
    model_version: str = Field(..., description="Versão do modelo usada na predição.")


class HealthResponse(BaseModel):
    """Resposta do endpoint de health check."""

    status: str
    model_version: str
