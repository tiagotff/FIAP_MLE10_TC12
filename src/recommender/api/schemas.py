"""Schemas de request/response da API de inferência (Pydantic)."""

from __future__ import annotations

from pydantic import BaseModel, Field

_EXAMPLE_PREDICT_PAYLOAD = {
    "user_id": 1,
    "product_id": 196,
    "purchase_count": 3,
    "days_since_last_order": 7,
    "order_hour_of_day": 10,
    "order_dow": 2,
    "basket_size": 8,
}


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

    model_config = {"json_schema_extra": {"example": _EXAMPLE_PREDICT_PAYLOAD}}


class PredictResponse(BaseModel):
    """Resultado de uma predição de recompra."""

    reorder_probability: float | None = Field(
        ...,
        description=(
            "Probabilidade de recompra, entre 0 e 1. `null` se o par "
            "usuário-produto for cold-start (só em respostas de lote)."
        ),
    )
    model_version: str = Field(..., description="Versão do modelo usada na predição.")

    model_config = {
        "json_schema_extra": {
            "example": {"reorder_probability": 0.73, "model_version": "1"}
        }
    }


class BatchPredictRequest(BaseModel):
    """Payload de entrada para predição em lote (um forward pass só)."""

    items: list[PredictRequest] = Field(
        ..., min_length=1, description="Lista de pares usuário-produto a avaliar."
    )

    model_config = {
        "json_schema_extra": {"example": {"items": [_EXAMPLE_PREDICT_PAYLOAD]}}
    }


class BatchPredictResponse(BaseModel):
    """Resultado de uma predição em lote, na mesma ordem da requisição."""

    results: list[PredictResponse] = Field(
        ..., description="Um resultado por item enviado, na mesma ordem."
    )


class ModelInfoResponse(BaseModel):
    """Metadados do modelo atualmente carregado pela API."""

    model_version: str = Field(..., description="Versão servida (tag de deploy).")
    num_users: int = Field(..., description="Nº de usuários no vocabulário de treino.")
    num_products: int = Field(
        ..., description="Nº de produtos no vocabulário de treino."
    )
    embedding_dim: int = Field(
        ..., description="Dimensão dos embeddings de usuário/produto."
    )
    mlp_hidden_dims: list[int] = Field(
        ..., description="Tamanhos das camadas ocultas da MLP."
    )
    feature_columns: list[str] = Field(
        ..., description="Ordem esperada das features tabulares no `predict`."
    )


class HealthResponse(BaseModel):
    """Resposta do endpoint de health check."""

    status: str = Field(..., description="'ok' quando o modelo já está carregado.")
    model_version: str = Field(..., description="Versão do modelo servido.")


class ErrorResponse(BaseModel):
    """Formato padrão de erro (usado na documentação OpenAPI)."""

    detail: str
