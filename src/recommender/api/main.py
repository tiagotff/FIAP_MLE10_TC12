"""API HTTP de inferência (FastAPI): serve predições de recompra.

Executada via `uvicorn recommender.api.main:app`. Carrega o modelo e os
encoders uma única vez, na subida do processo — não a cada requisição.

Endpoints:
    GET  /          — informações básicas e link para a documentação.
    GET  /health     — health check (usado pelo Cloud Run).
    GET  /model/info — metadados do modelo carregado.
    POST /predict    — predição para um par usuário-produto.
    POST /predict/batch — predição em lote (um único forward pass).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, status

from recommender.api.inference import InferenceService
from recommender.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)
from recommender.config.settings import get_settings

_service: InferenceService | None = None
_settings = get_settings()

_COLD_START_DETAIL = (
    "user_id ou product_id desconhecido (fora do vocabulário de treino)."
)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201
    """Carrega o modelo na subida da API; libera recursos ao encerrar."""
    global _service
    _service = InferenceService(Path(_settings.models_dir), _settings.device)
    yield
    _service = None


app = FastAPI(
    title="Instacart Recommender API",
    description=(
        "Prevê a probabilidade de um usuário recomprar um produto, usando "
        "o modelo híbrido (embeddings + MLP) registrado no MLflow Model "
        "Registry, stage Production. Projeto do Tech Challenge Fase 02 — "
        "FIAP MLE10."
    ),
    version=_settings.model_version,
    lifespan=lifespan,
    contact={"name": "Tiago de Freitas Faustino"},
    license_info={"name": "MIT"},
)


def _require_service() -> InferenceService:
    """Garante que o modelo já carregou; levanta 503 caso contrário."""
    if _service is None:
        raise HTTPException(status_code=503, detail="Modelo ainda carregando.")
    return _service


@app.get("/", tags=["Meta"], summary="Informações básicas da API")
def root() -> dict[str, Any]:
    """Ponto de entrada da API — aponta para a documentação interativa."""
    return {
        "name": "Instacart Recommender API",
        "docs": "/docs",
        "health": "/health",
        "model_version": _settings.model_version,
    }


@app.get(
    "/health",
    tags=["Meta"],
    summary="Health check",
    response_model=HealthResponse,
)
def health() -> HealthResponse:
    """Endpoint de health check (usado pelo Cloud Run e por monitoramento)."""
    status_value = "ok" if _service is not None else "loading"
    return HealthResponse(status=status_value, model_version=_settings.model_version)


@app.get(
    "/model/info",
    tags=["Meta"],
    summary="Metadados do modelo carregado",
    response_model=ModelInfoResponse,
)
def model_info() -> ModelInfoResponse:
    """Retorna arquitetura e vocabulário do modelo atualmente em produção."""
    service = _require_service()
    return ModelInfoResponse(model_version=_settings.model_version, **service.info())


@app.post(
    "/predict",
    tags=["Predição"],
    summary="Prevê a probabilidade de recompra de um par usuário-produto",
    response_model=PredictResponse,
    responses={
        422: {"model": ErrorResponse, "description": "user_id/product_id desconhecido"},
        503: {"model": ErrorResponse, "description": "Modelo ainda carregando"},
    },
)
def predict(request: PredictRequest) -> PredictResponse:
    """Prevê a probabilidade de recompra para um único par usuário-produto."""
    service = _require_service()
    encoded = service.encode_ids(request.user_id, request.product_id)
    if encoded is None:
        raise HTTPException(status_code=422, detail=_COLD_START_DETAIL)

    features = _feature_list(request)
    probability = service.predict(*encoded, features)
    return PredictResponse(
        reorder_probability=probability, model_version=_settings.model_version
    )


@app.post(
    "/predict/batch",
    tags=["Predição"],
    summary="Prevê a recompra para vários pares usuário-produto de uma vez",
    response_model=BatchPredictResponse,
    status_code=status.HTTP_200_OK,
)
def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    """Prevê em lote (um único forward pass) — mais eficiente que N chamadas a /predict.

    Pares cold-start (user_id/product_id fora do vocabulário) retornam
    `reorder_probability: null` em vez de derrubar a requisição inteira.
    """
    service = _require_service()
    raw_items = [
        (item.user_id, item.product_id, _feature_list(item)) for item in request.items
    ]
    probabilities = service.predict_batch(raw_items)
    results = [
        PredictResponse(reorder_probability=p, model_version=_settings.model_version)
        for p in probabilities
    ]
    return BatchPredictResponse(results=results)


def _feature_list(request: PredictRequest) -> list[float]:
    """Extrai as features tabulares do request, na ordem que o modelo espera."""
    return [
        request.purchase_count,
        request.days_since_last_order,
        request.order_hour_of_day,
        request.order_dow,
        request.basket_size,
    ]
