"""API HTTP de inferência (FastAPI): serve predições de recompra.

Executada via `uvicorn recommender.api.main:app`. Carrega o modelo e os
encoders uma única vez, na subida do processo — não a cada requisição.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from recommender.api.inference import InferenceService
from recommender.api.schemas import HealthResponse, PredictRequest, PredictResponse
from recommender.config.settings import get_settings

_service: InferenceService | None = None
_settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201
    """Carrega o modelo na subida da API; libera recursos ao encerrar."""
    global _service
    _service = InferenceService(Path(_settings.models_dir), _settings.device)
    yield
    _service = None


app = FastAPI(
    title="Instacart Recommender API",
    description="Prevê a probabilidade de recompra de um produto por um usuário.",
    version=_settings.model_version,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Endpoint de health check (usado pelo Cloud Run e por monitoramento)."""
    status = "ok" if _service is not None else "loading"
    return HealthResponse(status=status, model_version=_settings.model_version)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Prevê a probabilidade de recompra para um par usuário-produto."""
    if _service is None:
        raise HTTPException(status_code=503, detail="Modelo ainda carregando.")

    encoded = _service.encode_ids(request.user_id, request.product_id)
    if encoded is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "user_id ou product_id desconhecido "
                "(fora do vocabulário de treino)."
            ),
        )

    features = [
        request.purchase_count,
        request.days_since_last_order,
        request.order_hour_of_day,
        request.order_dow,
        request.basket_size,
    ]
    probability = _service.predict(*encoded, features)
    return PredictResponse(
        reorder_probability=probability, model_version=_settings.model_version
    )
