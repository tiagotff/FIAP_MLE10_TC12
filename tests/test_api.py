"""Testes de integração da API de inferência (via TestClient, sem rede real).

Depende de um modelo já treinado em `models/` (model.pt + encoders +
vocab_sizes.json) — roda como parte da suíte só quando esses artefatos
existem localmente, já que a API os carrega na subida.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

MODELS_DIR = Path("models")
pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "model.pt").exists(),
    reason="Requer um modelo treinado em models/ (rode o pipeline primeiro).",
)


@pytest.fixture
def client():  # noqa: ANN201
    """Cliente de teste com o lifespan da API executado (carrega o modelo)."""
    from recommender.api.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_ok_once_model_is_loaded(client) -> None:  # noqa: ANN001
    """O health check deve reportar status 'ok' após o modelo carregar."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_probability_between_zero_and_one(client) -> None:  # noqa: ANN001
    """Uma predição válida deve retornar uma probabilidade em [0, 1]."""
    response = client.post(
        "/predict",
        json={
            "user_id": 1,
            "product_id": 100,
            "purchase_count": 3,
            "days_since_last_order": 7,
            "order_hour_of_day": 10,
            "order_dow": 2,
            "basket_size": 8,
        },
    )
    assert response.status_code == 200
    probability = response.json()["reorder_probability"]
    assert 0.0 <= probability <= 1.0


def test_predict_returns_422_for_unknown_user(client) -> None:  # noqa: ANN001
    """Um user_id fora do vocabulário de treino deve retornar 422, não 500."""
    response = client.post(
        "/predict",
        json={
            "user_id": 999_999_999,
            "product_id": 100,
            "purchase_count": 3,
            "days_since_last_order": 7,
            "order_hour_of_day": 10,
            "order_dow": 2,
            "basket_size": 8,
        },
    )
    assert response.status_code == 422
