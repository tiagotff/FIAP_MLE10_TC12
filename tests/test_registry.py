"""Testes unitários para a lógica de registro/promoção do MLflow Model Registry.

Usa um `MlflowClient` simulado (mock) — não depende de um servidor
MLflow real rodando. O objetivo é validar a regra de negócio (quando
promove a Production, quando fica em Staging), não a integração com a
API do MLflow em si (essa é coberta pelo teste de ponta a ponta via
`dvc repro`).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from recommender.pipeline.registry import (
    REGISTERED_MODEL_NAME,
    _current_production_auc,
    _promote,
)


def test_current_production_auc_returns_none_when_no_production_version() -> None:
    """Sem versão em Production, deve retornar None (não deve levantar erro)."""
    client = MagicMock()
    client.get_latest_versions.return_value = []
    assert _current_production_auc(client) is None


def test_current_production_auc_reads_metric_from_the_versions_run() -> None:
    """Deve buscar o best_val_auc logado na run da versão em Production."""
    client = MagicMock()
    client.get_latest_versions.return_value = [MagicMock(run_id="run-123")]
    client.get_run.return_value.data.metrics = {"best_val_auc": 0.87}
    assert _current_production_auc(client) == 0.87


def test_promote_always_transitions_to_staging() -> None:
    """Toda promoção deve, no mínimo, mover a versão para Staging."""
    client = MagicMock()
    client.get_latest_versions.return_value = []
    _promote(client, version="1", new_auc=0.9)
    client.transition_model_version_stage.assert_any_call(
        REGISTERED_MODEL_NAME, "1", "Staging"
    )


def test_promote_to_production_when_no_current_production_version() -> None:
    """Sem versão prévia em Production, a nova versão deve ser promovida."""
    client = MagicMock()
    client.get_latest_versions.return_value = []
    _promote(client, version="1", new_auc=0.9)
    client.transition_model_version_stage.assert_any_call(
        REGISTERED_MODEL_NAME, "1", "Production", archive_existing_versions=True
    )


def test_promote_to_production_when_better_than_current() -> None:
    """Uma versão melhor que a atual em Production deve ser promovida."""
    client = MagicMock()
    client.get_latest_versions.return_value = [MagicMock(run_id="run-old")]
    client.get_run.return_value.data.metrics = {"best_val_auc": 0.80}
    _promote(client, version="2", new_auc=0.90)
    client.transition_model_version_stage.assert_any_call(
        REGISTERED_MODEL_NAME, "2", "Production", archive_existing_versions=True
    )


def test_does_not_promote_to_production_when_worse_than_current() -> None:
    """Uma versão pior (ou igual) que a atual em Production deve ficar só em Staging."""
    client = MagicMock()
    client.get_latest_versions.return_value = [MagicMock(run_id="run-old")]
    client.get_run.return_value.data.metrics = {"best_val_auc": 0.90}
    _promote(client, version="2", new_auc=0.85)

    calls = client.transition_model_version_stage.call_args_list
    assert len(calls) == 1  # só a chamada de Staging, nenhuma de Production
