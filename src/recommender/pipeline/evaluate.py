"""Estágio 4 do pipeline DVC: avalia o modelo treinado no split de validação.

Carrega os pesos salvos pelo estágio `train`, calcula um conjunto de
métricas (AUC-ROC, recall, precision, F1), grava `metrics.json` (lido
pelo `dvc metrics show`) e loga o resultado como uma run adicional no
MLflow.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from recommender.config.settings import get_settings
from recommender.data.dataset import InstacartReorderDataset
from recommender.models.factory import ModelFactory
from recommender.pipeline.common import build_model_config
from recommender.pipeline.feature_eng import FEATURE_COLUMNS, LABEL_COLUMN


def _predict(
    model: torch.nn.Module, loader: DataLoader, device: str
) -> tuple[list[int], list[float]]:
    """Roda o modelo no `loader` e retorna rótulos verdadeiros e probabilidades."""
    model.eval()
    labels, probs = [], []
    with torch.no_grad():
        for user_ids, product_ids, features, batch_labels in loader:
            logits = model(
                user_ids.to(device), product_ids.to(device), features.to(device)
            )
            probs.extend(torch.sigmoid(logits).cpu().tolist())
            labels.extend(batch_labels.tolist())
    return labels, probs


def _load_trained_model(
    models_dir: Path, device: str
) -> torch.nn.Module:
    """Reconstrói a arquitetura via ModelFactory e carrega os pesos salvos."""
    model_config = build_model_config(models_dir)
    model = ModelFactory.create(model_config).to(device)
    model.load_state_dict(torch.load(models_dir / "model.pt", map_location=device))
    return model


def _compute_metrics(labels: list[int], probs: list[float]) -> dict[str, float]:
    """Calcula AUC-ROC, recall, precision e F1 a partir das predições."""
    preds = [1 if p >= 0.5 else 0 for p in probs]
    return {
        "auc_roc": roc_auc_score(labels, probs),
        "recall": recall_score(labels, preds),
        "precision": precision_score(labels, preds),
        "f1": f1_score(labels, preds),
    }


def _log_metrics_to_mlflow(
    metrics: dict[str, float], metrics_path: Path, tracking_uri: str
) -> None:
    """Loga as métricas e o artefato `metrics.json` numa run do MLflow."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("instacart-recommender")
    with mlflow.start_run(run_name="evaluate"):
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(metrics_path))


def run(
    processed_dir: Path | None = None, models_dir: Path | None = None
) -> dict[str, float]:
    """Avalia o modelo salvo e grava `metrics.json`.

    Args:
        processed_dir: Override do diretório de dados processados.
        models_dir: Override do diretório de artefatos de modelo.

    Returns:
        Dicionário com as métricas calculadas.
    """
    settings = get_settings()
    processed_dir = processed_dir or Path(settings.data_processed_dir)
    models_dir = models_dir or Path(settings.models_dir)

    val_df = pd.read_parquet(processed_dir / "features_val.parquet")
    val_ds = InstacartReorderDataset.from_dataframe(
        val_df, FEATURE_COLUMNS, LABEL_COLUMN
    )
    val_loader = DataLoader(val_ds, batch_size=512)

    model = _load_trained_model(models_dir, settings.device)
    labels, probs = _predict(model, val_loader, settings.device)
    metrics = _compute_metrics(labels, probs)

    metrics_path = processed_dir.parent / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    _log_metrics_to_mlflow(metrics, metrics_path, settings.mlflow_tracking_uri)

    print(f"[evaluate] {metrics}")
    return metrics


if __name__ == "__main__":
    run()
