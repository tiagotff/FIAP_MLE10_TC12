"""Estágio do pipeline DVC: baseline Scikit-Learn para comparação com o modelo neural.

Treina uma Regressão Logística usando apenas as features tabulares
(sem os embeddings de user_id/product_id que o modelo neural usa),
calcula as mesmas 4 métricas do `evaluate.py` e loga o resultado como
uma run separada no MLflow — a base de comparação para o critério
"Rede neural: MLP funcional, early stopping, comparação com baselines".
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from recommender.config.settings import get_settings
from recommender.pipeline.feature_eng import FEATURE_COLUMNS, LABEL_COLUMN


def _load_splits(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega os splits de treino/validação gerados pelo feature_eng."""
    train_df = pd.read_parquet(processed_dir / "features_train.parquet")
    val_df = pd.read_parquet(processed_dir / "features_val.parquet")
    return train_df, val_df


def _train_logistic_regression(
    train_df: pd.DataFrame, random_seed: int
) -> LogisticRegression:
    """Treina uma Regressão Logística só com as features tabulares."""
    model = LogisticRegression(max_iter=1000, random_state=random_seed)
    model.fit(train_df[FEATURE_COLUMNS], train_df[LABEL_COLUMN])
    return model


def _compute_metrics(
    model: LogisticRegression, val_df: pd.DataFrame
) -> dict[str, float]:
    """Calcula AUC-ROC, recall, precision e F1 no conjunto de validação."""
    probs = model.predict_proba(val_df[FEATURE_COLUMNS])[:, 1]
    preds = model.predict(val_df[FEATURE_COLUMNS])
    labels = val_df[LABEL_COLUMN]
    return {
        "auc_roc": roc_auc_score(labels, probs),
        "recall": recall_score(labels, preds),
        "precision": precision_score(labels, preds),
        "f1": f1_score(labels, preds),
    }


def _log_baseline_to_mlflow(
    metrics: dict[str, float], metrics_path: Path, tracking_uri: str
) -> None:
    """Loga params/métricas/artefato da run de baseline no MLflow."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("instacart-recommender")
    with mlflow.start_run(run_name="baseline_logistic_regression"):
        mlflow.log_params(
            {"model_type": "logistic_regression", "features": FEATURE_COLUMNS}
        )
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(metrics_path))


def run(processed_dir: Path | None = None) -> dict[str, float]:
    """Treina o baseline, avalia e grava `metrics_baseline.json`.

    Args:
        processed_dir: Override do diretório de dados processados.

    Returns:
        Dicionário com as métricas calculadas.
    """
    settings = get_settings()
    processed_dir = processed_dir or Path(settings.data_processed_dir)

    train_df, val_df = _load_splits(processed_dir)
    model = _train_logistic_regression(train_df, settings.random_seed)
    metrics = _compute_metrics(model, val_df)

    metrics_path = processed_dir.parent / "metrics_baseline.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    _log_baseline_to_mlflow(metrics, metrics_path, settings.mlflow_tracking_uri)

    print(f"[baseline] {metrics}")
    return metrics


if __name__ == "__main__":
    run()
