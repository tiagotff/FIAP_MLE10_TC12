"""Funções compartilhadas entre os estágios `train` e `evaluate` do pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from recommender.config.model_config import (
    EmbeddingConfig,
    HybridModelConfig,
    MlpConfig,
)
from recommender.pipeline.feature_eng import FEATURE_COLUMNS

MODEL_CONFIG_PATH = Path("configs/model.yaml")


def _load_raw_model_yaml(config_path: Path) -> dict:
    """Lê `configs/model.yaml` como dict cru; retorna vazio se não existir."""
    if not config_path.exists():
        return {}
    return yaml.safe_load(config_path.read_text()) or {}


def _normalize_mlp_overrides(raw: dict) -> dict:
    """Extrai a seção `mlp` do YAML, convertendo `hidden_dims` para tupla."""
    mlp_overrides = dict(raw.get("mlp", {}))
    if "hidden_dims" in mlp_overrides:
        mlp_overrides["hidden_dims"] = tuple(mlp_overrides["hidden_dims"])
    return mlp_overrides


def build_model_config(
    models_dir: Path, config_path: Path = MODEL_CONFIG_PATH
) -> HybridModelConfig:
    """Monta a config do modelo combinando `configs/model.yaml` com o vocabulário.

    As dimensões de embedding e a arquitetura da MLP vêm do YAML (se
    existir; caso contrário usa os defaults das dataclasses); a
    cardinalidade do vocabulário (`num_users`, `num_products`) e o
    número de features tabulares são descobertos automaticamente a
    partir dos artefatos gerados pelo estágio `feature_eng` — não fazem
    sentido como valores fixos num YAML, já que dependem do dataset.

    Args:
        models_dir: Diretório onde `vocab_sizes.json` foi salvo pelo
            estágio `feature_eng`.
        config_path: Caminho do YAML de configuração do modelo
            (default: `configs/model.yaml`).

    Returns:
        Config pronta para `ModelFactory.create`.
    """
    vocab_sizes = json.loads((models_dir / "vocab_sizes.json").read_text())
    raw = _load_raw_model_yaml(config_path)

    return HybridModelConfig(
        embedding=EmbeddingConfig(**vocab_sizes, **raw.get("embedding", {})),
        mlp=MlpConfig(
            tabular_feature_dim=len(FEATURE_COLUMNS), **_normalize_mlp_overrides(raw)
        ),
        model_type=raw.get("model_type", "hybrid_mlp"),
    )
