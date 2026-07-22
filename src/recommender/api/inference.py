"""Carrega o modelo treinado e os encoders para servir predições.

Carregado uma única vez, na subida da API — não a cada requisição.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import torch

from recommender.models.factory import ModelFactory
from recommender.pipeline.common import build_model_config
from recommender.pipeline.feature_eng import FEATURE_COLUMNS


class InferenceService:
    """Encapsula o modelo carregado e os encoders, prontos para predição."""

    def __init__(self, models_dir: Path, device: str = "cpu") -> None:
        self.device = device
        self._user_index = self._load_id_index(models_dir / "user_encoder.joblib")
        self._product_index = self._load_id_index(models_dir / "product_encoder.joblib")

        model_config = build_model_config(models_dir)
        self.model = ModelFactory.create(model_config).to(device)
        self.model.load_state_dict(
            torch.load(models_dir / "model.pt", map_location=device)
        )
        self.model.eval()

    @staticmethod
    def _load_id_index(encoder_path: Path) -> dict[int, int]:
        """Carrega um encoder salvo e monta um dict {id_original: índice}."""
        encoder = joblib.load(encoder_path)
        return {int(raw_id): idx for idx, raw_id in enumerate(encoder.classes_)}

    def encode_ids(self, user_id: int, product_id: int) -> tuple[int, int] | None:
        """Traduz ids originais para os índices que o modelo conhece.

        Returns:
            `(user_idx, product_idx)`, ou `None` se algum dos dois nunca
            apareceu no treino (cold-start — o modelo não tem embedding
            para esse id).
        """
        user_idx = self._user_index.get(user_id)
        product_idx = self._product_index.get(product_id)
        if user_idx is None or product_idx is None:
            return None
        return user_idx, product_idx

    def predict(
        self, user_idx: int, product_idx: int, features: list[float]
    ) -> float:
        """Calcula a probabilidade de recompra para um par usuário-produto.

        Args:
            user_idx: Índice do usuário já codificado (ver `encode_ids`).
            product_idx: Índice do produto já codificado.
            features: Valores das features tabulares, na ordem de
                `FEATURE_COLUMNS`.

        Returns:
            Probabilidade de recompra, entre 0 e 1.
        """
        assert len(features) == len(FEATURE_COLUMNS)
        with torch.no_grad():
            logits = self.model(
                torch.tensor([user_idx], dtype=torch.long, device=self.device),
                torch.tensor([product_idx], dtype=torch.long, device=self.device),
                torch.tensor([features], dtype=torch.float32, device=self.device),
            )
            return float(torch.sigmoid(logits).item())


def load_reference_feature_order() -> list[str]:
    """Retorna a ordem esperada das features tabulares (para validação/docs)."""
    return list(FEATURE_COLUMNS)
