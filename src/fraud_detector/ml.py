import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient

from fraud_detector.core.config import Settings
from fraud_detector.schemas.tasks import Transaction

logger = logging.getLogger(__name__)


class Predictor(Protocol):
    version: str

    def predict(self, transaction: Transaction) -> float: ...


@dataclass
class MlflowPredictor:
    model: Any
    version: str
    inference_seconds: float

    def predict(self, transaction: Transaction) -> float:
        time.sleep(self.inference_seconds)
        frame = pd.DataFrame([transaction.model_dump(by_alias=True)])
        return float(self.model.predict_proba(frame)[0, 1])


def model_uri(settings: Settings) -> str:
    return f"models:/{settings.mlflow_model_name}@{settings.mlflow_model_alias}"


def _load_once(settings: Settings) -> MlflowPredictor:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    version = client.get_model_version_by_alias(
        settings.mlflow_model_name, settings.mlflow_model_alias
    ).version
    model = mlflow.sklearn.load_model(model_uri(settings))
    return MlflowPredictor(
        model=model, version=str(version), inference_seconds=settings.inference_seconds
    )


async def load_predictor(settings: Settings) -> MlflowPredictor:
    uri = model_uri(settings)
    for attempt in range(1, settings.model_load_attempts + 1):
        try:
            predictor = await asyncio.to_thread(_load_once, settings)
        except Exception as exc:
            logger.warning(
                "не удалось загрузить %s (попытка %s из %s): %s",
                uri,
                attempt,
                settings.model_load_attempts,
                exc,
            )
            if attempt == settings.model_load_attempts:
                raise RuntimeError(f"Не удалось загрузить модель {uri}") from exc
            await asyncio.sleep(settings.model_load_retry_seconds)
        else:
            logger.info("модель %s загружена, версия %s", uri, predictor.version)
            return predictor
    raise RuntimeError(f"Не удалось загрузить модель {uri}")
