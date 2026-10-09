import time
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from numpy.typing import NDArray

from fraud_detector import main

TRANSACTION = {
    "step": 1,
    "type": "TRANSFER",
    "amount": 1000.0,
    "oldbalanceOrg": 1000.0,
    "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 1000.0,
}


class FakePredictionModel:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, data: pd.DataFrame) -> NDArray[np.float64]:
        assert len(data) == 1
        return np.asarray(
            [[1.0 - self.probability, self.probability]],
            dtype=np.float64,
        )


class BrokenPredictionModel:
    def predict_proba(self, data: pd.DataFrame) -> NDArray[np.float64]:
        raise RuntimeError("Тестовая ошибка предсказания")


async def test_process_returns_fraud_prediction(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    app.state.fraud_model = FakePredictionModel(0.9)
    app.state.fraud_model_version = "1"

    response = await client.post("/process", json=TRANSACTION)

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == "FRAUD"
    assert body["is_fraud"] is True
    assert body["fraud_probability"] == pytest.approx(0.9)
    assert body["model_name"] == "FraudDetectionModel"
    assert body["model_version"] == "1"
    assert body["model_alias"] == "champion"


async def test_process_returns_normal_prediction(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    app.state.fraud_model = FakePredictionModel(0.1)
    app.state.fraud_model_version = "2"

    response = await client.post("/process", json=TRANSACTION)

    assert response.status_code == 200
    assert response.json()["prediction"] == "NORMAL"
    assert response.json()["is_fraud"] is False


async def test_process_returns_503_when_model_is_not_loaded(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    app.state.fraud_model = None
    app.state.fraud_model_version = None

    response = await client.post("/process", json=TRANSACTION)

    assert response.status_code == 503
    assert "Модель не загружена" in response.json()["detail"]


async def test_process_returns_500_when_prediction_fails(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    app.state.fraud_model = BrokenPredictionModel()
    app.state.fraud_model_version = "1"

    response = await client.post("/process", json=TRANSACTION)

    assert response.status_code == 500


class FakeModelVersion:
    def __init__(self, version: str = "2") -> None:
        self.version = version


class FakeRegistryClient:
    failures_before_success: int = 0
    always_fail: bool = False

    def __init__(self, tracking_uri: str) -> None:
        self.tracking_uri = tracking_uri
        self.calls = 0

    def get_model_version_by_alias(
        self,
        *,
        name: str,
        alias: str,
    ) -> FakeModelVersion:
        assert name == "FraudDetectionModel"
        assert alias == "champion"

        self.calls += 1
        if type(self).always_fail or self.calls <= type(self).failures_before_success:
            raise ConnectionError("MLflow временно недоступен")

        return FakeModelVersion()


def test_load_champion_model_retries_after_registry_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(FakeRegistryClient, "failures_before_success", 1)
    monkeypatch.setattr(FakeRegistryClient, "always_fail", False)
    monkeypatch.setattr(main, "MlflowClient", FakeRegistryClient)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda _: None)

    expected_model = object()
    loaded_uris: list[str] = []

    def fake_load_model(model_uri: str) -> Any:
        loaded_uris.append(model_uri)
        return expected_model

    monkeypatch.setattr(mlflow.sklearn, "load_model", fake_load_model)

    model, version = main.load_champion_model("http://mlflow.test")

    assert model is expected_model
    assert version == "2"
    assert loaded_uris == ["models:/FraudDetectionModel/2"]


def test_load_champion_model_raises_when_registry_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(FakeRegistryClient, "failures_before_success", 0)
    monkeypatch.setattr(FakeRegistryClient, "always_fail", True)
    monkeypatch.setattr(main, "MlflowClient", FakeRegistryClient)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda _: None)

    with pytest.raises(RuntimeError, match="Не удалось загрузить модель"):
        main.load_champion_model("http://mlflow.test")
