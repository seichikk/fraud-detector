from types import SimpleNamespace
from typing import Any

import pytest

from fraud_detector import ml
from fraud_detector.core.config import Settings
from fraud_detector.ml import MlflowPredictor, load_predictor, model_uri


@pytest.fixture
def settings() -> Settings:
    return Settings(
        mlflow_tracking_uri="http://mlflow.test:5000",
        mlflow_model_name="FraudDetectionModel",
        mlflow_model_alias="champion",
        model_load_attempts=3,
        model_load_retry_seconds=0,
        inference_seconds=0.1,
    )


def test_model_is_addressed_by_alias(settings: Settings) -> None:
    assert model_uri(settings) == "models:/FraudDetectionModel@champion"


def test_load_once_takes_version_from_registry(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[tuple[str, str]] = []
    loaded: list[str] = []

    class Registry:
        def __init__(self, tracking_uri: str) -> None:
            assert tracking_uri == settings.mlflow_tracking_uri

        def get_model_version_by_alias(self, name: str, alias: str) -> SimpleNamespace:
            asked.append((name, alias))
            return SimpleNamespace(version=4)

    def fake_load(uri: str) -> str:
        loaded.append(uri)
        return "модель"

    monkeypatch.setattr(ml, "MlflowClient", Registry)
    monkeypatch.setattr("mlflow.set_tracking_uri", lambda uri: None)
    monkeypatch.setattr("mlflow.sklearn.load_model", fake_load)

    predictor = ml._load_once(settings)

    assert asked == [("FraudDetectionModel", "champion")]
    assert loaded == ["models:/FraudDetectionModel@champion"]
    assert predictor == MlflowPredictor(model="модель", version="4", inference_seconds=0.1)


async def test_loading_retries_until_registry_answers(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts: list[int] = []
    ready = MlflowPredictor(model=None, version="2", inference_seconds=0)

    def flaky(current: Any) -> MlflowPredictor:
        attempts.append(1)
        if len(attempts) < 3:
            raise ConnectionError("mlflow ещё не поднялся")
        return ready

    monkeypatch.setattr(ml, "_load_once", flaky)

    assert await load_predictor(settings) is ready
    assert len(attempts) == 3


async def test_loading_gives_up_after_last_attempt(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(current: Any) -> MlflowPredictor:
        raise ConnectionError("mlflow недоступен")

    monkeypatch.setattr(ml, "_load_once", broken)

    with pytest.raises(RuntimeError, match="FraudDetectionModel@champion"):
        await load_predictor(settings)
