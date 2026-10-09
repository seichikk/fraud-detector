import json
from pathlib import Path

import mlflow
from mlflow import MlflowClient

PROJECT_DIR = Path(__file__).resolve().parents[1]
INFO_PATH = PROJECT_DIR / "artifacts" / "training" / "best_model_info.json"

TRACKING_URI = "http://localhost:5000"
MODEL_NAME = "FraudDetectionModel"
MODEL_ALIAS = "champion"


def main() -> None:
    if not INFO_PATH.exists():
        raise FileNotFoundError(
            f"Не найден файл с информацией о модели: {INFO_PATH}"
        )

    with INFO_PATH.open("r", encoding="utf-8") as file:
        model_info = json.load(file)

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    model_uri = model_info["model_uri"]
    source_run_id = model_info["run_id"]

    print(f"Выбранная модель: {model_info['run_name']}")
    print(f"PR-AUC: {model_info['pr_auc']:.6f}")
    print(f"Источник модели: {model_uri}")

    print("\nРегистрируем модель в MLflow Registry...")

    registered_version = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )

    version = str(registered_version.version)

    client.set_model_version_tag(
        name=MODEL_NAME,
        version=version,
        key="source_run_id",
        value=source_run_id,
    )

    client.set_model_version_tag(
        name=MODEL_NAME,
        version=version,
        key="selection_metric",
        value="pr_auc",
    )

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=MODEL_ALIAS,
        version=version,
    )

    active_version = client.get_model_version_by_alias(
        name=MODEL_NAME,
        alias=MODEL_ALIAS,
    )

    print("\nРегистрация завершена успешно.")
    print(f"Имя модели в Registry: {MODEL_NAME}")
    print(f"Зарегистрированная версия: {version}")
    print(f"Alias: {MODEL_ALIAS}")
    print(f"Версия, на которую указывает alias: {active_version.version}")
    print(f"Исходный Run ID: {source_run_id}")


if __name__ == "__main__":
    main()
