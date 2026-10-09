import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.data
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "raw" / "transactions.csv"
OUTPUT_DIR = PROJECT_DIR / "artifacts" / "training"

TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "fraud-detection"

SAMPLE_SIZE = 1_000_000
RANDOM_STATE = 42
TARGET = "isFraud"

NUMERIC_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

CATEGORICAL_FEATURES = ["type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

DTYPES = {
    "step": "float64",
    "type": "object",
    "amount": "float64",
    "oldbalanceOrg": "float64",
    "newbalanceOrig": "float64",
    "oldbalanceDest": "float64",
    "newbalanceDest": "float64",
    "isFraud": "int8",
}


def calculate_sha256(path: Path) -> str:
    """Вычисляет контрольную сумму исходного файла."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def create_preprocessor() -> ColumnTransformer:
    """Подготавливает числовые и категориальные признаки."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def save_diagnostics(
    y_true: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    output_dir: Path,
) -> None:
    """Сохраняет графики и текстовый отчёт о качестве модели."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Матрица ошибок показывает правильные и ошибочные предсказания.
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])

    figure, axis = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Обычная", "Мошенническая"],
    )
    display.plot(ax=axis, values_format=",d", colorbar=False)
    axis.set_title("Матрица ошибок")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(figure)

    # Кривая показывает компромисс между Precision и Recall.
    precision_values, recall_values, _ = precision_recall_curve(
        y_true,
        probabilities,
    )

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(recall_values, precision_values)
    axis.set_title("Кривая Precision-Recall")
    axis.set_xlabel("Полнота обнаружения (Recall)")
    axis.set_ylabel("Точность положительных прогнозов (Precision)")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "precision_recall_curve.png", dpi=150)
    plt.close(figure)

    # Подробный отчёт по двум классам.
    report = classification_report(
        y_true,
        predictions,
        labels=[0, 1],
        target_names=[
            "Обычная транзакция",
            "Мошенническая транзакция",
        ],
        zero_division=0,
    )

    (output_dir / "classification_report.txt").write_text(
        "Отчёт о качестве модели\n\n" + report,
        encoding="utf-8",
    )


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Не найден датасет: {DATA_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Этап 1 из 5. Загружаем PaySim...")

    df = pd.read_csv(
        DATA_PATH,
        usecols=[*FEATURES, TARGET],
        dtype=DTYPES,
    )

    if df.empty:
        raise ValueError("Датасет не содержит строк.")

    if df[TARGET].isna().any():
        raise ValueError("В целевом столбце isFraud есть пропуски.")

    if set(df[TARGET].unique()) != {0, 1}:
        raise ValueError("Для обучения должны присутствовать оба класса: 0 и 1.")

    print(f"Всего транзакций: {len(df):,}")
    print("Вычисляем контрольную сумму исходного файла...")

    file_digest = calculate_sha256(DATA_PATH)

    # Используем фиксированную случайную выборку для быстрого обучения.
    sample_size = min(SAMPLE_SIZE, len(df))

    if sample_size < len(df):
        df = df.sample(
            n=sample_size,
            random_state=RANDOM_STATE,
        ).reset_index(drop=True)

    X = df[FEATURES].copy()
    X["type"] = X["type"].astype(object)
    y = df[TARGET].astype("int8")

    print(f"Размер выборки: {len(df):,}")
    print(f"Мошеннических транзакций: {int(y.sum()):,}")
    print(f"Доля мошеннических транзакций: {y.mean():.6%}")

    # Подготавливаем описание фактически используемой выборки.
    dataset = mlflow.data.from_pandas(
        df,
        source=str(DATA_PATH.resolve()),
        name="PaySim",
        targets=TARGET,
        digest=file_digest[:32],
    )

    print("Этап 2 из 5. Разделяем данные...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Данные для обучения: {len(X_train):,}")
    print(f"Данные для проверки: {len(X_test):,}")

    print("Этап 3 из 5. Подготавливаем варианты моделей...")

    models = [
        (
            "logistic-baseline",
            LogisticRegression(
                max_iter=200,
                class_weight=None,
                random_state=RANDOM_STATE,
            ),
            {
                "model_type": "LogisticRegression",
                "class_weight": "none",
                "max_iter": 200,
            },
        ),
        (
            "logistic-balanced",
            LogisticRegression(
                max_iter=200,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            {
                "model_type": "LogisticRegression",
                "class_weight": "balanced",
                "max_iter": 200,
            },
        ),
        (
            "decision-tree-balanced",
            DecisionTreeClassifier(
                max_depth=12,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            {
                "model_type": "DecisionTreeClassifier",
                "max_depth": 12,
                "min_samples_leaf": 20,
                "class_weight": "balanced",
            },
        ),
    ]

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    results = []

    print("Этап 4 из 5. Обучаем модели и сохраняем результаты...")

    for run_name, classifier, model_params in models:
        print(f"\nОбучаем модель: {run_name}")

        model = Pipeline(
            steps=[
                ("preprocessor", create_preprocessor()),
                ("classifier", classifier),
            ]
        )

        variant_dir = OUTPUT_DIR / run_name
        variant_dir.mkdir(parents=True, exist_ok=True)

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tag("task", "fraud-detection")
            mlflow.set_tag("dataset_version", file_digest[:32])

            mlflow.log_params(
                {
                    **model_params,
                    "dataset_name": "PaySim",
                    "dataset_sha256": file_digest,
                    "sample_size": len(df),
                    "train_size": len(X_train),
                    "test_size": len(X_test),
                    "test_fraction": 0.2,
                    "random_state": RANDOM_STATE,
                    "target_column": TARGET,
                    "prediction_threshold": 0.5,
                }
            )

            # Связываем используемый датасет с текущим Run.
            mlflow.log_input(dataset, context="training")

            model.fit(X_train, y_train)

            probabilities = model.predict_proba(X_test)[:, 1]
            predictions = (probabilities >= 0.5).astype("int8")

            metrics = {
                "pr_auc": float(average_precision_score(y_test, probabilities)),
                "roc_auc": float(roc_auc_score(y_test, probabilities)),
                "precision": float(
                    precision_score(
                        y_test,
                        predictions,
                        zero_division=0,
                    )
                ),
                "recall": float(
                    recall_score(
                        y_test,
                        predictions,
                        zero_division=0,
                    )
                ),
                "f1": float(
                    f1_score(
                        y_test,
                        predictions,
                        zero_division=0,
                    )
                ),
                "accuracy": float(accuracy_score(y_test, predictions)),
            }

            mlflow.log_metrics(metrics)

            save_diagnostics(
                y_test,
                predictions,
                probabilities,
                variant_dir,
            )

            mlflow.log_artifacts(
                str(variant_dir),
                artifact_path="diagnostics",
            )

            # Сохраняем модель с примером входных данных и сигнатурой.
            input_example = X_train.head(5).copy()

            signature = infer_signature(
                input_example,
                model.predict(input_example),
            )

            # Разрешаем только конкретные типы, обнаруженные в нашей
            # собственной обученной модели. Это не означает, что следует
            # доверять моделям, полученным из неизвестных источников.
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                signature=signature,
                input_example=input_example,
                skops_trusted_types=[
                    "numpy.dtype",
                    "sklearn.tree._tree.Tree",
                ],
            )

            results.append(
                {
                    "run_name": run_name,
                    "run_id": run.info.run_id,
                    **model_params,
                    **metrics,
                }
            )

            print(f"Run ID: {run.info.run_id}")
            print(f"PR-AUC: {metrics['pr_auc']:.6f}")
            print(f"ROC-AUC: {metrics['roc_auc']:.6f}")
            print(f"Precision: {metrics['precision']:.6f}")
            print(f"Recall: {metrics['recall']:.6f}")
            print(f"F1: {metrics['f1']:.6f}")
            print(f"Accuracy: {metrics['accuracy']:.6f}")

    print("\nЭтап 5 из 5. Сравниваем модели...")

    comparison = pd.DataFrame(results).sort_values("pr_auc", ascending=False).reset_index(drop=True)

    comparison_path = OUTPUT_DIR / "model_comparison.csv"

    comparison.to_csv(
        comparison_path,
        index=False,
        encoding="utf-8-sig",
    )

    best = comparison.iloc[0].to_dict()

    best_model_info = {
        "run_name": str(best["run_name"]),
        "run_id": str(best["run_id"]),
        "model_uri": f"runs:/{best['run_id']}/model",
        "pr_auc": float(best["pr_auc"]),
        "target_column": TARGET,
        "sample_size": len(df),
        "dataset_sha256": file_digest,
        "registered_model_name": "FraudDetectionModel",
    }

    best_info_path = OUTPUT_DIR / "best_model_info.json"

    best_info_path.write_text(
        json.dumps(
            best_model_info,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Отдельный Run содержит итоговую сравнительную таблицу.
    with mlflow.start_run(run_name="model-comparison-summary") as summary_run:
        mlflow.log_artifact(str(comparison_path))
        mlflow.log_artifact(str(best_info_path))

        mlflow.log_metric(
            "best_pr_auc",
            float(best["pr_auc"]),
        )

        mlflow.set_tag(
            "best_model_run_id",
            str(best["run_id"]),
        )

        print(f"Run сводки сравнения: {summary_run.info.run_id}")

    print("\nСравнение моделей по PR-AUC:")

    print(
        comparison[
            [
                "run_name",
                "pr_auc",
                "roc_auc",
                "precision",
                "recall",
                "f1",
                "accuracy",
            ]
        ].to_string(index=False)
    )

    print("\nЛучшая модель по PR-AUC:")
    print(best["run_name"])
    print(f"PR-AUC: {best['pr_auc']:.6f}")
    print(f"Run ID: {best['run_id']}")

    print(f"\nТаблица сравнения: {comparison_path}")
    print(f"Информация о лучшей модели: {best_info_path}")
    print("Обучение и сравнение завершены успешно.")


if __name__ == "__main__":
    main()
