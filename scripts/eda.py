import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.data
import numpy as np
import pandas as pd

# Переключаем графики в режим без графического окна.
plt.switch_backend("Agg")

# Пути к файлам проекта.
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "raw" / "transactions.csv"
OUTPUT_DIR = PROJECT_DIR / "artifacts" / "eda"

# Настройки MLflow.
TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "fraud-detection"

# Для анализа не загружаем идентификаторы отправителя и получателя.
# Они не нужны для первоначального анализа распределений и могут
# занимать много памяти.
COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]

# Типы данных, позволяющие сократить использование памяти.
DTYPES = {
    "step": "int32",
    "type": "category",
    "amount": "float64",
    "oldbalanceOrg": "float64",
    "newbalanceOrig": "float64",
    "oldbalanceDest": "float64",
    "newbalanceDest": "float64",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}

NUMERIC_COLUMNS = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

NUMERIC_LABELS = {
    "amount": "Сумма операции (amount)",
    "oldbalanceOrg": "Баланс отправителя до операции",
    "newbalanceOrig": "Баланс отправителя после операции",
    "oldbalanceDest": "Баланс получателя до операции",
    "newbalanceDest": "Баланс получателя после операции",
}

STAT_LABELS = {
    "count": "Количество значений",
    "mean": "Среднее значение",
    "std": "Стандартное отклонение",
    "min": "Минимум",
    "25%": "25-й процентиль",
    "50%": "Медиана",
    "75%": "75-й процентиль",
    "max": "Максимум",
}


def calculate_sha256(path: Path) -> str:
    """Вычисляет SHA-256 для идентификации исходного файла."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def save_json(data: dict, path: Path) -> None:
    """Сохраняет результаты в формате JSON."""
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Не найден файл с данными: {DATA_PATH}\n"
            "Проверь, что CSV находится в папке data/raw "
            "и называется transactions.csv."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Этап 1 из 5. Читаем датасет PaySim...")

    df = pd.read_csv(
        DATA_PATH,
        usecols=COLUMNS,
        dtype=DTYPES,
    )

    if df.empty:
        raise ValueError("Файл CSV не содержит транзакций.")

    if not set(df["isFraud"].unique()).issubset({0, 1}):
        raise ValueError("В столбце isFraud обнаружены неожиданные значения.")

    print(f"Загружено транзакций: {len(df):,}")
    print(f"Столбцов для анализа: {len(df.columns)}")

    print("Этап 2 из 5. Рассчитываем основные показатели...")

    total_count = len(df)
    fraud_count = int(df["isFraud"].sum())
    normal_count = int(total_count - fraud_count)

    fraud_rate = fraud_count / total_count
    normal_rate = normal_count / total_count

    flagged_count = int(df["isFlaggedFraud"].sum())

    flagged_fraud_count = int(
        ((df["isFlaggedFraud"] == 1) & (df["isFraud"] == 1)).sum()
    )

    missing_count = int(df.isna().sum().sum())

    # SHA-256 исходного CSV нужен для идентификации его версии.
    print("Вычисляем контрольную сумму исходного файла...")
    file_digest = calculate_sha256(DATA_PATH)

    # Распределение транзакций по типам.
    type_summary = (
        df.groupby("type", observed=True)
        .agg(
            transactions=("isFraud", "size"),
            fraud_count=("isFraud", "sum"),
            fraud_rate=("isFraud", "mean"),
            total_amount=("amount", "sum"),
        )
        .reset_index()
    )

    type_summary_ru = type_summary.rename(
        columns={
            "type": "Тип транзакции",
            "transactions": "Всего транзакций",
            "fraud_count": "Мошеннических транзакций",
            "fraud_rate": "Доля мошеннических транзакций",
            "total_amount": "Общая сумма операций",
        }
    )

    # Количество пропущенных значений в каждом столбце.
    missing_values = (
        df.isna()
        .sum()
        .rename("Количество пропусков")
        .rename_axis("Столбец")
        .reset_index()
    )

    # Сопоставление флага существующей системы и фактического класса.
    fraud_flag_table = pd.crosstab(
        df["isFlaggedFraud"],
        df["isFraud"],
        rownames=["Флаг существующей системы (isFlaggedFraud)"],
        colnames=["Фактический класс (isFraud)"],
    )

    class_distribution = pd.DataFrame(
        {
            "Класс транзакции": ["Обычная", "Мошенническая"],
            "Количество транзакций": [normal_count, fraud_count],
            "Доля": [normal_rate, fraud_rate],
        }
    )

    # Описательная статистика числовых признаков.
    numeric_stats = (
        df[NUMERIC_COLUMNS]
        .describe()
        .rename(columns=NUMERIC_LABELS, index=STAT_LABELS)
    )

    # Сохраняем таблицы и сводку для дальнейшего изучения.
    class_distribution.to_csv(
        OUTPUT_DIR / "распределение_классов.csv",
        index=False,
        encoding="utf-8-sig",
    )

    type_summary_ru.to_csv(
        OUTPUT_DIR / "статистика_по_типам.csv",
        index=False,
        encoding="utf-8-sig",
    )

    missing_values.to_csv(
        OUTPUT_DIR / "пропущенные_значения.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fraud_flag_table.to_csv(
        OUTPUT_DIR / "сопоставление_флагов.csv",
        encoding="utf-8-sig",
    )

    numeric_stats.to_csv(
        OUTPUT_DIR / "статистика_числовых_признаков.csv",
        encoding="utf-8-sig",
    )

    summary = {
        "название_датасета": "PaySim",
        "исходный_файл": str(DATA_PATH.resolve()),
        "sha256_исходного_файла": file_digest,
        "количество_транзакций": total_count,
        "количество_анализируемых_столбцов": len(df.columns),
        "количество_обычных_транзакций": normal_count,
        "количество_мошеннических_транзакций": fraud_count,
        "доля_мошеннических_транзакций": fraud_rate,
        "количество_отмеченных_транзакций": flagged_count,
        "количество_пропущенных_значений": missing_count,
    }

    save_json(summary, OUTPUT_DIR / "сводка_анализа.json")

    # Формируем интерпретацию результатов на основе реальных данных.
    if fraud_rate < 0.05:
        balance_interpretation = (
            "Доля мошеннических транзакций ниже 5%. "
            "Это указывает на выраженный дисбаланс классов. "
            "При обучении и оценке моделей важно учитывать качество "
            "обнаружения редкого класса."
        )
    else:
        balance_interpretation = (
            "Доля мошеннических транзакций составляет не менее 5%. "
            "Перед обучением всё равно необходимо учитывать соотношение "
            "классов и оценивать качество обнаружения мошенничества."
        )

    if missing_count == 0:
        missing_interpretation = (
            "В проанализированных столбцах пропущенных значений нет."
        )
    else:
        missing_interpretation = (
            f"В проанализированных столбцах найдено пропущенных значений: "
            f"{missing_count:,}. Перед обучением нужно решить, как их обработать."
        )

    report = f"""ИССЛЕДОВАТЕЛЬСКИЙ АНАЛИЗ ДАННЫХ PAYSIM

1. ОБЩАЯ ИНФОРМАЦИЯ

Исходный файл: {DATA_PATH.name}
Количество транзакций: {total_count:,}
Количество столбцов, выбранных для анализа: {len(df.columns)}
Контрольная сумма SHA-256 исходного файла: {file_digest}

В исходном датасете PaySim 11 столбцов.
Для первоначального анализа не загружались идентификаторы отправителя
и получателя (nameOrig и nameDest), поскольку это идентификаторы счетов,
а не непосредственные количественные характеристики транзакции.

2. РАСПРЕДЕЛЕНИЕ КЛАССОВ

Обычные транзакции: {normal_count:,}
Мошеннические транзакции: {fraud_count:,}
Доля обычных транзакций: {normal_rate:.6%}
Доля мошеннических транзакций: {fraud_rate:.6%}

Интерпретация:
{balance_interpretation}

Целевой столбец модели: isFraud.
Значение 1 означает мошенническую транзакцию, а 0 означает обычную.

3. ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ

Общее количество пропущенных значений: {missing_count:,}

{missing_interpretation}

Подробная информация по столбцам сохранена в файле
«пропущенные_значения.csv».

4. ТРАНЗАКЦИИ ПО ТИПАМ

{type_summary_ru.to_string(index=False)}

5. ОПИСАТЕЛЬНАЯ СТАТИСТИКА ЧИСЛОВЫХ ПРИЗНАКОВ

{numeric_stats.to_string()}

6. ФЛАГ СУЩЕСТВУЮЩЕЙ СИСТЕМЫ

Транзакций с флагом isFlaggedFraud = 1: {flagged_count:,}
Среди них фактически мошеннических: {flagged_fraud_count:,}

isFlaggedFraud представляет собой существующий системный флаг.
Целевым столбцом остаётся isFraud. Флаг необходимо анализировать отдельно
и не включать автоматически в признаки модели, чтобы избежать утечки
информации и оценить способность модели самостоятельно обнаруживать
мошеннические транзакции.

7. ВЫВОДЫ ДЛЯ ДАЛЬНЕЙШЕГО ОБУЧЕНИЯ

Необходимо сравнить модели по метрикам, учитывающим качество обнаружения
мошеннических транзакций, например PR-AUC, Recall и F1.
Одной только Accuracy недостаточно, если классы сильно несбалансированы.

Графики и таблицы данного исследования сохранены в папке artifacts/eda.
"""

    (OUTPUT_DIR / "отчет_eda.txt").write_text(
        report,
        encoding="utf-8",
    )

    print("Этап 3 из 5. Строим графики...")

    # График 1. Количество обычных и мошеннических транзакций.
    plt.figure(figsize=(7, 5))

    bars = plt.bar(
        ["Обычные", "Мошеннические"],
        [normal_count, fraud_count],
    )

    plt.title("Распределение классов транзакций")
    plt.ylabel("Количество транзакций")
    plt.xlabel("Класс транзакции")

    plt.bar_label(bars, fmt="%.0f")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "распределение_классов.png",
        dpi=150,
    )
    plt.close()

    # График 2. Доля мошенничества в каждом типе транзакции.
    plot_data = type_summary.sort_values(
        "fraud_rate",
        ascending=False,
    )

    plt.figure(figsize=(8, 5))

    plt.bar(
        plot_data["type"].astype(str),
        plot_data["fraud_rate"] * 100,
    )

    plt.title("Доля мошеннических транзакций по типам операций")
    plt.ylabel("Мошеннические транзакции, %")
    plt.xlabel("Тип транзакции")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "мошенничество_по_типам.png",
        dpi=150,
    )
    plt.close()

    # График 3. Сравнение сумм обычных и мошеннических операций.
    # Для ограничения использования памяти берём выборку обычных операций.
    max_normal_samples = 300_000

    normal_amounts = df.loc[df["isFraud"] == 0, "amount"]
    fraud_amounts = df.loc[df["isFraud"] == 1, "amount"]

    if len(normal_amounts) > max_normal_samples:
        normal_amounts = normal_amounts.sample(
            n=max_normal_samples,
            random_state=42,
        )

    if len(fraud_amounts) > max_normal_samples:
        fraud_amounts = fraud_amounts.sample(
            n=max_normal_samples,
            random_state=42,
        )

    normal_log_amounts = np.log10(normal_amounts.clip(lower=0.01))
    fraud_log_amounts = np.log10(fraud_amounts.clip(lower=0.01))

    plt.figure(figsize=(8, 5))

    plt.hist(
        normal_log_amounts,
        bins=60,
        alpha=0.7,
        label="Обычные",
    )

    plt.hist(
        fraud_log_amounts,
        bins=60,
        alpha=0.7,
        label="Мошеннические",
    )

    plt.title("Распределение сумм транзакций")
    plt.xlabel("Логарифм суммы операции по основанию 10")
    plt.ylabel("Количество транзакций")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "распределение_сумм.png",
        dpi=150,
    )
    plt.close()

    print("Этап 4 из 5. Подготавливаем Dataset Tracking...")

    # Вычисляем метаданные набора данных для регистрации в MLflow.
    dataset = mlflow.data.from_pandas(
        df,
        source=str(DATA_PATH.resolve()),
        name="PaySim",
        targets="isFraud",
        digest=file_digest[:32],
    )

    print("Этап 5 из 5. Сохраняем результаты в MLflow...")

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="eda-paysim-v1") as run:
        mlflow.set_tag("stage", "eda")
        mlflow.set_tag("dataset_version", file_digest)

        mlflow.log_params(
            {
                "dataset_name": "PaySim",
                "source_filename": DATA_PATH.name,
                "target_column": "isFraud",
                "source_sha256": file_digest,
                "analysis_columns": len(df.columns),
            }
        )

        mlflow.log_metrics(
            {
                "total_transactions": total_count,
                "normal_transactions": normal_count,
                "fraudulent_transactions": fraud_count,
                "fraud_rate": fraud_rate,
                "flagged_transactions": flagged_count,
                "flagged_fraud_true_positives": flagged_fraud_count,
                "missing_values": missing_count,
            }
        )

        # Связываем этот запуск с набором данных.
        mlflow.log_input(dataset, context="eda")

        # Загружаем отчёт, таблицы и графики в Artifacts.
        mlflow.log_artifacts(
            str(OUTPUT_DIR),
            artifact_path="eda",
        )

        print(f"Идентификатор запуска MLflow: {run.info.run_id}")

    print("Анализ завершён успешно.")
    print(f"Отчёты и графики сохранены в: {OUTPUT_DIR}")
    print(f"Эксперимент MLflow: {EXPERIMENT_NAME}")


if __name__ == "__main__":
    main()
