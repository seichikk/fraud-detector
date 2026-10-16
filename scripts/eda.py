import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.data
import numpy as np
import pandas as pd

plt.switch_backend("Agg")

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "raw" / "transactions.csv"
OUTPUT_DIR = PROJECT_DIR / "artifacts" / "eda"

TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "fraud-detection"

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

NUMERIC_COLUMNS = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]

NUMERIC_LABELS = {
    "amount": "Сумма операции",
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
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def save_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(DATA_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH, usecols=COLUMNS, dtype=DTYPES)

    if df.empty:
        raise ValueError("Файл CSV не содержит транзакций.")

    if not set(df["isFraud"].unique()).issubset({0, 1}):
        raise ValueError("В столбце isFraud обнаружены неожиданные значения.")

    total_count = len(df)
    fraud_count = int(df["isFraud"].sum())
    normal_count = int(total_count - fraud_count)

    fraud_rate = fraud_count / total_count
    normal_rate = normal_count / total_count

    flagged_count = int(df["isFlaggedFraud"].sum())

    flagged_fraud_count = int(((df["isFlaggedFraud"] == 1) & (df["isFraud"] == 1)).sum())

    missing_count = int(df.isna().sum().sum())

    file_digest = calculate_sha256(DATA_PATH)

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

    missing_values = (
        df.isna().sum().rename("Количество пропусков").rename_axis("Столбец").reset_index()
    )

    fraud_flag_table = pd.crosstab(
        df["isFlaggedFraud"],
        df["isFraud"],
        rownames=["Флаг существующей системы"],
        colnames=["Фактический класс"],
    )

    class_distribution = pd.DataFrame(
        {
            "Класс транзакции": ["Обычная", "Мошенническая"],
            "Количество транзакций": [normal_count, fraud_count],
            "Доля": [normal_rate, fraud_rate],
        }
    )

    numeric_stats = df[NUMERIC_COLUMNS].describe().rename(columns=NUMERIC_LABELS, index=STAT_LABELS)

    class_distribution.to_csv(
        OUTPUT_DIR / "распределение_классов.csv", index=False, encoding="utf-8-sig"
    )

    type_summary_ru.to_csv(
        OUTPUT_DIR / "статистика_по_типам.csv", index=False, encoding="utf-8-sig"
    )

    missing_values.to_csv(
        OUTPUT_DIR / "пропущенные_значения.csv", index=False, encoding="utf-8-sig"
    )

    fraud_flag_table.to_csv(OUTPUT_DIR / "сопоставление_флагов.csv", encoding="utf-8-sig")

    numeric_stats.to_csv(OUTPUT_DIR / "статистика_числовых_признаков.csv", encoding="utf-8-sig")

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

    if fraud_rate < 0.05:
        balance_interpretation = (
            "мошеннических операций меньше 5%, классы сильно несбалансированы. "
            "при обучении и оценке главное, насколько хорошо модель ловит редкий класс."
        )
    else:
        balance_interpretation = (
            "мошеннических операций не меньше 5%, но соотношение классов "
            "всё равно нужно учитывать и смотреть, как модель находит мошенничество."
        )

    if missing_count == 0:
        missing_interpretation = "пропусков в анализируемых столбцах нет."
    else:
        missing_interpretation = (
            f"в анализируемых столбцах {missing_count:,} пропусков, "
            "до обучения нужно решить, что с ними делать."
        )

    report = f"""eda датасета paysim

1. общая информация

файл: {DATA_PATH.name}
транзакций: {total_count:,}
столбцов в анализе: {len(df.columns)}
sha-256 файла: {file_digest}

в paysim 11 столбцов. nameOrig и nameDest не загружались: это номера счетов
отправителя и получателя, а не числовые характеристики операции.

2. распределение классов

обычные: {normal_count:,} ({normal_rate:.6%})
мошеннические: {fraud_count:,} ({fraud_rate:.6%})

{balance_interpretation}

целевой столбец isFraud: 1 — мошенничество, 0 — обычная операция.

3. пропущенные значения

всего пропусков: {missing_count:,}
{missing_interpretation}
по столбцам: пропущенные_значения.csv

4. транзакции по типам

{type_summary_ru.to_string(index=False)}

5. статистика числовых признаков

{numeric_stats.to_string()}

6. флаг isFlaggedFraud

помечено флагом: {flagged_count:,}
из них действительно мошеннических: {flagged_fraud_count:,}

это флаг уже работающей системы, а целевой столбец — isFraud.
в признаки модели флаг не добавляем: иначе будет утечка, и мы не поймём,
находит ли модель мошенничество сама.

7. что дальше

модели сравниваем по метрикам для редкого класса: pr-auc, recall, f1.
accuracy при сильном дисбалансе мало что говорит.

графики и таблицы: artifacts/eda
"""

    (OUTPUT_DIR / "отчет_eda.txt").write_text(report, encoding="utf-8")

    plt.figure(figsize=(7, 5))

    bars = plt.bar(["Обычные", "Мошеннические"], [normal_count, fraud_count])

    plt.title("Распределение классов транзакций")
    plt.ylabel("Количество транзакций")
    plt.xlabel("Класс транзакции")

    plt.bar_label(bars, fmt="%.0f")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "распределение_классов.png", dpi=150)
    plt.close()

    plot_data = type_summary.sort_values("fraud_rate", ascending=False)

    plt.figure(figsize=(8, 5))

    plt.bar(plot_data["type"].astype(str), plot_data["fraud_rate"] * 100)

    plt.title("Доля мошеннических транзакций по типам операций")
    plt.ylabel("Мошеннические транзакции, %")
    plt.xlabel("Тип транзакции")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "мошенничество_по_типам.png", dpi=150)
    plt.close()

    max_normal_samples = 300_000

    normal_amounts = df.loc[df["isFraud"] == 0, "amount"]
    fraud_amounts = df.loc[df["isFraud"] == 1, "amount"]

    if len(normal_amounts) > max_normal_samples:
        normal_amounts = normal_amounts.sample(n=max_normal_samples, random_state=42)

    normal_log_amounts = np.log10(normal_amounts.clip(lower=0.01))
    fraud_log_amounts = np.log10(fraud_amounts.clip(lower=0.01))

    plt.figure(figsize=(8, 5))

    plt.hist(normal_log_amounts, bins=60, alpha=0.7, label="Обычные")

    plt.hist(fraud_log_amounts, bins=60, alpha=0.7, label="Мошеннические")

    plt.title("Распределение сумм транзакций")
    plt.xlabel("Логарифм суммы операции по основанию 10")
    plt.ylabel("Количество транзакций")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "распределение_сумм.png", dpi=150)
    plt.close()

    dataset = mlflow.data.from_pandas(
        df,
        source=str(DATA_PATH.resolve()),
        name="PaySim",
        targets="isFraud",
        digest=file_digest[:32],
    )

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

        mlflow.log_input(dataset, context="eda")

        mlflow.log_artifacts(str(OUTPUT_DIR), artifact_path="eda")

        print(f"eda run_id: {run.info.run_id}")

    print(f"артефакты: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
