import logging

import pandas as pd
from fastapi import APIRouter, HTTPException, Request, status

from fraud_detector.schemas.process import (
    PredictionResponse,
    TransactionRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["inference"])

MODEL_NAME = "FraudDetectionModel"
MODEL_ALIAS = "champion"
PREDICTION_THRESHOLD = 0.5


@router.post("/process", response_model=PredictionResponse)
async def process_transaction(
    transaction: TransactionRequest,
    request: Request,
) -> PredictionResponse:
    model = getattr(request.app.state, "fraud_model", None)
    model_version = getattr(request.app.state, "fraud_model_version", None)

    if model is None or model_version is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Модель не загружена. Проверь настройки MLflow.",
        )

    try:
        input_data = pd.DataFrame(
            [transaction.model_dump(by_alias=True)]
        )

        probability = float(model.predict_proba(input_data)[0, 1])
        is_fraud = probability >= PREDICTION_THRESHOLD

    except Exception as exc:
        logger.exception("Ошибка обработки транзакции")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Не удалось обработать транзакцию.",
        ) from exc

    return PredictionResponse(
        prediction="FRAUD" if is_fraud else "NORMAL",
        is_fraud=is_fraud,
        fraud_probability=probability,
        threshold=PREDICTION_THRESHOLD,
        model_name=MODEL_NAME,
        model_version=str(model_version),
        model_alias=MODEL_ALIAS,
    )
