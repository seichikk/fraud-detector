from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TransactionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: float = Field(ge=1, description="Номер временного шага")
    type: str = Field(min_length=1, max_length=32, description="Тип транзакции")
    amount: float = Field(ge=0, description="Сумма транзакции")
    oldbalance_org: float = Field(
        alias="oldbalanceOrg",
        ge=0,
        description="Баланс отправителя до операции",
    )
    newbalance_orig: float = Field(
        alias="newbalanceOrig",
        ge=0,
        description="Баланс отправителя после операции",
    )
    oldbalance_dest: float = Field(
        alias="oldbalanceDest",
        ge=0,
        description="Баланс получателя до операции",
    )
    newbalance_dest: float = Field(
        alias="newbalanceDest",
        ge=0,
        description="Баланс получателя после операции",
    )


class PredictionResponse(BaseModel):
    prediction: Literal["FRAUD", "NORMAL"]
    is_fraud: bool
    fraud_probability: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    model_name: str
    model_version: str
    model_alias: str
