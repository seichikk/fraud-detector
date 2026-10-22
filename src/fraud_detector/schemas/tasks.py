import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

TaskSource = Literal["api", "batch"]
TaskState = Literal["pending", "done"]


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, ser_json_inf_nan="constants")

    step: float = Field(ge=1, description="Номер временного шага")
    type: str = Field(min_length=1, max_length=32, description="Тип транзакции")
    amount: float = Field(ge=0, description="Сумма транзакции")
    oldbalance_org: float = Field(alias="oldbalanceOrg", ge=0, description="Баланс отправителя до")
    newbalance_orig: float = Field(
        alias="newbalanceOrig", ge=0, description="Баланс отправителя после"
    )
    oldbalance_dest: float = Field(alias="oldbalanceDest", ge=0, description="Баланс получателя до")
    newbalance_dest: float = Field(
        alias="newbalanceDest", ge=0, description="Баланс получателя после"
    )


class TaskMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: uuid.UUID
    source: TaskSource
    created_at: datetime
    transaction: Transaction


class TaskAccepted(BaseModel):
    task_id: uuid.UUID
    status: TaskState = "pending"


class TaskResult(BaseModel):
    is_fraud: bool
    fraud_probability: float = Field(ge=0, le=1)
    model_version: str
    started_at: datetime
    finished_at: datetime


class Prediction(BaseModel):
    prediction: Literal["FRAUD", "NORMAL"]
    is_fraud: bool
    fraud_probability: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    model_name: str
    model_version: str
    model_alias: str


class TaskView(BaseModel):
    task_id: uuid.UUID
    source: TaskSource
    status: TaskState
    created_at: datetime
    result: TaskResult | None = None


class Overloaded(BaseModel):
    detail: str
