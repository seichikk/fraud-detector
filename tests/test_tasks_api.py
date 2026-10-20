import asyncio
import uuid
from datetime import UTC, datetime

import pytest
from fakes import DEAD_RABBIT
from fastapi import FastAPI
from faststream.rabbit import RabbitBroker
from httpx import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine

from fraud_detector.api.v1 import tasks as tasks_api
from fraud_detector.core.config import Settings
from fraud_detector.db.tasks import TaskRepository, tasks
from fraud_detector.messaging import TaskRejectedError
from fraud_detector.schemas.tasks import TaskMessage, TaskResult, Transaction

TRANSACTION = {
    "step": 1,
    "type": "TRANSFER",
    "amount": 181.0,
    "oldbalanceOrg": 181.0,
    "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0,
}


@pytest.fixture
def settings(database_url: str) -> Settings:
    return Settings(
        database_url=database_url,
        rabbit_url=DEAD_RABBIT,
        queue_max_length=100,
        inference_seconds=0.2,
        worker_concurrency=4,
    )


@pytest.fixture
def published(monkeypatch: pytest.MonkeyPatch) -> list[TaskMessage]:
    messages: list[TaskMessage] = []

    async def fake_publish(broker: RabbitBroker, settings: Settings, message: TaskMessage) -> None:
        messages.append(message)

    monkeypatch.setattr(tasks_api, "publish_task", fake_publish)
    return messages


@pytest.fixture
def rejected(monkeypatch: pytest.MonkeyPatch) -> list[TaskMessage]:
    messages: list[TaskMessage] = []

    async def full_queue(broker: RabbitBroker, settings: Settings, message: TaskMessage) -> None:
        messages.append(message)
        raise TaskRejectedError("брокер отклонил сообщение")

    monkeypatch.setattr(tasks_api, "publish_task", full_queue)
    return messages


async def count_rows(engine: AsyncEngine, task_id: uuid.UUID) -> int:
    async with engine.connect() as connection:
        query = select(func.count()).select_from(tasks).where(tasks.c.task_id == task_id)
        return int((await connection.execute(query)).scalar_one())


async def test_accepted_task_is_pending_and_published(
    app: FastAPI, client: AsyncClient, published: list[TaskMessage]
) -> None:
    response = await client.post("/api/v1/tasks", json=TRANSACTION)

    assert response.status_code == 202
    task_id = uuid.UUID(response.json()["task_id"])
    assert response.headers["location"] == f"/api/v1/tasks/{task_id}"
    assert [message.task_id for message in published] == [task_id]

    view = await client.get(f"/api/v1/tasks/{task_id}")
    assert view.json()["status"] == "pending"
    assert view.json()["result"] is None


async def test_full_queue_gives_503_and_leaves_nothing_pending(
    app: FastAPI, client: AsyncClient, rejected: list[TaskMessage]
) -> None:
    response = await client.post("/api/v1/tasks", json=TRANSACTION)

    assert response.status_code == 503
    assert response.headers["retry-after"] == "5"
    assert len(rejected) == 1
    assert await count_rows(app.state.engine, rejected[0].task_id) == 0


async def test_finished_task_shows_result(
    app: FastAPI, client: AsyncClient, published: list[TaskMessage]
) -> None:
    response = await client.post("/api/v1/tasks", json=TRANSACTION)
    message = published[0]
    repository = TaskRepository(app.state.engine)
    result = TaskResult(
        is_fraud=True,
        fraud_probability=0.93,
        model_version="2",
        started_at=message.created_at,
        finished_at=message.created_at,
    )

    assert await repository.save_result(message, result) is True
    assert await repository.save_result(message, result) is False

    view = (await client.get(f"/api/v1/tasks/{response.json()['task_id']}")).json()
    assert view["status"] == "done"
    assert view["result"]["model_version"] == "2"
    assert await count_rows(app.state.engine, message.task_id) == 1
    assert (await repository.count_by_status())["done"] >= 1
    assert await repository.finished_since(message.created_at)


async def test_unknown_task_is_404(client: AsyncClient) -> None:
    response = await client.get(f"/api/v1/tasks/{uuid.uuid4()}")

    assert response.status_code == 404


async def test_invalid_transaction_is_422(
    client: AsyncClient, published: list[TaskMessage]
) -> None:
    response = await client.post("/api/v1/tasks", json={**TRANSACTION, "amount": -1})

    assert response.status_code == 422
    assert published == []


async def test_concurrent_redelivery_writes_single_row(
    app: FastAPI, published: list[TaskMessage]
) -> None:
    repository = TaskRepository(app.state.engine)
    message = TaskMessage(
        task_id=uuid.uuid4(),
        source="batch",
        created_at=datetime.now(UTC),
        transaction=Transaction.model_validate(TRANSACTION),
    )
    result = TaskResult(
        is_fraud=False,
        fraud_probability=0.1,
        model_version="1",
        started_at=message.created_at,
        finished_at=message.created_at,
    )

    outcomes = await asyncio.gather(*(repository.save_result(message, result) for _ in range(5)))

    assert sorted(outcomes) == [False, False, False, False, True]
    assert await count_rows(app.state.engine, message.task_id) == 1
