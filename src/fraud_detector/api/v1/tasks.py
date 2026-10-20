import logging
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Response, status

from fraud_detector.api.deps import BrokerDep, EngineDep, SettingsDep, TaskRepositoryDep
from fraud_detector.db.tasks import insert_pending
from fraud_detector.messaging import TaskRejectedError, publish_task
from fraud_detector.schemas.tasks import (
    Overloaded,
    TaskAccepted,
    TaskMessage,
    TaskView,
    Transaction,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": Overloaded}},
)
async def submit_task(
    transaction: Transaction,
    response: Response,
    settings: SettingsDep,
    engine: EngineDep,
    broker: BrokerDep,
) -> TaskAccepted:
    message = TaskMessage(
        task_id=uuid.uuid4(),
        source="api",
        created_at=datetime.now(UTC),
        transaction=transaction,
    )

    try:
        async with engine.begin() as connection:
            await insert_pending(connection, message)
            await publish_task(broker, settings, message)
    except TaskRejectedError as exc:
        logger.warning("задача %s не принята: %s", message.task_id, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Очередь задач заполнена, повторите запрос позже",
            headers={"Retry-After": str(settings.retry_after_seconds)},
        ) from exc

    response.headers["Location"] = f"/api/v1/tasks/{message.task_id}"
    return TaskAccepted(task_id=message.task_id)


@router.get(
    "/{task_id}", responses={status.HTTP_404_NOT_FOUND: {"description": "Нет такой задачи"}}
)
async def read_task(task_id: uuid.UUID, repository: TaskRepositoryDep) -> TaskView:
    view = await repository.get(task_id)
    if view is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Задача не найдена")
    return view
