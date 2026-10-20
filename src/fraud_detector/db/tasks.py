import uuid
from datetime import datetime
from typing import Any, Protocol

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    MetaData,
    String,
    Table,
    Uuid,
    func,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, insert
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from fraud_detector.schemas.tasks import TaskMessage, TaskResult, TaskView

metadata = MetaData()

tasks = Table(
    "tasks",
    metadata,
    Column("task_id", Uuid, primary_key=True),
    Column("source", String(16), nullable=False),
    Column("status", String(16), nullable=False),
    Column("payload", JSONB, nullable=False),
    Column("is_fraud", Boolean),
    Column("fraud_probability", Float),
    Column("model_version", String(32)),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("started_at", DateTime(timezone=True)),
    Column("finished_at", DateTime(timezone=True)),
    CheckConstraint("status in ('pending', 'done')", name="tasks_status_check"),
)

SCHEMA_LOCK_ID = 465282


class ResultStore(Protocol):
    async def save_result(self, message: TaskMessage, result: TaskResult) -> bool: ...


async def ensure_schema(engine: AsyncEngine) -> None:
    async with engine.begin() as connection:
        await connection.execute(text("select pg_advisory_xact_lock(:id)"), {"id": SCHEMA_LOCK_ID})
        await connection.run_sync(metadata.create_all, checkfirst=True)


def _payload(message: TaskMessage) -> dict[str, Any]:
    return message.transaction.model_dump(by_alias=True)


async def insert_pending(connection: AsyncConnection, message: TaskMessage) -> None:
    await connection.execute(
        tasks.insert().values(
            task_id=message.task_id,
            source=message.source,
            status="pending",
            payload=_payload(message),
            created_at=message.created_at,
        )
    )


class TaskRepository:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def save_result(self, message: TaskMessage, result: TaskResult) -> bool:
        statement = insert(tasks).values(
            task_id=message.task_id,
            source=message.source,
            status="done",
            payload=_payload(message),
            created_at=message.created_at,
            **result.model_dump(),
        )
        statement = statement.on_conflict_do_update(
            index_elements=[tasks.c.task_id],
            set_={
                "status": "done",
                **{name: statement.excluded[name] for name in TaskResult.model_fields},
            },
            where=tasks.c.status != "done",
        )
        async with self._engine.begin() as connection:
            outcome = await connection.execute(statement)
        return bool(outcome.rowcount)

    async def get(self, task_id: uuid.UUID) -> TaskView | None:
        async with self._engine.connect() as connection:
            row = (
                (await connection.execute(select(tasks).where(tasks.c.task_id == task_id)))
                .mappings()
                .first()
            )
        if row is None:
            return None

        result = None
        if row["status"] == "done":
            result = TaskResult.model_validate(dict(row))
        return TaskView(
            task_id=row["task_id"],
            source=row["source"],
            status=row["status"],
            created_at=row["created_at"],
            result=result,
        )

    async def count_by_status(self) -> dict[str, int]:
        async with self._engine.connect() as connection:
            rows = await connection.execute(
                select(tasks.c.status, func.count()).group_by(tasks.c.status)
            )
        return dict(rows.tuples().all())

    async def finished_since(self, moment: datetime) -> list[dict[str, Any]]:
        async with self._engine.connect() as connection:
            rows = await connection.execute(
                select(tasks.c.created_at, tasks.c.started_at, tasks.c.finished_at)
                .where(tasks.c.finished_at >= moment)
                .order_by(tasks.c.finished_at)
            )
        return [dict(row) for row in rows.mappings().all()]
