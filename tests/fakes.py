import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from fraud_detector.schemas.tasks import TaskMessage, TaskResult, Transaction

DEAD_DATABASE = "postgresql+asyncpg://nobody:nothing@127.0.0.1:1/nowhere"
DEAD_RABBIT = "amqp://nobody:nothing@127.0.0.1:1/"


def make_transaction(amount: float = 181.0) -> Transaction:
    return Transaction(
        step=1,
        type="TRANSFER",
        amount=amount,
        oldbalance_org=181.0,
        newbalance_orig=0.0,
        oldbalance_dest=0.0,
        newbalance_dest=0.0,
    )


def make_message(task_id: uuid.UUID | None = None, amount: float = 181.0) -> TaskMessage:
    return TaskMessage(
        task_id=task_id or uuid.uuid4(),
        source="api",
        created_at=datetime.now(UTC),
        transaction=make_transaction(amount),
    )


@dataclass
class FakePredictor:
    probability: float = 0.9
    version: str = "7"
    fail: bool = False
    delay: float = 0.0
    calls: int = 0

    def predict(self, transaction: Transaction) -> float:
        self.calls += 1
        time.sleep(self.delay)
        if self.fail:
            raise ValueError("модель не смогла посчитать")
        return self.probability


@dataclass
class MemoryStore:
    results: dict[uuid.UUID, TaskResult] = field(default_factory=dict)

    async def save_result(self, message: TaskMessage, result: TaskResult) -> bool:
        if message.task_id in self.results:
            return False
        self.results[message.task_id] = result
        return True
