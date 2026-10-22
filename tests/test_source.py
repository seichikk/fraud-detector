import asyncio
import csv
from pathlib import Path
from typing import cast

import pytest
from fakes import DEAD_RABBIT, make_message, make_transaction
from faststream.rabbit import RabbitBroker

from fraud_detector import source
from fraud_detector.core.config import Settings
from fraud_detector.messaging import TaskRejectedError
from fraud_detector.schemas.tasks import TaskMessage
from fraud_detector.source import (
    Backoff,
    SourceStats,
    parse_args,
    publish_with_backoff,
    read_transactions,
    run_batch,
    task_id_for,
    write_metrics,
)

HEADER = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]


def write_csv(path: Path, rows: int) -> Path:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(HEADER)
        for number in range(rows):
            writer.writerow(
                [1, "PAYMENT", 100 + number, "C1", 500, 400, "M1", 0, 0, 0, 0],
            )
    return path


class FlakyBroker:
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.sent: list[TaskMessage] = []


@pytest.fixture
def flaky(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    pauses: list[float] = []

    async def fake_publish(broker: FlakyBroker, settings: Settings, message: TaskMessage) -> None:
        if broker.failures > 0:
            broker.failures -= 1
            raise TaskRejectedError("очередь заполнена")
        broker.sent.append(message)

    async def fake_sleep(delay: float) -> None:
        pauses.append(delay)

    monkeypatch.setattr(source, "publish_task", fake_publish)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return pauses


def test_backoff_doubles_up_to_ceiling_and_relaxes_on_success() -> None:
    backoff = Backoff(minimum=0.01, maximum=0.08)

    pauses = [backoff.on_reject() for _ in range(5)]

    assert backoff.delay == pytest.approx(0.08)
    assert all(0 < pause <= 0.08 for pause in pauses)
    assert pauses[0] >= 0.005

    backoff.on_success()
    assert backoff.delay == pytest.approx(0.04)
    for _ in range(4):
        backoff.on_success()
    assert backoff.delay == 0


def test_transactions_are_read_lazily_and_limited(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "batch.csv", rows=5)

    transactions = list(read_transactions(path, limit=3))

    assert [item.amount for item in transactions] == [100, 101, 102]
    assert transactions[0].oldbalance_org == 500


def test_task_ids_are_stable_within_batch() -> None:
    assert task_id_for("b", 1) == task_id_for("b", 1)
    assert task_id_for("b", 1) != task_id_for("b", 2)
    assert task_id_for("b", 1) != task_id_for("c", 1)


async def test_rejected_publish_waits_and_retries_same_task(flaky: list[float]) -> None:
    broker = FlakyBroker(failures=3)
    stats = SourceStats()
    message = make_message()

    await publish_with_backoff(
        cast(RabbitBroker, broker),
        Settings(rabbit_url=DEAD_RABBIT),
        message,
        Backoff(0.01, 0.5),
        stats,
    )

    assert stats.rejected == 3
    assert stats.published == 1
    assert len(flaky) == 3
    assert all(pause > 0 for pause in flaky)
    assert flaky[2] > flaky[0] / 2
    assert [sent.task_id for sent in broker.sent] == [message.task_id]


async def test_stop_interrupts_retries_of_rejected_task(flaky: list[float]) -> None:
    broker = FlakyBroker(failures=1_000_000)
    stats = SourceStats()
    stop = asyncio.Event()
    stop.set()

    await publish_with_backoff(
        cast(RabbitBroker, broker),
        Settings(rabbit_url=DEAD_RABBIT),
        make_message(),
        Backoff(0.01, 0.5),
        stats,
        stop,
    )

    assert stats.rejected == 1
    assert stats.published == 0
    assert broker.sent == []


async def test_batch_publishes_every_row_once(flaky: list[float]) -> None:
    broker = FlakyBroker(failures=2)
    stats = SourceStats()

    await run_batch(
        cast(RabbitBroker, broker),
        Settings(rabbit_url=DEAD_RABBIT),
        iter([make_transaction(amount) for amount in (1, 2, 3)]),
        "batch-1",
        stats,
    )

    assert stats.published == 3
    assert len({message.task_id for message in broker.sent}) == 3
    assert broker.sent[0].task_id == task_id_for("batch-1", 0)


async def test_batch_stops_on_signal(flaky: list[float]) -> None:
    broker = FlakyBroker(failures=0)
    stop = asyncio.Event()
    stop.set()

    await run_batch(
        cast(RabbitBroker, broker),
        Settings(rabbit_url=DEAD_RABBIT),
        iter([make_transaction()]),
        "batch-2",
        SourceStats(),
        stop,
    )

    assert broker.sent == []


async def test_metrics_are_written_every_period(tmp_path: Path) -> None:
    stats = SourceStats(published=10)
    path = tmp_path / "metrics.csv"

    reporter = asyncio.create_task(write_metrics(stats, path, period=0.01))
    await asyncio.sleep(0.05)
    reporter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reporter

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert rows
    assert rows[0]["published"] == "10"


def test_arguments_have_sensible_defaults() -> None:
    args = parse_args([])

    assert args.file == Path("data/raw/transactions.csv")
    assert args.limit is None
    assert parse_args(["--limit", "5", "--batch-id", "x"]).batch_id == "x"
