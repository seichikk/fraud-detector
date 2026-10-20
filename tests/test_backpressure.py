import asyncio
import csv
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import aio_pika
import pytest
from aio_pika.abc import AbstractRobustConnection
from fakes import DEAD_DATABASE, FakePredictor, MemoryStore, make_message, make_transaction
from faststream.rabbit import RabbitBroker

from fraud_detector.core.config import Settings
from fraud_detector.messaging import (
    TaskRejectedError,
    dead_letter_queue_name,
    declare_topology,
    publish_task,
)
from fraud_detector.services.health import RabbitCheck
from fraud_detector.source import (
    COLUMNS,
    SourceStats,
    parse_args,
    run_batch,
    run_source,
    task_id_for,
)
from fraud_detector.worker import WorkerState, build_broker


@pytest.fixture
def settings(rabbit_url: str) -> Settings:
    suffix = uuid.uuid4().hex[:8]
    return Settings(
        rabbit_url=rabbit_url,
        database_url=DEAD_DATABASE,
        tasks_exchange=f"test-tasks-{suffix}",
        tasks_queue=f"test-tasks-{suffix}",
        queue_max_length=5,
        max_deliveries=3,
        retry_base_seconds=0,
        worker_concurrency=2,
        worker_graceful_timeout=10,
        source_min_backoff_seconds=0.005,
        source_max_backoff_seconds=0.05,
    )


@pytest.fixture
async def connection(rabbit_url: str) -> AsyncIterator[AbstractRobustConnection]:
    connection = await aio_pika.connect_robust(rabbit_url)
    yield connection
    await connection.close()


@pytest.fixture
async def publisher(
    settings: Settings, connection: AbstractRobustConnection
) -> AsyncIterator[RabbitBroker]:
    broker = RabbitBroker(settings.rabbit_url)
    async with broker:
        await declare_topology(broker, settings)
        yield broker
    channel = await connection.channel()
    await channel.queue_delete(settings.tasks_queue)
    await channel.queue_delete(dead_letter_queue_name(settings))
    await channel.exchange_delete(settings.tasks_exchange)
    await channel.exchange_delete(f"{settings.tasks_exchange}.dlx")
    await channel.close()


@pytest.fixture
def executor() -> Iterator[ThreadPoolExecutor]:
    pool = ThreadPoolExecutor(max_workers=2)
    yield pool
    pool.shutdown(wait=True)


async def ready_messages(connection: AbstractRobustConnection, queue: str) -> int:
    channel = await connection.channel()
    try:
        declared = await channel.declare_queue(queue, passive=True)
        return int(declared.declaration_result.message_count or 0)
    finally:
        await channel.close()


async def wait_until(check: Callable[[], Awaitable[bool]], within: float = 20.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + within
    while not await check():
        if loop.time() > deadline:
            raise AssertionError("не дождались нужного состояния очереди")
        await asyncio.sleep(0.05)


async def test_full_queue_rejects_publish(
    settings: Settings, publisher: RabbitBroker, connection: AbstractRobustConnection
) -> None:
    accepted = 0
    rejected = False
    for _ in range(settings.queue_max_length * 3):
        try:
            await publish_task(publisher, settings, make_message())
        except TaskRejectedError:
            rejected = True
            break
        accepted += 1

    depth = await ready_messages(connection, settings.tasks_queue)
    assert rejected
    assert settings.queue_max_length <= accepted <= settings.queue_max_length + 1
    assert depth == accepted


async def test_source_slows_down_and_loses_nothing(
    settings: Settings,
    publisher: RabbitBroker,
    executor: ThreadPoolExecutor,
) -> None:
    store = MemoryStore()
    state = WorkerState(
        settings=settings,
        executor=executor,
        predictor=FakePredictor(delay=0.02),
        store=store,
    )
    worker = build_broker(state)
    total = 60
    stats = SourceStats()

    async with worker:
        await worker.start()
        transactions = iter([make_transaction() for _ in range(total)])
        await run_batch(publisher, settings, transactions, "slow", stats)

        async def all_saved() -> bool:
            return len(store.results) == total

        await wait_until(all_saved)

    assert stats.published == total
    assert stats.rejected > 0
    assert set(store.results) == {task_id_for("slow", number) for number in range(total)}


async def test_poison_message_goes_to_dead_letters(
    settings: Settings,
    publisher: RabbitBroker,
    connection: AbstractRobustConnection,
    executor: ThreadPoolExecutor,
) -> None:
    predictor = FakePredictor(fail=True)
    state = WorkerState(
        settings=settings, executor=executor, predictor=predictor, store=MemoryStore()
    )
    worker = build_broker(state)
    dlq = dead_letter_queue_name(settings)

    async with worker:
        await worker.start()
        await publish_task(publisher, settings, make_message())

        async def dead_lettered() -> bool:
            return await ready_messages(connection, dlq) == 1

        await wait_until(dead_lettered)

    assert predictor.calls == settings.max_deliveries
    assert state.stats.failed == settings.max_deliveries
    assert await ready_messages(connection, settings.tasks_queue) == 0


async def test_garbage_is_rejected_without_retries(
    settings: Settings,
    publisher: RabbitBroker,
    connection: AbstractRobustConnection,
    executor: ThreadPoolExecutor,
) -> None:
    predictor = FakePredictor()
    state = WorkerState(
        settings=settings, executor=executor, predictor=predictor, store=MemoryStore()
    )
    worker = build_broker(state)
    dlq = dead_letter_queue_name(settings)

    async with worker:
        await worker.start()
        await publisher.publish(
            b"{not a json",
            exchange=settings.tasks_exchange,
            routing_key=settings.tasks_queue,
        )

        async def dead_lettered() -> bool:
            return await ready_messages(connection, dlq) == 1

        await wait_until(dead_lettered)

    channel = await connection.channel()
    queue = await channel.declare_queue(dlq, passive=True)
    message = await queue.get(no_ack=True)
    await channel.close()

    assert message.body == b"{not a json"
    assert message.headers["x-first-death-reason"] == "rejected"
    assert state.stats.rejected == 1
    assert predictor.calls == 0


async def test_stop_waits_for_tasks_in_progress(
    settings: Settings,
    publisher: RabbitBroker,
    connection: AbstractRobustConnection,
    executor: ThreadPoolExecutor,
) -> None:
    store = MemoryStore()
    state = WorkerState(
        settings=settings,
        executor=executor,
        predictor=FakePredictor(delay=0.5),
        store=store,
    )
    worker = build_broker(state)

    await worker.start()
    for _ in range(settings.worker_concurrency):
        await publish_task(publisher, settings, make_message())

    async def busy() -> bool:
        return state.stats.in_flight == settings.worker_concurrency

    await wait_until(busy)
    await worker.stop()

    assert state.stats.in_flight == 0
    assert len(store.results) == settings.worker_concurrency
    assert await ready_messages(connection, settings.tasks_queue) == 0


async def test_unroutable_task_is_refused(settings: Settings, publisher: RabbitBroker) -> None:
    lost = settings.model_copy(update={"tasks_queue": "nobody-listens-here"})

    with pytest.raises(TaskRejectedError, match="ни в одну очередь"):
        await publish_task(publisher, lost, make_message())


async def test_rabbit_health_check_reads_server_version(rabbit_url: str) -> None:
    version = await RabbitCheck(rabbit_url).fetch_version()

    assert version.split(".")[0].isdigit()


async def test_source_publishes_csv_file(
    settings: Settings,
    publisher: RabbitBroker,
    connection: AbstractRobustConnection,
    tmp_path: Path,
) -> None:
    data = tmp_path / "transactions.csv"
    with data.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[*COLUMNS, "isFraud"])
        writer.writeheader()
        for step in range(1, 6):
            writer.writerow(
                {
                    "step": step,
                    "type": "PAYMENT",
                    "amount": 10.0 * step,
                    "oldbalanceOrg": 100.0,
                    "newbalanceOrig": 90.0,
                    "oldbalanceDest": 0.0,
                    "newbalanceDest": 0.0,
                    "isFraud": 0,
                }
            )
    args = parse_args(["--file", str(data), "--limit", "3", "--metrics", str(tmp_path / "m.csv")])

    stats = await run_source(settings, args)

    assert stats.published == 3
    assert stats.rejected == 0
    assert await ready_messages(connection, settings.tasks_queue) == 3
