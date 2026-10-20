from collections.abc import AsyncIterator, Iterator
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from fakes import (
    DEAD_DATABASE,
    DEAD_RABBIT,
    FakePredictor,
    MemoryStore,
    make_message,
    make_transaction,
)
from faststream.rabbit import RabbitBroker, TestRabbitBroker
from httpx import ASGITransport, AsyncClient
from numpy.typing import NDArray

from fraud_detector.core.config import Settings
from fraud_detector.messaging import tasks_exchange
from fraud_detector.ml import MlflowPredictor
from fraud_detector.worker import (
    WorkerState,
    build_broker,
    create_worker_app,
    process_task,
    retry_delay,
)


@pytest.fixture
def worker_settings() -> Settings:
    return Settings(
        rabbit_url=DEAD_RABBIT,
        database_url=DEAD_DATABASE,
        retry_base_seconds=0,
        worker_concurrency=2,
    )


@pytest.fixture
def executor() -> Iterator[ThreadPoolExecutor]:
    pool = ThreadPoolExecutor(max_workers=2)
    yield pool
    pool.shutdown(wait=True)


@pytest.fixture
def state(worker_settings: Settings, executor: ThreadPoolExecutor) -> WorkerState:
    return WorkerState(
        settings=worker_settings,
        executor=executor,
        predictor=FakePredictor(),
        store=MemoryStore(),
    )


@pytest.fixture
async def broker(state: WorkerState) -> AsyncIterator[RabbitBroker]:
    real = build_broker(state)
    async with TestRabbitBroker(real) as test_broker:
        yield test_broker


async def publish(broker: RabbitBroker, settings: Settings, body: str | bytes) -> None:
    await broker.publish(body, exchange=tasks_exchange(settings), routing_key=settings.tasks_queue)


async def test_valid_task_is_scored_and_saved(broker: RabbitBroker, state: WorkerState) -> None:
    message = make_message()

    await publish(broker, state.settings, message.model_dump_json(by_alias=True))

    assert isinstance(state.store, MemoryStore)
    result = state.store.results[message.task_id]
    assert result.is_fraud is True
    assert result.fraud_probability == pytest.approx(0.9)
    assert result.model_version == "7"
    assert result.started_at <= result.finished_at
    assert state.stats.processed == 1
    assert state.stats.in_flight == 0


async def test_low_probability_is_not_fraud(broker: RabbitBroker, state: WorkerState) -> None:
    state.predictor = FakePredictor(probability=0.1)
    message = make_message()

    await publish(broker, state.settings, message.model_dump_json(by_alias=True))

    assert isinstance(state.store, MemoryStore)
    assert state.store.results[message.task_id].is_fraud is False


async def test_redelivery_does_not_duplicate_result(
    broker: RabbitBroker, state: WorkerState
) -> None:
    body = make_message().model_dump_json(by_alias=True)

    await publish(broker, state.settings, body)
    await publish(broker, state.settings, body)

    assert isinstance(state.store, MemoryStore)
    assert len(state.store.results) == 1
    assert state.stats.processed == 1
    assert state.stats.duplicates == 1


async def test_message_with_wrong_schema_is_rejected(
    broker: RabbitBroker, state: WorkerState
) -> None:
    await publish(broker, state.settings, b"{not a json")
    await publish(broker, state.settings, '{"task_id": "nope"}')

    assert isinstance(state.store, MemoryStore)
    assert state.store.results == {}
    assert state.stats.rejected == 2
    assert isinstance(state.predictor, FakePredictor)
    assert state.predictor.calls == 0


async def test_failed_inference_is_returned_to_queue(
    broker: RabbitBroker, state: WorkerState
) -> None:
    state.predictor = FakePredictor(fail=True)

    with pytest.raises(ValueError, match="модель не смогла"):
        await publish(broker, state.settings, make_message().model_dump_json(by_alias=True))

    assert state.stats.failed == 1
    assert state.stats.in_flight == 0


async def test_worker_without_model_refuses_to_process(state: WorkerState) -> None:
    state.predictor = None

    with pytest.raises(RuntimeError, match="не готов"):
        await process_task(state, make_message())


def test_retry_delay_grows_exponentially() -> None:
    settings = Settings(retry_base_seconds=0.5)

    assert [retry_delay(settings, attempt) for attempt in (1, 2, 3)] == [0.5, 1.0, 2.0]


def test_prefetch_keeps_one_task_ready_for_thread_pool() -> None:
    assert Settings(worker_concurrency=6).effective_prefetch == 7
    assert Settings(worker_concurrency=6, prefetch_count=0).effective_prefetch == 0


def test_mlflow_predictor_uses_model_probability() -> None:
    class Model:
        def predict_proba(self, frame: object) -> NDArray[np.float64]:
            return np.asarray([[0.3, 0.7]])

    predictor = MlflowPredictor(model=Model(), version="3", inference_seconds=0)

    assert predictor.predict(make_transaction()) == pytest.approx(0.7)


async def test_worker_http_endpoints_answer(worker_settings: Settings) -> None:
    app = create_worker_app(worker_settings)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://w") as http:
        liveness = await http.get("/healthz")
        stats = await http.get("/stats")

    assert liveness.status_code == 200
    assert liveness.json() == {"status": "ok"}
    assert stats.json()["prefetch"] == 3
    assert stats.json()["model_version"] is None
