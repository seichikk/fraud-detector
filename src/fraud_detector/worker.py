import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

import uvicorn
from faststream import AckPolicy
from faststream.asgi import AsgiFastStream, AsgiResponse, Request, get, make_ping_asgi, post
from faststream.asgi.types import Scope
from faststream.exceptions import RejectMessage
from faststream.rabbit import Channel, RabbitBroker
from faststream.rabbit.annotations import RabbitMessage
from pydantic import ValidationError

from fraud_detector.core.config import Settings, get_settings
from fraud_detector.core.logs import setup_logging
from fraud_detector.db.engine import create_engine
from fraud_detector.db.tasks import ResultStore, TaskRepository, ensure_schema
from fraud_detector.messaging import (
    dead_letter_queue_name,
    declare_dead_letters,
    tasks_exchange,
    tasks_queue,
)
from fraud_detector.ml import Predictor, load_predictor
from fraud_detector.schemas.tasks import Prediction, TaskMessage, TaskResult, Transaction

logger = logging.getLogger(__name__)


@dataclass
class WorkerStats:
    in_flight: int = 0
    processed: int = 0
    duplicates: int = 0
    rejected: int = 0
    failed: int = 0


@dataclass
class WorkerState:
    settings: Settings
    executor: ThreadPoolExecutor
    predictor: Predictor | None = None
    store: ResultStore | None = None
    stats: WorkerStats = field(default_factory=WorkerStats)


async def raw_body(message: RabbitMessage) -> bytes:
    return message.body


def retry_delay(settings: Settings, attempt: int) -> float:
    return float(settings.retry_base_seconds * 2 ** (attempt - 1))


def timed_predict(predictor: Predictor, message: TaskMessage) -> tuple[datetime, float]:
    started_at = datetime.now(UTC)
    return started_at, predictor.predict(message.transaction)


async def process_task(state: WorkerState, message: TaskMessage) -> bool:
    if state.predictor is None or state.store is None:
        raise RuntimeError("Воркер ещё не готов: нет модели или хранилища")

    loop = asyncio.get_running_loop()
    started_at, probability = await loop.run_in_executor(
        state.executor, timed_predict, state.predictor, message
    )
    result = TaskResult(
        is_fraud=probability >= state.settings.prediction_threshold,
        fraud_probability=probability,
        model_version=state.predictor.version,
        started_at=started_at,
        finished_at=datetime.now(UTC),
    )
    return await state.store.save_result(message, result)


def build_broker(state: WorkerState) -> RabbitBroker:
    settings = state.settings
    broker = RabbitBroker(
        settings.rabbit_url,
        graceful_timeout=settings.worker_graceful_timeout,
        log_level=logging.DEBUG,
    )

    @broker.subscriber(
        tasks_queue(settings),
        tasks_exchange(settings),
        channel=Channel(prefetch_count=settings.effective_prefetch),
        ack_policy=AckPolicy.NACK_ON_ERROR,
        decoder=raw_body,
    )
    async def handle_task(body: bytes, raw: RabbitMessage) -> None:
        try:
            message = TaskMessage.model_validate_json(body)
        except ValidationError as exc:
            state.stats.rejected += 1
            logger.warning(
                "сообщение не подходит под схему (%s ошибок), уходит в %s",
                exc.error_count(),
                dead_letter_queue_name(settings),
            )
            raise RejectMessage from exc

        state.stats.in_flight += 1
        try:
            saved = await process_task(state, message)
        except Exception:
            state.stats.failed += 1
            attempt = int(raw.headers.get("x-delivery-count", 0)) + 1
            logger.exception(
                "задача %s упала на попытке %s из %s",
                message.task_id,
                attempt,
                settings.max_deliveries,
            )
            if attempt < settings.max_deliveries:
                await asyncio.sleep(retry_delay(settings, attempt))
            raise
        finally:
            state.stats.in_flight -= 1

        if saved:
            state.stats.processed += 1
        else:
            state.stats.duplicates += 1
            logger.info("задача %s уже обработана, повтор пропущен", message.task_id)

    return broker


def json_response(payload: object, status_code: int = 200) -> AsgiResponse:
    return AsgiResponse(
        json.dumps(payload, ensure_ascii=False).encode(),
        status_code=status_code,
        headers={"content-type": "application/json"},
    )


def create_worker_app(settings: Settings, state: WorkerState | None = None) -> AsgiFastStream:
    if state is None:
        executor = ThreadPoolExecutor(
            max_workers=settings.worker_concurrency, thread_name_prefix="inference"
        )
        state = WorkerState(settings=settings, executor=executor)
    broker = build_broker(state)
    engine = create_engine(settings.database_url)

    async def prepare() -> None:
        await ensure_schema(engine)
        state.store = TaskRepository(engine)
        state.predictor = await load_predictor(settings)
        await broker.connect()
        await declare_dead_letters(broker, settings)
        logger.info(
            "воркер готов: потоков %s, prefetch %s, inference %s с",
            settings.worker_concurrency,
            settings.effective_prefetch,
            settings.inference_seconds,
        )

    async def release() -> None:
        await asyncio.to_thread(state.executor.shutdown, wait=True)
        await engine.dispose()
        logger.info("воркер остановлен: %s", asdict(state.stats))

    @get
    async def liveness(scope: Scope) -> AsgiResponse:
        return json_response({"status": "ok"})

    @get
    async def stats(scope: Scope) -> AsgiResponse:
        return json_response(
            {
                **asdict(state.stats),
                "model_version": state.predictor.version if state.predictor else None,
                "concurrency": settings.worker_concurrency,
                "prefetch": settings.effective_prefetch,
            }
        )

    @post
    async def process(scope: Scope, request: Request) -> AsgiResponse:
        predictor = state.predictor
        if predictor is None:
            return json_response({"detail": "модель ещё не загружена"}, status_code=503)
        try:
            transaction = Transaction.model_validate_json(await request.body())
        except ValidationError as exc:
            errors = exc.errors(include_url=False, include_context=False, include_input=False)
            return json_response({"detail": errors}, status_code=422)

        loop = asyncio.get_running_loop()
        probability = await loop.run_in_executor(state.executor, predictor.predict, transaction)
        is_fraud = probability >= settings.prediction_threshold
        prediction = Prediction(
            prediction="FRAUD" if is_fraud else "NORMAL",
            is_fraud=is_fraud,
            fraud_probability=probability,
            threshold=settings.prediction_threshold,
            model_name=settings.mlflow_model_name,
            model_version=predictor.version,
            model_alias=settings.mlflow_model_alias,
        )
        return json_response(prediction.model_dump(mode="json"))

    return AsgiFastStream(
        broker,
        asgi_routes=[
            ("/healthz", liveness),
            ("/health", make_ping_asgi(broker, timeout=settings.health_timeout_seconds)),
            ("/stats", stats),
            ("/process", process),
        ],
        logger=logger,
        on_startup=[prepare],
        after_shutdown=[release],
    )


def main() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    uvicorn.run(
        create_worker_app(settings),
        host=settings.worker_host,
        port=settings.worker_port,
        log_config=None,
        access_log=False,
    )
