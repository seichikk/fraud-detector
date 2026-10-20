from aiormq.abc import DeliveredMessage
from aiormq.exceptions import AMQPError
from faststream.rabbit import ExchangeType, QueueType, RabbitBroker, RabbitExchange, RabbitQueue
from faststream.rabbit.schemas.queue import QuorumQueueArgs

from fraud_detector.core.config import Settings
from fraud_detector.schemas.tasks import TaskMessage


class TaskRejectedError(Exception):
    pass


def tasks_exchange(settings: Settings) -> RabbitExchange:
    return RabbitExchange(settings.tasks_exchange, type=ExchangeType.DIRECT, durable=True)


def dead_letter_exchange(settings: Settings) -> RabbitExchange:
    return RabbitExchange(f"{settings.tasks_exchange}.dlx", type=ExchangeType.DIRECT, durable=True)


def dead_letter_queue_name(settings: Settings) -> str:
    return f"{settings.tasks_queue}.dlq"


def tasks_queue(settings: Settings) -> RabbitQueue:
    arguments: QuorumQueueArgs = {
        "x-dead-letter-exchange": dead_letter_exchange(settings).name,
        "x-dead-letter-routing-key": dead_letter_queue_name(settings),
        "x-dead-letter-strategy": "at-least-once",
        "x-overflow": "reject-publish",
        "x-delivery-limit": settings.max_deliveries - 1,
    }
    if settings.bounded_queue:
        arguments["x-max-length"] = settings.queue_max_length

    return RabbitQueue(
        settings.tasks_queue,
        queue_type=QueueType.QUORUM,
        durable=True,
        arguments=arguments,
        routing_key=settings.tasks_queue,
    )


def dead_letter_queue(settings: Settings) -> RabbitQueue:
    name = dead_letter_queue_name(settings)
    return RabbitQueue(name, queue_type=QueueType.QUORUM, durable=True, routing_key=name)


async def declare_dead_letters(broker: RabbitBroker, settings: Settings) -> None:
    dlx = await broker.declare_exchange(dead_letter_exchange(settings))
    dlq = await broker.declare_queue(dead_letter_queue(settings))
    await dlq.bind(dlx, routing_key=dead_letter_queue_name(settings))


async def declare_topology(broker: RabbitBroker, settings: Settings) -> None:
    await declare_dead_letters(broker, settings)
    exchange = await broker.declare_exchange(tasks_exchange(settings))
    queue = await broker.declare_queue(tasks_queue(settings))
    await queue.bind(exchange, routing_key=settings.tasks_queue)


async def publish_task(broker: RabbitBroker, settings: Settings, message: TaskMessage) -> None:
    try:
        confirmation = await broker.publish(
            message.model_dump_json(by_alias=True),
            exchange=tasks_exchange(settings),
            routing_key=settings.tasks_queue,
            persist=True,
            mandatory=True,
            message_id=str(message.task_id),
            content_type="application/json",
            timeout=settings.publish_timeout_seconds,
        )
    except TimeoutError as exc:
        raise TaskRejectedError("брокер не подтвердил публикацию вовремя") from exc
    except (AMQPError, ConnectionError) as exc:
        raise TaskRejectedError("брокер отклонил сообщение") from exc

    if isinstance(confirmation, DeliveredMessage):
        raise TaskRejectedError("сообщение не попало ни в одну очередь")
