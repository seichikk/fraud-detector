import argparse
import asyncio
import uuid
from datetime import UTC, datetime

from faststream.rabbit import RabbitBroker

from fraud_detector.core.config import get_settings
from fraud_detector.messaging import publish_task, tasks_exchange
from fraud_detector.schemas.tasks import TaskMessage, Transaction


async def send(kind: str) -> None:
    settings = get_settings()
    async with RabbitBroker(settings.rabbit_url) as broker:
        if kind == "garbage":
            await broker.publish(
                b"{not a json",
                exchange=tasks_exchange(settings),
                routing_key=settings.tasks_queue,
                persist=True,
            )
            print("отправлено сообщение не по схеме")
            return

        message = TaskMessage(
            task_id=uuid.uuid4(),
            source="api",
            created_at=datetime.now(UTC),
            transaction=Transaction(
                step=1,
                type="TRANSFER",
                amount=float("inf"),
                oldbalanceOrg=0,
                newbalanceOrig=0,
                oldbalanceDest=0,
                newbalanceDest=0,
            ),
        )
        await publish_task(broker, settings, message)
        print(f"отправлена ядовитая задача {message.task_id}: схема верная, модель её не посчитает")


def main() -> None:
    parser = argparse.ArgumentParser(description="отправка заведомо плохого сообщения в очередь")
    parser.add_argument("--kind", choices=["garbage", "poison"], default="garbage")
    asyncio.run(send(parser.parse_args().kind))


if __name__ == "__main__":
    main()
