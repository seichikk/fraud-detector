import argparse
import asyncio
import csv
import logging
import secrets
import signal
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from faststream.rabbit import RabbitBroker

from fraud_detector.core.config import Settings, get_settings
from fraud_detector.core.logs import setup_logging
from fraud_detector.messaging import TaskRejectedError, declare_topology, publish_task
from fraud_detector.schemas.tasks import TaskMessage, Transaction

logger = logging.getLogger(__name__)

TASK_NAMESPACE = uuid.UUID("5b1f3c3e-8a0e-4a8e-9c2a-7f6d4b1e2a90")
COLUMNS = [field.alias or name for name, field in Transaction.model_fields.items()]


class Backoff:
    def __init__(self, minimum: float, maximum: float) -> None:
        self.minimum = minimum
        self.maximum = maximum
        self.delay = 0.0

    def on_reject(self) -> float:
        self.delay = min(self.maximum, max(self.minimum, self.delay * 2))
        jitter = 0.5 + secrets.randbelow(501) / 1000
        return self.delay * jitter

    def on_success(self) -> None:
        self.delay = self.delay / 2 if self.delay / 2 >= self.minimum else 0.0


@dataclass
class SourceStats:
    published: int = 0
    rejected: int = 0
    pause_seconds: float = 0.0


def read_transactions(path: Path, limit: int | None = None) -> Iterator[Transaction]:
    with path.open(newline="", encoding="utf-8") as file:
        for number, row in enumerate(csv.DictReader(file)):
            if limit is not None and number >= limit:
                return
            yield Transaction.model_validate({column: row[column] for column in COLUMNS})


def task_id_for(batch_id: str, number: int) -> uuid.UUID:
    return uuid.uuid5(TASK_NAMESPACE, f"{batch_id}:{number}")


async def publish_with_backoff(
    broker: RabbitBroker,
    settings: Settings,
    message: TaskMessage,
    backoff: Backoff,
    stats: SourceStats,
    stop: asyncio.Event | None = None,
) -> None:
    while True:
        attempt = message.model_copy(update={"created_at": datetime.now(UTC)})
        try:
            await publish_task(broker, settings, attempt)
        except TaskRejectedError:
            stats.rejected += 1
            pause = backoff.on_reject()
            stats.pause_seconds = pause
            await asyncio.sleep(pause)
            if stop is not None and stop.is_set():
                return
        else:
            stats.published += 1
            backoff.on_success()
            stats.pause_seconds = backoff.delay
            return


async def run_batch(
    broker: RabbitBroker,
    settings: Settings,
    transactions: Iterator[Transaction],
    batch_id: str,
    stats: SourceStats,
    stop: asyncio.Event | None = None,
) -> None:
    backoff = Backoff(settings.source_min_backoff_seconds, settings.source_max_backoff_seconds)
    for number, transaction in enumerate(transactions):
        if stop is not None and stop.is_set():
            logger.info("получен сигнал остановки, источник прекращает чтение файла")
            return
        message = TaskMessage(
            task_id=task_id_for(batch_id, number),
            source="batch",
            created_at=datetime.now(UTC),
            transaction=transaction,
        )
        await publish_with_backoff(broker, settings, message, backoff, stats, stop)


async def write_metrics(stats: SourceStats, path: Path | None, period: float = 1.0) -> None:
    started = time.monotonic()
    last_moment, last_published = started, 0
    metrics = path.open("w", newline="", encoding="utf-8") if path else None
    writer = csv.writer(metrics, lineterminator="\n") if metrics else None
    if writer:
        writer.writerow(["elapsed", "published", "rejected", "rate", "pause_seconds"])

    def record() -> float:
        nonlocal last_moment, last_published
        now = time.monotonic()
        interval = now - last_moment
        rate = (stats.published - last_published) / interval if interval > 0.05 else 0.0
        last_moment, last_published = now, stats.published
        if writer and metrics:
            row = [round(now - started, 2), stats.published, stats.rejected, round(rate, 1)]
            writer.writerow([*row, round(stats.pause_seconds, 4)])
            metrics.flush()
        return rate

    try:
        while True:
            await asyncio.sleep(period)
            rate = record()
            if round(last_moment - started) % 5 == 0:
                logger.info(
                    "опубликовано %s, отказов брокера %s, скорость %.1f задач/с, пауза %.0f мс",
                    stats.published,
                    stats.rejected,
                    rate,
                    stats.pause_seconds * 1000,
                )
    finally:
        record()
        if metrics:
            metrics.close()


async def run_source(settings: Settings, args: argparse.Namespace) -> SourceStats:
    stats = SourceStats()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    batch_id = args.batch_id or f"{args.file.name}:{args.file.stat().st_size}"
    broker = RabbitBroker(settings.rabbit_url, log_level=logging.DEBUG)
    async with broker:
        await declare_topology(broker, settings)
        reporter = asyncio.create_task(write_metrics(stats, args.metrics))
        try:
            await run_batch(
                broker,
                settings,
                read_transactions(args.file, args.limit),
                batch_id,
                stats,
                stop,
            )
        finally:
            reporter.cancel()
    logger.info("источник закончил: опубликовано %s, отказов %s", stats.published, stats.rejected)
    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="пакетный скоринг csv через очередь задач")
    parser.add_argument("--file", type=Path, default=Path("data/raw/transactions.csv"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-id", default=None)
    parser.add_argument("--metrics", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    asyncio.run(run_source(settings, parse_args(argv)))
