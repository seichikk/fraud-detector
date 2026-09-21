import asyncio
import logging
import time
from collections.abc import Sequence
from typing import Protocol

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from fraud_detector.schemas.system import ComponentHealth

logger = logging.getLogger(__name__)


class HealthCheck(Protocol):
    name: str

    async def fetch_version(self) -> str: ...


class PostgresCheck:
    name = "postgres"

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def fetch_version(self) -> str:
        async with self._engine.connect() as connection:
            result = await connection.execute(text("select current_setting('server_version')"))
            return str(result.scalar_one())


def elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


async def probe(check: HealthCheck, limit_seconds: float) -> ComponentHealth:
    started = time.perf_counter()
    try:
        async with asyncio.timeout(limit_seconds):
            version = await check.fetch_version()
    except TimeoutError:
        logger.warning("%s не ответил за %.1f с", check.name, limit_seconds)
        return ComponentHealth(
            name=check.name,
            status="error",
            latency_ms=elapsed_ms(started),
            error="превышено время ожидания",
        )
    except Exception as exc:
        logger.warning("%s недоступен: %r", check.name, exc)
        return ComponentHealth(
            name=check.name,
            status="error",
            latency_ms=elapsed_ms(started),
            error=type(exc).__name__,
        )
    return ComponentHealth(
        name=check.name,
        status="ok",
        version=version,
        latency_ms=elapsed_ms(started),
    )


async def run_checks(checks: Sequence[HealthCheck], limit_seconds: float) -> list[ComponentHealth]:
    return list(await asyncio.gather(*(probe(check, limit_seconds) for check in checks)))
