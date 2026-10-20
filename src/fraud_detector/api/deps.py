from typing import Annotated

from fastapi import Depends, Request
from faststream.rabbit import RabbitBroker
from sqlalchemy.ext.asyncio import AsyncEngine

from fraud_detector.core.config import Settings
from fraud_detector.db.tasks import TaskRepository
from fraud_detector.services.health import HealthCheck, PostgresCheck, RabbitCheck


def get_app_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_engine(request: Request) -> AsyncEngine:
    engine: AsyncEngine = request.app.state.engine
    return engine


def get_broker(request: Request) -> RabbitBroker:
    broker: RabbitBroker = request.app.state.broker
    return broker


def get_task_repository(request: Request) -> TaskRepository:
    return TaskRepository(get_engine(request))


def get_health_checks(request: Request) -> list[HealthCheck]:
    settings = get_app_settings(request)
    return [PostgresCheck(get_engine(request)), RabbitCheck(settings.rabbit_url)]


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
EngineDep = Annotated[AsyncEngine, Depends(get_engine)]
BrokerDep = Annotated[RabbitBroker, Depends(get_broker)]
TaskRepositoryDep = Annotated[TaskRepository, Depends(get_task_repository)]
HealthChecksDep = Annotated[list[HealthCheck], Depends(get_health_checks)]
