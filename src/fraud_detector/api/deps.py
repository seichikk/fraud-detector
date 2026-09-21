from typing import Annotated

from fastapi import Depends, Request

from fraud_detector.core.config import Settings
from fraud_detector.services.health import HealthCheck, PostgresCheck


def get_app_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_health_checks(request: Request) -> list[HealthCheck]:
    return [PostgresCheck(request.app.state.engine)]


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
HealthChecksDep = Annotated[list[HealthCheck], Depends(get_health_checks)]
