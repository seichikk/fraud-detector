from fastapi import APIRouter, Response, status

from fraud_detector.api.deps import HealthChecksDep, SettingsDep
from fraud_detector.schemas.system import HealthResponse
from fraud_detector.services.health import run_checks

router = APIRouter(tags=["meta"])


@router.get("/health", responses={503: {"model": HealthResponse}})
async def read_health(
    checks: HealthChecksDep,
    settings: SettingsDep,
    response: Response,
) -> HealthResponse:
    components = await run_checks(checks, settings.health_timeout_seconds)
    everything_ok = all(component.status == "ok" for component in components)
    if not everything_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(status="ok" if everything_ok else "degraded", components=components)
