from fastapi import APIRouter

from fraud_detector.api.deps import SettingsDep
from fraud_detector.core.meta import DISTRIBUTION_NAME, get_app_version
from fraud_detector.schemas.system import VersionResponse

router = APIRouter(tags=["meta"])


@router.get("/version")
async def read_version(settings: SettingsDep) -> VersionResponse:
    return VersionResponse(
        name=DISTRIBUTION_NAME,
        version=get_app_version(),
        commit=settings.git_sha,
    )
