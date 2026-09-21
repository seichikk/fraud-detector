from fastapi import APIRouter

from fraud_detector.schemas.system import LivenessResponse

router = APIRouter(tags=["system"])


@router.get("/healthz")
async def healthz() -> LivenessResponse:
    return LivenessResponse()
