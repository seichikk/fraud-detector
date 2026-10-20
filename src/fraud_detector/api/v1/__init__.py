from fastapi import APIRouter

from fraud_detector.api.v1 import health, tasks, version

router = APIRouter(prefix="/api/v1")
router.include_router(version.router)
router.include_router(health.router)
router.include_router(tasks.router)
