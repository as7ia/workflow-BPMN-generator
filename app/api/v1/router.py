"""Version 1 route aggregation."""

from fastapi import APIRouter

from app.api.v1.routes.generation import router as generation_router
from app.api.v1.routes.health import router as health_router
from app.api.v1.routes.text_generation import router as text_generation_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(generation_router)
api_router.include_router(text_generation_router)

