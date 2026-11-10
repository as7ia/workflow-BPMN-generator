"""FastAPI application entry point."""

from fastapi import FastAPI

from app.api.v1.router import api_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title="BPMN Generation API",
        version="0.1.0",
        description="REST API for a BPMN 2.0 generation engine.",
    )
    application.include_router(api_router, prefix="/api/v1")
    return application


app = create_app()

