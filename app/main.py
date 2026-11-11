"""FastAPI application entry point."""

import asyncio

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError

from app.api.errors import error_response
from app.api.middleware import RequestBodyLimitMiddleware
from app.api.v1.router import api_router


def create_app(
    *,
    layout_timeout_seconds: float = 10.0,
    max_concurrent_layouts: int = 2,
    max_concurrent_llm_calls: int = 1,
    max_request_bytes: int = 1_048_576,
) -> FastAPI:
    """Create and configure the FastAPI application."""
    if layout_timeout_seconds <= 0:
        raise ValueError("layout_timeout_seconds must be positive")
    if max_concurrent_layouts <= 0:
        raise ValueError("max_concurrent_layouts must be positive")
    if max_concurrent_llm_calls <= 0:
        raise ValueError("max_concurrent_llm_calls must be positive")
    if max_request_bytes <= 0:
        raise ValueError("max_request_bytes must be positive")
    application = FastAPI(
        title="BPMN Generation API",
        version="0.1.0",
        description="REST API for a BPMN 2.0 generation engine.",
    )
    application.state.layout_timeout_seconds = layout_timeout_seconds
    application.state.layout_semaphore = asyncio.Semaphore(max_concurrent_layouts)
    application.state.llm_semaphore = asyncio.Semaphore(max_concurrent_llm_calls)
    application.include_router(api_router, prefix="/api/v1")
    application.add_middleware(
        RequestBodyLimitMiddleware, max_body_bytes=max_request_bytes
    )

    @application.exception_handler(RequestValidationError)
    async def request_validation_error(
        _request: Request, error: RequestValidationError
    ):
        details = [
            {
                "location": list(item["loc"]),
                "message": item["msg"],
                "type": item["type"],
            }
            for item in error.errors()
        ]
        return error_response(
            422,
            "schema_validation_failed",
            "The request does not match ProcessModel schema 1.0.",
            details=details,
        )

    return application


app = create_app()

