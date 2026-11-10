"""Stable sanitized HTTP error responses."""

from typing import Any

from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


class ApiError(BaseModel):
    """Machine-readable API error description."""

    code: str
    message: str
    details: list[dict[str, Any]] | dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Top-level error response envelope."""

    error: ApiError


def error_response(
    status_code: int,
    code: str,
    message: str,
    *,
    details: list[dict[str, Any]] | dict[str, Any] | None = None,
) -> JSONResponse:
    """Build a consistent JSON error without internal exception data."""
    body = ErrorResponse(
        error=ApiError(code=code, message=message, details=details or {})
    )
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))
