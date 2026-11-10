"""Version 1 JSON-to-BPMN generation endpoint."""

import asyncio
from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.api.errors import ErrorResponse, error_response
from app.bpmn import BpmnLayoutError, BpmnSerializationError
from app.domain import ProcessGraphValidationError, ProcessModel
from app.services import generate_bpmn

router = APIRouter(tags=["generation"])


class GenerationResponse(BaseModel):
    """Successful JSON-to-BPMN response."""

    status: Literal["success"]
    bpmn_xml: str


_ERROR_RESPONSES = {
    413: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
    500: {"model": ErrorResponse},
    502: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
    504: {"model": ErrorResponse},
}


def _layout_error_response(error: BpmnLayoutError):
    mappings = {
        "worker_unavailable": (503, "worker_unavailable", "The BPMN layout worker is unavailable."),
        "layout_timeout": (504, "layout_timeout", "BPMN layout exceeded the configured timeout."),
        "expanded_subprocess_unsupported": (422, "expanded_subprocess_unsupported", "Expanded subprocess layout is not supported."),
        "input_too_large": (413, "layout_input_too_large", "Serialized BPMN exceeds the layout input limit."),
    }
    status, code, message = mappings.get(
        error.code,
        (502, "layout_failed", "BPMN layout failed validation or execution."),
    )
    safe_details: dict[str, object] = {}
    if error.code == "layout_timeout" and "timeout_seconds" in error.details:
        safe_details["timeout_seconds"] = error.details["timeout_seconds"]
    if error.code == "input_too_large":
        for key in ("input_bytes", "max_input_bytes"):
            if key in error.details:
                safe_details[key] = error.details[key]
    if error.code == "expanded_subprocess_unsupported" and "subprocess_id" in error.details:
        safe_details["subprocess_id"] = error.details["subprocess_id"]
    return error_response(status, code, message, details=safe_details)


@router.post(
    "/generate-from-json",
    response_model=GenerationResponse,
    responses=_ERROR_RESPONSES,
)
async def generate_from_json(
    model: ProcessModel, request: Request
) -> GenerationResponse:
    """Generate fully laid-out BPMN XML from a schema 1.0 process model."""
    try:
        async with request.app.state.layout_semaphore:
            bpmn_xml = await asyncio.to_thread(
                generate_bpmn,
                model,
                timeout_seconds=request.app.state.layout_timeout_seconds,
            )
    except ProcessGraphValidationError as error:
        details = [
            {
                "code": issue.code,
                "location": list(issue.location),
                "message": issue.message,
            }
            for issue in error.issues
        ]
        return error_response(
            422,
            "graph_validation_failed",
            "The process graph is invalid.",
            details=details,
        )
    except BpmnSerializationError:
        return error_response(
            500,
            "serialization_failed",
            "The validated process could not be serialized.",
        )
    except BpmnLayoutError as error:
        return _layout_error_response(error)
    return GenerationResponse(status="success", bpmn_xml=bpmn_xml)
