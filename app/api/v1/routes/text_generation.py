"""Version 1 English-text-to-BPMN generation endpoint."""

import asyncio
from typing import Annotated, Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, StringConstraints

from app.api.errors import ErrorResponse, error_response
from app.bpmn import BpmnLayoutError, BpmnSerializationError
from app.domain import ProcessGraphValidationError
from app.llm import OllamaExtractionError, OllamaSettings, extract_process
from app.services import generate_bpmn

router = APIRouter(tags=["generation"])

Description = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=12_000),
]


class TextGenerationRequest(BaseModel):
    """Bounded English process description."""

    description: Description


class TextGenerationResponse(BaseModel):
    """Successful candidate BPMN response requiring human review."""

    status: Literal["success"]
    bpmn_xml: str
    review_required: Literal[True]


_ERROR_RESPONSES = {
    413: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
    500: {"model": ErrorResponse},
    502: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
    504: {"model": ErrorResponse},
}


def _extraction_error_response(error: OllamaExtractionError):
    mappings = {
        "unsupported_description": (
            422,
            "unsupported_description",
            "The description is ambiguous or unsupported by the current process model.",
        ),
        "input_too_large": (
            413,
            "description_too_large",
            "The description exceeds the configured extraction limit.",
        ),
        "model_unavailable": (
            503,
            "ollama_model_unavailable",
            "The configured Ollama model is unavailable.",
        ),
        "ollama_unavailable": (
            503,
            "ollama_unavailable",
            "The local Ollama service is unavailable.",
        ),
        "ollama_timeout": (
            504,
            "ollama_timeout",
            "Process extraction exceeded the configured timeout.",
        ),
        "schema_validation_failed": (
            502,
            "llm_schema_invalid",
            "The extracted process did not match ProcessModel schema 1.0.",
        ),
        "graph_validation_failed": (
            502,
            "llm_graph_invalid",
            "The extracted process graph failed deterministic validation.",
        ),
    }
    status, code, message = mappings.get(
        error.code,
        (502, "llm_output_invalid", "The local model returned unusable process data."),
    )
    safe_details: dict[str, object] = {}
    if error.code == "input_too_large" and "max_input_chars" in error.details:
        safe_details["max_input_chars"] = error.details["max_input_chars"]
    if error.code in {"schema_validation_failed", "graph_validation_failed"}:
        issues = error.details.get("issues")
        if isinstance(issues, list):
            safe_details["issues"] = issues
    return error_response(status, code, message, details=safe_details)


def _layout_error_response(error: BpmnLayoutError):
    mappings = {
        "worker_unavailable": (
            503,
            "worker_unavailable",
            "The BPMN layout worker is unavailable.",
        ),
        "layout_timeout": (
            504,
            "layout_timeout",
            "BPMN layout exceeded the configured timeout.",
        ),
        "input_too_large": (
            413,
            "layout_input_too_large",
            "Serialized BPMN exceeds the layout input limit.",
        ),
    }
    status, code, message = mappings.get(
        error.code,
        (502, "layout_failed", "BPMN layout failed validation or execution."),
    )
    safe_details: dict[str, object] = {}
    if error.code == "layout_timeout" and "timeout_seconds" in error.details:
        safe_details["timeout_seconds"] = error.details["timeout_seconds"]
    return error_response(status, code, message, details=safe_details)


@router.post(
    "/generate-from-text",
    response_model=TextGenerationResponse,
    responses=_ERROR_RESPONSES,
)
async def generate_from_text(
    payload: TextGenerationRequest, request: Request
) -> TextGenerationResponse:
    """Extract a candidate process and return laid-out BPMN requiring review."""
    try:
        settings = OllamaSettings.from_env()
    except (TypeError, ValueError):
        return error_response(
            500,
            "ollama_configuration_invalid",
            "The local Ollama configuration is invalid.",
        )

    try:
        async with request.app.state.llm_semaphore:
            model = await asyncio.to_thread(
                extract_process,
                payload.description,
                settings=settings,
            )
    except OllamaExtractionError as error:
        return _extraction_error_response(error)

    try:
        async with request.app.state.layout_semaphore:
            bpmn_xml = await asyncio.to_thread(
                generate_bpmn,
                model,
                timeout_seconds=request.app.state.layout_timeout_seconds,
            )
    except ProcessGraphValidationError:
        return error_response(
            502,
            "llm_graph_invalid",
            "The extracted process graph failed deterministic validation.",
        )
    except BpmnSerializationError:
        return error_response(
            500,
            "serialization_failed",
            "The validated process could not be serialized.",
        )
    except BpmnLayoutError as error:
        return _layout_error_response(error)

    return TextGenerationResponse(
        status="success",
        bpmn_xml=bpmn_xml,
        review_required=True,
    )
