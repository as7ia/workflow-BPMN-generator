# Engineering Rules

These rules apply to the entire repository.

## Scope and architecture

- Keep the backend modular: HTTP routing belongs in `app/api`, while future domain logic must live outside the transport layer.
- Mount public endpoints beneath the versioned `/api/v1` prefix.
- Keep application startup in `app/main.py` and avoid import-time side effects.
- Do not add BPMN generation, LLM integration, persistence, or external-service code until a mission explicitly requires it.
- Add a dependency only when the standard library or an existing dependency cannot reasonably satisfy the requirement.

## Python coding rules

- Support the Python version declared in `pyproject.toml`.
- Use type annotations for public functions and meaningful names for modules, functions, and variables.
- Prefer small, single-purpose modules and functions.
- Keep API response schemas explicit when endpoints grow beyond trivial health checks.
- Never hard-code credentials, tokens, secrets, or environment-specific URLs.
- Write all source code, API fields, documentation, and user-facing process input handling in English.

## API rules

- Preserve backward compatibility within an API version unless a breaking change is explicitly authorized.
- Return appropriate HTTP status codes and stable JSON response shapes.
- Keep FastAPI's OpenAPI and Swagger documentation enabled unless explicitly instructed otherwise.

## Testing rules

- Every endpoint or behavior change requires deterministic pytest coverage.
- Unit tests must not call external services, require credentials, use a GPU, or depend on network access.
- Use FastAPI's in-process test client for API tests.
- Do not use real time, randomness, or external state without deterministic control.
- Run the complete test suite before reporting completion; never claim tests passed unless the command was actually run successfully.
- A failing test must be fixed or reported exactly; do not hide, skip, or weaken tests merely to obtain a passing run.

## Delivery rules

- Keep changes limited to the active mission.
- Report every modified file and all remaining errors.
- Do not commit or push unless the user explicitly requests it.

