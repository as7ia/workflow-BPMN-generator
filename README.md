# BPMN Generation API

A minimal, modular FastAPI backend that will host a BPMN 2.0 generation engine. This initial version provides only service health checking; BPMN generation and AI integration are intentionally out of scope.

## Requirements

- Python 3.11 or newer

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the application and test dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Run locally

Start the development server:

```powershell
python -m uvicorn app.main:app --reload
```

The API is available at `http://127.0.0.1:8000`. Useful routes:

- Health check: `http://127.0.0.1:8000/api/v1/health`
- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI schema: `http://127.0.0.1:8000/openapi.json`

## Tests

Run the deterministic test suite without external services:

```powershell
python -m pytest
```

