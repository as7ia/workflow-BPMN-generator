# BPMN Generation API

A modular FastAPI backend for a BPMN 2.0 generation engine. The current implementation provides service health checking, a typed intermediate process model, deterministic graph validation, semantic BPMN XML serialization, and an offline BPMN DI layout adapter. AI integration is not implemented yet.

The typed intermediate process representation and its deterministic validators are documented in [docs/intermediate-schema.md](docs/intermediate-schema.md).
The semantic XML serializer and its validation boundaries are documented in [docs/bpmn-serializer.md](docs/bpmn-serializer.md). AI process understanding remains out of scope.

## Requirements

- Python 3.11 or newer
- Node.js 20.12 or newer for BPMN DI layout

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

## Offline BPMN DI layout worker

Install the exact locked Node dependency once while package access is
available:

```powershell
Set-Location app\bpmn\layout_worker
npm.cmd ci --ignore-scripts
Set-Location ..\..\..
```

After installation, `app.bpmn.layout.layout_bpmn()` runs locally without
network access. It supports collapsed subprocesses only and rejects warnings,
semantic changes, expanded subprocesses, and incomplete BPMN DI.

The normal pytest suite mocks the worker. Run the real worker tests explicitly:

```powershell
$env:RUN_BPMN_LAYOUT_INTEGRATION = "1"
python -m pytest -m layout_integration tests/test_bpmn_layout_integration.py
```

