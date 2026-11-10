# Local Ollama process extraction

`app.llm.extract_process()` makes one bounded request to a configured Ollama
`/api/chat` endpoint. It asks for the current `ProcessModel` JSON Schema and
then independently performs JSON parsing, Pydantic validation, and deterministic
graph validation. Invalid output is rejected; it is never repaired.

The prompt maps only the supported schema 1.0 concepts: participants, lanes,
tasks, start and end events, exclusive and parallel gateways, descriptive
branch labels, sequence-flow loops, task/subprocess message endpoints, and
collapsed subprocess boundaries. It never requests BPMN XML or BPMN DI.

## Configuration

Copy `.env.example` values into the environment used to start the application.
The project does not load `.env` files automatically and stores no credentials.

| Variable | Default |
|---|---|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` |
| `OLLAMA_MODEL` | `qwen3:8b` |
| `OLLAMA_TIMEOUT_SECONDS` | `60` |
| `OLLAMA_MAX_INPUT_CHARS` | `12000` |
| `OLLAMA_MAX_RESPONSE_BYTES` | `2097152` |
| `OLLAMA_NUM_PREDICT` | `4096` |
| `OLLAMA_NUM_CTX` | `8192` |

The default endpoint is loopback-only. Changing `OLLAMA_BASE_URL` to a remote
host changes the trust boundary: process descriptions and generated candidates
will leave the local machine. Use only an explicitly trusted Ollama deployment.
URLs containing credentials are rejected.

## Model installation and offline operation

Install Ollama separately using its platform instructions. Model weights must
also be installed explicitly by the operator, for example:

```powershell
ollama pull qwen3:8b
```

The application never runs `ollama pull` and never downloads weights. After
Ollama and the selected model are installed, inference can operate offline on
the configured local service. GPU use is optional and is not required by the
adapter or its tests.

## Validity and fidelity

Schema validity means fields and types match `ProcessModel`. Graph validity
means references, ownership, reachability, gateway rules, and scope invariants
pass deterministic checks. Neither proves that the candidate faithfully
captures the author's intent. Ambiguous, implicit, or unsupported business
facts remain an LLM risk and require human review. The model is instructed to
refuse rather than guess, and refusals are returned as
`unsupported_description`.

This adapter does not claim complete BPMN 2.0 coverage. Expanded subprocesses,
event variants, boundary events, choreography, compensation, executable
conditions, and other concepts outside schema 1.0 are unsupported.

## Tests

Fast tests inject mocked transport responses and never start Ollama or load a
model:

```powershell
python -m pytest tests/test_ollama_extraction.py
```

Passing mocked tests demonstrates adapter behavior only. It does not
demonstrate English understanding or successful Qwen inference.
